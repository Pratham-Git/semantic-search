from openai import OpenAI
import pandas as pd
import numpy as np
import redis
import time
import logging
from redis.commands.search.field import VectorField, TextField, NumericField
from redis.commands.search.index_definition import IndexDefinition, IndexType
import pickle
import os
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedOpenAIEmbeddingBuilder:
    def __init__(self, api_key, redis_host="localhost", redis_port=6969):
        self.openai_client = OpenAI(api_key=api_key)
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
        self.index_name = "restaurant_idx_768"
        self.embedding_model = "text-embedding-3-small"
        self.embedding_dimensions = 768
        
        self.requests_per_minute = 3000  # Adjust as per your OpenAI tier
        self.tokens_per_minute = 1000000
        self.batch_size = 100
        self.max_retries = 3
        
        self.cache_file = "openai_embedding_cache.pkl"
        self.progress_file = "openai_embedding_progress.txt"
        
    def load_progress(self) -> int:
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return int(f.read().strip())
        return 0
    
    def save_progress(self, index: int) -> None:
        with open(self.progress_file, 'w') as f:
            f.write(str(index))
    
    def load_cached_embeddings(self) -> Dict:
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        return {}
    
    def save_embeddings_to_cache(self, embeddings_dict: Dict) -> None:
        with open(self.cache_file, 'wb') as f:
            pickle.dump(embeddings_dict, f)
    
    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4
    
    def get_embeddings_batch_optimized(self, texts: List[str], start_idx: int = 0) -> List[List[float]]:
        def get_single_batch(batch_texts: List[str], batch_idx: int) -> Optional[List[List[float]]]:
            for attempt in range(self.max_retries):
                try:
                    total_tokens = sum(self.estimate_tokens(text) for text in batch_texts)
                    logger.debug(f"Batch {batch_idx} estimated tokens: {total_tokens}")
                    
                    response = self.openai_client.embeddings.create(
                        model=self.embedding_model,
                        input=batch_texts,
                        encoding_format="float",
                        dimensions=self.embedding_dimensions
                    )
                    return [item.embedding for item in response.data]
                except Exception as e:
                    error_msg = str(e).lower()
                    if "rate limit" in error_msg or "quota" in error_msg:
                        wait_time = (attempt + 1) * 60
                        logger.warning(f"Rate limit hit on batch {batch_idx}, attempt {attempt + 1}/{self.max_retries}. Waiting {wait_time}s...")
                    elif "timeout" in error_msg:
                        wait_time = (attempt + 1) * 10
                        logger.warning(f"Timeout on batch {batch_idx}, attempt {attempt + 1}/{self.max_retries}. Waiting {wait_time}s...")
                    else:
                        wait_time = (attempt + 1) * 5
                        logger.warning(f"Error on batch {batch_idx}, attempt {attempt + 1}/{self.max_retries}: {e}. Waiting {wait_time}s...")
                    if attempt < self.max_retries - 1:
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed to process batch {batch_idx} after {self.max_retries} attempts")
                        raise e
            return None
        
        cached_embeddings = self.load_cached_embeddings()
        all_embeddings = []
        
        requests_this_minute = 0
        tokens_this_minute = 0
        minute_start_time = time.time()
        
        total_batches = (len(texts) - start_idx + self.batch_size - 1) // self.batch_size
        processed_batches = start_idx // self.batch_size
        
        logger.info(f"🚀 Processing {len(texts)} texts in batches of {self.batch_size}")
        logger.info(f"📋 Starting from index {start_idx} (batch {processed_batches + 1})")
        
        for i in range(0, start_idx, self.batch_size):
            batch_idx = i // self.batch_size
            cache_key = f"batch_{batch_idx}"
            if cache_key in cached_embeddings:
                all_embeddings.extend(cached_embeddings[cache_key])
        
        for i in range(start_idx, len(texts), self.batch_size):
            batch_idx = i // self.batch_size
            cache_key = f"batch_{batch_idx}"
            current_batch = batch_idx - processed_batches + 1
            
            if cache_key in cached_embeddings:
                logger.info(f"📋 Using cached batch {current_batch}/{total_batches} (batch_idx: {batch_idx})")
                all_embeddings.extend(cached_embeddings[cache_key])
                continue
            
            elapsed_time = time.time() - minute_start_time
            if elapsed_time >= 60:
                requests_this_minute = 0
                tokens_this_minute = 0
                minute_start_time = time.time()
            elif requests_this_minute >= self.requests_per_minute:
                sleep_time = 60 - elapsed_time
                logger.info(f"⏳ Rate limit reached, sleeping for {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                requests_this_minute = 0
                tokens_this_minute = 0
                minute_start_time = time.time()
            
            batch_texts = texts[i:i+self.batch_size]
            estimated_tokens = sum(self.estimate_tokens(text) for text in batch_texts)
            
            if tokens_this_minute + estimated_tokens > self.tokens_per_minute:
                sleep_time = 60 - elapsed_time
                logger.info(f"⏳ Token limit would be exceeded, sleeping for {sleep_time:.1f}s...")
                time.sleep(max(sleep_time, 1))
                requests_this_minute = 0
                tokens_this_minute = 0
                minute_start_time = time.time()
            
            try:
                batch_embeddings = get_single_batch(batch_texts, batch_idx)
                if batch_embeddings:
                    all_embeddings.extend(batch_embeddings)
                    cached_embeddings[cache_key] = batch_embeddings
                    self.save_progress(i + self.batch_size)
                    self.save_embeddings_to_cache(cached_embeddings)
                    requests_this_minute += 1
                    tokens_this_minute += estimated_tokens
                    logger.info(f"✅ Batch {current_batch}/{total_batches} completed "
                                f"(batch_idx: {batch_idx}, {len(batch_embeddings)} embeddings)")
                    time.sleep(0.1)
                else:
                    logger.error(f"❌ Failed to process batch {batch_idx}")
                    break
            except Exception as e:
                logger.error(f"❌ Critical error processing batch {batch_idx}: {e}")
                break
        
        logger.info(f"🎯 Generated {len(all_embeddings)} embeddings total")
        return all_embeddings
    
    def create_redis_schema_768(self) -> None:
        try:
            info = self.redis_client.ft(self.index_name).info()
            existing_docs = info.get('num_docs', 0)
            
            recreate = input(f"Index '{self.index_name}' exists with {existing_docs} documents. Recreate? (y/N): ").lower() == 'y'
            
            if recreate:
                logger.info("🗑️  Dropping existing index...")
                self.redis_client.ft(self.index_name).dropindex(delete_documents=True)
            else:
                logger.info("📋 Using existing index")
                return
        except Exception:
            logger.info("📋 Index doesn't exist, creating new one...")
        
        try:
            schema = (
                TextField("restaurant_name", sortable=True),
                TextField("restaurant_address"),
                TextField("location_city", sortable=True),
                TextField("region_name"),
                TextField("features"),
                TextField("cuisines", sortable=True),
                TextField("deal_title"),
                NumericField("cost_for_two", sortable=True),
                TextField("start_time"),
                TextField("end_time"),
                NumericField("latitude"),
                NumericField("longitude"),
                VectorField("embedding", "HNSW", {
                    "TYPE": "FLOAT32",
                    "DIM": self.embedding_dimensions,
                    "DISTANCE_METRIC": "COSINE",
                    "INITIAL_CAP": 20000,
                    "M": 16,
                    "EF_CONSTRUCTION": 200
                })
            )
            index_definition = IndexDefinition(
                prefix=["rest:"],
                index_type=IndexType.HASH
            )
            self.redis_client.ft(self.index_name).create_index(
                schema,
                definition=index_definition
            )
            logger.info(f"✅ Created Redis index '{self.index_name}' with {self.embedding_dimensions}D vectors")
        except Exception as e:
            logger.error(f"❌ Failed to create index: {e}")
            raise
    
    def build_optimized_index(self, df: pd.DataFrame) -> bool:
        logger.info(f"🚀 Starting index build for {len(df)} restaurants")
        
        texts = []
        for _, row in df.iterrows():
            text_parts = []
            
            if pd.notna(row.get('restaurant_name')):
                text_parts.append(f"Restaurant: {row['restaurant_name']}")
            if pd.notna(row.get('cuisines')):
                text_parts.append(f"Serves {row['cuisines']}")
            if pd.notna(row.get('features')):
                text_parts.append(f"Features: {row['features']}")
            if pd.notna(row.get('deal_title')):
                text_parts.append(f"Deal: {row['deal_title']}")
            if pd.notna(row.get('cost_for_two')):
                text_parts.append(f"Cost ₹{row['cost_for_two']} for two")
            if pd.notna(row.get('location_city')):
                text_parts.append(f"Located in {row['location_city']}")
            if pd.notna(row.get('region_name')):
                text_parts.append(f"Area: {row['region_name']}")
            
            texts.append(". ".join(text_parts))
        
        start_idx = self.load_progress()
        remaining_texts = len(texts) - start_idx
        total_batches = (remaining_texts + self.batch_size - 1) // self.batch_size
        estimated_minutes = total_batches * 60 / self.requests_per_minute
        
        if remaining_texts > 0:
            logger.info(f"📊 Estimation:")
            logger.info(f"   • Remaining texts: {remaining_texts}")
            logger.info(f"   • Batches needed: {total_batches}")
            logger.info(f"   • Estimated time: {estimated_minutes:.1f} minutes")
            
            if estimated_minutes > 5:
                proceed = input(f"This will take approximately {estimated_minutes:.1f} minutes. Continue? (y/N): ")
                if proceed.lower() != 'y':
                    logger.info("❌ Cancelled by user")
                    return False
        
        self.create_redis_schema_768()
        
        logger.info("🔄 Generating embeddings...")
        embeddings = self.get_embeddings_batch_optimized(texts, start_idx)
        
        if len(embeddings) < len(texts):
            logger.error(f"❌ Only generated {len(embeddings)}/{len(texts)} embeddings")
            return False
        
        logger.info("📤 Uploading to Redis...")
        success = self.upload_to_redis(df, embeddings)
        
        if success:
            for cache_file in [self.cache_file, self.progress_file]:
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                    logger.info(f"🗑️  Removed {cache_file}")
            logger.info("🎉 Index build completed successfully!")
            self.verify_index()
        
        return success
    
    def upload_to_redis(self, df: pd.DataFrame, embeddings: List[List[float]]) -> bool:
        try:
            pipe = self.redis_client.pipeline(transaction=False)
            upload_count = 0
            
            for idx, (_, row) in enumerate(df.iterrows()):
                if idx >= len(embeddings):
                    logger.warning(f"⚠️  Stopping upload at index {idx}, no more embeddings available")
                    break
                
                doc_data = {
                    "restaurant_name": str(row.get('restaurant_name', '')),
                    "restaurant_address": str(row.get('restaurant_address', '')),
                    "location_city": str(row.get('location_city', '')),
                    "region_name": str(row.get('region_name', '')),
                    "features": str(row.get('features', '')),
                    "cuisines": str(row.get('cuisines', '')),
                    "deal_title": str(row.get('deal_title', '')),
                    "start_time": str(row.get('start_time', '')),
                    "end_time": str(row.get('end_time', '')),
                    "cost_for_two": float(row['cost_for_two']) if pd.notna(row.get('cost_for_two')) else 0.0,
                    "latitude": float(row.get('latitude', 0)) if pd.notna(row.get('latitude')) else 0.0,
                    "longitude": float(row.get('longitude', 0)) if pd.notna(row.get('longitude')) else 0.0,
                    "embedding": np.array(embeddings[idx], dtype='float32').tobytes()
                }
                
                doc_id = f"rest:{row.get('restaurant_id', idx)}"
                pipe.hset(doc_id, mapping=doc_data)
                upload_count += 1
                
                if upload_count % 1000 == 0:
                    pipe.execute()
                    pipe = self.redis_client.pipeline(transaction=False)
                    logger.info(f"📤 Uploaded {upload_count}/{len(df)} documents...")
            
            if upload_count % 1000 != 0:
                pipe.execute()
            
            logger.info(f"✅ Successfully uploaded {upload_count} documents to Redis")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to upload to Redis: {e}")
            return False
    
    def verify_index(self) -> None:
        try:
            info = self.redis_client.ft(self.index_name).info()
            doc_count = info.get('num_docs', 0)
            logger.info(f"✅ Index verification: {doc_count} documents indexed")
            
            from redis.commands.search.query import Query
            test_query = Query("*").paging(0, 1)
            result = self.redis_client.ft(self.index_name).search(test_query)
            
            if result.docs:
                logger.info(f"✅ Sample document: {result.docs[0].id}")
            
        except Exception as e:
            logger.warning(f"⚠️  Index verification failed: {e}")

def quick_test_build(api_key: str, sample_size: int = 100, redis_host: str = "localhost", redis_port: int = 6969) -> bool:
    try:
        df = pd.read_csv("restaurants_data.csv").head(sample_size)
        builder = OptimizedOpenAIEmbeddingBuilder(api_key, redis_host, redis_port)
        return builder.build_optimized_index(df)
    except Exception as e:
        logger.error(f"❌ Quick test failed: {e}")
        return False

if __name__ == "__main__":
    API_KEY = os.getenv("OPENAI_API_KEY")
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in environment")

    try:
        df = pd.read_csv("C:/Users/jaisw/semantic-search/final/restaurants_data.csv")
        builder = OptimizedOpenAIEmbeddingBuilder(API_KEY, redis_host="localhost", redis_port=6969)
        success = builder.build_optimized_index(df)

        if success:
            logger.info("🎯 SUCCESS! Restaurant search index is ready!")
        else:
            logger.error("❌ FAILED! Check the logs above for details.")
    except FileNotFoundError:
        logger.error("❌ restaurants_data.csv not found. Please ensure the file exists.")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
