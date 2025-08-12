import pandas as pd
import numpy as np
import redis
import json
import re
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
import pytz
from redis.commands.search.query import Query
from openai import OpenAI
import google.generativeai as genai
import os

def safe_str(value, default=None):
    if value is None or pd.isna(value): return default
    return value.decode('utf-8') if isinstance(value, bytes) else str(value)

def safe_float(value, default=None):
    if value is None or pd.isna(value): return default
    if isinstance(value, bytes):
        try: value = value.decode('utf-8')
        except: return default
    try: return float(value)
    except: return default

def parse_time(time_str):
    if not isinstance(time_str, str): return None
    for fmt in ('%H:%M', '%H:%M:%S'):
        try: return datetime.strptime(time_str.strip(), fmt).time()
        except: continue
    return None

def is_currently_open(start_str, end_str, now_time):
    start, end = parse_time(start_str), parse_time(end_str)
    if start is None or end is None: return None
    return start <= now_time < end if start <= end else now_time >= start or now_time < end

class RestaurantSearchSystem:
    def __init__(self, redis_host="serene_euclid", redis_port=6379, redis_password=None, openai_api_key=None, gemini_api_key=None):
        # Initialize OpenAI for embeddings only
        openai_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not openai_key:
            raise ValueError("OpenAI API key required for embeddings. Set OPENAI_API_KEY environment variable.")
        
        # Initialize Gemini for chat/prompts
        gemini_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        if not gemini_key:
            raise ValueError("Gemini API key required for chat. Set GEMINI_API_KEY environment variable.")
        
        self.openai_client = OpenAI(api_key=openai_key)
        genai.configure(api_key=gemini_key)
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, password=redis_password, decode_responses=False)
        self.index_name = "restaurant_idx_768"
        self.embedding_model = "text-embedding-3-small"
        self.verify_system_ready()

    def verify_system_ready(self):
        try:
            info = self.redis_client.ft(self.index_name).info()
            doc_count = info.get('num_docs', 0)
            print(f"✅ Index '{self.index_name}' contains: {doc_count} documents")
            return doc_count > 0
        except Exception as e: 
            print(f"❌ System verification failed: {e}")
            return False

    def extract_constraints(self, query):
        try:
            prompt = f"""Parse this restaurant query and return JSON only: '{query}'
                        
                        Extract these constraints:
                        - cost_min/cost_max: For "under 2000" set cost_max: 2000, for "above 1500" set cost_min: 1500
                        - sort_by: "cost_asc" for cheap/budget, "cost_desc" for expensive/premium  
                        - cuisines: List of cuisine types mentioned
                        - features: List of any restaurant features/amenities mentioned (parking, bar, outdoor seating, etc.)
                        - name: Specific restaurant name if mentioned
                        - is_open_now: true if "open now" mentioned
                        - rewritten_query: Clean search query for embedding
                        
                        Return only valid JSON without markdown formatting or explanations."""

            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0,
                    max_output_tokens=300,
                )
            )
            
            response_text = response.text.strip()
            cleaned = re.search(r"\{.*\}", response_text, re.DOTALL)
            constraints = json.loads(cleaned.group(0)) if cleaned else {}
            print(f"🔍 Extracted: {constraints}")
            return constraints
            
        except Exception as e:
            print(f"❌ Constraint extraction failed: {e}")
            return {"rewritten_query": query}

    def embed_query(self, text):
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model, input=text, 
                encoding_format="float", dimensions=768
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            return embedding / np.linalg.norm(embedding)
        except Exception as e:
            print(f"❌ Embedding failed: {e}")
            return np.zeros(768, dtype=np.float32)

    def calculate_distance_km(self, user_lat, user_lon, rest_lat, rest_lon):
        try:
            coords = [float(c) for c in [user_lat, user_lon, rest_lat, rest_lon]]
            R = 6371.0
            lat1_rad, lon1_rad, lat2_rad, lon2_rad = [radians(c) for c in coords]
            dlon, dlat = lon2_rad - lon1_rad, lat2_rad - lat1_rad
            a = sin(dlat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)**2
            return round(R * 2 * atan2(sqrt(a), sqrt(1-a)), 2)
        except: 
            return None

    def _sanitize_record(self, record):
        sanitized = {(k.decode('utf-8') if isinstance(k, bytes) else k): 
                    (v.decode('utf-8') if isinstance(v, bytes) else v) for k, v in record.items()}
        return {
            "restaurant_name": safe_str(sanitized.get("restaurant_name")),
            "cost_for_two": safe_float(sanitized.get("cost_for_two")),
            "cuisines": safe_str(sanitized.get("cuisines")),
            "features": safe_str(sanitized.get("features")),
            "location_city": safe_str(sanitized.get("location_city")),
            "region_name": safe_str(sanitized.get("region_name")),
            "address": safe_str(sanitized.get("address")),
            "start_time": safe_str(sanitized.get("start_time")),
            "end_time": safe_str(sanitized.get("end_time")),
            "latitude": safe_float(sanitized.get("latitude")),
            "longitude": safe_float(sanitized.get("longitude")),
        }

    def _passes_filters(self, meta, constraints, now_ist):
        # Cost filtering
        cost = meta.get('cost_for_two')
        if constraints.get('cost_min') and (cost is None or cost < constraints['cost_min']): return False
        if constraints.get('cost_max') and (cost is None or cost > constraints['cost_max']): return False
        
        # Time filtering
        if constraints.get("is_open_now") and not is_currently_open(meta.get('start_time'), meta.get('end_time'), now_ist): 
            return False
        
        # Name filtering
        if constraints.get('name') and constraints['name'].lower() not in safe_str(meta.get('restaurant_name'), '').lower(): 
            return False
        
        # Feature filtering - flexible matching
        required_features = constraints.get('features')
        if required_features:
            restaurant_features = safe_str(meta.get('features'), '').lower()
            required_features = [required_features] if isinstance(required_features, str) else required_features
            
            # Check if any required feature matches (partial matching)
            matched = False
            for feature in required_features:
                if feature and any(keyword.lower() in restaurant_features for keyword in feature.split()):
                    matched = True
                    break
            
            if not matched:
                print(f"❌ {meta.get('restaurant_name')}: missing features {required_features}")
                return False
        
        # Cuisine filtering - flexible matching  
        required_cuisines = constraints.get('cuisines')
        if required_cuisines:
            restaurant_cuisines = safe_str(meta.get('cuisines'), '').lower()
            required_cuisines = [required_cuisines] if isinstance(required_cuisines, str) else required_cuisines
            
            # Check if any required cuisine matches
            matched = False
            for cuisine in required_cuisines:
                if cuisine and cuisine.lower() in restaurant_cuisines:
                    matched = True
                    break
                    
            if not matched:
                print(f"❌ {meta.get('restaurant_name')}: missing cuisines {required_cuisines}")
                return False
        
        return True

    def search(self, query, user_lat=None, user_lon=None, top_k=10, debug=True):
        try:
            print(f"🔍 Searching: '{query}'")
            
            if not self.verify_system_ready():
                return []
            
            constraints = self.extract_constraints(query)
            refined_query = constraints.get("rewritten_query", query)
            
            # Generate embedding for semantic search using OpenAI
            query_embedding = self.embed_query(refined_query)
            if np.allclose(query_embedding, 0): 
                return []
            
            # Vector search
            search_limit = min(500, top_k * 20)
            q = Query(f"*=>[KNN {search_limit} @embedding $vec_param AS vector_score]")\
                .return_fields("restaurant_name", "cost_for_two", "cuisines", "features", 
                              "location_city", "region_name", "address", "start_time", 
                              "end_time", "latitude", "longitude", "vector_score")\
                .dialect(2).paging(0, search_limit).sort_by("vector_score")
            
            result = self.redis_client.ft(self.index_name).search(
                q, query_params={"vec_param": query_embedding.tobytes()}
            )
            
            print(f"📊 Found {len(result.docs)} candidates")
            
            now_ist = datetime.now(pytz.timezone('Asia/Kolkata')).time()
            results = []
            
            for doc in result.docs:
                meta = {k: doc.__dict__.get(k) for k in doc.__dict__ if k != 'id'}
                sanitized_meta = self._sanitize_record(meta)
                
                if not self._passes_filters(sanitized_meta, constraints, now_ist): 
                    continue
                
                # Calculate distance if coordinates provided
                dist = None
                if all([user_lat, user_lon, sanitized_meta.get('latitude'), sanitized_meta.get('longitude')]):
                    dist = self.calculate_distance_km(
                        user_lat, user_lon, 
                        sanitized_meta['latitude'], sanitized_meta['longitude']
                    )
                
                results.append({
                    **sanitized_meta, 
                    'similarity_score': float(getattr(doc, 'vector_score', 0)), 
                    'distance_km': dist
                })
            
            print(f"✅ {len(results)} restaurants after filtering")
            
            # Sort results
            sort_by = constraints.get("sort_by")
            if sort_by == 'cost_asc':
                results.sort(key=lambda r: r.get('cost_for_two') or float('inf'))
            elif sort_by == 'cost_desc':
                results.sort(key=lambda r: r.get('cost_for_two') or 0, reverse=True)
            else:
                # Sort by distance first (if available), then similarity
                results.sort(key=lambda r: (
                    r['distance_km'] if r['distance_km'] is not None else 999,
                    -r['similarity_score']
                ))
            
            return results[:top_k]
        
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []

    def get_restaurant_recommendations(self, query, user_lat=None, user_lon=None, top_k=10):
        results = self.search(query, user_lat, user_lon, top_k, debug=True)
        return {
            "query": query, 
            "total_results": len(results),
            "user_location": {"latitude": user_lat, "longitude": user_lon} if user_lat and user_lon else None,
            "restaurants": results,
            "search_metadata": {
                "search_time": datetime.now().isoformat(), 
                "index_used": self.index_name,
                "embedding_model": f"OpenAI {self.embedding_model}", 
                "query_processing": "Google Gemini 1.5 Flash"
            }
        }

    def test_search_system(self):
        test_queries = [
            "italian restaurants with parking", 
            "cheap chinese food under 1000", 
            "restaurants with bar and outdoor seating", 
            "romantic fine dining"
        ]
        
        for query in test_queries:
            print(f"\n{'='*50}")
            print(f"🧪 Testing: '{query}'")
            print('='*50)
            
            results = self.search(query, top_k=3, debug=True)
            for i, r in enumerate(results, 1):
                print(f"   {i}. {r.get('restaurant_name')} - {r.get('cuisines')} "
                      f"(₹{r.get('cost_for_two')}) - Score: {r.get('similarity_score', 0):.3f}")

# Example usage
if __name__ == "__main__":
    try:
        search_system = RestaurantSearchSystem()
        
        # Test the system
        results = search_system.get_restaurant_recommendations(
            "best pizza place with outdoor seating and parking", 
            user_lat=28.6139, user_lon=77.2090, top_k=5
        )
        
        print(f"\n🎯 Found {results['total_results']} restaurants:")
        for i, restaurant in enumerate(results['restaurants'], 1):
            name = restaurant.get('restaurant_name')
            cuisines = restaurant.get('cuisines')
            cost = restaurant.get('cost_for_two')
            score = restaurant.get('similarity_score', 0)
            distance = restaurant.get('distance_km')
            
            dist_str = f" ({distance}km)" if distance else ""
            print(f"{i}. {name} - {cuisines} (₹{cost}){dist_str} - Score: {score:.3f}")
        
        print("✅ Search system working!")
        
    except Exception as e: 
        print(f"❌ System test failed: {e}")
        import traceback
        traceback.print_exc()