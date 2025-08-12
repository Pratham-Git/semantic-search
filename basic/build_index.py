!pip install sentence-transformers faiss-cpu pandas numpy

import pandas as pd
import numpy as np
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from google.colab import files as colab_files
import zipfile

class RestaurantIndexBuilder:
    def __init__(self):
        print("Loading sentence transformer model on GPU...")
        self.model = SentenceTransformer('intfloat/e5-large-v2', device='cuda')
        self.all_features = set()
        self.all_cuisines = set()

    def extract_all_features_and_cuisines(self, df):
        """Extract all unique features and cuisines from the dataset"""
        all_features = set()
        all_cuisines = set()
        print("Extracting features and cuisines...")

        for features in df['features'].dropna():
            feature_list = str(features).lower().replace(',', '|').replace(';', '|').replace(' and ', '|').split('|')
            for feature in feature_list:
                feature = feature.strip()
                if feature and len(feature) > 2:
                    all_features.add(feature)

        for cuisines in df['cuisines'].dropna():
            cuisine_list = str(cuisines).lower().replace(',', '|').replace(';', '|').replace(' and ', '|').split('|')
            for cuisine in cuisine_list:
                cuisine = cuisine.strip()
                if cuisine and len(cuisine) > 2:
                    all_cuisines.add(cuisine)

        self.all_features = all_features
        self.all_cuisines = all_cuisines

        print(f"Found {len(all_features)} unique features and {len(all_cuisines)} cuisine types")
        return all_features, all_cuisines

    def create_searchable_text(self, row):
        """Create ONLY restaurant-specific text"""
        parts = []
        if pd.notna(row['restaurant_name']):
            parts.append(f"Restaurant: {row['restaurant_name']}")
        if pd.notna(row['cuisines']):
            parts.append(f"Serves {str(row['cuisines']).lower()} cuisine food")
        if pd.notna(row['features']):
            parts.append(f"Has {str(row['features']).lower()}")
        if pd.notna(row['cost_for_two']):
            cost = float(row['cost_for_two'])
            cost_desc = self.get_cost_description(cost)
            parts.append(f"Price range is {cost_desc}, costs ₹{cost} for two people")
        return ". ".join(parts)

    def get_cost_description(self, cost):
        """Convert cost to natural language descriptions"""
        if cost < 500:
            return "very cheap and budget-friendly"
        elif cost < 1000:
            return "affordable and moderately priced"
        elif cost < 2000:
            return "mid-range pricing"
        elif cost < 3000:
            return "expensive and premium"
        else:
            return "very expensive luxury dining"

    def build_and_save_index(self, df, save_path="restaurant_search_index"):
        """Build the complete search index and save all components"""
        print("Starting index building process...")
        self.extract_all_features_and_cuisines(df)

        print("Creating restaurant-specific descriptions...")
        searchable_texts = []
        metadata = []
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"Processing restaurant {idx}/{len(df)}")
            searchable_text = self.create_searchable_text(row)
            searchable_texts.append(searchable_text)
            metadata.append({
                'restaurant_name': row['restaurant_name'],
                'cuisines': str(row['cuisines']) if pd.notna(row['cuisines']) else '',
                'features': str(row['features']) if pd.notna(row['features']) else '',
                'cost_for_two': float(row['cost_for_two']) if pd.notna(row['cost_for_two']) else 0,
                'searchable_text': searchable_text
            })

        print("Generating embeddings on GPU...")
        batch_size = 128 
        embeddings = []
        for i in range(0, len(searchable_texts), batch_size):
            batch = searchable_texts[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(searchable_texts) + batch_size - 1)//batch_size}")
            batch_embeddings = self.model.encode(
                batch,
                convert_to_tensor=True,
                device='cuda',
                show_progress_bar=True
            ).cpu().numpy()
            embeddings.extend(batch_embeddings)

        embeddings = np.array(embeddings).astype('float32')
        print(f"Generated embeddings shape: {embeddings.shape}")

        print("Creating FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

        os.makedirs(save_path, exist_ok=True)

        print("Saving index components...")
        faiss.write_index(index, f"{save_path}/index.faiss")
        with open(f"{save_path}/metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        with open(f"{save_path}/embeddings.pkl", 'wb') as f:
            pickle.dump(embeddings, f)
        with open(f"{save_path}/features.pkl", 'wb') as f:
            pickle.dump({'all_features': self.all_features, 'all_cuisines': self.all_cuisines}, f)
        with open(f"{save_path}/model_info.txt", 'w') as f:
            f.write(f"Model: intfloat/e5-large-v2\n")
            f.write(f"Dataset size: {len(df)} restaurants\n")
            f.write(f"Embedding dimension: {dimension}\n")

        print(f"Index saved successfully to {save_path}/")
        return save_path

def main():
    print("=== Restaurant Search Index Builder for Google Colab (FAST GPU) ===")
    print("Please upload your restaurants_data.csv file:")
    uploaded = colab_files.upload()
    csv_filename = list(uploaded.keys())[0]
    print(f"Uploaded file: {csv_filename}")

    try:
        df = pd.read_csv(csv_filename)
        print(f"Loaded {len(df)} restaurants")
        print(f"Columns: {list(df.columns)}")
        required_cols = ['restaurant_name', 'cuisines', 'features', 'cost_for_two']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"ERROR: Missing required columns: {missing_cols}")
            return
    except Exception as e:
        print(f"Error loading CSV: {str(e)}")
        return

    builder = RestaurantIndexBuilder()
    save_path = builder.build_and_save_index(df)

    zip_filename = "restaurant_search_index.zip"
    print(f"Creating zip file: {zip_filename}")
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for root, dirs, file_list in os.walk(save_path):
            for file in file_list:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, save_path)
                zipf.write(file_path, arcname)

    print(f"Index built successfully! Downloading {zip_filename}...")
    colab_files.download(zip_filename)

if __name__ == "__main__":
    main()
