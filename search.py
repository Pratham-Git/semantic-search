import streamlit as st
import pandas as pd
import numpy as np
import faiss
import pickle
import re
from sentence_transformers import SentenceTransformer
import os
import ollama, json

class RestaurantSearchSystem:
    def __init__(self):
        self.model = None
        self.index = None
        self.metadata = None
        self.embeddings = None
        self.all_features = set()
        self.all_cuisines = set()

    def load_model(self):
        if self.model is None:
            with st.spinner("Loading AI model..."):
                self.model = SentenceTransformer('intfloat/e5-large-v2')
        return self.model

    def load_index(self, index_path="restaurant_search_index"):
        try:
            index_file = os.path.join(index_path, "index.faiss")
            metadata_file = os.path.join(index_path, "metadata.pkl")
            embeddings_file = os.path.join(index_path, "embeddings.pkl")
            features_file = os.path.join(index_path, "features.pkl")

            if not os.path.exists(index_file):
                return False, "Index file not found."
            self.index = faiss.read_index(index_file)
            with open(metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
            with open(embeddings_file, 'rb') as f:
                self.embeddings = pickle.load(f)
            with open(features_file, 'rb') as f:
                features_data = pickle.load(f)
                self.all_features = features_data['all_features']
                self.all_cuisines = features_data['all_cuisines']
            return True, f"Loaded {len(self.metadata)} restaurants"
        except Exception as e:
            return False, str(e)

    def extract_constraints(self, query):
        features_list = ['5-star dining', 'air condition', 'alcohol served', 'authentic japanese cuisine', 'award winners', 'bar', 'barbeque', 'bars & pubs', 'breakfast buffet', 'buffet', 'cafe', 'dance floor', 'dessert', 'disabled friendly', 'dj', 'eatout', 'exotic cocktails', 'formal attire', 'great breakfasts', 'happy hours', 'healthy food', 'home delivery', 'hookah', 'karaoke', 'kebabs', 'kids allowed', 'live kitchen', 'live music', 'live sports screening', 'luxury dining', 'mall parking', 'microbrewery', 'movies', 'new year', 'nightlife', 'outdoor seating', 'parking', 'pet friendly', 'pocket friendly', 'premium imported ingredients', 'pure veg', 'romantic', 'rooftops', 'sake collection', 'seafood', 'shisha', 'smoking area', 'sports bar', 'stags allowed', 'sunday brunches', 'take-away', 'thali', 'vaccinated staff', 'valet parking', 'vegan', 'wheelchair accessible']
        features_str = ", ".join(f'"{f}"' for f in features_list)

        response = ollama.chat(
            model="qwen2.5:7b",
            messages=[
                {
                    "role": "system",
                    "content": (
                    "You are a restaurant search query parser. Extract information and return valid JSON only.\n\n"
                    "Rules:\n"
                    "- If query mentions a single price/budget (e.g., '1k budget', 'priced at 1000') set BOTH cost_min AND cost_max to that value\n"
                    "- If query mentions a range (e.g., '500 to 1000'), set cost_min=500, cost_max=1000\n"
                    "- cost_min/cost_max: Convert 1k→1000, 2.5k→2500\n"
                    f"- features: Choose from these terms: {features_str}\n"
                    "- Only use features that exist in the list above\n\n"
                    "Return JSON with keys: cost_min, cost_max, cuisines, features, name"
                )
                },
                {"role": "user", "content": query}
            ]
        )
        raw_output = response["message"]["content"].strip()
        print(raw_output)
        cleaned_output = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if cleaned_output:
            raw_output = cleaned_output.group(0)

        try:
            data = json.loads(raw_output)
        except json.JSONDecodeError:
            data = {}

        if not data.get("cost_min") and not data.get("cost_max"):
            nums = re.findall(r"\d+", query)
            if nums:
                val = int(nums[0])
                data["cost_min"] = val
                data["cost_max"] = val

        return {
            "cost_min": data.get("cost_min"),
            "cost_max": data.get("cost_max"),
            "cuisines": data.get("cuisines", []),
            "features": data.get("features", []),
            "name": data.get("name")
        }



    def apply_constraints(self, constraints):
        valid_indices = []
        cost_min = constraints.get('cost_min')
        cost_max = constraints.get('cost_max')

        try:
            cost_min = float(cost_min) if cost_min is not None else None
        except (TypeError, ValueError):
            cost_min = None
        try:
            cost_max = float(cost_max) if cost_max is not None else None
        except (TypeError, ValueError):
            cost_max = None

        for i, meta in enumerate(self.metadata):
            try:
                cost = float(meta['cost_for_two']) 
            except (TypeError, ValueError):
                continue

            if cost_min is not None and cost < cost_min:
                continue
            if cost_max is not None and cost > cost_max:
                continue
            if constraints['features'] and not all(f in meta['features'].lower() for f in constraints['features']):
                continue
            if constraints['cuisines'] and not any(c in meta['cuisines'].lower() for c in constraints['cuisines']):
                continue

            valid_indices.append(i)

        if not valid_indices:
            valid_indices = list(range(len(self.metadata)))

        return valid_indices




    def search(self, query, top_k=10):
        if self.index is None or self.metadata is None: 
            return []
        model = self.load_model()
        constraints = self.extract_constraints(query)
        valid_indices = self.apply_constraints(constraints)
        if not valid_indices: 
            return []
        
        query_embedding = model.encode([query], convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        if len(valid_indices) == len(self.metadata):
            scores, indices = self.index.search(query_embedding, top_k)
            results_indices = indices[0]
        else:
            valid_embeddings = self.embeddings[valid_indices]
            temp_index = faiss.IndexFlatIP(valid_embeddings.shape[1])
            temp_index.add(valid_embeddings)
            scores, indices = temp_index.search(query_embedding, min(top_k, len(valid_indices)))
            results_indices = [valid_indices[idx] for idx in indices[0] if idx != -1]
        
        results = []
        for i, idx in enumerate(results_indices):
            if idx < len(self.metadata):
                meta = self.metadata[idx]
                try:
                    cost_for_two = float(meta['cost_for_two'])
                except (TypeError, ValueError):
                    cost_for_two = 0.0
                    
                results.append({
                    'restaurant_name': meta['restaurant_name'],
                    'cuisines': meta['cuisines'],
                    'features': meta['features'],
                    'cost_for_two': cost_for_two,
                    'similarity_score': float(scores[0][i]) if i < len(scores[0]) else 0.0
                })
        
        if constraints['cost_min'] and constraints['cost_max'] and constraints['cost_min'] == constraints['cost_max']:
            target_cost = constraints['cost_min']
            results.sort(key=lambda r: (abs(r['cost_for_two'] - target_cost), -r['similarity_score']))
        else:
            results.sort(key=lambda r: -r['similarity_score']) 

        return results


@st.cache_resource
def get_search_system():
    s = RestaurantSearchSystem()
    success, msg = s.load_index("restaurant_search_index")
    if success: 
        st.success(msg)
        return s
    else: 
        st.error(msg)
        st.stop()

def main():
    st.set_page_config(page_title="Restaurant Search", layout="centered")
    st.title("Restaurant Search")
    search_system = get_search_system()

    query = st.text_input("Search for restaurants:", placeholder="e.g., cheap pizza with parking or Domino's")
    if query:
        with st.spinner("Searching..."):
            results = search_system.search(query, top_k=10)
        if results:
            for i, r in enumerate(results, 1):
                st.markdown(f"### {i}. {r['restaurant_name']} — ₹{r['cost_for_two']:.0f}")
                st.write(f"**Cuisine:** {r['cuisines']}")
                st.write(f"**Features:** {r['features']}")
                st.caption(f"Score: {r['similarity_score']:.3f}")
                st.divider()
        else:
            st.warning("No results found.")

if __name__ == "__main__":
    main()
