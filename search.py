import streamlit as st
import pandas as pd
import numpy as np
import faiss
import pickle
import re
from sentence_transformers import SentenceTransformer
import os
import ollama
import json

class RestaurantSearchSystem:
    def __init__(self):
        """Initializes the search system components."""
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
            return False, f"Error loading index: {str(e)}"

    def extract_constraints(self, query):
        features_list = ['5-star dining', 'air condition', 'alcohol served', 'authentic japanese cuisine', 'award winners', 'bar', 'barbeque', 'bars & pubs', 'breakfast buffet', 'buffet', 'cafe', 'dance floor', 'dessert', 'disabled friendly', 'dj', 'eatout', 'exotic cocktails', 'formal attire', 'great breakfasts', 'happy hours', 'healthy food', 'home delivery', 'hookah', 'karaoke', 'kebabs', 'kids allowed', 'live kitchen', 'live music', 'live sports screening', 'luxury dining', 'mall parking', 'microbrewery', 'movies', 'new year', 'nightlife', 'outdoor seating', 'parking', 'pet friendly', 'pocket friendly', 'premium imported ingredients', 'pure veg', 'romantic', 'rooftops', 'sake collection', 'seafood', 'shisha', 'smoking area', 'sports bar', 'stags allowed', 'sunday brunches', 'take-away', 'thali', 'vaccinated staff', 'valet parking', 'vegan', 'wheelchair accessible']
        features_str = ", ".join(f'"{f}"' for f in features_list)

        try:
            response = ollama.chat(
                model="qwen2.5:7b",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a restaurant search query parser. Extract information and return valid JSON only.\n\n"
                            "Rules:\n"
                            "1.  **Price Filters (Specific Numbers):**\n"
                            "    - If a user gives a single number (e.g., 'for 1500', 'around 1k'), you MUST set BOTH `cost_min` AND `cost_max` to that exact value.\n"
                            "    - If a user gives a range (e.g., 'under 2000', 'between 1000-2000'), set `cost_min` and `cost_max` accordingly.\n"
                            "2.  **Sorting Preferences (Vague Terms):**\n"
                            "    - If a user uses a vague price term, set the `sort_by` field and leave `cost_min`/`cost_max` as null.\n"
                            "    - 'cheap', 'affordable', 'budget' -> `sort_by: 'cost_asc'`\n"
                            "    - 'mediocre', 'mid-range', 'average price' -> `sort_by: 'cost_mid'`\n"
                            "    - 'expensive', 'fancy', 'luxury' -> `sort_by: 'cost_desc'`\n"
                            "3.  **Output Structure:**\n"
                            "    - You MUST return a JSON object containing ALL of the following keys, using `null` if a value is not present: `cost_min`, `cost_max`, `sort_by`, `cuisines`, `features`, `name`.\n"
                            "4.  **Features:**\n"
                            f"   - Choose from these terms only: {features_str}\n"
                        )
                    },
                    {"role": "user", "content": query}
                ]
            )
            raw_output = response["message"]["content"].strip()
            print(f"Raw output from LLM: {raw_output}")
            
            cleaned_output = re.search(r"\{.*\}", raw_output, re.DOTALL)
            if cleaned_output:
                raw_output = cleaned_output.group(0)

            data = json.loads(raw_output)
        except Exception as e:
            st.error(f"Could not parse constraints from LLM: {e}")
            data = {}

        cuisines_list = data.get("cuisines") or []
        features_list = data.get("features") or []

        return {
            "cost_min": data.get("cost_min"),
            "cost_max": data.get("cost_max"),
            "sort_by": data.get("sort_by"),
            "cuisines": [c.lower() for c in cuisines_list],
            "features": [f.lower() for f in features_list],
            "name": data.get("name")
        }

    def search(self, query, top_k=10):
        if self.index is None or self.metadata is None:
            return []

        model = self.load_model()
        constraints = self.extract_constraints(query)
        
        query_embedding = model.encode([query], convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        filtered_results = []
        processed_indices = set()
        
        num_candidates = max(top_k * 20, 200)
        max_search_size = self.index.ntotal // 2 

        while len(filtered_results) < top_k and num_candidates <= max_search_size:
            initial_scores, initial_indices = self.index.search(query_embedding, num_candidates)
            
            cost_min = constraints.get('cost_min')
            cost_max = constraints.get('cost_max')

            for score, idx in zip(initial_scores[0], initial_indices[0]):
                if idx == -1 or idx in processed_indices:
                    continue
                
                processed_indices.add(idx)
                meta = self.metadata[idx]
                
                try:
                    cost = float(meta['cost_for_two'])
                    if cost_min is not None and cost < float(cost_min): continue
                    if cost_max is not None and cost > float(cost_max): continue
                except (TypeError, ValueError):
                    if cost_min is not None or cost_max is not None: continue


                meta_features = meta.get('features', '').lower()
                if constraints['features'] and not all(f in meta_features for f in constraints['features']):
                    continue

                meta_cuisines = meta.get('cuisines', '').lower()
                if constraints['cuisines'] and not any(c in meta_cuisines for c in constraints['cuisines']):
                    continue

                filtered_results.append({
                    'restaurant_name': meta['restaurant_name'],
                    'cuisines': meta['cuisines'],
                    'features': meta['features'],
                    'cost_for_two': float(meta.get('cost_for_two', 0.0)),
                    'similarity_score': float(score)
                })

            if len(filtered_results) >= top_k:
                break
            
            num_candidates *= 2

        sort_by = constraints.get('sort_by')
        cost_min = constraints.get('cost_min')
        cost_max = constraints.get('cost_max')

        if cost_min is not None and cost_min == cost_max:
            target_cost = float(cost_min)
            filtered_results.sort(key=lambda r: (abs(r['cost_for_two'] - target_cost), -r['similarity_score']))
        elif sort_by == 'cost_asc':
            filtered_results.sort(key=lambda r: (r['cost_for_two'], -r['similarity_score']))
        elif sort_by == 'cost_desc':
            filtered_results.sort(key=lambda r: (-r['cost_for_two'], -r['similarity_score']))
        elif sort_by == 'cost_mid':
            middle_cost = 1200.0 
            filtered_results.sort(key=lambda r: (abs(r['cost_for_two'] - middle_cost), -r['similarity_score']))
        else:
            filtered_results.sort(key=lambda r: -r['similarity_score'])

        return filtered_results[:top_k]


@st.cache_resource
def get_search_system():
    """Cached function to load the search system once."""
    s = RestaurantSearchSystem()
    success, msg = s.load_index("restaurant_search_index")
    if success:
        st.success(msg)
        return s
    else:
        st.error(msg)
        st.stop()

def main():
    """Main function to run the Streamlit app."""
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
            st.warning("No results found matching your criteria.")

if __name__ == "__main__":
    main()
