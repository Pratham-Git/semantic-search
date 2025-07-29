import streamlit as st
import pandas as pd
import numpy as np
import faiss
import pickle
import re
from sentence_transformers import SentenceTransformer
import os

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
        query_lower = query.lower()
        constraints = {'features': [], 'cuisines': [], 'cost_min': None, 'cost_max': None}
        cost_patterns = [
            (r'under (\d+)', 'max'), (r'below (\d+)', 'max'), (r'less than (\d+)', 'max'),
            (r'above (\d+)', 'min'), (r'over (\d+)', 'min'), (r'more than (\d+)', 'min'),
            (r'around (\d+)', 'both')
        ]
        for pattern, ctype in cost_patterns:
            match = re.search(pattern, query_lower)
            if match:
                val = int(match.group(1))
                if ctype == 'max': constraints['cost_max'] = val
                elif ctype == 'min': constraints['cost_min'] = val
                elif ctype == 'both':
                    constraints['cost_min'] = int(val * 0.8)
                    constraints['cost_max'] = int(val * 1.2)
        for feature in self.all_features:
            if feature in query_lower: constraints['features'].append(feature)
        for cuisine in self.all_cuisines:
            if cuisine in query_lower: constraints['cuisines'].append(cuisine)
        if any(w in query_lower for w in ['cheap','budget','affordable']):
            constraints['cost_max'] = constraints['cost_max'] or 800
        if any(w in query_lower for w in ['expensive','luxury','costly']):
            constraints['cost_min'] = constraints['cost_min'] or 2000
        return constraints

    def apply_constraints(self, constraints):
        valid_indices = []
        for i, meta in enumerate(self.metadata):
            cost = meta['cost_for_two']
            if constraints['cost_min'] and cost < constraints['cost_min']: continue
            if constraints['cost_max'] and cost > constraints['cost_max']: continue
            if constraints['features'] and not all(f in meta['features'].lower() for f in constraints['features']): continue
            if constraints['cuisines'] and not any(c in meta['cuisines'].lower() for c in constraints['cuisines']): continue
            valid_indices.append(i)
        return valid_indices

    def search(self, query, top_k=10):
        if self.index is None or self.metadata is None: return []
        model = self.load_model()
        constraints = self.extract_constraints(query)
        valid_indices = self.apply_constraints(constraints)
        if not valid_indices: return []
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
                results.append({
                    'restaurant_name': meta['restaurant_name'],
                    'cuisines': meta['cuisines'],
                    'features': meta['features'],
                    'cost_for_two': meta['cost_for_two'],
                    'similarity_score': float(scores[0][i]) if i < len(scores[0]) else 0.0
                })
        return results

@st.cache_resource
def get_search_system():
    s = RestaurantSearchSystem()
    success, msg = s.load_index("restaurant_search_index")
    if success: st.success(msg); return s
    else: st.error(msg); st.stop()

def main():
    st.set_page_config(page_title="Restaurant Search", layout="centered")
    st.title("🍽️ Restaurant Search")
    search_system = get_search_system()

    query = st.text_input("Search for restaurants:", placeholder="e.g., cheap pizza with parking")
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
