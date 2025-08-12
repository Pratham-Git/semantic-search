import streamlit as st
import pandas as pd
import numpy as np
import faiss
import pickle
import re
import ollama
import json
import requests
from sentence_transformers import SentenceTransformer
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
import os
import pytz # Added for timezone handling

# The definitive, correct import statement for this library
from streamlit_geolocation import streamlit_geolocation

class RestaurantSearchSystem:
    """
    Handles loading the search index, model, and performing the search.
    This class is robust and does not need changes.
    """
    def __init__(self):
        self.model = None
        self.index = None
        self.metadata = None

    @st.cache_resource(show_spinner="Loading embedding model...")
    def load_model(_self):
        try:
            model = SentenceTransformer('intfloat/e5-large-v2')
            return model
        except Exception as e:
            st.error(f"Fatal: Failed to load embedding model: {str(e)}")
            st.stop()

    def load_index(self, index_path="restaurant_search_index"):
        try:
            index_file = os.path.join(index_path, "index.faiss")
            metadata_file = os.path.join(index_path, "metadata.pkl")
            
            if not os.path.exists(index_file) or not os.path.exists(metadata_file):
                st.error(f"Fatal: Index or metadata file not found in '{os.path.abspath(index_path)}'.")
                st.info("Please ensure 'index.faiss' and 'metadata.pkl' are present.")
                st.stop()
                
            self.index = faiss.read_index(index_file)
            with open(metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
            return True, f"Loaded {len(self.metadata)} restaurants from index."
        except Exception as e:
            st.error(f"Fatal: Error loading index files: {str(e)}")
            st.stop()

    def extract_constraints(self, query):
        features_list = ['5-star dining', 'air condition', 'alcohol served', 'authentic japanese cuisine', 'award winners', 'bar', 'barbeque', 'bars & pubs', 'breakfast buffet', 'buffet', 'cafe', 'dance floor', 'dessert', 'disabled friendly', 'dj', 'eatout', 'exotic cocktails', 'formal attire', 'great breakfasts', 'happy hours', 'healthy food', 'home delivery', 'hookah', 'karaoke', 'kebabs', 'kids allowed', 'live kitchen', 'live music', 'live sports screening', 'luxury dining', 'mall parking', 'microbrewery', 'movies', 'new year', 'nightlife', 'outdoor seating', 'parking', 'pet friendly', 'pocket friendly', 'premium imported ingredients', 'pure veg', 'romantic', 'rooftops', 'sake collection', 'seafood', 'shisha', 'smoking area', 'sports bar', 'stags allowed', 'sunday brunches', 'take-away', 'thali', 'vaccinated staff', 'valet parking', 'vegan', 'wheelchair accessible']
        features_str = ", ".join(f'"{f}"' for f in features_list)
        try:
            with st.spinner("Analyzing your query with local AI..."):
                response = ollama.chat(
                    model="qwen2.5:7b",
                    messages=[{
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
                            "    - 'expensive', 'fancy', 'luxury' -> `sort_by: 'cost_desc'`\n"
                            "3.  **Output Structure:**\n"
                            "    - You MUST return a JSON object containing ALL of the following keys, using `null` if a value is not present: `cost_min`, `cost_max`, `sort_by`, `cuisines`, `features`, `name`, `time_after`, `time_before`, `is_open_now`.\n" # Added is_open_now
                            "4.  **Time:** Extract times like 'after 10pm' into `time_after: '22:00'` or 'before 7pm' into `time_before: '19:00'`. Use 24-hour format.\n"
                            "5.  **Current Status:** If the user asks for places 'open now' or 'open right now', set `is_open_now: true`.\n" # New rule
                            "6.  **Features:**\n"
                            f"   - Choose from these terms only: {features_str}\n"
                        )
                    }, {"role": "user", "content": query}]
                )
            raw_output = response["message"]["content"].strip()
            cleaned = re.search(r"\{.*\}", raw_output, re.DOTALL)
            data = json.loads(cleaned.group(0)) if cleaned else {}
            return data
        except Exception:
            st.toast("Could not analyze query with AI. Using basic keyword search.")
            return {}

    def calculate_distance_km(self, user_lat, user_lon, rest_lat, rest_lon):
        try:
            coords = [float(c) for c in [user_lat, user_lon, rest_lat, rest_lon]]
            R = 6371.0
            lat1_rad, lon1_rad = radians(coords[0]), radians(coords[1])
            lat2_rad, lon2_rad = radians(coords[2]), radians(coords[3])
            dlon, dlat = lon2_rad - lon1_rad, lat2_rad - lat1_rad
            a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            return round(R * c, 2)
        except (ValueError, TypeError):
            return None

    def search(self, query, user_lat=None, user_lon=None, top_k=10):
        if not self.index or not self.model: 
            st.error("Search system not initialized.")
            return []
        
        constraints = self.extract_constraints(query)
        refined_query = constraints.get("rewritten_query", query)
        radius_km = constraints.get("radius_km", 5)

        try:
            query_emb = self.model.encode([refined_query])
            faiss.normalize_L2(query_emb)
            scores, indices = self.index.search(query_emb, min(len(self.metadata), 500))
        except Exception as e:
            st.error(f"Vector search failed: {e}")
            return []

        user_has_location = bool(user_lat and user_lon)
        
        # Get current time for filtering
        now_ist = datetime.now(pytz.timezone('Asia/Kolkata')).time()

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1: continue
            meta = self.metadata[idx]
            
            # --- ADDED TIME FILTERING LOGIC ---
            passes_time_filter = True
            if constraints.get("time_after"):
                if not is_open_after(constraints["time_after"], meta.get('start_time'), meta.get('end_time')):
                    passes_time_filter = False
            if constraints.get("time_before"):
                if not is_open_before(constraints["time_before"], meta.get('start_time'), meta.get('end_time')):
                    passes_time_filter = False
            
            # --- NEW 'OPEN NOW' FILTER ---
            if constraints.get("is_open_now") is True:
                if not is_currently_open(meta.get('start_time'), meta.get('end_time'), now_ist):
                    passes_time_filter = False
            # --- END NEW FILTER ---

            if not passes_time_filter:
                continue
            # --- END OF TIME FILTERING ---

            dist = None
            if user_has_location and meta.get('latitude') and meta.get('longitude'):
                dist = self.calculate_distance_km(user_lat, user_lon, meta['latitude'], meta['longitude'])
            
            results.append({**meta, 'similarity_score': float(score), 'distance_km': dist})
        
        def sort_key(r):
            dist = r['distance_km']
            if dist is not None and dist <= radius_km: return (1, dist, -r['similarity_score'])
            if dist is not None: return (2, dist, -r['similarity_score'])
            return (3, -r['similarity_score'], 0)
        
        results.sort(key=sort_key)
        return results[:top_k]

@st.cache_resource(show_spinner="Initializing search system...")
def get_search_system():
    s = RestaurantSearchSystem()
    success, msg = s.load_index("restaurant_search_index")
    if success:
        st.success(msg, icon="✅")
        s.model = s.load_model()
        return s

def safe_str(value, default='N/A'):
    if value is None or pd.isna(value) or str(value).lower() == 'nan':
        return default
    return str(value)

def parse_time(time_str):
    if not isinstance(time_str, str): return None
    for fmt in ('%H:%M', '%H:%M:%S'):
        try:
            return datetime.strptime(time_str.strip(), fmt).time()
        except (ValueError, TypeError): continue
    return None

def is_currently_open(start_time_str, end_time_str, now_time):
    start_time, end_time = parse_time(start_time_str), parse_time(end_time_str)
    if start_time is None or end_time is None: return None
    if start_time <= end_time: return start_time <= now_time < end_time
    else: return now_time >= start_time or now_time < end_time

def is_open_after(query_time_str, start_time_str, end_time_str):
    query_time = parse_time(query_time_str)
    start_time = parse_time(start_time_str)
    end_time = parse_time(end_time_str)
    if query_time is None or start_time is None or end_time is None: return True
    if start_time <= end_time: return query_time < end_time
    else: return True

def is_open_before(query_time_str, start_time_str, end_time_str):
    query_time = parse_time(query_time_str)
    start_time = parse_time(start_time_str)
    if query_time is None or start_time is None: return True
    return query_time > start_time

def main():
    st.set_page_config(page_title="Smart Restaurant Search", layout="centered")
    st.title("🍽️ Smart Restaurant Search")
    
    search_system = get_search_system()

    IST = pytz.timezone('Asia/Kolkata')
    now_ist = datetime.now(IST).time()

    st.subheader("1. Your Location")
    st.info("Your browser may ask for permission to access your location for a precise search.")
    
    location_data = streamlit_geolocation()

    user_lat, user_lon = None, None
    if location_data and location_data.get('latitude'):
        user_lat, user_lon = location_data['latitude'], location_data['longitude']
        st.success(f"📍 Location Acquired: {user_lat:.4f}, {user_lon:.4f}", icon="✅")
    else:
        st.warning("Location permission not granted. Search will be based on relevance only.", icon="⚠️")

    st.subheader("2. Your Search")
    query = st.text_input("What are you looking for?", placeholder="e.g., cafe open right now", help="Try queries like 'cheap chinese food near me' or 'open after 11pm'.")

    if query:
        with st.spinner("Searching for the best spots..."):
            results = search_system.search(query, user_lat=user_lat, user_lon=user_lon)
        
        if results:
            st.success(f"Found {len(results)} relevant restaurants!", icon="🎉")
            for i, r in enumerate(results, 1):
                title = f"{i}. {safe_str(r.get('restaurant_name'))} — ₹{r.get('cost_for_two', 0):.0f}"
                if r.get('distance_km') is not None:
                    title += f" ({r['distance_km']:.1f} km away)"
                
                with st.expander(title, expanded=(i<=3)):
                    open_status = is_currently_open(r.get('start_time'), r.get('end_time'), now_ist)
                    timings_str = f"**Timings:** {safe_str(r.get('start_time'), 'N/A')} - {safe_str(r.get('end_time'), 'N/A')}"
                    if open_status is True:
                        timings_str += " <span style='color: green; font-weight: bold;'>(Open Now)</span>"
                    elif open_status is False:
                        timings_str += " <span style='color: red; font-weight: bold;'>(Closed)</span>"

                    col1, col2 = st.columns([2, 3])
                    with col1:
                        st.markdown(f"**Cuisine:** {safe_str(r.get('cuisines'))}")
                        if r.get('distance_km') is not None:
                            st.markdown(f"**Distance:** {r['distance_km']:.2f} km")
                        else:
                            st.markdown("**Distance:** N/A")
                        st.markdown(timings_str, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"**Features:** {safe_str(r.get('features'))}")
                        st.markdown(f"**Location:** {safe_str(r.get('region_name'))}, {safe_str(r.get('location_city'))}")
                    
                    address = safe_str(r.get('address'))
                    if address != 'N/A':
                        st.caption(f"Address: {address}")
        else:
            st.error("No matching results found. Please try a different search.", icon="🤷")

if __name__ == "__main__":
    main()
