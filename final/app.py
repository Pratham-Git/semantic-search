# FILE: app.py
from fastapi import FastAPI, HTTPException, Query
from typing import Optional, List
from pydantic import BaseModel
from search_logic import RestaurantSearchSystem  # Updated import
import uvicorn

class Restaurant(BaseModel):
    restaurant_name: Optional[str]
    cost_for_two: Optional[float]
    cuisines: Optional[str]
    features: Optional[str]
    location_city: Optional[str]
    region_name: Optional[str]
    address: Optional[str]
    start_time: Optional[str]
    end_time: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    similarity_score: float
    distance_km: Optional[float]

class SearchResponse(BaseModel):
    message: str
    data: List[Restaurant]

app = FastAPI(
    title="Smart Restaurant Search API",
    description="An API for semantic search of restaurants with location and time-based filtering.",
    version="1.0.0"
)

# Initialize using environment variables managed inside RestaurantSearchSystem
search_system = RestaurantSearchSystem()
@app.get("/", tags=["General"])
async def read_root():
    return {"message": "Welcome to the Smart Restaurant Search API. Go to /docs for interactive documentation."}

@app.get("/search/", response_model=SearchResponse, tags=["Search"])
async def search_restaurants(
    query: str,
    latitude: Optional[float] = Query(None, description="User's current latitude (e.g., 28.4595)"),
    longitude: Optional[float] = Query(None, description="User's current longitude (e.g., 77.0266)"),
    top_k: int = Query(10, description="Number of results to return", ge=1, le=50)
):
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter cannot be empty.")
        
    try:
        results = search_system.search(
            query=query,
            user_lat=latitude,
            user_lon=longitude,
            top_k=top_k
        )
        if not results:
            return {"message": "No matching results found.", "data": []}
        return {"message": f"Found {len(results)} relevant restaurants.", "data": results}
    except Exception as e:
        print(f"An error occurred during search: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
