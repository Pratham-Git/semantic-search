from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    name: str
    age: int

@app.get("/")
async def home():
    return {"message": "Hello World"}

@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.get("/user")
async def user_info(name: str, age: int = 18):
    return {"name": name, "age": age}


@app.post("/create_user")
async def create_user(user: User):
    return {"message": f"User {user.name} created successfully,", "data": user}