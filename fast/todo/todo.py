from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Task(BaseModel):
    title: str
    description: str

tasks = []

@app.post("/tasks/")
async def create_task(task: Task):
    tasks.append(task)
    return {"message": "Task created successfully"}

@app.get("/tasks/")
async def get_tasks():
    return tasks