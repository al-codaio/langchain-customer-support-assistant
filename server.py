from fastapi import FastAPI, HTTPException
from langserve import add_routes
import uvicorn
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

try:
    from main import graph
except ImportError as e:
    print(f"Error importing graph from main.py: {e}")
    print("Please ensure 'graph' is a top-level variable in main.py after workflow.compile().")
    exit(1)

app = FastAPI(
    title="Customer Support Assistant LangServe",
    version="1.0",
    description="A LangGraph-powered customer support AI.",
)

origins = [
    "http://localhost",
    "http://localhost:8001",
    "http://127.0.0.1",
    "http://127.0.0.1:8001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# The path `/customer-support` means API will be available at http://localhost:8000/customer-support/invoke
add_routes(
    app,
    graph,
    path="/customer-support",
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)