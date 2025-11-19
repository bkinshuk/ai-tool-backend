from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from recommend import recommend_tools
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="AI Tool Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    prompt: str
    top_k: int = 5

@app.post("/recommend")
def recommend_endpoint(query: Query):
    results = recommend_tools(query.prompt, top_k=query.top_k)
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
