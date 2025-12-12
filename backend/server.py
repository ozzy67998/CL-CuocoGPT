import os
import sys
import asyncio
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add ../src to path and import normally
HERE = os.path.dirname(__file__)
SRC_DIR = os.path.normpath(os.path.join(HERE, "src"))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from ollama import pipeline_backend as recipe_bot  # type: ignore

app = FastAPI(title="Recipe Chatbot API")

# allow frontend (local browser)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
print("pingas")
# preload pipeline once
pipeline_loaded = False

def ensure_loaded():
    global pipeline_loaded
    if not pipeline_loaded:
        recipe_bot._load_pipeline()
        pipeline_loaded = True

class ChatQuery(BaseModel):
    query: str

@app.get("/health")
def health():
    return {"status": "ok", "pipeline_loaded": pipeline_loaded}

@app.post("/chat")
async def chat(payload: ChatQuery):
    await asyncio.to_thread(ensure_loaded)
    summary = await asyncio.to_thread(recipe_bot._process_query, payload.query)
    return {"response": summary}    


@app.post("/clear")
async def clear_conversation():
    """Clear the backend conversation context stored in memory."""
    await asyncio.to_thread(ensure_loaded)
    await asyncio.to_thread(recipe_bot.clear_conversation_context)
    return {"status": "cleared"}

