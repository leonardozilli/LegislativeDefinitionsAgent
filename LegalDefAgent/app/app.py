from typing import Any, List, Union, Optional

import uvicorn
from fastapi import FastAPI

from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

from langserve import add_routes

import sys
sys.path.insert(1, '../LegalDefAgent')

import src.config as config
import src.models as models
from src.agent import LegalDefAgent


app = FastAPI(
    title="LegalDefAgent",
    version="1.0",
    description=""
)

class QueryRequest(BaseModel):
    query: str

# Some input schema
class Input(BaseModel):
    input: str

# Some output schema
class Output(BaseModel):
    output: Any

defagent = LegalDefAgent(model=models.groq)

def inp(question: str) -> dict:
    return {
            "question": question,
            "messages": [
                ("system", ''),
                ("user", question),
            ]
        }

def out(state: dict):
    result = state[1]["answer"]  # 0: retrieve, 1: generate
    return result


add_routes(
    app,
    RunnableLambda(inp) | defagent.graph,
    playground_type="chat",
    path='/defagent',
)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)