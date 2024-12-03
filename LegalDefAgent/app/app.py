from typing import Any, List, Union, Optional

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field
from langserve import add_routes
from langgraph.graph.message import add_messages
from typing import Annotated, Literal, Sequence, List, Any, Dict
from typing_extensions import TypedDict
from pprint import pprint

from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage, AnyMessage

import sys
sys.path.insert(1, '../LegalDefAgent')


import LegalDefAgent.src.config as config
import LegalDefAgent.src.models as models
from LegalDefAgent.src.agent import LegalDefAgent


app = FastAPI(
    title="LegalDefAgent",
    version="1.0",
    description=""
)

from fastapi.middleware.cors import CORSMiddleware

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

class QueryRequest(BaseModel):
    query: str

class Input(BaseModel):
    input: str

class Output(BaseModel):
    output: Any

defagent = LegalDefAgent(model=models._get_model('groq', streaming=False))

def inp(question: str) -> dict:
    return {
            "input": question,
            "messages": [
                SystemMessage(content=''),
                HumanMessage(content=question)
            ]
        }

def out(state: dict) -> str:
    return state['output']

class InputChat(BaseModel):
    """Input for the chat endpoint."""

    input: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question: str
    definendum: str
    relevant_defs: List[str]
    definiens: List[str]

class OutputChat(BaseModel):
    """Output for the chat endpoint."""

    input: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question: str
    definendum: str
    relevant_defs: List[str]
    definiens: List[str]

add_routes(
    app,
    RunnableLambda(inp) | defagent.graph | RunnableLambda(out),
    #defagent.graph.with_types(input_type=InputChat, output_type=OutputChat),
    playground_type="default",
    path='/defagent',
)

add_routes(
    app,
    #chain.with_types(input_type=InputChat),
    RunnableLambda(inp) | defagent.graph.with_types(input_type=dict,output_type=str),
    #enable_feedback_endpoint=True,
    #enable_public_trace_link_endpoint=True,
    playground_type="chat",
    path='/defagent_chat',
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="localhost", port=8000, reload=True)