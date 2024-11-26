from typing import Any, List, Union

import uvicorn
from fastapi import FastAPI

from langchain_core.runnables import RunnableLambda
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.messages import AIMessage, FunctionMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.utils.function_calling import format_tool_to_openai_tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from langserve import add_routes

import src.config as config
import src.models as models
from src.agent import LefalDefAgent

app = FastAPI(
    title="LegalDefAgent",
    version="1.0",
    description=""
)

class QueryRequest(BaseModel):
    query: str

@app.post("/api/definition")
async def get_definition(request: QueryRequest):
    # Initialize your DefAgent here (or better, make it a singleton)
    defagent = DefAgent(model=models.groq)
    
    # Get the result from your agent
    result = defagent.invoke(request.query)
    
    # Format the response
    return {
        "definition": result,
        "sources": []  # Add sources if available from your agent
    }

defagent = DefAgent(model=models.groq)

add_routes(
    app,
    RunnableLambda(inp) | defagent.workflow,
    playground_type="default",
    path='/defagent'
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)