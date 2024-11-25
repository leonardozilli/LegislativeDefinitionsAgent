from pydantic import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain.tools.retriever import create_retriever_tool
from typing import List 

from utils import setup_vectorstore

@tool
def vector_search(query: str) -> str:
    """
    Searches and retrieves the most similar definitions to the given query in a vector DB.

    Args:
        query: The query to search for in the vector DB.
    """

    retriever = setup_vectorstore()
    docsList = retriever.invoke(input=query)

    return docsList

@tool
def retriever(query: str) -> List:
    retriever = setup_vectorstore()
    docsList = retriever.invoke(query)

    return docsList


def retriever_tool():
    retriever_tool = create_retriever_tool(
        setup_vectorstore(),
        "retrieve_definitions",
        "Search and return legal definitions of terms from a vector store.",
    )

    return retriever_tool


