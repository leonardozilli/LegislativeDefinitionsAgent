from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, StructuredTool, tool, InjectedToolArg
from langchain.tools.retriever import create_retriever_tool
from typing import List , Annotated

from LegalDefAgent.src.retriever import setup_vectorstore

def create_vector_search_tool(vectorstore):
    @tool
    def vector_search(input: str) -> str:
        """
        Searches and retrieves the most similar definitions to the given query in a vector DB.

        Args:
            query: The query to search for in the vector DB.
        """

        docsList = vectorstore.invoke(input)

        return docsList
    
    return vector_search


def retriever_tool():
    retriever_tool = create_retriever_tool(
        setup_vectorstore(),
        "retrieve_definitions",
        "Search and return legal definitions of terms from a vector store.",
    )

    return retriever_tool


