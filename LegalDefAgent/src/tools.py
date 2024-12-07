from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, StructuredTool, tool, InjectedToolArg
from langchain.tools.retriever import create_retriever_tool
from typing import List , Annotated

from LegalDefAgent.src.retriever import vector_store


def create_vector_search_tool(retriever):
    @tool
    def vector_search(input: str) -> str:
        """
        Searches and retrieves the most similar definitions to the given query in a vector DB.

        Args:
            input: The input sentence from the user.
        """

        docsList = retriever.invoke(input)

        return docsList
    
    return vector_search


#def retriever_tool():
    #retriever_tool = create_retriever_tool(
        #vector_store.setup_vectorstore(),
        #"retrieve_definitions",
        #"Search and return legal definitions of terms from a vector store.",
    #)

    #return retriever_tool


def references_resolver(definitions):
    raise NotImplementedError

