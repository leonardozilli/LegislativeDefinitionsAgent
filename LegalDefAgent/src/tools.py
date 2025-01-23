from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, StructuredTool, tool, InjectedToolArg
from langchain.tools.retriever import create_retriever_tool
from typing import List , Annotated
import xml.etree.ElementTree as ET
from pathlib import Path
import re

from .retriever import vector_store
from .settings import settings
from .utils import definition_obj_to_path


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



def extract_definition_from_xmldb(definition_metadata: dict) -> str:
    """
    Extracts the content of an XML element with the specified defines attribute.

    Args:
        definition_metadata (dict): A dictionary containing the metadata of the definition to retrieve.

    Returns:
        str: The text content of the matching element, or None if no match is found.
    """

    namespace = settings.DB_CONFIG.NAMESPACES[definition_metadata['dataset']]

    file_path = definition_obj_to_path(definition_metadata)
    def_n = definition_metadata['def_n']
    try:
        # Parse the XML file
        tree = ET.parse(Path(settings.DB_CONFIG.XML_DATA_DIR) / file_path)
        root = tree.getroot()

        # Find the element with the specified defines attribute
        res = root.find(f".//akn:*[@defines='{def_n}']", namespace)
        if res is not None:
            text = ''.join(res.itertext()).strip()
            return re.sub(r'\s+', ' ', text)
        # Return None if no match is found
        return None

    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return None
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return None