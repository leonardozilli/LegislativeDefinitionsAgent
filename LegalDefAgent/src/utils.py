import tiktoken
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from IPython.display import Image, display
import logging
import sys
import os
from datetime import datetime
from .settings import settings

existdb_settings = settings.EXIST_CONFIG

load_dotenv(find_dotenv())

from typing import List, Optional, Any
from pyexistdb import db


class ExistDBHandler:
    """Handler for executing XQueries against an eXist-db instance."""
    
    def __init__(self, server_url: str, username: str, password: str):
        """
        Initialize connection to eXist-db.
        
        Args:
            server_url: Full URL to eXist-db server
            username: eXist-db username
            password: eXist-db password
        """
        self.db = db.ExistDB(server_url, username, password)
    
    def execute_query(self, query: str) -> List[str]:
        """
        Execute an XQuery and return all results.
        
        Args:
            query: XQuery string to execute
            
        Returns:
            List of results as strings
            
        Raises:
            Exception: If query execution fails
        """
        try:
            results = []
            query_result = self.db.executeQuery(query)
            hits = self.db.getHits(query_result)
            
            for i in range(hits):
                result = self.db.retrieve(query_result, i)
                results.append(result)
                
            return results
            
        except Exception as e:
            raise Exception(f"Query execution failed: {str(e)}")


def get_exist_handler():
    EXISTDB_SERVER_URL = f"http://{existdb_settings.XDB_HOST}:{existdb_settings.XDB_PORT}/exist/"
    return ExistDBHandler(
        server_url=EXISTDB_SERVER_URL,
        username=existdb_settings.XDB_USER,
        password=existdb_settings.XDB_PASSWORD
    )


def get_token_count(string: str, llm_name: str) -> int:
    """
    Returns the number of tokens in a given string for a specified language model.
    :param string: The input text to be tokenized.
    :param llm_name: The name of the language model to determine the encoding.
    :return: The count of tokens in the input string.
    """
    encoding = tiktoken.encoding_for_model(llm_name)
    return len(encoding.encode(string))


def chatqa(query: str, llm_name: str) -> str:
    """
    Basic OpenAI inference through langchain.
    :param query: The user's query.
    :param llm_name: The name of the OpenAI model to be used.
    :return: The model's output
    """

    llm = ChatOpenAI(model_name=llm_name, temperature=0)
    chain = LLMChain(llm=llm, prompt=PromptTemplate(
        template="""{q}""", input_variables=["q"]))
    res = chain.apply([{'q': query}])[0]['text']

    return res


def merge_dicts(*dicts: dict) -> dict:
    _ = {}
    for dict_ in dicts:
        _.update(dict_)
    return _

def parse_date_filters(date_filters):
    time_point = date_filters.get('time_point')
    from_date = date_filters.get('from_date')
    to_date = date_filters.get('to_date')
    
    if time_point:
        return datetime.strptime(time_point, '%Y-%m-%d').date()
    elif from_date and to_date:
        return datetime.strptime(from_date, '%Y-%m-%d').date(), datetime.strptime(to_date, '%Y-%m-%d').date()
    elif from_date:
        return datetime.strptime(from_date, '%Y-%m-%d').date(), datetime.today().date()
    elif to_date:
        return datetime.strptime('0001-01-01', '%Y-%m-%d').date(), datetime.strptime(to_date, '%Y-%m-%d').date()


def docs_to_json(docs):
    return {doc.to_json().get('kwargs') for doc in docs}


def json_to_aimessage(parsed_output):
    """Convert parsed output back to AIMessage."""
    content = f"{parsed_output}"
    return AIMessage(content=content)


def definition_obj_to_path(definition_obj):
    return os.path.join(definition_obj['dataset'], definition_obj['document_id'])


def draw_graph(graph):
    try:
        display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
    except Exception:
        # This requires some extra dependencies and is optional
        pass


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    node = event
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)


def get_uri_date(uri):
    at_split = uri.split("@")[-1]
    date = at_split.split("/")[0]
    return date



def setup_logging(log_level=logging.DEBUG):
    """
    Configures logging to output messages directly in Jupyter Notebook cells.

    Args:
        log_level (int): Logging level (e.g., logging.DEBUG, logging.INFO).
    """
    # Remove any existing handlers attached to the root logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure the logging system
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)  # Ensure logs are shown in notebook output
        ],
    )

    logging.info("Logging configured")