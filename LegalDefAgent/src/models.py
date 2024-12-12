import os
from dotenv import find_dotenv, load_dotenv
from functools import lru_cache

load_dotenv(find_dotenv())

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_ollama.chat_models import ChatOllama

from . import config

from typing import List, Any, Dict
from langchain_core.callbacks.base import BaseCallbackHandler

class CustomHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        formatted_prompts = "\n".join(prompts)
        print(f"-> PROMPT:\n{formatted_prompts}\n")

@lru_cache(maxsize=4)
def _get_model(model_name: str, streaming: bool = False):
    match model_name:
        case "gpt":
            return ChatOpenAI(model="gpt-4o-mini-2024-07-18", api_key=config.OPENAI_API_KEY, temperature=0, streaming=streaming)
        case "groq":
            return ChatGroq(model="llama3-70b-8192", api_key=config.GROQ_API_KEY, temperature=0, streaming=streaming)
        case "groq_tool":
            return ChatGroq(model="llama3-groq-8b-8192-tool-use-preview", api_key=config.GROQ_API_KEY, temperature=0, streaming=streaming)
        case "mistral":
            return ChatMistralAI(model="open-mistral-nemo", api_key=config.MISTRAL_API_KEY, temperature=0, streaming=streaming)
        case "gemma":
            return ChatOllama(model="gemma2:2b", temperature=0)
        case "llama32":
            return ChatOllama(model='llama3.2', temperature=0)
        case _:
            raise ValueError(f"Unsupported model type: {model_name}")