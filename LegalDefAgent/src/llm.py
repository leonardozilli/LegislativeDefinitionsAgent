from typing import TypeAlias
from dotenv import find_dotenv, load_dotenv
from functools import cache

load_dotenv(find_dotenv())

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_ollama.chat_models import ChatOllama
from langchain_community.chat_models import FakeListChatModel

from .schema.models import (
    AllModelEnum,
    FakeModelName,
    MistralModelName,
    GroqModelName,
    OpenAIModelName,
    OllamaModelName,
)

#def _get_model(model_name: str, streaming: bool = False):
    #match model_name:
        #case "gpt":
            #return ChatOpenAI(model="gpt-4o-mini", api_key=config.OPENAI_API_KEY, temperature=0, streaming=streaming)
        #case "groq":
            #return ChatGroq(model="llama3-70b-8192", api_key=config.GROQ_API_KEY, temperature=0, streaming=streaming)
        #case "groq_tool":
            #return ChatGroq(model="llama3-groq-8b-8192-tool-use-preview", api_key=config.GROQ_API_KEY, temperature=0, streaming=streaming)
        #case "mistral":
            #return ChatMistralAI(model="open-mistral-nemo", api_key=config.MISTRAL_API_KEY, temperature=0, streaming=streaming)
        #case "gemma":
            #return ChatOllama(model="gemma2:2b", temperature=0)
        #case "llama32":
            #return ChatOllama(model='llama3.2', temperature=0)
        #case _:
            #raise ValueError(f"Unsupported model type: {model_name}")


_MODEL_TABLE = {
    OpenAIModelName.GPT_4O_MINI: "gpt-4o-mini",
    OpenAIModelName.GPT_4O: "gpt-4o",
    GroqModelName.LLAMA_3_8B: "llama3-8b-8192",
    GroqModelName.LLAMA_3_70B: "llama3-70b-8192",
    GroqModelName.LLAMA_3_8B_TOOL: "llama3-groq-8b-8192-tool-use-preview",
    GroqModelName.LLAMA_33_70B: "llama-3.3-70b-versatile",
    MistralModelName.NEMO_12B: "open-mistral-nemo",
    OllamaModelName.GEMMA2_2B: "gemma2:2b",
    OllamaModelName.LLAMA_32_3B: "llama3.2",
    OllamaModelName.PHI3_4B: "phi3",
    FakeModelName.FAKE: "fake",
}

ModelT: TypeAlias = ChatOpenAI | ChatGroq | ChatMistralAI | ChatOllama


@cache
def get_model(model_name: AllModelEnum, /) -> ModelT:
    # NOTE: models with streaming=True will send tokens as they are generated
    # if the /stream endpoint is called with stream_tokens=True (the default)
    api_model_name = _MODEL_TABLE.get(model_name)
    
    if not api_model_name:
        raise ValueError(f"Unsupported model: {model_name}")

    if model_name in OpenAIModelName:
        return ChatOpenAI(model=api_model_name, temperature=0, streaming=True)
    if model_name in GroqModelName:
        return ChatGroq(model=api_model_name, temperature=0, streaming=True)
    if model_name in MistralModelName:
        return ChatMistralAI(model=api_model_name, temperature=0, streaming=True)
    if model_name in OllamaModelName:
        return ChatOllama(model=api_model_name, temperature=0)
    if model_name in FakeModelName:
        return FakeListChatModel(responses=["This is a test response from the fake model."])
