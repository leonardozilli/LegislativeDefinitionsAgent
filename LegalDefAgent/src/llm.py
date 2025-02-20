from functools import cache
from typing import TypeAlias
from dotenv import find_dotenv, load_dotenv

from langchain_together import ChatTogether
from langchain_community.chat_models import FakeListChatModel
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama.chat_models import ChatOllama
from langchain_mistralai import ChatMistralAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI

from .settings import settings
from .schema.models import (
    AllModelEnum,
    FakeModelName,
    MistralModelName,
    GroqModelName,
    OpenAIModelName,
    OllamaModelName,
    DeepSeekModelName,
    GoogleModelName,
    AnthropicModelName,
    TogetherModelName,
)

load_dotenv(find_dotenv())


_MODEL_TABLE = {
    OpenAIModelName.GPT_4O_MINI: "gpt-4o-mini",
    OpenAIModelName.GPT_4O: "gpt-4o",
    GroqModelName.LLAMA_33_70B: "llama-3.3-70b-versatile",
    GroqModelName.LLAMA_31_8B: "llama-3.1-8b-instant",
    GroqModelName.GEMMA2_9B_IT: "gemma2-9b-it",
    MistralModelName.NEMO_12B: "open-mistral-nemo",
    OllamaModelName.LLAMA_32_3B: "llama3.2",
    DeepSeekModelName.DEEPSEEK_CHAT: "deepseek-chat",
    GoogleModelName.GEMINI_15_FLASH: "gemini-1.5-flash",
    GoogleModelName.GEMINI_15_FLASH_8B: "gemini-1.5-flash-8b",
    GoogleModelName.GEMINI_15_PRO: "gemini-1.5-pro",
    GoogleModelName.GEMMA2_2B_IT: "gemma2-2b-it",
    GoogleModelName.GEMMA2_9B_IT: "gemma2-9b-it",
    GoogleModelName.GEMMA2_27B_IT: "gemma2-27b-it",
    AnthropicModelName.HAIKU_35: "claude-3-5-haiku-20241022",
    AnthropicModelName.SONNET_35: "claude-3-5-sonnet-20241022",
    TogetherModelName.LLAMA_33_70B: "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    TogetherModelName.NEMOTRON_70B: "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    TogetherModelName.LLAMA_31_8B: "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
}

ModelT: TypeAlias = (
    ChatOpenAI | ChatGroq | ChatMistralAI | ChatOllama |
    FakeListChatModel | ChatGoogleGenerativeAI |
    ChatAnthropic | ChatTogether
)


@cache
def get_model(model_name: AllModelEnum, /) -> ModelT:
    # NOTE: models with streaming=True will send tokens as they are generated
    # if the /stream endpoint is called with stream_tokens=True (the default)
    api_model_name = _MODEL_TABLE.get(model_name)

    if not api_model_name:
        raise ValueError(f"Unsupported model: {model_name}")
    if model_name in TogetherModelName:
        return ChatTogether(model=api_model_name, temperature=0.3, streaming=True)
    if model_name in OpenAIModelName:
        return ChatOpenAI(model=api_model_name, temperature=0.3, streaming=True)
    if model_name in GroqModelName:
        return ChatGroq(model=api_model_name, temperature=0.3, streaming=True)
    if model_name in DeepSeekModelName:
        return BaseChatOpenAI(model=api_model_name, openai_api_key=settings.DEEPSEEK_API_KEY,
                              openai_api_base='https://api.deepseek.com', max_tokens=1024)
    if model_name in MistralModelName:
        return ChatMistralAI(model=api_model_name, temperature=0.3, streaming=True)
    if model_name in OllamaModelName:
        return ChatOllama(model=api_model_name, temperature=0.3)
    if model_name in FakeModelName:
        return FakeListChatModel(responses=["This is a test response from the fake model."])
    if model_name in GoogleModelName:
        return ChatGoogleGenerativeAI(model=api_model_name)
    if model_name in AnthropicModelName:
        return ChatAnthropic(model=api_model_name)
