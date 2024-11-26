import os
import dotenv

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_ollama.chat_models import ChatOllama

import src.config as config

gpt = ChatOpenAI(model="gpt-4o-mini-2024-07-18", api_key=config.OPENAI_API_KEY)

groq = ChatGroq(model="llama3-8b-8192", api_key=config.GROQ_API_KEY)

mix = ChatMistralAI(model="open-mistral-nemo", api_key=config.MISTRAL_API_KEY)

gemma = ChatOllama(model="gemma2:2b")

llama32 = ChatOllama(model='llama3.2')
