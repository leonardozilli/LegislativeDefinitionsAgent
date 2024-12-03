import os
from dotenv import find_dotenv, load_dotenv
from functools import lru_cache

load_dotenv(find_dotenv())

LOG_PATH = os.getenv("LOG_PATH", "logs/")
MILVUSDB_URI = os.getenv("MILVUSDB_URI", "/home/leo/Desktop/dhdk/Master thesis/.project/LegalDefAgent/vec_db/definitions_vectors.db")
MILVUSDB_COLLECTION_NAME = os.getenv("MILVUSDB_COLLECTION_NAME", "Definitions")

LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")

DB_CONFIG = {
    'DATA_DIR': os.getenv("DB_DATA_DIR", "data/datasets/"),
    'OUTPUT_DIR': os.getenv("DB_OUTPUT_DIR", "data/definitions/"),
    'DATASETS': ['EurLex', 'Normattiva', 'PDL'],
    'MAX_DEFINITION_LENGTH': int(os.getenv("MAX_DEFINITION_LENGTH", "5000")),
    'BATCH_SIZE': int(os.getenv("DB_BATCH_SIZE", "50")),
    'NAMESPACES': {
        'EurLex': {'akn': 'http://docs.oasis-open.org/legaldocml/ns/akn/3.0'},
        'Normattiva': {'akn': 'http://docs.oasis-open.org/legaldocml/ns/akn/3.0'},
        'PDL': {'akn': 'http://docs.oasis-open.org/legaldocml/ns/akn/3.0/WD17'}
    }
}