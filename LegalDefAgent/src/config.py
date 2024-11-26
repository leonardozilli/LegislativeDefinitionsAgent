from dotenv import load_dotenv
import os

load_dotenv()

LOG_PATH = os.getenv("LOG_PATH", "logs/")
MILVUSDB_URI = os.getenv("MILVUSDB_URI", "./vec_db/definitions_vectors.db")
MILVUSDB_COLLECTION_NAME = os.getenv("MILVUSDB_COLLECTION_NAME", "Definitions")

LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")

