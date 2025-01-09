from typing import Annotated, Any, List, Dict

from dotenv import find_dotenv, load_dotenv
from pydantic import BeforeValidator, HttpUrl, SecretStr, TypeAdapter, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .schema.models import (
    AllModelEnum,
    AnthropicModelName,
    AWSModelName,
    MistralModelName,
    GoogleModelName,
    GroqModelName,
    OpenAIModelName,
    Provider,
)


load_dotenv(find_dotenv(raise_error_if_not_found=True))


def check_str_is_http(x: str) -> str:
    http_url_adapter = TypeAdapter(HttpUrl)
    return str(http_url_adapter.validate_python(x))

class DBConfig(BaseSettings):
    XML_DATA_DIR: str | None = None
    DEFINITIONS_OUTPUT_DIR: str | None = None
    VDB_OUTPUT_DIR: str | None = None
    DATASETS: List[str] = ['EurLex', 'Normattiva', 'PDL']
    MAX_DEFINITION_LENGTH: int = 5000
    BATCH_SIZE: int = 50
    NAMESPACES: Dict[str, Dict[str, str]] = {
        'EurLex': {'akn': 'http://docs.oasis-open.org/legaldocml/ns/akn/3.0'},
        'Normattiva': {'akn': 'http://docs.oasis-open.org/legaldocml/ns/akn/3.0'},
        'PDL': {'akn': 'http://docs.oasis-open.org/legaldocml/ns/akn/3.0/WD17'}
    }

class eXistDBConfig(BaseSettings):
    XDB_HOST: str = "0.0.0.0"
    XDB_PORT: int = 8080
    XDB_USER: str | None = None
    XDB_PASSWORD: str | None = None
    DATASETS: List[str] = ['EurLex', 'NormaAttiva', 'portal-camera']
    NAMESPACES: Dict[str, Dict[str, str]] = {
        'EurLex': {'akn': 'http://docs.oasis-open.org/legaldocml/ns/akn/3.0'},
        'NormaAttiva': {'akn': 'http://docs.oasis-open.org/legaldocml/ns/akn/3.0'},
        'portal-camera': {'akn': 'http://docs.oasis-open.org/legaldocml/ns/akn/3.0/WD17'}
    }
    COLLECTION_NAMES_MAP: Dict[str, str] = {
        'EurLex': 'EurLex',
        'Normattiva': 'NormaAttiva',
        'PDL': 'portal-camera'
    }

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        validate_default=False,
    )
    MODE: str | None = "false"

    MILVUSDB_URI: str | None = None
    MILVUSDB_COLLECTION_NAME: str | None = None

    DB_CONFIG: DBConfig = DBConfig()
    EXIST_CONFIG: eXistDBConfig = eXistDBConfig()

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    AUTH_SECRET: SecretStr | None = None

    OPENAI_API_KEY: SecretStr | None = None
    GROQ_API_KEY: SecretStr | None = None
    MISTRAL_API_KEY: SecretStr | None = None


    # If DEFAULT_MODEL is None, it will be set in model_post_init
    DEFAULT_MODEL: AllModelEnum | None = None  # type: ignore[assignment]


    LANGCHAIN_TRACING_V2: bool | None = None
    LANGCHAIN_PROJECT: str | None = None
    LANGCHAIN_ENDPOINT: Annotated[str, BeforeValidator(check_str_is_http)] = (
        "https://api.smith.langchain.com"
    )
    LANGCHAIN_API_KEY: SecretStr | None = None

    def model_post_init(self, __context: Any) -> None:
        api_keys = {
            Provider.OPENAI: self.OPENAI_API_KEY,
            Provider.GROQ: self.GROQ_API_KEY,
            Provider.MISTRAL: self.MISTRAL_API_KEY,
        }
        active_keys = {k for k, v in api_keys.items() if v}
        if not active_keys:
            raise ValueError("At least one LLM API key must be provided.")

        if self.DEFAULT_MODEL is None:
            first_provider = next(iter(active_keys))
            match first_provider:
                case Provider.GROQ:
                    self.DEFAULT_MODEL = GroqModelName.LLAMA_3_8B
                case Provider.OPENAI:
                    self.DEFAULT_MODEL = OpenAIModelName.GPT_4O_MINI
                case Provider.MISTRAL:
                    self.DEFAULT_MODEL = MistralModelName.NEMO_12B
                case Provider.ANTHROPIC:
                    self.DEFAULT_MODEL = AnthropicModelName.HAIKU_3
                case Provider.GOOGLE:
                    self.DEFAULT_MODEL = GoogleModelName.GEMINI_15_FLASH
                case Provider.AWS:
                    self.DEFAULT_MODEL = AWSModelName.BEDROCK_HAIKU
                case _:
                    raise ValueError(f"Unknown provider: {first_provider}")


    def is_dev(self) -> bool:
        return self.MODE == "dev"


settings = Settings()
