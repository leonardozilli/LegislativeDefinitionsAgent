from typing import Any, Dict, List

from dotenv import find_dotenv, load_dotenv
from pydantic import HttpUrl, SecretStr, TypeAdapter, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from legaldefagent.schema.models import (AllModelEnum, AnthropicModelName,
                                         DeepSeekModelName, GoogleModelName,
                                         GroqModelName, MistralModelName,
                                         OpenAIModelName, Provider,
                                         TogetherModelName, VLLMModelName)

load_dotenv(find_dotenv(raise_error_if_not_found=True), override=False)


def check_str_is_http(x: str) -> str:
    http_url_adapter = TypeAdapter(HttpUrl)
    return str(http_url_adapter.validate_python(x))


class eXistDBConfig(BaseSettings):
    EXISTDB_HOST: str = "0.0.0.0"
    EXISTDB_PORT: int = 8080
    EXISTDB_USER: str | None = None
    EXISTDB_PASSWORD: str | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    @field_validator("COLLECTIONS_NAMES", "COLLECTIONS_LEGID", "COLLECTIONS_NAMESPACES", mode="before")
    @classmethod
    def split_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return v

    COLLECTIONS_NAMES: List[str] | None
    COLLECTIONS_LEGID: List[str] | None
    COLLECTIONS_NAMESPACES: List[str] | None
    CONSOLIDATED_COLLECTION: str | None
    COLLECTIONS: Dict | None = None

    @field_validator("COLLECTIONS", mode="before")
    @classmethod
    def set_collections(cls, v, info):
        collections = {}
        collections_names = info.data.get("COLLECTIONS_NAMES")
        for i, name in enumerate(collections_names):
            collections[name] = {
                "legid": info.data.get("COLLECTIONS_LEGID", [])[i] if info.data.get("COLLECTIONS_LEGID") else None,
                "namespace": info.data.get("COLLECTIONS_NAMESPACES", [])[i]
                if info.data.get("COLLECTIONS_NAMESPACES")
                else None,
            }
        return collections

    MILVUSDB_PATH: str = "data/vdb/definitions.db"
    MILVUSDB_COLLECTION_NAME: str = "Definitions"

    EXIST_CONFIG: eXistDBConfig = eXistDBConfig()

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    FRONTEND_HOST: str = "localhost"
    FRONTEND_PORT: int = 3000

    AUTH_SECRET: SecretStr | None = None

    OPENAI_API_KEY: SecretStr | None = None
    GROQ_API_KEY: SecretStr | None = None
    MISTRAL_API_KEY: SecretStr | None = None
    ANTHROPIC_API_KEY: SecretStr | None = None
    GOOGLE_API_KEY: SecretStr | None = None
    DEEPSEEK_API_KEY: SecretStr | None = None
    TOGETHER_API_KEY: SecretStr | None = None

    VLLM: bool | None = False

    # If DEFAULT_MODEL is None, it will be set in model_post_init
    DEFAULT_MODEL: AllModelEnum | None = None  # type: ignore[assignment]
    AVAILABLE_MODELS: set[AllModelEnum] = set()  # type: ignore[assignment]

    def model_post_init(self, __context: Any) -> None:
        api_keys = {
            Provider.OPENAI: self.OPENAI_API_KEY,
            Provider.GROQ: self.GROQ_API_KEY,
            Provider.DEEPSEEK: self.DEEPSEEK_API_KEY,
            Provider.MISTRAL: self.MISTRAL_API_KEY,
            Provider.ANTHROPIC: self.ANTHROPIC_API_KEY,
            Provider.GOOGLE: self.GOOGLE_API_KEY,
            Provider.TOGETHER: self.TOGETHER_API_KEY,
        }
        active_keys = [k for k, v in api_keys.items() if v]
        if not active_keys:
            raise ValueError("At least one LLM API key must be provided.")

        for provider in active_keys:
            match provider:
                case Provider.GROQ:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = GroqModelName.LLAMA_33_70B
                    self.AVAILABLE_MODELS.update(set(GroqModelName))
                case Provider.TOGETHER:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = TogetherModelName.LLAMA_33_70B
                    self.AVAILABLE_MODELS.update(set(TogetherModelName))
                case Provider.GOOGLE:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = GoogleModelName.GEMINI_15_FLASH
                    self.AVAILABLE_MODELS.update(set(GoogleModelName))
                case Provider.OPENAI:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = OpenAIModelName.GPT_4O_MINI
                    self.AVAILABLE_MODELS.update(set(OpenAIModelName))
                case Provider.DEEPSEEK:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = DeepSeekModelName.DEEPSEEK_CHAT
                    self.AVAILABLE_MODELS.update(set(DeepSeekModelName))
                case Provider.MISTRAL:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = MistralModelName.NEMO_12B
                    self.AVAILABLE_MODELS.update(set(MistralModelName))
                case Provider.ANTHROPIC:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = AnthropicModelName.HAIKU_35
                    self.AVAILABLE_MODELS.update(set(AnthropicModelName))
                case _:
                    raise ValueError(f"Unknown provider: {provider}")

        if self.VLLM:
            if self.DEFAULT_MODEL is None:
                self.DEFAULT_MODEL = VLLMModelName.LLAMA_33_70B
            self.AVAILABLE_MODELS.update(set(VLLMModelName))


settings = Settings()


if __name__ == "__main__":
    for name in settings.COLLECTIONS:
        print(name)
