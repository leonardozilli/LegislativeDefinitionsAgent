from pathlib import Path
from typing import Any, Dict, Set

import yaml
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

from legaldefagent.core.schema.models import (
    AllModelEnum,
    AnthropicModelName,
    DeepSeekModelName,
    GoogleModelName,
    GroqModelName,
    MistralModelName,
    OpenAIModelName,
    Provider,
    TogetherModelName,
    VLLMModelName,
)
from legaldefagent.core.utils import get_available_vllm_models


load_dotenv(find_dotenv(raise_error_if_not_found=True), override=False)


class CollectionConfig(BaseModel):
    namespace: str | None = None
    jurisdiction: str | None = None


class ExistDBConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    user: str | None = None
    password: str | None = None


class MilvusDBConfig(BaseModel):
    path: str = "data/vdb/definitions.db"
    collection_name: str = "Definitions"


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class UIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 3000


class VLLMConfig(BaseModel):
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8001


def yaml_config_settings_source() -> Dict[str, Any]:
    """
    Loads configuration from config.yaml.
    Returns an empty dict if the file is missing.
    """
    yaml_file = Path("config.yaml")
    if not yaml_file.exists():
        return {}

    with open(yaml_file, encoding="utf-8") as f:
        return yaml.safe_load(f)


class Settings(BaseSettings):
    collections: Dict[str, CollectionConfig] = {}
    consolidated_collection: str | None = None

    existdb: ExistDBConfig = ExistDBConfig()
    api: APIConfig = APIConfig()
    ui: UIConfig = UIConfig()

    langchain_tracing: bool = False

    auth_secret: SecretStr | None = None

    openai_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    mistral_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    deepseek_api_key: SecretStr | None = None
    together_api_key: SecretStr | None = None

    vllm: VLLMConfig = VLLMConfig()
    milvusdb: MilvusDBConfig = MilvusDBConfig()

    default_model: AllModelEnum | str | None = Field(default=None, alias="default_model")
    available_models: Set[AllModelEnum] = set()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple:
        """Defines the priority of config sources:

        1. Constructor arguments (init)
        2. Environment Variables (.env)
        3. YAML Config file (config.yaml)
        4. Defaults
        """
        return (
            init_settings,
            env_settings,
            yaml_config_settings_source,
            file_secret_settings,
        )

    def model_post_init(self, __context: Any) -> None:
        api_keys = {
            Provider.OPENAI: self.openai_api_key,
            Provider.GROQ: self.groq_api_key,
            Provider.DEEPSEEK: self.deepseek_api_key,
            Provider.MISTRAL: self.mistral_api_key,
            Provider.ANTHROPIC: self.anthropic_api_key,
            Provider.GOOGLE: self.google_api_key,
            Provider.TOGETHER: self.together_api_key,
        }

        active_keys = [k for k, v in api_keys.items() if v]

        if not active_keys and not self.vllm.enabled:
            raise ValueError("At least one LLM API key must be provided unless VLLM is enabled.")

        for provider in active_keys:
            match provider:
                case Provider.GROQ:
                    if self.default_model is None:
                        self.default_model = GroqModelName.LLAMA_33_70B
                    self.available_models.update(set(GroqModelName))
                case Provider.TOGETHER:
                    if self.default_model is None:
                        self.default_model = TogetherModelName.LLAMA_33_70B
                    self.available_models.update(set(TogetherModelName))
                case Provider.GOOGLE:
                    if self.default_model is None:
                        self.default_model = GoogleModelName.GEMINI_15_FLASH
                    self.available_models.update(set(GoogleModelName))
                case Provider.OPENAI:
                    if self.default_model is None:
                        self.default_model = OpenAIModelName.GPT_4O_MINI
                    self.available_models.update(set(OpenAIModelName))
                case Provider.DEEPSEEK:
                    if self.default_model is None:
                        self.default_model = DeepSeekModelName.DEEPSEEK_CHAT
                    self.available_models.update(set(DeepSeekModelName))
                case Provider.MISTRAL:
                    if self.default_model is None:
                        self.default_model = MistralModelName.NEMO_12B
                    self.available_models.update(set(MistralModelName))
                case Provider.ANTHROPIC:
                    if self.default_model is None:
                        self.default_model = AnthropicModelName.HAIKU_35
                    self.available_models.update(set(AnthropicModelName))
                case _:
                    raise ValueError(f"Unknown provider: {provider}")

        if self.vllm.enabled:
            try:
                available_vllm_models = get_available_vllm_models(self.vllm.host, self.vllm.port)
            except RuntimeError as e:
                raise ValueError(f"Failed to retrieve VLLM models: {e}") from e
            if not available_vllm_models:
                raise ValueError("No VLLM models are available from the VLLM endpoint.")

            self.available_models.update(set(available_vllm_models))

            if self.default_model is None:
                self.default_model = available_vllm_models[0]

        if self.default_model is None:
            raise ValueError("No default model could be determined.")


settings = Settings()
