from enum import StrEnum, auto
from typing import TypeAlias


class Provider(StrEnum):
    OPENAI = auto()
    ANTHROPIC = auto()
    GOOGLE = auto()
    GROQ = auto()
    OLLAMA = auto()
    MISTRAL = auto()
    DEEPSEEK = auto()
    AWS = auto()
    FAKE = auto()


class OpenAIModelName(StrEnum):
    """https://platform.openai.com/docs/models/gpt-4o"""

    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"


class DeepSeekModelName(StrEnum):
    """https://api-docs.deepseek.com/quick_start/pricing/"""

    DEEPSEEK_CHAT = "deepseek-chat"


class AnthropicModelName(StrEnum):
    """https://docs.anthropic.com/en/docs/about-claude/models#model-names"""

    HAIKU_35 = "claude-3-5-haiku-20241022"
    SONNET_35 = "claude-3-5-sonnet-20241022"


class GoogleModelName(StrEnum):
    """https://ai.google.dev/gemini-api/docs/models/gemini"""

    GEMINI_15_FLASH = "gemini-1.5-flash"
    GEMINI_15_FLASH_8B = "gemini-1.5-flash-8b"
    GEMINI_15_PRO = "gemini-1.5-pro"


class GroqModelName(StrEnum):
    """https://console.groq.com/docs/models"""

    LLAMA_33_70B = "groq-llama-3.3-70b-versatile"
    LLAMA_31_8B = "groq-llama-3.1-8b-instant"
    #GEMMA2_9B_IT = "groq-gemma2-9b-it"


class MistralModelName(StrEnum):
    """https://docs.mistral.ai/getting-started/models/models_overview/"""

    NEMO_12B = "open-mistral-nemo"


class OllamaModelName(StrEnum):
    """https://ollama.com/library"""

    GEMMA2_2B = "ollama-gemma2:2b"
    LLAMA_32_3B = "ollama-llama3.2"
    PHI3_4B = "ollama-phi3"


class AWSModelName(StrEnum):
    """https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html"""

    BEDROCK_HAIKU = "bedrock-3.5-haiku"


class FakeModelName(StrEnum):
    """Fake model for testing."""

    FAKE = "fake"


AllModelEnum: TypeAlias = (
    OpenAIModelName
    | AnthropicModelName
    | GoogleModelName
    | GroqModelName
    | AWSModelName
    | FakeModelName
    | MistralModelName
    | OllamaModelName
    | DeepSeekModelName
)
