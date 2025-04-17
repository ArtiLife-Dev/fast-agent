"""
Type definitions for LLM providers.
"""

from enum import Enum


class Provider(Enum):
    """Supported LLM providers"""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    FAST_AGENT = "fast-agent"
    DEEPSEEK = "deepseek"
    GENERIC = "generic"
    OPENROUTER = "openrouter"
    GOOGLE = "google"  # Add Google provider
    AZUREOPENAI = "azureopenai"  # Add Azure OpenAI provider
    