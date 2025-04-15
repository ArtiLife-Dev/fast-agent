import os

from mcp_agent.core.exceptions import ProviderKeyError
from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.providers.augmented_llm_openai import OpenAIAugmentedLLM

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseekchat"  # current Deepseek only has two type models


class DeepSeekAugmentedLLM(OpenAIAugmentedLLM):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["provider_name"] = "Deepseek"  # Set provider name in kwargs
        super().__init__(*args, **kwargs)  # Properly pass args and kwargs to parent

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize Deepseek-specific default parameters"""
        chosen_model = kwargs.get("model", DEFAULT_DEEPSEEK_MODEL)

        return RequestParams(
            model=chosen_model,
            systemPrompt=self.instruction,
            parallel_tool_calls=True,
            max_iterations=10,
            use_history=True,
        )

    def _api_key(self) -> str:
        config = self.context.config
        api_key = None

        if config and config.deepseek:
            api_key = config.deepseek.api_key
            if api_key == "<your-api-key-here>":
                api_key = None

        if api_key is None:
            api_key = os.getenv("DEEPSEEK_API_KEY")

        if not api_key:
            raise ProviderKeyError(
                "DEEPSEEK API key not configured",
                "The DEEKSEEK API key is required but not set.\n"
                "Add it to your configuration file under deepseek.api_key\n"
                "Or set the DEEPSEEK_API_KEY environment variable",
            )
        return api_key

    def _base_url(self) -> str:
        if self.context.config and self.context.config.deepseek:
            base_url = self.context.config.deepseek.base_url

        return base_url if base_url else DEEPSEEK_BASE_URL
