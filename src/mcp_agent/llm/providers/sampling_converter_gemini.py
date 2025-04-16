from google.genai import types as genai_types
from google.genai.types import GenerateContentResponse
from mcp import StopReason
from mcp.types import PromptMessage

from mcp_agent.llm.providers.multipart_converter_gemini import GeminiConverter
from mcp_agent.llm.sampling_format_converter import ProviderFormatConverter
from mcp_agent.logging.logger import get_logger

_logger = get_logger(__name__)


class GeminiSamplingConverter(
    ProviderFormatConverter[genai_types.ContentDict, GenerateContentResponse]
):
    """
    Convert between Google Gemini and MCP types for sampling.
    """

    @classmethod
    def from_prompt_message(cls, message: PromptMessage) -> genai_types.ContentDict:
        """Convert an MCP PromptMessage to a Gemini ContentDict."""
        return GeminiConverter.convert_prompt_message_to_gemini(message)

    # Note: Stop reason conversion logic will likely reside in the
    # GeminiAugmentedLLM class when processing the response, as Gemini's
    # response structure differs from Anthropic's regarding stop reasons.


def mcp_stop_reason_to_gemini_finish_reason(stop_reason: StopReason):
    """Maps MCP StopReason to potential Gemini finish reasons (conceptual)."""
    # This mapping is conceptual as Gemini's finish reasons are outcomes, not inputs.
    if not stop_reason:
        return None
    elif stop_reason == "endTurn":
        return "STOP"
    elif stop_reason == "maxTokens":
        return "MAX_TOKENS"
    elif stop_reason == "stopSequence":
        # Gemini uses stop_sequences parameter, finish reason would be STOP
        return "STOP"
    elif stop_reason == "toolUse":
        # Gemini indicates tool use via function_call in response, not a finish reason
        return None # Or perhaps STOP if that's the typical reason alongside tool use?
    elif stop_reason == "error":
        return "OTHER" # Or SAFETY/RECITATION depending on context
    else:
        # Handle potential future MCP stop reasons or unknown values
        _logger.warning(f"Unknown MCP StopReason: {stop_reason}")
        return "OTHER"


def gemini_finish_reason_to_mcp_stop_reason(
    finish_reason: str | None, has_function_call: bool = False
) -> StopReason:
    """Maps Gemini finish reason to MCP StopReason."""
    # Prioritize tool use if present
    if has_function_call:
        return "toolUse"

    if not finish_reason:
        return "endTurn" # Default if no reason provided

    finish_reason_upper = finish_reason.upper()

    if finish_reason_upper == "STOP":
        return "endTurn"
    elif finish_reason_upper == "MAX_TOKENS":
        return "maxTokens"
    elif finish_reason_upper == "SAFETY":
        _logger.warning("Gemini response stopped due to safety settings.")
        return "error" # Map safety blocks to error for now
    elif finish_reason_upper == "RECITATION":
        _logger.warning("Gemini response stopped due to recitation.")
        return "error" # Map recitation blocks to error for now
    elif finish_reason_upper == "OTHER":
        _logger.warning("Gemini response stopped due to 'OTHER' reason.")
        return "error"
    else:
        _logger.warning(f"Unknown Gemini finish reason: {finish_reason}")
        return "error" # Map unknown reasons to error
