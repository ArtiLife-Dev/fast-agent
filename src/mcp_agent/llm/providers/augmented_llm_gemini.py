import os
from typing import TYPE_CHECKING, List, Optional

import google.genai as genai
from google.genai import types as genai_types
from google.genai.types import ContentDict, FunctionCall, GenerateContentResponse
from google.protobuf.json_format import MessageToDict  # Needed for tool arg conversion
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    EmbeddedResource,
    ImageContent,
    TextContent,
)
from rich.text import Text  # For displaying messages

from mcp_agent.core.exceptions import ProviderKeyError
from mcp_agent.llm.augmented_llm import AugmentedLLM, RequestParams
from mcp_agent.llm.providers.multipart_converter_gemini import GeminiConverter
from mcp_agent.llm.providers.sampling_converter_gemini import (
    GeminiSamplingConverter,
    gemini_finish_reason_to_mcp_stop_reason,
)
from mcp_agent.logging.logger import get_logger

# Import necessary content helpers
from mcp_agent.mcp.helpers.content_helpers import get_text, is_text_content
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

if TYPE_CHECKING:
    from mcp import ListToolsResult

    from mcp_agent.config import Settings


# Default model if not specified
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"


class GeminiAugmentedLLM(AugmentedLLM[ContentDict, GenerateContentResponse]):
    """
    AugmentedLLM implementation for Google Gemini models.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.provider = "Google"
        self.logger = get_logger(__name__)
        # Initialize client once during instantiation
        self._client: Optional[genai.GenerativeModel] = None
        super().__init__(*args, type_converter=GeminiSamplingConverter, **kwargs)

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize Gemini-specific default parameters"""
        return RequestParams(
            model=kwargs.get("model", DEFAULT_GEMINI_MODEL),
            # Gemini uses 'max_output_tokens' in generation_config
            maxTokens=kwargs.get("max_output_tokens", 8192), # Default for gemini-1.5-flash
            systemPrompt=self.instruction, # Gemini uses 'system_instruction' in generation_config
            # parallel_tool_calls=True, # Gemini handles parallel calls implicitly
            max_iterations=10, # Internal loop limit for tool calls
            use_history=True,
        )

    def _api_key(self, config: "Settings") -> str:
        """Retrieve the Google API key."""
        api_key = None

        if hasattr(config, "google") and config.google:
            api_key = config.google.api_key
            if api_key == "<your-api-key-here>":
                api_key = None

        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY") # Gemini SDK uses GOOGLE_API_KEY

        if not api_key:
            raise ProviderKeyError(
                "Google API key not configured",
                "The Google API key is required but not set.\n"
                "Add it to your configuration file under google.api_key "
                "or set the GOOGLE_API_KEY environment variable.",
            )

        return api_key

    def _get_client(self, model_name: str) -> genai.GenerativeModel:
        """Initializes and returns the Gemini client."""
        if self._client and self._client.model_name == f"models/{model_name}":
             return self._client

        api_key = self._api_key(self.context.config)
        genai.configure(api_key=api_key)
        # TODO: Add support for Vertex AI client initialization if needed later
        # client = genai.Client(vertexai=True, project='your-project-id', location='us-central1')
        self._client = genai.GenerativeModel(model_name)
        return self._client

    async def generate_internal(
        self,
        message_param: ContentDict, # This represents the latest user message content
        request_params: RequestParams | None = None,
    ) -> list[TextContent | ImageContent | EmbeddedResource]:
        """
        Process a query using the Gemini LLM and available tools.
        Handles history, tool calls, and response conversion.
        """
        params = self.get_request_params(request_params)
        model_name = params.model or DEFAULT_GEMINI_MODEL
        client = self._get_client(model_name)

        # Prepare history and current message
        history: List[ContentDict] = []
        if params.use_history:
             # Convert MCP history (which should be List[ContentDict] for Gemini)
             # Assuming self.history.get() returns the correct format for Gemini
             history = self.history.get(include_history=True) # Ensure history is List[ContentDict]

        # The message_param is the latest user message, already in ContentDict format
        current_contents = history + [message_param]

        # --- Tool Handling ---
        tool_list: ListToolsResult = await self.aggregator.list_tools()
        available_mcp_tools: List[genai_types.Tool] = []
        if tool_list.tools:
            available_mcp_tools = GeminiConverter.convert_mcp_tools_to_gemini(tool_list.tools)
            self.logger.debug(f"Converted MCP tools to Gemini format: {available_mcp_tools}")

        # --- Built-in Tool Handling ---
        gemini_native_tools = []
        # Check model name variant and add built-in tools
        # Note: Using simple string checks. More robust parsing might be needed if names get complex.
        # Extract base model name for checks if needed, e.g., model_name.replace("-code","").replace("-search","")
        if "-code" in model_name:
            self.logger.debug(f"Enabling Code Execution for model: {model_name}")
            gemini_native_tools.append(genai_types.Tool(code_execution=genai_types.CodeExecution())) # Correct type
        if "-search" in model_name:
             self.logger.debug(f"Enabling Google Search for model: {model_name}")
             # Use GoogleSearch for 2.0+ models, GoogleSearchRetrieval for 1.5
             # Assuming model names clearly indicate version or using a helper to determine this.
             # For now, let's assume GoogleSearch is appropriate for the target models (2.0 flash, 2.5 pro).
             gemini_native_tools.append(genai_types.Tool(google_search=genai_types.GoogleSearch()))
             # If targeting 1.5 models specifically:
             # gemini_native_tools.append(genai_types.Tool(google_search_retrieval=genai_types.GoogleSearchRetrieval()))

        all_tools = available_mcp_tools + gemini_native_tools

        # Prepare generation config
        generation_config = genai_types.GenerationConfig(
            # candidate_count=params.candidate_count or 1, # If needed
            # stop_sequences=params.stopSequences, # If needed
            max_output_tokens=params.maxTokens,
            temperature=params.temperature,
            # top_p=params.top_p, # If needed
            # top_k=params.top_k, # If needed
        )
        safety_settings = None # TODO: Add safety settings if needed

        system_instruction = params.systemPrompt or self.instruction

        # --- Main Loop for Tool Calls (Simplified for now) ---
        final_response_content: list[TextContent | ImageContent | EmbeddedResource] = []
        current_turn = 1 # Track turns for logging/debugging

        while current_turn <= params.max_iterations:
            self._log_chat_progress(self.chat_turn(), model=model_name)
            self.logger.debug(f"Calling Gemini. Contents: {current_contents}, Config: {generation_config}, Tools: {all_tools}")

            try:
                response = await client.generate_content_async(
                    contents=current_contents,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    tools=all_tools if all_tools else None,
                    system_instruction=system_instruction if system_instruction else None,
                )
                self.logger.debug(f"Gemini response turn {current_turn}:", data=response)

            except Exception as e:
                self.logger.error(f"Error calling Gemini API: {e}", exc_info=True)
                # TODO: Improve error handling (e.g., specific API errors)
                final_response_content.append(TextContent(type="text", text=f"[Error: {e}]"))
                break # Exit loop on error

            # --- Process Response ---
            if not response.candidates:
                 self.logger.warning("Gemini response has no candidates.")
                 final_response_content.append(TextContent(type="text", text="[No response from model]"))
                 break

            candidate = response.candidates[0]
            content = candidate.content # This is a ContentDict

            # Check for function calls and code execution results
            # Use getattr for safer access
            has_function_call = any(getattr(part, 'function_call', None) for part in content.parts)
            has_code_execution = any(getattr(part, 'executable_code', None) or getattr(part, 'code_execution_result', None) for part in content.parts)


            # Extract text and other content (Phase 3 & 5)
            turn_responses: list[TextContent | ImageContent | EmbeddedResource] = []
            for part in content.parts:
                if getattr(part, 'text', None):
                    turn_responses.append(TextContent(type="text", text=part.text))
                elif getattr(part, 'inline_data', None):
                    # Handle inline data (Images, Blobs from Phase 3)
                    blob = part.inline_data
                    # Attempt to create ImageContent if mime type matches, else skip/log
                    if GeminiConverter._is_supported_image_type(blob.mime_type):
                         # Assuming get_image_data can handle base64 string directly if needed
                         # Or just pass the blob data if ImageContent accepts bytes
                         # Ensure data is bytes
                         img_data = blob.data if isinstance(blob.data, bytes) else b''
                         if img_data:
                              turn_responses.append(ImageContent(type="image", mimeType=blob.mime_type, data=img_data))
                         else:
                              self.logger.warning(f"Received image blob with no data: {blob.mime_type}")
                              turn_responses.append(TextContent(type="text", text=f"[Empty Image Data: {blob.mime_type}]"))
                    else:
                          # Handle other blob types (PDF, Audio, Video) - maybe as EmbeddedResource?
                          # For now, represent as text placeholder
                          self.logger.warning(f"Received unhandled inline_data blob: {blob.mime_type}")
                          turn_responses.append(TextContent(type="text", text=f"[Inline Data: {blob.mime_type}]"))
                elif getattr(part, 'executable_code', None):
                    # Handle Code Execution Request (Phase 5)
                    # Format as markdown code block for display
                    code_text = f"\n```python\n# Code Execution Request:\n{part.executable_code.code}\n```\n"
                    turn_responses.append(TextContent(type="text", text=code_text))
                elif getattr(part, 'code_execution_result', None):
                    # Handle Code Execution Result (Phase 5)
                    # Format as markdown block for display
                    result_text = f"\n```\n# Code Execution Result:\n{part.code_execution_result.output}\n```\n"
                    turn_responses.append(TextContent(type="text", text=result_text))
                elif getattr(part, 'function_call', None):
                    # Function calls are handled in the tool use loop, don't add to text response here
                    pass
                else:
                    # Log unexpected part types
                    try:
                        part_dict = MessageToDict(part._pb)
                        self.logger.warning(f"Unhandled Gemini response part type: {part_dict}")
                        turn_responses.append(TextContent(type="text", text=f"[Unhandled Part: {part_dict.get('dataType', 'Unknown')}]"))
                    except Exception:
                         self.logger.warning(f"Unhandled Gemini response part type: {type(part)}")
                         turn_responses.append(TextContent(type="text", text="[Unhandled Part Type]"))

            final_response_content.extend(turn_responses) # Add text/media from this turn

            # Determine stop reason (consider code execution as a form of 'tool use' for stopping)
            # Note: Gemini API might return STOP even with code execution if it's the final step.
            # We prioritize checking for function calls first. If none, check finish reason.
            is_tool_use_stop = has_function_call # Add OR has_code_execution if needed? Check API behavior.

            mcp_stop_reason = gemini_finish_reason_to_mcp_stop_reason(
                candidate.finish_reason, has_function_call=is_tool_use_stop
            )
            self.logger.debug(f"Turn {current_turn}: FinishReason='{candidate.finish_reason}', HasFunctionCall={has_function_call}, HasCodeExec={has_code_execution} -> MCP StopReason='{mcp_stop_reason}'")


            if mcp_stop_reason != "toolUse":
                # Normal stop (endTurn, maxTokens, error, etc.)
                await self.show_assistant_message(
                    "".join(get_text(r) for r in turn_responses if is_text_content(r))
                )
                if mcp_stop_reason == "maxTokens":
                     # Add max tokens warning if needed
                     pass
                break # Exit loop

            # --- Handle Tool Use ---
            self.logger.info(f"Turn {current_turn}: Tool use detected.")

            # Extract function calls from the response content parts
            function_calls_to_process: List[FunctionCall] = []
            for part in content.parts:
                if part.function_call:
                    function_calls_to_process.append(part.function_call)

            if not function_calls_to_process:
                 self.logger.warning("Stop reason was 'toolUse' but no function calls found in parts.")
                 break # Avoid infinite loop if something went wrong

            tool_results_for_next_turn = []
            assistant_message_text = "".join(get_text(r) for r in turn_responses if is_text_content(r))

            for i, func_call in enumerate(function_calls_to_process):
                tool_name = func_call.name
                # Gemini args are Struct/dict-like, convert if necessary or pass directly
                # The Gemini SDK returns args as a google.protobuf.struct_pb2.Struct
                # Convert it to a standard Python dict for MCP
                tool_args = MessageToDict(func_call.args) if func_call.args else {}

                if i == 0: # Show assistant message only before the first tool call of the turn
                    display_text = assistant_message_text if assistant_message_text else Text("the assistant requested tool calls", style="dim green italic")
                    await self.show_assistant_message(display_text, tool_name)

                # Find the corresponding MCP tool info for display (optional but helpful)
                mcp_tool_info = next((t for t in tool_list.tools if t.name == tool_name), None)
                self.show_tool_call(
                    available_tools=[mcp_tool_info] if mcp_tool_info else [], # Pass tool info for display
                    tool_name=tool_name,
                    tool_args=tool_args
                )

                # Create MCP CallToolRequest
                tool_call_request = CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name=tool_name, arguments=tool_args),
                )

                # Execute the MCP tool
                # Note: Gemini doesn't have a tool_call_id like Anthropic. We use the function name.
                result = await self.call_tool(
                    request=tool_call_request, tool_call_id=tool_name # Use name as identifier
                )
                self.show_tool_result(result)

                # Store result for the next turn's message
                tool_results_for_next_turn.append((tool_name, result))
                final_response_content.extend(result.content) # Add tool result content to final output

            # Prepare the 'tool' role message with results
            tool_response_message = GeminiConverter.create_tool_results_message(
                tool_results_for_next_turn
            )

            # Append the model's response (containing the function calls)
            # and the tool results message to the conversation history for the next iteration
            current_contents.append(content) # Model's response with function_call parts
            current_contents.append(tool_response_message) # User/Tool response with function_response parts

            current_turn += 1 # Increment turn counter and continue loop

        # --- End Loop ---

        # Save history if enabled
        if params.use_history:
            # Assuming current_contents now holds the full conversation for this call
            # Need to filter out the initial prompt messages if they were included
            prompt_messages_count = len(self.history.get(include_history=False))
            new_messages = current_contents[prompt_messages_count:]
            self.history.set(new_messages) # Ensure history stores List[ContentDict]

        self._log_chat_finished(model=model_name)
        return final_response_content


    async def generate_messages(
        self,
        message_param: ContentDict, # Represents the latest user message
        request_params: RequestParams | None = None,
    ) -> PromptMessageMultipart:
        """
        Process a query using the Gemini LLM and return PromptMessageMultipart.
        """
        res = await self.generate_internal(
            message_param=message_param,
            request_params=request_params,
        )
        # Assuming the response is primarily text for now
        return PromptMessageMultipart.assistant(*res)

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: List["PromptMessageMultipart"],
        request_params: RequestParams | None = None,
    ) -> PromptMessageMultipart:
        """
        Apply a prompt (potentially multi-message) to the Gemini LLM.
        Sets the history and calls generate_messages if the last message is from the user.
        """
        if not multipart_messages:
            self.logger.warning("Attempted to apply an empty prompt.")
            # Or raise an error? Returning empty assistant for now.
            return PromptMessageMultipart.assistant()

        # Convert all MCP messages to Gemini ContentDict format
        # This assumes the history mechanism expects List[ContentDict]
        gemini_history_content: List[ContentDict] = []
        for msg in multipart_messages:
             try:
                 converted_msg = GeminiConverter.convert_to_gemini(msg)
                 gemini_history_content.append(converted_msg)
             except Exception as e:
                 self.logger.error(f"Failed to convert prompt message {msg.role}: {e}", exc_info=True)
                 # Skip problematic message or handle error? Skipping for now.

        if not gemini_history_content:
             self.logger.error("Failed to convert any prompt messages.")
             return PromptMessageMultipart.assistant(TextContent(type="text", text="[Error converting prompt]"))


        # Check the role of the last *successfully converted* message
        last_message_role = gemini_history_content[-1].get("role")

        # Determine messages to add to history vs. the message to generate from
        messages_to_set_in_history = []
        message_to_generate_from = None

        if last_message_role == "user":
            messages_to_set_in_history = gemini_history_content[:-1]
            message_to_generate_from = gemini_history_content[-1]
            self.logger.debug(f"Applying prompt ending with user message. History size: {len(messages_to_set_in_history)}")
        elif last_message_role == "model":
            # If the prompt ends with a model message, set the full prompt as history
            # and return the last model message directly without calling the LLM.
            messages_to_set_in_history = gemini_history_content
            self.logger.debug(f"Applying prompt ending with model message. History size: {len(messages_to_set_in_history)}")
            # Convert the last Gemini ContentDict back to MCP PromptMessageMultipart
            last_model_message_content = messages_to_set_in_history[-1].get("parts", [])
            mcp_content = []
            for part in last_model_message_content:
                 # Basic conversion back - assumes text for now
                 if part.get("text"):
                     mcp_content.append(TextContent(type="text", text=part["text"]))
                 # TODO: Add reverse conversion for other types if needed
            self.history.set(messages_to_set_in_history, is_prompt=True)
            return PromptMessageMultipart(role="assistant", content=mcp_content)
        else:
            # Handle unexpected last role (e.g., 'tool' shouldn't be the last in a prompt)
             self.logger.warning(f"Prompt ended with unexpected role: {last_message_role}. Treating as complete.")
             messages_to_set_in_history = gemini_history_content
             self.history.set(messages_to_set_in_history, is_prompt=True)
             return PromptMessageMultipart.assistant(TextContent(type="text", text="[Prompt ended unexpectedly]"))

        # Set the history (excluding the last user message)
        self.history.set(messages_to_set_in_history, is_prompt=True)

        # Generate response based on the last user message
        # generate_internal handles adding the message_to_generate_from to the history it gets
        return await self.generate_messages(message_to_generate_from, request_params)

    @classmethod
    def convert_message_to_message_param(cls, message: GenerateContentResponse, **kwargs) -> ContentDict:
        """
        Convert a Gemini response object (GenerateContentResponse) to the
        input format (ContentDict) needed for history.
        """
        # Gemini's history format is List[ContentDict].
        # A response needs to be converted back into a ContentDict with role='model'.
        if message.candidates:
            # Use the content from the first candidate.
            # The .content attribute is already a ContentDict.
            candidate_content = message.candidates[0].content
            # Ensure the role is explicitly 'model' for history consistency.
            # The candidate.content should already have role='model'.
            if candidate_content.role != "model":
                 _logger = get_logger(__name__) # Get logger instance locally
                 _logger.warning(f"Gemini response candidate content had unexpected role '{candidate_content.role}', forcing to 'model'.")
                 # Create a new dict to avoid modifying the original response object
                 return {"role": "model", "parts": candidate_content.parts}
            return candidate_content
        else:
            # Handle cases where there are no candidates (e.g., blocked response)
            _logger = get_logger(__name__) # Get logger instance locally
            _logger.warning("Gemini response has no candidates, creating empty model message for history.")
            # Returning an empty ContentDict might cause issues, return one with empty text part.
            return {"role": "model", "parts": [{"text": ""}]}
