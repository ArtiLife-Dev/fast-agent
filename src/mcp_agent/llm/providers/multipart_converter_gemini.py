import json
from typing import Any, Dict, List, Sequence, Union, cast

from google.generativeai import types as genai_types
from mcp.types import ( # MCP Tool related types
    ListToolsResult,
    Schema,
    ToolInfo,
    # MCP Message related types
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    PromptMessage,
    TextContent,
    TextResourceContents,
)

from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.helpers.content_helpers import (
    get_blob_data,
    get_image_data,
    get_resource_uri,
    get_text,
    is_image_content,
    is_resource_content,
    is_text_content,
)
from mcp_agent.mcp.mime_utils import guess_mime_type, is_text_mime_type
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.mcp.schemas import normalize_schema_properties # Helper for schema normalization

_logger = get_logger("multipart_converter_gemini")

# List of image MIME types supported by Gemini API (add more as needed based on docs)
SUPPORTED_IMAGE_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/heic",
    "image/heif",
}
# List of audio MIME types supported by Gemini API
SUPPORTED_AUDIO_MIME_TYPES = {
    "audio/wav",
    "audio/mp3",
    "audio/aiff",
    "audio/aac",
    "audio/ogg",
    "audio/flac",
}
# List of video MIME types supported by Gemini API
SUPPORTED_VIDEO_MIME_TYPES = {
    "video/mp4",
    "video/mpeg",
    "video/mov",
    "video/avi",
    "video/x-flv",
    "video/mpg",
    "video/webm",
    "video/wmv",
    "video/3gpp",
}


class GeminiConverter:
    """Converts MCP message types to Google Gemini API format."""

    @staticmethod
    def _is_supported_image_type(mime_type: str) -> bool:
        return mime_type in SUPPORTED_IMAGE_MIME_TYPES

    @staticmethod
    def _is_supported_audio_type(mime_type: str) -> bool:
        return mime_type in SUPPORTED_AUDIO_MIME_TYPES

    @staticmethod
    def _is_supported_video_type(mime_type: str) -> bool:
        return mime_type in SUPPORTED_VIDEO_MIME_TYPES

    @staticmethod
    def convert_to_gemini(
        multipart_msg: PromptMessageMultipart,
    ) -> genai_types.ContentDict:
        """
        Convert a PromptMessageMultipart message to Gemini API ContentDict format.

        Args:
            multipart_msg: The PromptMessageMultipart message to convert

        Returns:
            A Gemini API ContentDict object
        """
        role = multipart_msg.role
        # Gemini uses 'model' for assistant role
        if role == "assistant":
            role = "model"
        # Gemini uses 'tool' for function response role
        # elif role == "tool": # Handled by create_tool_results_message
        #     role = "tool"

        gemini_parts: List[genai_types.Part] = []

        for content_item in multipart_msg.content:
            try:
                if is_text_content(content_item):
                    text = get_text(content_item)
                    if text is not None:
                        gemini_parts.append(genai_types.Part(text=text))
                    else:
                        _logger.warning("Skipping TextContent with None text.")

                elif is_image_content(content_item):
                    image_content: ImageContent = content_item
                    if GeminiConverter._is_supported_image_type(image_content.mimeType):
                        image_data = get_image_data(image_content)
                        if image_data:
                            gemini_parts.append(
                                genai_types.Part(
                                    inline_data=genai_types.Blob(
                                        mime_type=image_content.mimeType, data=image_data
                                    )
                                )
                            )
                        else:
                            _logger.warning(f"Skipping ImageContent with no data: {image_content.mimeType}")
                    else:
                        _logger.warning(
                            f"Skipping unsupported image type: {image_content.mimeType}"
                        )
                        gemini_parts.append(
                            genai_types.Part(text=f"[Unsupported Image: {image_content.mimeType}]")
                        )

                elif is_resource_content(content_item):
                    resource: EmbeddedResource = content_item
                    resource_content = resource.resource
                    uri_str = get_resource_uri(resource)
                    mime_type = getattr(resource_content, "mimeType", None)
                    if not mime_type and uri_str:
                        mime_type = guess_mime_type(uri_str) or "application/octet-stream"
                    elif not mime_type:
                         mime_type = "application/octet-stream" # Default if no URI or guess

                    # Handle specific resource types
                    if mime_type == "application/pdf":
                        blob_data = get_blob_data(resource)
                        if blob_data:
                             gemini_parts.append(
                                 genai_types.Part(
                                     inline_data=genai_types.Blob(
                                         mime_type=mime_type, data=blob_data
                                     )
                                 )
                             )
                        else:
                             _logger.warning(f"Skipping PDF resource with no data: {uri_str}")
                             gemini_parts.append(genai_types.Part(text=f"[PDF Resource without data: {uri_str}]"))
                    elif GeminiConverter._is_supported_audio_type(mime_type):
                        blob_data = get_blob_data(resource)
                        if blob_data:
                             gemini_parts.append(
                                 genai_types.Part(
                                     inline_data=genai_types.Blob(
                                         mime_type=mime_type, data=blob_data
                                     )
                                 )
                             )
                        else:
                             _logger.warning(f"Skipping Audio resource with no data: {uri_str}")
                             gemini_parts.append(genai_types.Part(text=f"[Audio Resource without data: {uri_str}]"))
                    elif GeminiConverter._is_supported_video_type(mime_type):
                        blob_data = get_blob_data(resource)
                        if blob_data:
                             gemini_parts.append(
                                 genai_types.Part(
                                     inline_data=genai_types.Blob(
                                         mime_type=mime_type, data=blob_data
                                     )
                                 )
                             )
                        else:
                             _logger.warning(f"Skipping Video resource with no data: {uri_str}")
                             gemini_parts.append(genai_types.Part(text=f"[Video Resource without data: {uri_str}]"))
                    elif is_text_mime_type(mime_type) or isinstance(resource_content, TextResourceContents):
                         # Handle other text-based resources (HTML, CSV, TXT, etc.)
                         text = get_text(resource)
                         if text is not None:
                             gemini_parts.append(genai_types.Part(text=text))
                         else:
                             _logger.warning(f"Could not extract text from resource: {uri_str} ({mime_type})")
                             gemini_parts.append(genai_types.Part(text=f"[Unreadable Text Resource: {uri_str}]"))
                    elif GeminiConverter._is_supported_image_type(mime_type):
                         # Handle images embedded as resources
                         image_data = get_image_data(resource) # Handles BlobResourceContents too
                         if image_data:
                             gemini_parts.append(
                                 genai_types.Part(
                                     inline_data=genai_types.Blob(
                                         mime_type=mime_type, data=image_data
                                     )
                                 )
                             )
                         else:
                             _logger.warning(f"Skipping Image resource with no data: {uri_str}")
                             gemini_parts.append(genai_types.Part(text=f"[Image Resource without data: {uri_str}]"))
                    else:
                        # Attempt to extract text as a last resort for unknown resources
                        fallback_text = get_text(resource)
                        if fallback_text:
                             _logger.warning(
                                 f"Converting unsupported resource type {mime_type} ({uri_str}) to text."
                             )
                             gemini_parts.append(genai_types.Part(text=fallback_text))
                        else:
                             _logger.warning(
                                 f"Skipping unsupported resource type in Gemini conversion: {mime_type} ({uri_str})"
                             )
                             gemini_parts.append(genai_types.Part(text=f"[Unsupported Resource: {mime_type} at {uri_str}]"))

                # Note: FunctionCall parts are generated by the model, not sent by the user.
                # FunctionResponse parts are handled by create_tool_results_message.
                else:
                     _logger.warning(f"Skipping unknown/unhandled content item type: {type(content_item)}")

            except Exception as e:
                _logger.error(f"Error converting content item: {content_item}. Error: {e}", exc_info=True)
                gemini_parts.append(genai_types.Part(text="[Error during content conversion]"))

        # Handle empty content - Gemini expects at least one part? Check SDK behavior.
        # For now, if parts list is empty, add an empty text part.
        if not gemini_parts:
            gemini_parts.append(genai_types.Part(text=""))

        return genai_types.ContentDict(role=role, parts=gemini_parts)

    @staticmethod
    def convert_prompt_message_to_gemini(
        message: PromptMessage,
    ) -> genai_types.ContentDict:
        """
        Convert a standard PromptMessage to Gemini API ContentDict format.

        Args:
            message: The PromptMessage to convert

        Returns:
            A Gemini API ContentDict object
        """
        multipart = PromptMessageMultipart(role=message.role, content=[message.content])
        return GeminiConverter.convert_to_gemini(multipart)

    @staticmethod
    def _convert_mcp_schema_to_gemini_schema(mcp_schema: Schema) -> genai_types.Schema:
        """Recursively convert MCP Schema dictionary to Gemini Schema object."""
        gemini_type_str = str(mcp_schema.get("type", "any")).upper()
        try:
            gemini_type = genai_types.Type[gemini_type_str]
        except KeyError:
            _logger.warning(f"Unsupported MCP schema type '{mcp_schema.get('type')}', defaulting to STRING.")
            gemini_type = genai_types.Type.STRING # Default fallback

        gemini_schema_args = {
            "type_": gemini_type,
            "description": mcp_schema.get("description"),
            "nullable": mcp_schema.get("nullable"), # Pass along if present
            "enum": mcp_schema.get("enum"),
            "format": mcp_schema.get("format"), # Pass along if present
        }

        if gemini_type == genai_types.Type.OBJECT and "properties" in mcp_schema:
            gemini_schema_args["properties"] = {
                prop_name: GeminiConverter._convert_mcp_schema_to_gemini_schema(prop_schema)
                for prop_name, prop_schema in mcp_schema["properties"].items()
            }
            gemini_schema_args["required"] = mcp_schema.get("required")
        elif gemini_type == genai_types.Type.ARRAY and "items" in mcp_schema:
            gemini_schema_args["items"] = GeminiConverter._convert_mcp_schema_to_gemini_schema(
                mcp_schema["items"]
            )

        # Filter out None values before creating the Schema object
        filtered_args = {k: v for k, v in gemini_schema_args.items() if v is not None}

        return genai_types.Schema(**filtered_args)


    @staticmethod
    def convert_mcp_tools_to_gemini(mcp_tools: List[ToolInfo]) -> List[genai_types.Tool]:
        """Convert a list of MCP ToolInfo objects to Gemini Tool objects."""
        gemini_tools: List[genai_types.Tool] = []
        function_declarations: List[genai_types.FunctionDeclaration] = []

        for tool in mcp_tools:
            try:
                # Normalize schema before conversion
                normalized_schema = normalize_schema_properties(tool.inputSchema or {})

                parameters = None
                if normalized_schema:
                     parameters = GeminiConverter._convert_mcp_schema_to_gemini_schema(
                         cast(Schema, normalized_schema) # Cast after normalization
                     )

                declaration = genai_types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description or "", # Ensure description is not None
                    parameters=parameters,
                )
                function_declarations.append(declaration)
            except Exception as e:
                _logger.error(f"Failed to convert MCP tool '{tool.name}' to Gemini format: {e}", exc_info=True)

        if function_declarations:
            gemini_tools.append(genai_types.Tool(function_declarations=function_declarations))

        return gemini_tools

    @staticmethod
    def convert_tool_result_to_gemini_part(
        tool_result: CallToolResult, function_name: str
    ) -> genai_types.Part:
        """
        Convert an MCP CallToolResult to a Gemini FunctionResponse Part.
        Serializes the result content, prioritizing text.
        """
        response_content: Any = None
        error_occurred = tool_result.isError

        if tool_result.content:
            # Prioritize the first text content block if available
            text_parts = [item for item in tool_result.content if is_text_content(item)]
            if text_parts:
                response_content = get_text(text_parts[0])
                # Attempt to parse if it looks like JSON, otherwise keep as string
                if isinstance(response_content, str):
                    stripped_text = response_content.strip()
                    if (stripped_text.startswith('{') and stripped_text.endswith('}')) or \
                       (stripped_text.startswith('[') and stripped_text.endswith(']')):
                        try:
                            response_content = json.loads(stripped_text)
                        except json.JSONDecodeError:
                            _logger.debug("Tool result text looked like JSON but failed to parse, keeping as string.")
                            pass # Keep as string if parsing fails
            else:
                # Fallback: Create a summary string if no text content
                summary = f"[Tool '{function_name}' executed"
                if error_occurred:
                    summary += " with error"
                summary += f", {len(tool_result.content)} non-text parts returned]"
                response_content = summary
                _logger.warning(f"Tool result for '{function_name}' has no text content. Using summary: {summary}")
        elif error_occurred:
             response_content = f"[Tool '{function_name}' failed with no specific output]"
        else:
             response_content = f"[Tool '{function_name}' executed successfully with no output]"


        # Gemini expects the 'response' field in FunctionResponse
        # to contain the structured data or error message.
        function_response_data = {"result": response_content}
        if error_occurred:
            function_response_data = {"error": str(response_content)} # Structure error clearly

        return genai_types.Part(
            function_response=genai_types.FunctionResponse(
                name=function_name,
                response=function_response_data,
            )
        )

    @staticmethod
    def create_tool_results_message(
        tool_results: List[tuple[str, CallToolResult]], # List of (function_name, CallToolResult)
    ) -> genai_types.ContentDict:
        """
        Create a 'tool' role message containing function results for Gemini.
        """
        tool_response_parts: List[genai_types.Part] = []
        for function_name, result in tool_results:
            try:
                part = GeminiConverter.convert_tool_result_to_gemini_part(
                    tool_result=result, function_name=function_name
                )
                tool_response_parts.append(part)
            except Exception as e:
                 _logger.error(f"Failed to convert tool result for '{function_name}': {e}", exc_info=True)
                 # Add an error part to the message
                 tool_response_parts.append(genai_types.Part(
                     function_response=genai_types.FunctionResponse(
                         name=function_name,
                         response={"error": f"Failed to process tool result: {e}"}
                     )
                 ))

        # Gemini expects tool responses in a message with role 'tool'
        return genai_types.ContentDict(role="tool", parts=tool_response_parts)

    # TODO: Implement creation of tool role message in Phase 4
    # @staticmethod
    # def convert_tool_result_to_gemini(
    #     tool_result: CallToolResult, function_name: str
    # ) -> genai_types.Part:
    #     """
    #     Convert an MCP CallToolResult to a Gemini FunctionResponse Part.
    #     """
    #     pass

    # TODO: Implement creation of tool role message in Phase 4
    # @staticmethod
    # def create_tool_results_message(
    #     tool_results: List[tuple[str, CallToolResult]], # Assuming str is function_name called
    # ) -> genai_types.ContentDict:
    #     """
    #     Create a 'tool' role message containing function results for Gemini.
    #     """
    #     pass
