from typing import TYPE_CHECKING

import pytest

from mcp_agent.core.prompt import Prompt
from mcp_agent.llm.augmented_llm_passthrough import (
    CALL_TOOL_INDICATOR,
    FIXED_RESPONSE_INDICATOR,
    PassthroughLLM,
)

if TYPE_CHECKING:
    from mcp_agent.mcp.interfaces import AugmentedLLMProtocol
    from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


@pytest.mark.asyncio
async def test_simple_return():
    llm: AugmentedLLMProtocol = PassthroughLLM()
    response = await llm.generate(multipart_messages=[Prompt.user("playback message")])
    assert "assistant" == response.role
    assert "playback message" == response.first_text()


@pytest.mark.asyncio
async def test_concatenates_text_for_multiple_parts():
    llm: AugmentedLLMProtocol = PassthroughLLM()
    response = await llm.generate(
        multipart_messages=[
            Prompt.user("123abc"),
            Prompt.assistant("456def"),
            Prompt.user("789ghi"),
        ]
    )
    assert "assistant" == response.role
    assert "789ghi" in response.first_text()
    assert "456def" in response.first_text()
    assert "123abc" in response.first_text()


@pytest.mark.asyncio
async def test_set_fixed_return():
    llm: AugmentedLLMProtocol = PassthroughLLM()
    response: PromptMessageMultipart = await llm.generate(
        multipart_messages=[Prompt.user(f"{FIXED_RESPONSE_INDICATOR} foo")]
    )
    assert "foo" == response.first_text()

    response: PromptMessageMultipart = await llm.generate(
        multipart_messages=[Prompt.user("other messages respond with foo")]
    )
    assert "foo" == response.first_text()


@pytest.mark.asyncio
async def test_set_fixed_return_ignores_not_set():
    llm: AugmentedLLMProtocol = PassthroughLLM()
    response: PromptMessageMultipart = await llm.generate(
        multipart_messages=[Prompt.user(f"{FIXED_RESPONSE_INDICATOR}")]
    )
    assert "***FIXED_RESPONSE" == response.first_text()

    response: PromptMessageMultipart = await llm.generate(
        multipart_messages=[Prompt.user("ignored message")]
    )
    assert "ignored message" == response.first_text()


@pytest.mark.asyncio
async def test_parse_tool_call_no_args():
    llm: AugmentedLLMProtocol = PassthroughLLM()
    name, args = llm._parse_tool_command(f"{CALL_TOOL_INDICATOR} mcp_tool_name")
    assert "mcp_tool_name" == name
    assert None is args


@pytest.mark.asyncio
async def test_parse_tool_call_with_args():
    llm: AugmentedLLMProtocol = PassthroughLLM()
    name, args = llm._parse_tool_command(
        f'{CALL_TOOL_INDICATOR} mcp_tool_name_args {{"arg": "value"}}'
    )
    assert "mcp_tool_name_args" == name
    assert args is not None
    assert "value" == args["arg"]


# actual tool calling is covered in the integration tests
