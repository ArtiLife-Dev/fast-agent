from typing import TYPE_CHECKING

import pytest

from mcp_agent.agents.agent import Agent
from mcp_agent.core.agent_types import AgentConfig
from mcp_agent.core.prompt import Prompt
from mcp_agent.llm.augmented_llm_playback import PlaybackLLM
from mcp_agent.llm.model_factory import ModelFactory

if TYPE_CHECKING:
    from mcp_agent.mcp.interfaces import AugmentedLLMProtocol


@pytest.mark.asyncio
async def test_model_factory_creates_playback():
    """Test that ModelFactory correctly creates a PlaybackLLM instance"""
    # Create a factory for the playback model
    factory = ModelFactory.create_factory("playback")

    # Verify the factory is callable
    assert callable(factory)

    # Create an instance using the factory
    instance = factory(
        Agent(
            AgentConfig(name="playback_agent", instruction="Helpful AI Agent", servers=[]),
            context=None,
        )
    )

    # Verify the instance is a PlaybackLLM
    assert isinstance(instance, PlaybackLLM)


@pytest.mark.asyncio
async def test_basic_playback_no_mock():
    """Test that ModelFactory correctly creates a PlaybackLLM instance"""

    llm: AugmentedLLMProtocol = PlaybackLLM()
    result = await llm.generate([Prompt.user("hello, world!")])
    assert "HISTORY LOADED" == result.first_text()


@pytest.mark.asyncio
async def test_simple_playback_functionality():
    llm: AugmentedLLMProtocol = PlaybackLLM()
    await llm.generate(
        [
            Prompt.user("message 1"),
            Prompt.assistant("response 1"),
            Prompt.user("message 2"),
            Prompt.assistant("response 2"),
        ],
    )
    response1 = await llm.generate([Prompt.user("evalstate")])
    response2 = await llm.generate([Prompt.user("llmindset")])
    assert "response 1" == response1.first_text()
    assert "response 2" == response2.first_text()


@pytest.mark.asyncio
async def test_exhaustion_behaviour():
    llm: AugmentedLLMProtocol = PlaybackLLM()
    await llm.generate(
        [
            Prompt.user("message 1"),
            Prompt.assistant("response 1"),
        ],
    )
    response1 = await llm.generate([Prompt.user("evalstate")])
    response2 = await llm.generate([Prompt.user("llmindset")])
    assert "response 1" == response1.first_text()
    assert "MESSAGES EXHAUSTED" in response2.first_text()
    assert "(0 overage)" in response2.first_text()

    for _ in range(3):
        overage = await llm.generate([Prompt.user("overage?")])
        assert f"({_ + 1} overage)" in overage.first_text()
