"""
Unit tests for agent types and their interactions with the interactive prompt.
"""

from mcp_agent.agents.agent import Agent
from mcp_agent.core.agent_types import AgentConfig, AgentType


def test_agent_type_default():
    """Test that agent_type defaults to AgentType.BASIC.value"""
    agent = Agent(config=AgentConfig(name="test_agent"))
    assert agent.agent_type == AgentType.BASIC.value


def test_agent_type_custom():
    """Test that agent_type can be set to a custom value"""
    agent = Agent(config=AgentConfig(name="test_agent", agent_type="custom_type"))
    assert agent.agent_type == "custom_type"


def test_agent_type_from_enum():
    """Test that agent_type can be set from an enum value"""
    agent = Agent(config=AgentConfig(name="test_agent", agent_type=AgentType.ROUTER.value))
    assert agent.agent_type == AgentType.ROUTER.value