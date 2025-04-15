import pytest

from mcp_agent.core.prompt import Prompt


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chaining_routers(fast_agent):
    """Check that the router routes"""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    @fast.agent(name="target1")
    @fast.agent(name="target2")
    @fast.agent(name="target3")
    @fast.router(name="router1", agents=["target1", "target2"])
    @fast.chain(name="chain", sequence=["router1", "target3"], cumulative=True)
    async def agent_function():
        async with fast.run() as agent:
            await agent.router1._llm.generate(
                [
                    Prompt.user(
                        """***FIXED_RESPONSE 
                        {"agent": "target2",
                        "confidence": "high",
                        "reasoning": "Test Request"}"""
                    )
                ]
            )
            result = await agent.chain.send("github.com/varaarul")
            assert "github.com/varaarul" in result
            assert "target3" in result

    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_router_selects_parallel(fast_agent):
    """Check that the router routes"""
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent

    @fast.agent(name="target1")
    @fast.agent(name="target2")
    @fast.agent(name="target3")
    @fast.parallel(name="parallel", fan_out=["target2", "target3"])
    @fast.router(name="router1", agents=["target1", "parallel"])
    async def agent_function():
        async with fast.run() as agent:
            await agent.router1._llm.generate(
                [
                    Prompt.user(
                        """***FIXED_RESPONSE 
                        {"agent": "parallel",
                        "confidence": "high",
                        "reasoning": "Test Request"}"""
                    )
                ]
            )
            result = await agent.router1.send("github.com/varaarul")
            assert "github.com/varaarul" in result

    await agent_function()
