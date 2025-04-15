"""
Router agent implementation using the BaseAgent adapter pattern.

This provides a simplified implementation that routes messages to agents
by determining the best agent for a request and dispatching to it.
"""

from typing import TYPE_CHECKING, List, Optional, Tuple, Type

from mcp.types import TextContent
from pydantic import BaseModel

from mcp_agent.agents.agent import Agent
from mcp_agent.agents.base_agent import BaseAgent
from mcp_agent.core.agent_types import AgentConfig, AgentType
from mcp_agent.core.exceptions import AgentConfigError
from mcp_agent.core.prompt import Prompt
from mcp_agent.core.request_params import RequestParams
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.interfaces import ModelT
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

if TYPE_CHECKING:
    from mcp_agent.context import Context

logger = get_logger(__name__)

# Simple system instruction for the router
ROUTING_SYSTEM_INSTRUCTION = """
You are a highly accurate request router that directs incoming requests to the most appropriate agent.
Analyze each request and determine which specialized agent would be best suited to handle it based on their capabilities.

Follow these guidelines:
- Carefully match the request's needs with each agent's capabilities and description
- Select the single most appropriate agent for the request
- Provide your confidence level (high, medium, low) and brief reasoning for your selection
"""

# Default routing instruction with placeholders for context and request
DEFAULT_ROUTING_INSTRUCTION = """
You are a highly accurate request router that directs incoming requests to the most appropriate agent.

<fastagent:data>
<fastagent:agents>
{context}
</fastagent:agents>

<fastagent:request>
{request}
</fastagent:request>
</fastagent:data>

Your task is to analyze the request and determine the most appropriate agent from the options above.

<fastagent:instruction>
Respond with JSON following the schema below:
{{
    "type": "object",
    "required": ["agent", "confidence", "reasoning"],
    "properties": {{
        "agent": {{
            "type": "string",
            "description": "The exact name of the selected agent"
        }},
        "confidence": {{
            "type": "string",
            "enum": ["high", "medium", "low"],
            "description": "Your confidence level in this selection"
        }},
        "reasoning": {{
            "type": "string",
            "description": "Brief explanation for your selection"
        }}
    }}
}}

Supply only the JSON with no preamble. Use "reasoning" field to describe actions. NEVER EMIT CODE FENCES. 

</fastagent:instruction>
"""


class RoutingResponse(BaseModel):
    """Model for the structured routing response from the LLM."""

    agent: str
    confidence: str
    reasoning: Optional[str] = None


class RouterResult(BaseModel):
    """Router result with agent reference and confidence rating."""

    result: BaseAgent
    confidence: str
    reasoning: Optional[str] = None

    # Allow Agent objects to be stored without serialization
    model_config = {"arbitrary_types_allowed": True}


class RouterAgent(BaseAgent):
    """
    A simplified router that uses an LLM to determine the best agent for a request,
    then dispatches the request to that agent and returns the response.
    """

    @property
    def agent_type(self) -> str:
        """Return the type of this agent."""
        return AgentType.ROUTER.value

    def __init__(
        self,
        config: AgentConfig,
        agents: List[Agent],
        routing_instruction: Optional[str] = None,
        context: Optional["Context"] = None,
        default_request_params: Optional[RequestParams] = None,
        **kwargs,
    ) -> None:
        """
        Initialize a RouterAgent.

        Args:
            config: Agent configuration or name
            agents: List of agents to route between
            routing_instruction: Optional custom routing instruction
            context: Optional application context
            default_request_params: Optional default request parameters
            **kwargs: Additional keyword arguments to pass to BaseAgent
        """
        super().__init__(config=config, context=context, **kwargs)

        if not agents:
            raise AgentConfigError("At least one agent must be provided")

        self.agents = agents
        self.routing_instruction = routing_instruction
        self.agent_map = {agent.name: agent for agent in agents}

        # Set up base router request parameters
        base_params = {"systemPrompt": ROUTING_SYSTEM_INSTRUCTION, "use_history": False}

        # Merge with provided defaults if any
        if default_request_params:
            # Start with defaults and override with router-specific settings
            merged_params = default_request_params.model_copy(update=base_params)
        else:
            merged_params = RequestParams(**base_params)

        self._default_request_params = merged_params

    async def initialize(self) -> None:
        """Initialize the router and all agents."""
        if not self.initialized:
            await super().initialize()

            # Initialize all agents if not already initialized
            for agent in self.agents:
                if not getattr(agent, "initialized", False):
                    await agent.initialize()

            self.initialized = True

    async def shutdown(self) -> None:
        """Shutdown the router and all agents."""
        await super().shutdown()

        # Shutdown all agents
        for agent in self.agents:
            try:
                await agent.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down agent: {str(e)}")

    async def _get_routing_result(
        self,
        messages: List[PromptMessageMultipart],
    ) -> Optional[RouterResult]:
        """
        Common method to extract request and get routing result.

        Args:
            messages: The messages to extract request from

        Returns:
            RouterResult containing the selected agent, or None if no suitable agent found
        """
        if not self.initialized:
            await self.initialize()

        # Extract the request text from the last message
        request = messages[-1].all_text() if messages else ""

        # Determine which agent to route to
        routing_result = await self._route_request(request)

        if not routing_result:
            logger.warning("Could not determine appropriate agent for this request")

        return routing_result

    async def generate(
        self,
        multipart_messages: List[PromptMessageMultipart],
        request_params: Optional[RequestParams] = None,
    ) -> PromptMessageMultipart:
        """
        Route the request to the most appropriate agent and return its response.

        Args:
            multipart_messages: Messages to route
            request_params: Optional request parameters

        Returns:
            The response from the selected agent
        """
        routing_result = await self._get_routing_result(multipart_messages)

        if not routing_result:
            return PromptMessageMultipart(
                role="assistant",
                content=[
                    TextContent(
                        type="text", text="Could not determine appropriate agent for this request."
                    )
                ],
            )

        # Get the selected agent
        selected_agent = routing_result.result

        # Log the routing decision
        logger.info(
            f"Routing request to agent: {selected_agent.name} (confidence: {routing_result.confidence})"
        )

        # Dispatch the request to the selected agent
        return await selected_agent.generate(multipart_messages, request_params)

    async def structured(
        self,
        prompt: List[PromptMessageMultipart],
        model: Type[ModelT],
        request_params: Optional[RequestParams] = None,
    ) -> Tuple[ModelT | None, PromptMessageMultipart]:
        """
        Route the request to the most appropriate agent and parse its response.

        Args:
            prompt: Messages to route
            model: Pydantic model to parse the response into
            request_params: Optional request parameters

        Returns:
            The parsed response from the selected agent, or None if parsing fails
        """
        routing_result = await self._get_routing_result(prompt)

        if not routing_result:
            return None, Prompt.assistant("No routing result")

        # Get the selected agent
        selected_agent = routing_result.result

        # Log the routing decision
        logger.info(
            f"Routing structured request to agent: {selected_agent.name} (confidence: {routing_result.confidence})"
        )

        # Dispatch the request to the selected agent
        return await selected_agent.structured(prompt, model, request_params)

    async def _route_request(self, request: str) -> Optional[RouterResult]:
        """
        Determine which agent to route the request to.

        Args:
            request: The request to route

        Returns:
            RouterResult containing the selected agent, or None if no suitable agent was found
        """
        if not self.agents:
            logger.warning("No agents available for routing")
            return None

        # If only one agent is available, use it directly
        if len(self.agents) == 1:
            return RouterResult(
                result=self.agents[0], confidence="high", reasoning="Only one agent available"
            )

        # Generate agent descriptions for the context
        agent_descriptions = []
        for i, agent in enumerate(self.agents, 1):
            description = agent.instruction if isinstance(agent.instruction, str) else ""
            agent_descriptions.append(f"{i}. Name: {agent.name} - {description}")

        context = "\n\n".join(agent_descriptions)

        # Format the routing prompt
        routing_instruction = self.routing_instruction or DEFAULT_ROUTING_INSTRUCTION
        prompt_text = routing_instruction.format(context=context, request=request)

        # Create multipart message for the router
        prompt = PromptMessageMultipart(
            role="user", content=[TextContent(type="text", text=prompt_text)]
        )

        # Get structured response from LLM
        assert self._llm
        response, _ = await self._llm.structured(
            [prompt], RoutingResponse, self._default_request_params
        )

        if not response:
            logger.warning("No routing response received from LLM")
            return None

        # Look up the agent by name
        selected_agent = self.agent_map.get(response.agent)

        if not selected_agent:
            logger.warning(f"Agent '{response.agent}' not found in available agents")
            return None

        return RouterResult(
            result=selected_agent, confidence=response.confidence, reasoning=response.reasoning
        )
