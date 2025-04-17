import asyncio
import re  # Added for tag parsing
from typing import Dict, List, Optional, Tuple

from mcp.types import TextContent  # Added PromptMessage, MessageRole

from mcp_agent.agents.agent import Agent
from mcp_agent.agents.base_agent import BaseAgent
from mcp_agent.core.agent_types import AgentConfig, AgentType
from mcp_agent.core.request_params import RequestParams
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.interfaces import ModelT
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


class PeersAgent(BaseAgent):
    """
    A workflow where multiple agents collaborate like peers in a group chat.

    Agents can choose to participate in each round, and the conversation continues
    until a consensus is reached or a maximum number of rounds is exceeded.
    The shared history only includes messages explicitly contributed by agents.
    """

    @property
    def agent_type(self) -> str:
        """Return the type of this agent."""
        return AgentType.PEERS.value

    def __init__(
        self,
        config: AgentConfig,
        peer_agents: List[Agent], # Includes coordinator
        coordinator_agent_name: str,
        max_rounds: int = 5,
        # consensus_mechanism: str = "max_rounds", # Removed
        **kwargs,
    ) -> None:
        """
        Initialize a PeersAgent agent.

        Args:
            config: Agent configuration.
            peer_agents: List of all agents participating, including the coordinator.
            coordinator_agent_name: The name of the agent designated as coordinator.
            max_rounds: Maximum number of conversational rounds.
            **kwargs: Additional keyword arguments to pass to BaseAgent.
        """
        super().__init__(config, **kwargs)
        self.all_agents = peer_agents # Store all agents
        self.max_rounds = max_rounds
        # self.consensus_mechanism = consensus_mechanism # Removed
        self._shared_history: List[PromptMessageMultipart] = []
        self.logger = get_logger(f"{__name__}.{self.name}")

        # Find the coordinator agent
        self.coordinator_agent: Optional[Agent] = None
        for agent in self.all_agents:
            if agent.name == coordinator_agent_name:
                self.coordinator_agent = agent
                break
        if not self.coordinator_agent:
            # This should ideally be caught by the factory, but double-check
            raise ValueError(f"Coordinator agent '{coordinator_agent_name}' not found in the provided peer_agents list.")

        # Separate peers from coordinator for easier iteration
        self.peer_agents = [agent for agent in self.all_agents if agent.name != coordinator_agent_name]

        self.logger.info(
            f"Initialized PeersAgent '{self.name}' with coordinator '{coordinator_agent_name}' "
            f"and {len(self.peer_agents)} other peers. Max rounds: {max_rounds}."
        )

    # --- Helper Methods ---

    def _extract_tag_content(self, text: str, tag: str) -> Optional[str]:
        """Extracts content from the first occurrence of <tag>content</tag>."""
        match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else None

    def _check_tag_presence(self, text: str, tag: str) -> bool:
        """Checks if <tag/> or <tag>...</tag> exists."""
        return bool(re.search(rf"<{tag}\s*/>|<{tag}>", text, re.IGNORECASE))

    def _add_to_shared_history(self, agent_name: str, message: PromptMessageMultipart):
        """Adds a message to shared history with prepended name."""
        # Create a new message to avoid modifying the original
        shared_msg = message.model_copy(deep=True)
        # Prepend name to the first text content block if it exists
        if shared_msg.content and isinstance(shared_msg.content[0], TextContent):
             original_text = shared_msg.content[0].text
             shared_msg.content[0].text = f"[{agent_name}]: {original_text}"
        elif shared_msg.content: # If no text content, add a text block with the name
             shared_msg.content.insert(0, TextContent(type="text", text=f"[{agent_name}]:"))
        else: # If no content at all
             shared_msg.content = [TextContent(type="text", text=f"[{agent_name}]:")]

        self._shared_history.append(shared_msg)
        self.logger.debug(f"Added message from {agent_name} to shared history.")

    def _get_agent_internal_history(self, agent: Agent) -> List[PromptMessageMultipart]:
        """Safely gets the message history from an agent."""
        if hasattr(agent, 'message_history') and isinstance(agent.message_history, list):
            return agent.message_history.copy() # Return a copy
        return []

    def _build_individual_history(self, target_agent: Agent) -> List[PromptMessageMultipart]:
        """Builds the history view for a specific agent."""
        individual_history: List[PromptMessageMultipart] = []
        # agent_internal_history = self._get_agent_internal_history(target_agent)
        # Map shared history messages to full internal messages for the target agent
        # This is complex because shared history only contains the <share> part.
        # Simplification: We will provide the shared history + a prompt asking
        # the agent to consider its internal thoughts. A more robust solution
        # might involve passing message IDs.

        # Start with the shared history, adjusting roles
        for shared_msg in self._shared_history:
            msg_copy = shared_msg.model_copy(deep=True)
            # Extract sender name (assuming format "[SenderName]: ...")
            sender_name_match = re.match(r"\[(.*?)\]:", msg_copy.first_text() or "")
            sender_name = sender_name_match.group(1) if sender_name_match else "Unknown" # Handle potential format errors

            if sender_name == target_agent.name:
                # How to get the *full* message corresponding to this shared part?
                # For now, we'll just mark it as assistant role.
                # TODO: Enhance this to link back to full internal message if possible.
                msg_copy.role = 'assistant'
            else:
                msg_copy.role = 'user'
            individual_history.append(msg_copy)

        # Alternative: Append internal history *after* shared history?
        # individual_history.extend(agent_internal_history) # This might duplicate info

        return individual_history


    # --- Main Workflow Logic ---

    async def generate(
        self,
        multipart_messages: List[PromptMessageMultipart],
        request_params: Optional[RequestParams] = None,
    ) -> PromptMessageMultipart:
        """
        Execute the peer collaboration workflow with a coordinator.

        Args:
            multipart_messages: The initial message(s) starting the conversation (usually from user).
            request_params: Optional parameters to configure the request (applied to peers).

        Returns:
            The final message from the coordinator agent.
        """
        if not self.coordinator_agent:
             # Should be caught in init, but safety check
             raise RuntimeError("Coordinator agent is not set.")

        self.logger.info(f"Peers workflow '{self.name}' starting with coordinator '{self.coordinator_agent.name}'.")
        self._shared_history = []  # Reset shared history

        # 1. Initial User Message(s)
        initial_user_messages = [msg for msg in multipart_messages if msg.role == 'user']
        if initial_user_messages:
            # Add user messages directly to shared history (no name prepended yet)
            # Or should the coordinator see them first? Let's add them directly for now.
            self._shared_history.extend(initial_user_messages)
            self.logger.debug(f"Added {len(initial_user_messages)} initial user message(s) to shared history.")

        # 2. Coordinator Introduction (Round 0)
        intro_prompt_text = "You are the coordinator for a group discussion. Please introduce the topic based on the initial user request and list the participating agents:\n"
        intro_prompt_text += f"- Coordinator: {self.coordinator_agent.name}\n"
        for peer in self.peer_agents:
            intro_prompt_text += f"- Peer: {peer.name}\n"
        intro_prompt_text += "Initial User Request:\n"
        for msg in initial_user_messages:
             intro_prompt_text += f"{msg.all_text()}\n"

        intro_prompt = PromptMessageMultipart(role='user', content=[TextContent(type="text", text=intro_prompt_text)])
        try:
            coordinator_intro = await self.coordinator_agent.generate([intro_prompt], request_params)
            self._add_to_shared_history(self.coordinator_agent.name, coordinator_intro)
        except Exception as e:
            self.logger.error(f"Coordinator {self.coordinator_agent.name} failed during introduction: {e}")
            # Return an error message?
            return PromptMessageMultipart(role='assistant', content=[TextContent(type="text", text=f"Error: Coordinator failed during introduction: {e}")])


        # 3. Main Discussion Rounds
        current_round = 0
        while current_round < self.max_rounds:
            current_round += 1
            self.logger.info(f"Starting round {current_round}/{self.max_rounds}")

            # round_contributions = False
            peer_responses: Dict[str, PromptMessageMultipart] = {} # Store full responses this round

            # --- Peer Agent Turns ---
            peer_tasks = []
            for peer_agent in self.peer_agents:
                # Build individualized history for this peer
                individual_history = self._build_individual_history(peer_agent)

                # Add prompt asking for contribution with tags
                participation_prompt = PromptMessageMultipart(
                    role='user',
                    content=[TextContent(type="text", text=(
                        "Considering the discussion history from your perspective and your internal thoughts/tool use, "
                        "do you have a contribution to share with the group? \n"
                        "If yes, wrap the part you want to share in <share>...</share> tags within your response. "
                        "If you have nothing to add this round, include the <abstain/> tag in your response. "
                        # "Remember to include your reasoning/tool use outside the <share> tags for your own history."
                    ))]
                )
                individual_history.append(participation_prompt)

                # Schedule the agent call
                peer_tasks.append(peer_agent.generate(individual_history, request_params))

            # Run peers in parallel
            results = await asyncio.gather(*peer_tasks, return_exceptions=True)

            # Process peer responses
            for i, result in enumerate(results):
                peer_agent = self.peer_agents[i]
                if isinstance(result, Exception):
                    self.logger.error(f"Agent {peer_agent.name} failed in round {current_round}: {result}")
                    continue # Skip this agent

                full_response: PromptMessageMultipart = result
                peer_responses[peer_agent.name] = full_response # Store full response
                response_text = full_response.all_text()

                # Check for abstention
                if self._check_tag_presence(response_text, "abstain"):
                    self.logger.debug(f"Agent {peer_agent.name} abstained in round {current_round}.")
                    continue

                # Check for shared content
                shared_content = self._extract_tag_content(response_text, "share")
                if shared_content:
                    # Create a new message with only the shared content
                    shared_message = PromptMessageMultipart(
                        role='assistant', # Role in shared history is assistant
                        content=[TextContent(type="text", text=shared_content)]
                        # name=peer_agent.name # Name is prepended in _add_to_shared_history
                    )
                    self._add_to_shared_history(peer_agent.name, shared_message)
                    # round_contributions = True
                else:
                     self.logger.debug(f"Agent {peer_agent.name} responded but did not share content in round {current_round}.")


            # --- Coordinator Check ---
            self.logger.debug("Asking coordinator to check for consensus.")
            coordinator_history = self._build_individual_history(self.coordinator_agent)
            coordinator_prompt = PromptMessageMultipart(
                 role='user',
                 content=[TextContent(type="text", text=(
                     "Review the latest round of discussion. Summarize the progress. "
                     "Has a consensus or final answer been reached? \n"
                     "If yes, clearly state the conclusion and include the tag <consensus_reached/> in your response. \n"
                     "If not, identify disagreements or next steps and include the tag <continue_discussion/>."
                 ))]
            )
            coordinator_history.append(coordinator_prompt)

            try:
                coordinator_response = await self.coordinator_agent.generate(coordinator_history, request_params)
                coordinator_response_text = coordinator_response.all_text()
                self._add_to_shared_history(self.coordinator_agent.name, coordinator_response) # Add coordinator's full check to history

                # Check for consensus tag
                if self._check_tag_presence(coordinator_response_text, "consensus_reached"):
                    self.logger.info(f"Coordinator '{self.coordinator_agent.name}' signaled consensus reached in round {current_round}.")
                    # Return the coordinator's final message
                    return coordinator_response
                elif self._check_tag_presence(coordinator_response_text, "continue_discussion"):
                     self.logger.info(f"Coordinator '{self.coordinator_agent.name}' signaled to continue discussion.")
                     # Continue to next round
                else:
                     self.logger.warning(f"Coordinator '{self.coordinator_agent.name}' did not explicitly signal consensus or continuation. Continuing by default.")
                     # Continue by default if no clear signal

            except Exception as e:
                self.logger.error(f"Coordinator {self.coordinator_agent.name} failed during consensus check: {e}")
                # Decide how to handle coordinator failure - maybe end the workflow?
                return PromptMessageMultipart(role='assistant', content=[TextContent(type="text", text=f"Error: Coordinator failed during consensus check: {e}")])


        # Max rounds reached
        self.logger.warning(f"Peers workflow '{self.name}' finished after reaching max rounds ({self.max_rounds}).")
        # Return the last message from the coordinator (which might indicate lack of consensus)
        last_message = self._shared_history[-1] if self._shared_history else \
                       PromptMessageMultipart(role='assistant', content=[TextContent(type="text", text="Workflow ended: Max rounds reached.")])
        return last_message


    # Removed _check_consensus method as logic is now handled by coordinator


    async def structured(
        self,
        prompt: List[PromptMessageMultipart],
        model: type[ModelT],
        request_params: Optional[RequestParams] = None,
    ) -> Tuple[ModelT | None, PromptMessageMultipart]:
        """
        Structured output is not the primary goal of the Peers workflow.
        It returns the final coordinator message. Trying to parse structured
        data from the final coordinator message might be possible but is not
        guaranteed.
        """
        self.logger.warning("Structured output requested for PeersAgent. Returning final coordinator message.")
        final_message = await self.generate(prompt, request_params)
        # Attempt to parse structured data from the final message if needed,
        # but for now, return None for the model part.
        # You could potentially call coordinator_agent.structured here if appropriate.
        return None, final_message

    async def initialize(self) -> None:
        """
        Initialize the PeersAgent and ALL participating agents (peers + coordinator).
        """
        await super().initialize()
        self.logger.debug(f"Initializing {len(self.all_agents)} agents for '{self.name}'.")
        init_tasks = []
        for agent in self.all_agents: # Use all_agents now
            if not getattr(agent, "initialized", False):
                 init_tasks.append(agent.initialize())
            else:
                 self.logger.debug(f"Agent '{agent.name}' already initialized.")
        if init_tasks:
             await asyncio.gather(*init_tasks)
        self.logger.debug(f"All agents for '{self.name}' initialized.")


    async def shutdown(self) -> None:
        """
        Shutdown the PeersAgent and ALL participating agents (peers + coordinator).
        """
        await super().shutdown()
        self.logger.debug(f"Shutting down {len(self.all_agents)} agents for '{self.name}'.")
        shutdown_tasks = []
        for agent in self.all_agents: # Use all_agents now
            # Check if shutdown method exists and is callable
            if hasattr(agent, "shutdown") and callable(getattr(agent, "shutdown")):
                 # Check if it's already shut down (simple check, might need refinement)
                 if getattr(agent, "initialized", True): # Assume initialized if no flag
                     shutdown_tasks.append(agent.shutdown())
                 else:
                     self.logger.debug(f"Agent '{agent.name}' already shut down or not initialized.")
            else:
                 self.logger.warning(f"Agent '{agent.name}' does not have a shutdown method.")

        if shutdown_tasks:
             results = await asyncio.gather(*shutdown_tasks, return_exceptions=True)
             for i, result in enumerate(results):
                 # Find the corresponding agent name for error logging
                 agent_name = "Unknown"
                 if i < len(self.all_agents):
                     agent_name = self.all_agents[i].name
                 if isinstance(result, Exception):
                     self.logger.error(f"Error shutting down agent {agent_name}: {result}")
        self.logger.debug(f"All agents for '{self.name}' shut down.")
