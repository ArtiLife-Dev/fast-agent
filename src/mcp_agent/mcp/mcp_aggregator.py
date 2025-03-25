from asyncio import Lock, gather
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
)

from mcp import GetPromptResult, ReadResourceResult
from mcp.client.session import ClientSession
from mcp.server.lowlevel.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    ListToolsResult,
    Prompt,
    TextContent,
    Tool,
)
from pydantic import AnyUrl, BaseModel, ConfigDict

from mcp_agent.context_dependent import ContextDependent
from mcp_agent.event_progress import ProgressAction
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.gen_client import gen_client
from mcp_agent.mcp.mcp_agent_client_session import MCPAgentClientSession
from mcp_agent.mcp.mcp_connection_manager import MCPConnectionManager

if TYPE_CHECKING:
    from mcp_agent.context import Context


logger = get_logger(__name__)  # This will be replaced per-instance when agent_name is available

SEP = "-"

# Define type variables for the generalized method
T = TypeVar("T")
R = TypeVar("R")


class NamespacedTool(BaseModel):
    """
    A tool that is namespaced by server name.
    """

    tool: Tool
    server_name: str
    namespaced_tool_name: str


class MCPAggregator(ContextDependent):
    """
    Aggregates multiple MCP servers. When a developer calls, e.g. call_tool(...),
    the aggregator searches all servers in its list for a server that provides that tool.
    """

    initialized: bool = False
    """Whether the aggregator has been initialized with tools and resources from all servers."""

    connection_persistence: bool = False
    """Whether to maintain a persistent connection to the server."""

    server_names: List[str]
    """A list of server names to connect to."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    async def __aenter__(self):
        if self.initialized:
            return self

        # Keep a connection manager to manage persistent connections for this aggregator
        if self.connection_persistence:
            # Try to get existing connection manager from context
            if not hasattr(self.context, "_connection_manager"):
                self.context._connection_manager = MCPConnectionManager(self.context.server_registry)
                await self.context._connection_manager.__aenter__()
            self._persistent_connection_manager = self.context._connection_manager

        await self.load_servers()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def __init__(
        self,
        server_names: List[str],
        connection_persistence: bool = True,  # Default to True for better stability
        context: Optional["Context"] = None,
        name: str = None,
        **kwargs,
    ) -> None:
        """
        :param server_names: A list of server names to connect to.
        :param connection_persistence: Whether to maintain persistent connections to servers (default: True).
        Note: The server names must be resolvable by the gen_client function, and specified in the server registry.
        """
        super().__init__(
            context=context,
            **kwargs,
        )

        self.server_names = server_names
        self.connection_persistence = connection_persistence
        self.agent_name = name
        self._persistent_connection_manager: MCPConnectionManager = None

        # Set up logger with agent name in namespace if available
        global logger
        logger_name = f"{__name__}.{name}" if name else __name__
        logger = get_logger(logger_name)

        # Maps namespaced_tool_name -> namespaced tool info
        self._namespaced_tool_map: Dict[str, NamespacedTool] = {}
        # Maps server_name -> list of tools
        self._server_to_tool_map: Dict[str, List[NamespacedTool]] = {}
        self._tool_map_lock = Lock()

        # Cache for prompt objects, maps server_name -> list of prompt objects
        self._prompt_cache: Dict[str, List[Prompt]] = {}
        self._prompt_cache_lock = Lock()

    async def close(self) -> None:
        """
        Close all persistent connections when the aggregator is deleted.
        """
        if self.connection_persistence and self._persistent_connection_manager:
            try:
                # Only attempt cleanup if we own the connection manager
                if hasattr(self.context, "_connection_manager") and self.context._connection_manager == self._persistent_connection_manager:
                    logger.info("Shutting down all persistent connections...")
                    await self._persistent_connection_manager.disconnect_all()
                    await self._persistent_connection_manager.__aexit__(None, None, None)
                    delattr(self.context, "_connection_manager")
                self.initialized = False
            except Exception as e:
                logger.error(f"Error during connection manager cleanup: {e}")

    @classmethod
    async def create(
        cls,
        server_names: List[str],
        connection_persistence: bool = False,
    ) -> "MCPAggregator":
        """
        Factory method to create and initialize an MCPAggregator.
        Use this instead of constructor since we need async initialization.
        If connection_persistence is True, the aggregator will maintain a
        persistent connection to the servers for as long as this aggregator is around.
        By default we do not maintain a persistent connection.
        """

        logger.info(f"Creating MCPAggregator with servers: {server_names}")

        instance = cls(
            server_names=server_names,
            connection_persistence=connection_persistence,
        )

        try:
            await instance.__aenter__()

            logger.debug("Loading servers...")
            await instance.load_servers()

            logger.debug("MCPAggregator created and initialized.")
            return instance
        except Exception as e:
            logger.error(f"Error creating MCPAggregator: {e}")
            await instance.__aexit__(None, None, None)

    async def load_servers(self) -> None:
        """
        Discover tools from each server in parallel and build an index of namespaced tool names.
        Also populate the prompt cache.
        """
        if self.initialized:
            logger.debug("MCPAggregator already initialized.")
            return

        async with self._tool_map_lock:
            self._namespaced_tool_map.clear()
            self._server_to_tool_map.clear()

        async with self._prompt_cache_lock:
            self._prompt_cache.clear()

        for server_name in self.server_names:
            if self.connection_persistence:
                logger.info(
                    f"Creating persistent connection to server: {server_name}",
                    data={
                        "progress_action": ProgressAction.STARTING,
                        "server_name": server_name,
                        "agent_name": self.agent_name,
                    },
                )

                await self._persistent_connection_manager.get_server(server_name, client_session_factory=MCPAgentClientSession)

            logger.info(
                f"MCP Servers initialized for agent '{self.agent_name}'",
                data={
                    "progress_action": ProgressAction.INITIALIZED,
                    "agent_name": self.agent_name,
                },
            )

        async def fetch_tools(client: ClientSession):
            try:
                result: ListToolsResult = await client.list_tools()
                return result.tools or []
            except Exception as e:
                logger.error(f"Error loading tools from server '{server_name}'", data=e)
                return []

        async def fetch_prompts(client: ClientSession, server_name: str) -> List[Prompt]:
            # Only fetch prompts if the server supports them
            capabilities = await self.get_capabilities(server_name)
            if not capabilities or not capabilities.prompts:
                logger.debug(f"Server '{server_name}' does not support prompts")
                return []

            try:
                result = await client.list_prompts()
                return getattr(result, "prompts", [])
            except Exception as e:
                logger.debug(f"Error loading prompts from server '{server_name}': {e}")
                return []

        async def load_server_data(server_name: str):
            tools: List[Tool] = []
            prompts: List[Prompt] = []

            if self.connection_persistence:
                server_connection = await self._persistent_connection_manager.get_server(server_name, client_session_factory=MCPAgentClientSession)
                tools = await fetch_tools(server_connection.session)
                prompts = await fetch_prompts(server_connection.session, server_name)
            else:
                async with gen_client(server_name, server_registry=self.context.server_registry) as client:
                    tools = await fetch_tools(client)
                    prompts = await fetch_prompts(client, server_name)

            return server_name, tools, prompts

        # Gather data from all servers concurrently
        results = await gather(
            *(load_server_data(server_name) for server_name in self.server_names),
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, BaseException):
                logger.error(f"Error loading server data: {result}")
                continue

            server_name, tools, prompts = result

            # Process tools
            self._server_to_tool_map[server_name] = []
            for tool in tools:
                namespaced_tool_name = f"{server_name}{SEP}{tool.name}"
                namespaced_tool = NamespacedTool(
                    tool=tool,
                    server_name=server_name,
                    namespaced_tool_name=namespaced_tool_name,
                )

                self._namespaced_tool_map[namespaced_tool_name] = namespaced_tool
                self._server_to_tool_map[server_name].append(namespaced_tool)

            # Process prompts
            async with self._prompt_cache_lock:
                self._prompt_cache[server_name] = prompts

            logger.debug(
                f"MCP Aggregator initialized for server '{server_name}'",
                data={
                    "progress_action": ProgressAction.INITIALIZED,
                    "server_name": server_name,
                    "agent_name": self.agent_name,
                    "tool_count": len(tools),
                    "prompt_count": len(prompts),
                },
            )

        self.initialized = True

    async def get_capabilities(self, server_name: str):
        """Get server capabilities if available."""
        if not self.connection_persistence:
            # For non-persistent connections, we can't easily check capabilities
            return None

        try:
            server_conn = await self._persistent_connection_manager.get_server(server_name, client_session_factory=MCPAgentClientSession)
            # server_capabilities is a property, not a coroutine
            return server_conn.server_capabilities
        except Exception as e:
            logger.debug(f"Error getting capabilities for server '{server_name}': {e}")
            return None

    async def list_servers(self) -> List[str]:
        """Return the list of server names aggregated by this agent."""
        if not self.initialized:
            await self.load_servers()

        return self.server_names

    async def list_tools(self) -> ListToolsResult:
        """
        :return: Tools from all servers aggregated, and renamed to be dot-namespaced by server name.
        """
        if not self.initialized:
            await self.load_servers()

        return ListToolsResult(
            tools=[
                namespaced_tool.tool.model_copy(update={"name": namespaced_tool_name})
                for namespaced_tool_name, namespaced_tool in self._namespaced_tool_map.items()
            ]
        )

    async def _execute_on_server(
        self,
        server_name: str,
        operation_type: str,
        operation_name: str,
        method_name: str,
        method_args: Dict[str, Any] = None,
        error_factory: Callable[[str], R] = None,
    ) -> R:
        """
        Generic method to execute operations on a specific server.

        Args:
            server_name: Name of the server to execute the operation on
            operation_type: Type of operation (for logging) e.g., "tool", "prompt"
            operation_name: Name of the specific operation being called (for logging)
            method_name: Name of the method to call on the client session
            method_args: Arguments to pass to the method
            error_factory: Function to create an error return value if the operation fails

        Returns:
            Result from the operation or an error result
        """

        async def try_execute(client: ClientSession):
            try:
                method = getattr(client, method_name)
                return await method(**method_args)
            except Exception as e:
                error_msg = f"Failed to {method_name} '{operation_name}' on server '{server_name}': {e}"
                logger.error(error_msg)
                return error_factory(error_msg) if error_factory else None

        if self.connection_persistence:
            server_connection = await self._persistent_connection_manager.get_server(server_name, client_session_factory=MCPAgentClientSession)
            return await try_execute(server_connection.session)
        else:
            logger.debug(
                f"Creating temporary connection to server: {server_name}",
                data={
                    "progress_action": ProgressAction.STARTING,
                    "server_name": server_name,
                    "agent_name": self.agent_name,
                },
            )
            async with gen_client(server_name, server_registry=self.context.server_registry) as client:
                result = await try_execute(client)
                logger.debug(
                    f"Closing temporary connection to server: {server_name}",
                    data={
                        "progress_action": ProgressAction.SHUTDOWN,
                        "server_name": server_name,
                        "agent_name": self.agent_name,
                    },
                )
                return result

    async def _parse_resource_name(self, name: str, resource_type: str) -> tuple[str, str]:
        """
        Parse a possibly namespaced resource name into server name and local resource name.

        Args:
            name: The resource name, possibly namespaced
            resource_type: Type of resource (for error messages), e.g. "tool", "prompt"

        Returns:
            Tuple of (server_name, local_resource_name)
        """
        server_name = None
        local_name = None

        if SEP in name:  # Namespaced resource name
            server_name, local_name = name.split(SEP, 1)
        else:
            # For tools, search all servers for the tool
            if resource_type == "tool":
                for _, tools in self._server_to_tool_map.items():
                    for namespaced_tool in tools:
                        if namespaced_tool.tool.name == name:
                            server_name = namespaced_tool.server_name
                            local_name = name
                            break
                    if server_name:
                        break
            # For all other resource types, use the first server
            # (prompt resource type is specially handled in get_prompt)
            else:
                local_name = name
                server_name = self.server_names[0] if self.server_names else None

        return server_name, local_name

    async def call_tool(self, name: str, arguments: dict | None = None) -> CallToolResult:
        """
        Call a namespaced tool, e.g., 'server_name.tool_name'.
        """
        if not self.initialized:
            await self.load_servers()

        server_name, local_tool_name = await self._parse_resource_name(name, "tool")

        if server_name is None or local_tool_name is None:
            logger.error(f"Error: Tool '{name}' not found")
            return CallToolResult(
                isError=True,
                content=[TextContent(type="text", text=f"Tool '{name}' not found")],
            )

        logger.info(
            "Requesting tool call",
            data={
                "progress_action": ProgressAction.CALLING_TOOL,
                "tool_name": local_tool_name,
                "server_name": server_name,
                "agent_name": self.agent_name,
            },
        )

        return await self._execute_on_server(
            server_name=server_name,
            operation_type="tool",
            operation_name=local_tool_name,
            method_name="call_tool",
            method_args={"name": local_tool_name, "arguments": arguments},
            error_factory=lambda msg: CallToolResult(isError=True, content=[TextContent(type="text", text=msg)]),
        )

    async def get_prompt(self, prompt_name: str | None, arguments: dict[str, str] | None) -> GetPromptResult:
        """
        Get a prompt from a server.

        :param prompt_name: Name of the prompt, optionally namespaced with server name
                           using the format 'server_name-prompt_name'
        :param arguments: Optional dictionary of string arguments to pass to the prompt template
                         for templating
        :return: GetPromptResult containing the prompt description and messages
                 with a namespaced_name property for display purposes
        """
        if not self.initialized:
            await self.load_servers()

        # Handle the case where prompt_name is None
        if not prompt_name:
            server_name = self.server_names[0] if self.server_names else None
            local_prompt_name = None
            namespaced_name = None
        # Handle namespaced prompt name
        elif SEP in prompt_name:
            server_name, local_prompt_name = prompt_name.split(SEP, 1)
            namespaced_name = prompt_name  # Already namespaced
        # Plain prompt name - will use cache to find the server
        else:
            local_prompt_name = prompt_name
            server_name = None
            namespaced_name = None  # Will be set when server is found

        # If we have a specific server to check
        if server_name:
            if server_name not in self.server_names:
                logger.error(f"Error: Server '{server_name}' not found")
                return GetPromptResult(
                    description=f"Error: Server '{server_name}' not found",
                    messages=[],
                )

            # Check if server supports prompts
            capabilities = await self.get_capabilities(server_name)
            if not capabilities or not capabilities.prompts:
                logger.debug(f"Server '{server_name}' does not support prompts")
                return GetPromptResult(
                    description=f"Server '{server_name}' does not support prompts",
                    messages=[],
                )

            # Check the prompt cache to avoid unnecessary errors
            if local_prompt_name:
                async with self._prompt_cache_lock:
                    if server_name in self._prompt_cache:
                        # Check if any prompt in the cache has this name
                        prompt_names = [prompt.name for prompt in self._prompt_cache[server_name]]
                        if local_prompt_name not in prompt_names:
                            logger.debug(f"Prompt '{local_prompt_name}' not found in cache for server '{server_name}'")
                            return GetPromptResult(
                                description=f"Prompt '{local_prompt_name}' not found on server '{server_name}'",
                                messages=[],
                            )

            # Try to get the prompt from the specified server
            method_args = {"name": local_prompt_name} if local_prompt_name else {}
            if arguments:
                method_args["arguments"] = arguments

            result = await self._execute_on_server(
                server_name=server_name,
                operation_type="prompt",
                operation_name=local_prompt_name or "default",
                method_name="get_prompt",
                method_args=method_args,
                error_factory=lambda msg: GetPromptResult(description=msg, messages=[]),
            )

            # Add namespaced name and source server to the result
            if result and result.messages:
                result.namespaced_name = namespaced_name or f"{server_name}{SEP}{local_prompt_name}"

                # Store the arguments in the result for display purposes
                if arguments:
                    result.arguments = arguments

            return result

        # No specific server - use the cache to find servers that have this prompt
        logger.debug(f"Searching for prompt '{local_prompt_name}' using cache")

        # Find potential servers from the cache
        potential_servers = []
        async with self._prompt_cache_lock:
            for s_name, prompt_list in self._prompt_cache.items():
                prompt_names = [prompt.name for prompt in prompt_list]
                if local_prompt_name in prompt_names:
                    potential_servers.append(s_name)

        if potential_servers:
            logger.debug(f"Found prompt '{local_prompt_name}' in cache for servers: {potential_servers}")

            # Try each server from the cache
            for s_name in potential_servers:
                # Check if this server supports prompts
                capabilities = await self.get_capabilities(s_name)
                if not capabilities or not capabilities.prompts:
                    logger.debug(f"Server '{s_name}' does not support prompts, skipping")
                    continue

                try:
                    method_args = {"name": local_prompt_name}
                    if arguments:
                        method_args["arguments"] = arguments

                    result = await self._execute_on_server(
                        server_name=s_name,
                        operation_type="prompt",
                        operation_name=local_prompt_name,
                        method_name="get_prompt",
                        method_args=method_args,
                        error_factory=lambda _: None,  # Return None instead of an error
                    )

                    # If we got a successful result with messages, return it
                    if result and result.messages:
                        logger.debug(f"Successfully retrieved prompt '{local_prompt_name}' from server '{s_name}'")
                        # Add namespaced name using the actual server where found
                        result.namespaced_name = f"{s_name}{SEP}{local_prompt_name}"

                        # Store the arguments in the result for display purposes
                        if arguments:
                            result.arguments = arguments

                        return result

                except Exception as e:
                    logger.debug(f"Error retrieving prompt from server '{s_name}': {e}")
        else:
            logger.debug(f"Prompt '{local_prompt_name}' not found in any server's cache")

            # If not in cache, perform a full search as fallback (cache might be outdated)
            # First identify servers that support prompts
            supported_servers = []
            for s_name in self.server_names:
                capabilities = await self.get_capabilities(s_name)
                if capabilities and capabilities.prompts:
                    supported_servers.append(s_name)
                else:
                    logger.debug(f"Server '{s_name}' does not support prompts, skipping from fallback search")

            # Try all supported servers in order
            for s_name in supported_servers:
                try:
                    # Use a quiet approach - don't log errors if not found
                    method_args = {"name": local_prompt_name}
                    if arguments:
                        method_args["arguments"] = arguments

                    result = await self._execute_on_server(
                        server_name=s_name,
                        operation_type="prompt",
                        operation_name=local_prompt_name,
                        method_name="get_prompt",
                        method_args=method_args,
                        error_factory=lambda _: None,  # Return None instead of an error
                    )

                    # If we got a successful result with messages, return it
                    if result and result.messages:
                        logger.debug(f"Found prompt '{local_prompt_name}' on server '{s_name}' (not in cache)")
                        # Add namespaced name using the actual server where found
                        result.namespaced_name = f"{s_name}{SEP}{local_prompt_name}"

                        # Store the arguments in the result for display purposes
                        if arguments:
                            result.arguments = arguments

                        # Update the cache - need to fetch the prompt object to store in cache
                        try:
                            prompt_list_result = await self._execute_on_server(
                                server_name=s_name,
                                operation_type="prompts-list",
                                operation_name="",
                                method_name="list_prompts",
                                error_factory=lambda _: None,
                            )

                            prompts = getattr(prompt_list_result, "prompts", [])
                            matching_prompts = [p for p in prompts if p.name == local_prompt_name]
                            if matching_prompts:
                                async with self._prompt_cache_lock:
                                    if s_name not in self._prompt_cache:
                                        self._prompt_cache[s_name] = []
                                    # Add if not already in the cache
                                    prompt_names_in_cache = [p.name for p in self._prompt_cache[s_name]]
                                    if local_prompt_name not in prompt_names_in_cache:
                                        self._prompt_cache[s_name].append(matching_prompts[0])
                        except Exception:
                            # Ignore errors when updating cache
                            pass

                        return result

                except Exception:
                    # Don't log errors during fallback search
                    pass

        # If we get here, we couldn't find the prompt on any server
        logger.info(f"Prompt '{local_prompt_name}' not found on any server")
        return GetPromptResult(
            description=f"Prompt '{local_prompt_name}' not found on any server",
            messages=[],
        )

    async def list_prompts(self, server_name: str = None):
        """
        List available prompts from one or all servers.

        :param server_name: Optional server name to list prompts from. If not provided,
                           lists prompts from all servers.
        :return: Dictionary mapping server names to lists of available prompts
        """
        if not self.initialized:
            await self.load_servers()

        results = {}

        # If we already have the data in cache and not requesting a specific server,
        # we can use the cache directly
        if not server_name:
            async with self._prompt_cache_lock:
                if all(s_name in self._prompt_cache for s_name in self.server_names):
                    # Return the cached prompt objects
                    for s_name, prompt_list in self._prompt_cache.items():
                        results[s_name] = prompt_list
                    logger.debug("Returning cached prompts for all servers")
                    return results

        # If server_name is provided, only list prompts from that server
        if server_name:
            if server_name in self.server_names:
                # Check if we can use the cache
                async with self._prompt_cache_lock:
                    if server_name in self._prompt_cache:
                        results[server_name] = self._prompt_cache[server_name]
                        logger.debug(f"Returning cached prompts for server '{server_name}'")
                        return results

                # Check if server supports prompts
                capabilities = await self.get_capabilities(server_name)
                if not capabilities or not capabilities.prompts:
                    logger.debug(f"Server '{server_name}' does not support prompts")
                    results[server_name] = []
                    return results

                # If not in cache and server supports prompts, fetch from server
                result = await self._execute_on_server(
                    server_name=server_name,
                    operation_type="prompts-list",
                    operation_name="",
                    method_name="list_prompts",
                    error_factory=lambda _: [],
                )

                # Update cache with the result
                async with self._prompt_cache_lock:
                    self._prompt_cache[server_name] = getattr(result, "prompts", [])

                results[server_name] = result
            else:
                logger.error(f"Server '{server_name}' not found")
        else:
            # We need to filter the servers that support prompts
            supported_servers = []
            for s_name in self.server_names:
                capabilities = await self.get_capabilities(s_name)
                if capabilities and capabilities.prompts:
                    supported_servers.append(s_name)
                else:
                    logger.debug(f"Server '{s_name}' does not support prompts, skipping")
                    # Add empty list to results for this server
                    results[s_name] = []

            # Process servers sequentially to ensure proper resource cleanup
            # This helps prevent resource leaks especially on Windows
            if supported_servers:
                server_results = []
                for s_name in supported_servers:
                    try:
                        result = await self._execute_on_server(
                            server_name=s_name,
                            operation_type="prompts-list",
                            operation_name="",
                            method_name="list_prompts",
                            error_factory=lambda _: [],
                        )
                        server_results.append(result)
                    except Exception as e:
                        logger.debug(f"Error fetching prompts from {s_name}: {e}")
                        server_results.append(e)

                for i, result in enumerate(server_results):
                    if isinstance(result, BaseException):
                        continue

                    s_name = supported_servers[i]
                    results[s_name] = result

                    # Update cache with the result
                    async with self._prompt_cache_lock:
                        self._prompt_cache[s_name] = getattr(result, "prompts", [])

        logger.debug(f"Available prompts across servers: {results}")
        return results

    async def get_resource(self, server_name: str, resource_uri: str) -> ReadResourceResult:
        """
        Get a resource directly from an MCP server by URI.

        Args:
            server_name: Name of the MCP server to retrieve the resource from
            resource_uri: URI of the resource to retrieve

        Returns:
            ReadResourceResult object containing the resource content

        Raises:
            ValueError: If the server doesn't exist or the resource couldn't be found
        """
        if not self.initialized:
            await self.load_servers()

        if server_name not in self.server_names:
            raise ValueError(f"Server '{server_name}' not found")

        logger.info(
            "Requesting resource",
            data={
                "progress_action": ProgressAction.CALLING_TOOL,
                "resource_uri": resource_uri,
                "server_name": server_name,
                "agent_name": self.agent_name,
            },
        )

        try:
            uri = AnyUrl(resource_uri)
        except Exception as e:
            raise ValueError(f"Invalid resource URI: {resource_uri}. Error: {e}")

        # Use the _execute_on_server method to call read_resource on the server
        return await self._execute_on_server(
            server_name=server_name,
            operation_type="resource",
            operation_name=resource_uri,
            method_name="read_resource",
            method_args={"uri": uri},
            error_factory=lambda msg: ValueError(f"Failed to retrieve resource: {msg}"),
        )


class MCPCompoundServer(Server):
    """
    A compound server (server-of-servers) that aggregates multiple MCP servers and is itself an MCP server
    """

    def __init__(self, server_names: List[str], name: str = "MCPCompoundServer") -> None:
        super().__init__(name)
        self.aggregator = MCPAggregator(server_names)

        # Register handlers for tools, prompts, and resources
        self.list_tools()(self._list_tools)
        self.call_tool()(self._call_tool)
        self.get_prompt()(self._get_prompt)
        self.list_prompts()(self._list_prompts)

    async def _list_tools(self) -> List[Tool]:
        """List all tools aggregated from connected MCP servers."""
        tools_result = await self.aggregator.list_tools()
        return tools_result.tools

    async def _call_tool(self, name: str, arguments: dict | None = None) -> CallToolResult:
        """Call a specific tool from the aggregated servers."""
        try:
            result = await self.aggregator.call_tool(name=name, arguments=arguments)
            return result.content
        except Exception as e:
            return CallToolResult(
                isError=True,
                content=[TextContent(type="text", text=f"Error calling tool: {e}")],
            )

    async def _get_prompt(self, name: str = None, arguments: dict[str, str] = None) -> GetPromptResult:
        """
        Get a prompt from the aggregated servers.

        Args:
            name: Name of the prompt to get (optionally namespaced)
            arguments: Optional dictionary of string arguments for prompt templating
        """
        try:
            result = await self.aggregator.get_prompt(prompt_name=name, arguments=arguments)
            return result
        except Exception as e:
            return GetPromptResult(description=f"Error getting prompt: {e}", messages=[])

    async def _list_prompts(self, server_name: str = None) -> Dict[str, List[str]]:
        """List available prompts from the aggregated servers."""
        try:
            return await self.aggregator.list_prompts(server_name=server_name)
        except Exception as e:
            logger.error(f"Error listing prompts: {e}")
            return {}

    async def run_stdio_async(self) -> None:
        """Run the server using stdio transport."""
        async with stdio_server() as (read_stream, write_stream):
            await self.run(
                read_stream=read_stream,
                write_stream=write_stream,
                initialization_options=self.create_initialization_options(),
            )
