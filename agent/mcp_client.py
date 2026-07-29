"""
agent/mcp_client.py
MCP Client Manager: connects to external MCP servers and translates tool schemas.
"""
from __future__ import annotations

import os
import re
import logging
from typing import Any, Dict, List, Optional

from config.loader import MCPServerConnectionConfig

logger = logging.getLogger(__name__)


def _substitute_env_vars(env_dict: Dict[str, str]) -> Dict[str, str]:
    """Substitute ${VAR} patterns with os.environ values."""
    res = {}
    pattern = re.compile(r"\$\{([A-Za-z0-9_]+)\}")
    for k, v in env_dict.items():
        def repl(match):
            var_name = match.group(1)
            return os.getenv(var_name, "")
        res[k] = pattern.sub(repl, v)
    return res


class MCPClientManager:
    def __init__(self, server_configs: List[MCPServerConnectionConfig]):
        self.server_configs = [c for c in server_configs if c.enabled]
        self._sessions: Dict[str, Any] = {}
        self._exit_stack: Optional[Any] = None
        self._tool_map: Dict[str, Dict[str, Any]] = {}  # namespaced_name -> {server, tool_name, schema}

    async def connect_all(self) -> None:
        """Connects to all enabled MCP servers and lists their tools."""
        try:
            from contextlib import AsyncExitStack
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            logger.warning("mcp package is not installed. External MCP tools will be unavailable.")
            return

        self._exit_stack = AsyncExitStack()

        for s_cfg in self.server_configs:
            try:
                if s_cfg.transport == "stdio":
                    if not s_cfg.command:
                        logger.warning(f"MCP server '{s_cfg.name}' has transport 'stdio' but no command specified.")
                        continue
                    env = {**os.environ, **_substitute_env_vars(s_cfg.env)}
                    params = StdioServerParameters(
                        command=s_cfg.command,
                        args=s_cfg.args,
                        env=env,
                    )
                    read_stream, write_stream = await self._exit_stack.enter_async_context(stdio_client(params))
                    session = await self._exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
                    await session.initialize()
                    self._sessions[s_cfg.name] = session
                    
                    # Fetch tools
                    tools_response = await session.list_tools()
                    for tool in tools_response.tools:
                        if s_cfg.tool_allowlist and tool.name not in s_cfg.tool_allowlist:
                            continue
                        namespaced_name = f"{s_cfg.name}__{tool.name}"
                        self._tool_map[namespaced_name] = {
                            "server": s_cfg.name,
                            "tool_name": tool.name,
                            "raw_tool": tool,
                            "schema": {
                                "type": "function",
                                "function": {
                                    "name": namespaced_name,
                                    "description": f"[{s_cfg.name}] {tool.description or ''}".strip(),
                                    "parameters": tool.inputSchema if hasattr(tool, "inputSchema") else {"type": "object", "properties": {}},
                                },
                            },
                        }
            except Exception as e:
                logger.warning(f"Failed to connect to MCP server '{s_cfg.name}': {e}")

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Returns all discovered tool schemas for OpenAI function calling."""
        return [item["schema"] for item in self._tool_map.values()]

    async def call_tool(self, namespaced_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Calls a namespaced MCP tool and returns result or error dict."""
        if namespaced_name not in self._tool_map:
            return {"error": f"Tool '{namespaced_name}' not found."}

        info = self._tool_map[namespaced_name]
        server_name = info["server"]
        tool_name = info["tool_name"]
        session = self._sessions.get(server_name)

        if not session:
            return {"error": f"Session for server '{server_name}' is not connected."}

        try:
            res = await session.call_tool(tool_name, arguments)
            # Format content output
            text_outputs = []
            if hasattr(res, "content") and res.content:
                for item in res.content:
                    if hasattr(item, "text"):
                        text_outputs.append(item.text)
                    else:
                        text_outputs.append(str(item))
            output_str = "\n".join(text_outputs) if text_outputs else str(res)
            return {"result": output_str, "is_error": getattr(res, "isError", False)}
        except Exception as e:
            return {"error": f"MCP tool call failed: {str(e)}"}

    async def close_all(self) -> None:
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
        self._sessions.clear()
        self._tool_map.clear()
