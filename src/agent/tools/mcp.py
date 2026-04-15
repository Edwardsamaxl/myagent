from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import dataclass
from typing import Any

import requests

from .schemas import Tool, ToolSchema, ToolSource
from .context import ToolUseContext


@dataclass
class MCPServerConfig:
    url: str  # HTTP URL or "stdio" for local servers
    name: str  # Server name (e.g., "filesystem")
    command: str | None = None  # For stdio mode: the command to execute
    headers: dict[str, str] | None = None


class MCPClient:
    """MCP client that connects to MCP servers via stdio or HTTP.

    MCP protocol: JSON-RPC 2.0 over stdio or HTTP.
    Tool discovery via 'tools/list' endpoint.
    Tool execution via 'tools/call' endpoint.
    """

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self._tools: dict[str, Tool] = {}

    async def connect(self) -> None:
        """Connect to MCP server and discover available tools."""
        if self.config.url == "stdio":
            await self._connect_stdio()
        else:
            await self._connect_http()

    async def _connect_http(self) -> None:
        """Connect via HTTP."""
        try:
            resp = requests.get(
                f"{self.config.url}/tools/list",
                headers=self.config.headers,
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json()

            for tool_def in data.get("tools", []):
                schema = ToolSchema(
                    name=tool_def["name"],
                    description=tool_def.get("description", ""),
                    input_schema=tool_def.get("input_schema", {"type": "object", "properties": {}}),
                    output_schema=tool_def.get("output_schema"),
                )
                tool = Tool(
                    schema=schema,
                    source=ToolSource.MCP,
                    handler=self._create_mcp_handler(tool_def["name"]),
                    mcp_server=self.config.name,
                )
                self._tools[schema.name] = tool
        except Exception as exc:
            raise ConnectionError(f"Failed to connect to MCP server {self.config.name}: {exc}")

    async def _connect_stdio(self) -> None:
        """Connect via stdio using subprocess."""
        if not self.config.command:
            raise ConnectionError(f"stdio mode requires command for server {self.config.name}")

        try:
            # Parse command into args
            cmd_parts = self.config.command.split()
            proc = await asyncio.create_subprocess_exec(
                *cmd_parts,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Send tools/list request
            request = json.dumps({"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 1})
            proc.stdin.write(request.encode() + b"\n")
            await proc.stdin.drain()

            # Read response
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=30)
            data = json.loads(line.decode())

            for tool_def in data.get("result", {}).get("tools", []):
                schema = ToolSchema(
                    name=tool_def["name"],
                    description=tool_def.get("description", ""),
                    input_schema=tool_def.get("input_schema", {"type": "object", "properties": {}}),
                )
                tool = Tool(
                    schema=schema,
                    source=ToolSource.MCP,
                    handler=self._create_mcp_handler(tool_def["name"]),
                    mcp_server=self.config.name,
                )
                self._tools[schema.name] = tool

        except Exception as exc:
            raise ConnectionError(f"Failed to connect to MCP server via stdio: {exc}")

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute an MCP tool with given arguments."""
        if self.config.url == "stdio":
            return await self._call_stdio(name, arguments)
        return await self._call_http(name, arguments)

    async def _call_http(self, name: str, arguments: dict[str, Any]) -> str:
        try:
            resp = requests.post(
                f"{self.config.url}/tools/execute",
                json={"name": name, "arguments": arguments},
                headers=self.config.headers,
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json().get("result", "")
        except Exception as exc:
            return f"MCP tool call failed: {exc}"

    async def _call_stdio(self, name: str, arguments: dict[str, Any]) -> str:
        # Simplified stdio call - requires process persistence
        return f"stdio call not fully implemented for {name}"

    def _create_mcp_handler(self, tool_name: str):
        """Create a handler closure that calls this tool via MCP."""
        async def handler(ctx: ToolUseContext, **kwargs) -> str:
            return await self.call_tool(tool_name, kwargs)
        return handler

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())


class MCPToolManager:
    """Manages MCP server connections and tool discovery."""

    def __init__(self) -> None:
        self._clients: dict[str, MCPClient] = {}
        self._servers: dict[str, MCPServerConfig] = {}

    def add_server(
        self,
        name: str,
        url: str,
        command: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Add an MCP server configuration.

        Args:
            name: Server name (e.g., "filesystem")
            url: "stdio" for local servers, or HTTP URL
            command: For stdio mode: the command to execute (e.g., "npx @modelcontextprotocol/server-filesystem ./workspace")
            headers: Optional HTTP headers for HTTP mode
        """
        self._servers[name] = MCPServerConfig(
            url=url, name=name, command=command, headers=headers
        )

    async def connect_server(self, name: str) -> list[Tool]:
        """Connect to an MCP server and discover tools."""
        if name not in self._servers:
            raise ValueError(f"Unknown MCP server: {name}")

        client = MCPClient(self._servers[name])
        await client.connect()
        self._clients[name] = client
        return client.list_tools()

    async def connect_all(self) -> dict[str, list[Tool]]:
        """Connect to all configured MCP servers."""
        all_tools = {}
        for name in self._servers:
            try:
                tools = await self.connect_server(name)
                all_tools[name] = tools
            except Exception as exc:
                all_tools[name] = []
                print(f"Warning: Failed to connect to MCP server {name}: {exc}")
        return all_tools

    def get_client(self, name: str) -> MCPClient | None:
        return self._clients.get(name)

    def list_servers(self) -> list[str]:
        return list(self._servers.keys())
