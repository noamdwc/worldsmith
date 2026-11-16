"""Launch the AutoResearch MCP server over stdio."""

from .research_mcp import mcp


if __name__ == "__main__":
    mcp.run(transport="stdio")
