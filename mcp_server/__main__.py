"""
mcp_server/__main__.py
Entrypoint for running the RAG MCP server as a standalone process.
Usage:
    python -m mcp_server [--config config/phase4.yaml]
"""
from __future__ import annotations

import argparse
from mcp_server.server import create_server


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline MCP Server")
    parser.add_argument(
        "--config",
        default="config/phase4.yaml",
        help="Path to config YAML (default: config/phase4.yaml)",
    )
    args = parser.parse_args()

    server = create_server(config_path=args.config)
    server.run()


if __name__ == "__main__":
    main()
