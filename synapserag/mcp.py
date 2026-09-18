"""
CLI Entry Point for the SynapseRAG Native MCP Server.
Usage:
    python -m synapserag.mcp --storage-dir ./data/synapse_db
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

from .config import SynapseConfig
from .engine import SynapseEngine
from .connectors.mcp_server import create_mcp_server


def main():
    parser = argparse.ArgumentParser(description="Run SynapseRAG Native MCP Server")
    parser.add_argument(
        "--storage-dir",
        type=str,
        default="./data/synapse_db",
        help="Path to the local storage directory for vectors, graph, and sparse index."
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="synapse-rag",
        help="Name of the MCP server"
    )
    args = parser.parse_args()

    config = SynapseConfig(storage_dir=args.storage_dir)
    engine = SynapseEngine(config=config)
    mcp = create_mcp_server(engine=engine, server_name=args.server_name)

    # Run FastMCP stdio server
    mcp.run()


if __name__ == "__main__":
    main()
