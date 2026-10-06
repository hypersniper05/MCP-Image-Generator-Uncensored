#!/usr/bin/env bash
# Stop Image Gen MCP (either profile).
cd "$(dirname "$0")"
docker compose --profile gpu --profile cpu down
