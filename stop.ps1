# Stop Image Gen MCP (either profile).
Set-Location -Path $PSScriptRoot
docker compose --profile gpu --profile cpu down
