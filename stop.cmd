@echo off
rem Stop Image Gen MCP.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop.ps1" %*
