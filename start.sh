#!/usr/bin/env bash
# Start Image Gen MCP. Reads `device:` from config.yaml (gpu or cpu),
# builds the matching image if it does not exist yet and waits until the model is loaded.
# --build rebuilds (or, with IMAGE_REPO set, pulls) the image even if it exists.
set -euo pipefail
BUILD=0
[ "${1:-}" = "--build" ] && BUILD=1
cd "$(dirname "$0")"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed. See README.md -> Requirements." >&2
  exit 1
fi

# config.yaml is local (not tracked by git): create it from the tracked example on the first run.
[ -f config.yaml ] || { cp config.example.yaml config.yaml; echo "==> created config.yaml from config.example.yaml (edit it to change GPUs, sizes, ...)"; }

DEVICE=$(sed -nE 's/^[[:space:]]*device[[:space:]]*:[[:space:]]*"?([A-Za-z]+)"?.*/\1/p' config.yaml | head -n1 | tr '[:upper:]' '[:lower:]')
case "$DEVICE" in
  gpu|cuda) PROFILE=gpu ;;
  cpu) PROFILE=cpu ;;
  *) echo "config.yaml: 'device' must be gpu or cpu (found '$DEVICE')" >&2; exit 1 ;;
esac

mkdir -p models outputs inputs
[ -f .env ] || cp .env.example .env
# server.port in config.yaml (the only "port:" key in the file)
PORT=$(sed -nE 's/^[[:space:]]+port[[:space:]]*:[[:space:]]*([0-9]+).*/\1/p' config.yaml | head -n1)
PORT=${PORT:-5005}

# Remember the profile and port so plain `docker compose logs/restart/up` use the same settings.
{ grep -v -E '^(COMPOSE_PROFILES|MCP_PORT)=' .env || true; echo "COMPOSE_PROFILES=$PROFILE"; echo "MCP_PORT=$PORT"; } > .env.tmp && mv .env.tmp .env

echo "==> device: $PROFILE"
# Build (or pull) first, while any running instance keeps serving.
IMAGE_REPO=$(sed -nE 's/^[[:space:]]*IMAGE_REPO[[:space:]]*=[[:space:]]*([^[:space:]#]+).*/\1/p' .env | head -n1)
IMAGE="${IMAGE_REPO:-imagegen-mcp}:$PROFILE"
if [ "$BUILD" = 1 ] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  if [ -n "$IMAGE_REPO" ]; then
    docker compose --profile "$PROFILE" pull
  else
    docker compose --profile "$PROFILE" build
  fi
else
  echo "==> using the existing image $IMAGE (./start.sh --build rebuilds it)"
  if [ -z "$IMAGE_REPO" ]; then
    BUILT=$(date -d "$(docker image inspect -f '{{.Created}}' "$IMAGE")" +%s 2>/dev/null || echo 0)
    if [ "$BUILT" -gt 0 ]; then  # GNU date only; skipped where the timestamp cannot be parsed (macOS)
      NEWER=$(find src pyproject.toml Dockerfile docker/patches -type f -newermt "@$BUILT" 2>/dev/null | head -n1)
      [ -n "$NEWER" ] && echo "    note: $NEWER changed after this image was built; run ./start.sh --build to use the new code"
    fi
  fi
fi

# Replace the running container (either profile) with the new one.
docker compose --profile gpu --profile cpu down --remove-orphans >/dev/null 2>&1 || true
docker rm -f imagegen-mcp >/dev/null 2>&1 || true
if ! docker compose --profile "$PROFILE" up -d --no-build; then
  docker ps --format '{{.Names}}' | grep -q '^imagegen-mcp$' || { echo "docker compose up failed" >&2; exit 1; }
  echo "    (compose reported an error but the container is running; continuing)"
fi

echo "==> waiting for the server (first start downloads ~11.8 GB of models)"
last=""
while true; do
  # /api/status always answers 200 ({"state":...} first); /health returns 503 on errors.
  body=$(curl -fsS "http://localhost:${PORT}/api/status" 2>/dev/null || true)
  if [ -n "$body" ]; then
    state=$(printf '%s' "$body" | sed -nE 's/^\{"state":"([a-z]+)".*/\1/p')
    pct=$(printf '%s' "$body" | sed -nE 's/.*"percent":([0-9.]+).*/\1/p')
    msg="state: ${state:-?}${pct:+ (download ${pct}%)}"
    [ "$msg" != "$last" ] && echo "    $msg" && last="$msg"
    [ "$state" = "ready" ] && break
    if [ "$state" = "error" ]; then
      echo "Startup failed. Details: docker compose logs --tail 80" >&2
      printf '%s\n' "$body" >&2
      exit 1
    fi
  fi
  if ! docker ps --format '{{.Names}}' | grep -q '^imagegen-mcp$'; then
    echo "The container stopped. Details: docker compose logs --tail 80" >&2
    exit 1
  fi
  sleep 5
done

echo
echo "Ready. MCP endpoint (Streamable HTTP, no auth):"
echo "    http://localhost:${PORT}/mcp"
echo "From other machines use this host's IP address instead of localhost."
echo "Logs: docker compose logs -f     Stop: ./stop.sh"
