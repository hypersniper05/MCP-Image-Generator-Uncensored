# Start Image Gen MCP on Windows. Reads `device:` from config.yaml (gpu or cpu),
# builds the matching image if it does not exist yet and waits until the model is loaded.
# -Build rebuilds (or, with IMAGE_REPO set, pulls) the image even if it exists.
param([switch]$Build)
$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error 'Docker is not installed. See README.md -> Requirements.'
    exit 1
}

# config.yaml is local (not tracked by git): create it from the tracked example on the first run.
if (-not (Test-Path config.yaml)) {
    Copy-Item config.example.yaml config.yaml
    Write-Host '==> created config.yaml from config.example.yaml (edit it to change GPUs, sizes, ...)'
}

$deviceLine = Get-Content config.yaml | Where-Object { $_ -match '^\s*device\s*:' } | Select-Object -First 1
if ($deviceLine -match '^\s*device\s*:\s*"?([A-Za-z]+)') { $device = $Matches[1].ToLower() } else { $device = '' }
switch ($device) {
    'gpu'  { $composeProfile = 'gpu' }
    'cuda' { $composeProfile = 'gpu' }
    'cpu'  { $composeProfile = 'cpu' }
    default { Write-Error "config.yaml: 'device' must be gpu or cpu (found '$device')"; exit 1 }
}

foreach ($d in 'models', 'outputs', 'inputs') { if (-not (Test-Path $d)) { New-Item -ItemType Directory $d | Out-Null } }
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# server.port in config.yaml (the only "port:" key in the file)
$port = 5005
$portLine = Get-Content config.yaml | Where-Object { $_ -match '^\s+port\s*:\s*(\d+)' } | Select-Object -First 1
if ($portLine -match '^\s+port\s*:\s*(\d+)') { $port = [int]$Matches[1] }

# Remember the profile and port so plain `docker compose logs/restart/up` use the same settings.
$envLines = @(Get-Content .env | Where-Object { $_ -notmatch '^(COMPOSE_PROFILES|MCP_PORT)=' }) +
    "COMPOSE_PROFILES=$composeProfile" + "MCP_PORT=$port"
Set-Content -Path .env -Value $envLines -Encoding ascii

Write-Host "==> device: $composeProfile"
# Build (or pull) first, while any running instance keeps serving.
$imageRepo = ''
$repoLine = Get-Content .env | Where-Object { $_ -match '^\s*IMAGE_REPO\s*=\s*(\S+)' } | Select-Object -First 1
if ($repoLine -match '^\s*IMAGE_REPO\s*=\s*([^\s#]+)') { $imageRepo = $Matches[1] }
$image = if ($imageRepo) { "${imageRepo}:$composeProfile" } else { "imagegen-mcp:$composeProfile" }
$prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
docker image inspect $image 2>$null | Out-Null
$haveImage = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = $prev
if ($Build -or -not $haveImage) {
    if ($imageRepo) { docker compose --profile $composeProfile pull } else { docker compose --profile $composeProfile build }
    if ($LASTEXITCODE -ne 0) { Write-Error 'docker compose build/pull failed'; exit 1 }
} else {
    Write-Host "==> using the existing image $image (start.cmd -Build rebuilds it)"
    if (-not $imageRepo) {
        $built = [datetime](docker image inspect -f '{{.Created}}' $image)
        $newer = Get-ChildItem -Recurse -File src, pyproject.toml, Dockerfile, docker/patches |
                 Where-Object { $_.LastWriteTimeUtc -gt $built.ToUniversalTime() } | Select-Object -First 1
        if ($newer) {
            Write-Host "    note: $($newer.Name) changed after this image was built; run start.cmd -Build to use the new code" -ForegroundColor Yellow
        }
    }
}

# Replace the running container (either profile) with the new one.
$prev = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
docker compose --profile gpu --profile cpu down --remove-orphans 2>$null | Out-Null
docker rm -f imagegen-mcp 2>$null | Out-Null
$ErrorActionPreference = $prev
docker compose --profile $composeProfile up -d --no-build
if ($LASTEXITCODE -ne 0) {
    $running = docker ps --format '{{.Names}}' | Where-Object { $_ -eq 'imagegen-mcp' }
    if (-not $running) { Write-Error 'docker compose up failed'; exit 1 }
    Write-Host '    (compose reported an error but the container is running; continuing)'
}

Write-Host '==> waiting for the server (first start downloads ~11.8 GB of models)'
$last = ''
while ($true) {
    # /api/status always answers 200 with the state (/health returns 503 on errors, which
    # Invoke-WebRequest turns into an exception).
    try {
        $r = Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:$port/api/status" -TimeoutSec 5 -ErrorAction Stop
        $body = $r.Content
    } catch { $body = $null }
    if ($body) {
        $h = $body | ConvertFrom-Json
        $msg = "state: $($h.state)"
        if ($h.download -and $h.download.percent -ne $null -and $h.state -eq 'downloading') { $msg += " (download $($h.download.percent)%)" }
        if ($msg -ne $last) { Write-Host "    $msg"; $last = $msg }
        if ($h.state -eq 'ready') { break }
        if ($h.state -eq 'error') {
            Write-Host "Startup failed: $($h.error)" -ForegroundColor Red
            Write-Host 'Details: docker compose logs --tail 80'
            exit 1
        }
    }
    $running = docker ps --format '{{.Names}}' | Where-Object { $_ -eq 'imagegen-mcp' }
    if (-not $running) { Write-Host 'The container stopped. Details: docker compose logs --tail 80' -ForegroundColor Red; exit 1 }
    Start-Sleep -Seconds 5
}

Write-Host ''
Write-Host 'Ready. MCP endpoint (Streamable HTTP, no auth):' -ForegroundColor Green
Write-Host "    http://localhost:$port/mcp"
Write-Host "From other machines use this computer's IP address instead of localhost."
Write-Host 'Logs: docker compose logs -f     Stop: stop.cmd'
