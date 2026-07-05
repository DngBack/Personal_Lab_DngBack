param(
    [string]$Image = $env:VLLM_PARTIAL_PREFILL_IMAGE,
    [string]$BaseImage = "vllm/vllm-openai:v0.24.0-ubuntu2404",
    [switch]$SkipImageCheck,
    [switch]$Push
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Image)) {
    $Image = "dngback/vllm-v0.24.0-ubuntu2404-concurrent-prefill:latest"
}
if ($Push -and $Image -notmatch "/") {
    throw "Push requires a Docker Hub namespace, e.g. `$env:VLLM_PARTIAL_PREFILL_IMAGE='your_user/vllm-openai:v0.24.0-ubuntu2404-concurrent-prefill'"
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Test-DockerDaemonReady {
    $job = Start-Job -ScriptBlock {
        docker info *> $null
        return $LASTEXITCODE
    }
    if (-not (Wait-Job $job -Timeout 30)) {
        Stop-Job $job -ErrorAction SilentlyContinue
        Remove-Job $job -Force -ErrorAction SilentlyContinue
        return $false
    }
    $exitCode = Receive-Job $job
    Remove-Job $job -Force -ErrorAction SilentlyContinue
    return ($exitCode -eq 0)
}

if (-not (Test-DockerDaemonReady)) {
    throw "Docker daemon is not ready. Start Docker Desktop and ensure the Linux engine is running."
}

docker build `
    --build-arg BASE_IMAGE=$BaseImage `
    -f (Join-Path $ScriptDir "Dockerfile.vllm-concurrent-prefill") `
    -t $Image `
    $ScriptDir
if ($LASTEXITCODE -ne 0) {
    throw "docker build failed with exit code $LASTEXITCODE"
}

if (-not $SkipImageCheck) {
    docker run --rm --entrypoint python3 $Image `
        /opt/dng-vllm-patches/patch_vllm_concurrent_prefill_v024.py --check
    if ($LASTEXITCODE -ne 0) {
        throw "patched image sanity check failed with exit code $LASTEXITCODE"
    }
}

if ($Push) {
    docker push $Image
    if ($LASTEXITCODE -ne 0) {
        throw "docker push failed with exit code $LASTEXITCODE"
    }
}

Write-Host "Built image: $Image"
