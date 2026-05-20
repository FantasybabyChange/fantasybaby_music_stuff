param(
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8767,
    [string]$OutputDir = "output/ui",
    [ValidateSet("DEBUG", "INFO", "WARNING", "ERROR")]
    [string]$LogLevel = "INFO",
    [switch]$Open,
    [switch]$DryRun,
    [switch]$Sync
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv is not installed or not available on PATH."
}

$srcPath = Join-Path $repoRoot "src"
if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$srcPath;$env:PYTHONPATH"
} else {
    $env:PYTHONPATH = $srcPath
}

$arguments = @(
    "run",
    "--no-sync",
    "python",
    "-m",
    "music_stuff.cli",
    "--log-level",
    $LogLevel,
    "ui",
    "--host",
    $HostName,
    "--port",
    $Port,
    "--out",
    $OutputDir
)

if ($Sync) {
    $arguments = $arguments | Where-Object { $_ -ne "--no-sync" }
}

if ($Open) {
    $arguments += "--open"
}

if ($DryRun) {
    $arguments += "--dry-run"
}

Write-Host "Starting Music Stuff UI at http://$HostName`:$Port"
uv @arguments
