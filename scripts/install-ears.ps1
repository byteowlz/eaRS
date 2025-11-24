# Install script for ears on Windows
# Run with: powershell -ExecutionPolicy Bypass -File scripts\install-ears.ps1

$ErrorActionPreference = "Stop"

$ROOT_DIR = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

$FEATURES = @()

# Platform/Acceleration selection
Write-Host "Select hardware acceleration:"
Write-Host "  1) none (CPU only)"
Write-Host "  2) nvidia (CUDA)"
Write-Host "  3) apple (Metal/CoreML)"
Write-Host "  4) amd (ROCm)"
Write-Host "  5) directml (Windows)"
$platform_choice = Read-Host "Enter choice [1-5]"

switch ($platform_choice) {
    "2" { $FEATURES += "nvidia" }
    "3" { $FEATURES += "apple" }
    "4" { $FEATURES += "amd" }
    "5" { $FEATURES += "directml" }
}

# Engine selection
Write-Host ""
Write-Host "Select additional speech recognition engine:"
Write-Host "(Kyutai moshi is always enabled)"
Write-Host "  1) none"
Write-Host "  2) whisper"
Write-Host "  3) parakeet"
$engine_choice = Read-Host "Enter choice [1-3]"

switch ($engine_choice) {
    "2" { $FEATURES += "whisper" }
    "3" { $FEATURES += "parakeet" }
}

# Hooks selection
Write-Host ""
$hooks_choice = Read-Host "Enable script execution hooks for dictation? [y/N]"
if ($hooks_choice -match "^[Yy]$") {
    $FEATURES += "hooks"
}

# Build feature flag
if ($FEATURES.Count -eq 0) {
    $FEATURE_FLAG = @()
    Write-Host ""
    Write-Host "Building with default features (CPU, moshi)..."
} else {
    $FEATURE_STRING = $FEATURES -join ","
    $FEATURE_FLAG = @("--features", $FEATURE_STRING)
    Write-Host ""
    Write-Host "Building with features: $FEATURE_STRING"
}

cargo build --release @FEATURE_FLAG
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Installing ears..."
cargo install --path "$ROOT_DIR" --force @FEATURE_FLAG
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Installation complete. Binaries are available in $env:USERPROFILE\.cargo\bin."
Write-Host "Installed binaries: ears, ears-server, ears-dictation"
