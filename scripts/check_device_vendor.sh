#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../vendor/device"
shasum -a 256 -c SHA256SUMS
