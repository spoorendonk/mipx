#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
exec python3 "${ROOT_DIR}/tests/perf/generate_highs_baselines.py" "$@"
