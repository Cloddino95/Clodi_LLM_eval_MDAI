#!/usr/bin/env bash
# Deprecated compatibility wrapper. Study settings live only in YAML.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python "$PROJECT_ROOT/run_study.py" "$@"
