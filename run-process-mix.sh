#!/usr/bin/env bash
#
# Wrapper to run process-mix.py with the project's virtualenv.
#
# Usage: ./run-process-mix.sh <input> [output.mp3] [extra args...]
#   input  - audio file (WAV, MP3, ...) or video (MP4, MOV, M4V, MKV)
#   output - optional; defaults to the input's name with .mp3
#
set -euo pipefail

# Resolve the directory this script lives in (handles spaces in the path).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"
SCRIPT="$SCRIPT_DIR/process-mix.py"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <input> [output.mp3] [extra args...]" >&2
    exit 1
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "Error: virtualenv python not found at $VENV_PYTHON" >&2
    echo "Create it first, e.g.: python -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi

if [[ ! -f "$1" ]]; then
    echo "Error: input file not found: $1" >&2
    exit 1
fi

exec "$VENV_PYTHON" "$SCRIPT" "$@"
