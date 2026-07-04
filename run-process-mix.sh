#!/usr/bin/env bash
#
# Wrapper to run process-mix.py with the project's virtualenv.
#
# Usage: ./run-process-mix.sh <input.wav> <output.mp3> [extra args...]
#
set -euo pipefail

# Resolve the directory this script lives in (handles spaces in the path).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"
SCRIPT="$SCRIPT_DIR/process-mix.py"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <input.wav> <output.mp3> [extra args...]" >&2
    exit 1
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "Error: virtualenv python not found at $VENV_PYTHON" >&2
    echo "Create it first, e.g.: python -m venv .venv && .venv/bin/pip install pedalboard scipy numpy pydub pyloudnorm" >&2
    exit 1
fi

INPUT="$1"
OUTPUT="$2"
shift 2

if [[ ! -f "$INPUT" ]]; then
    echo "Error: input file not found: $INPUT" >&2
    exit 1
fi

exec "$VENV_PYTHON" "$SCRIPT" "$INPUT" "$OUTPUT" "$@"
