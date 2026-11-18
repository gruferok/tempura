#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
PYTHON="${PYTHON:-.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    echo 'Create .venv first: python3 -m venv .venv && .venv/bin/python -m pip install -r requirements-dev.txt' >&2
    exit 1
fi
# This sandbox has a locally built native runtime; normal desktop installs use
# the OS libportaudio2 (Linux) or the sounddevice wheel's DLL/dylib.
if [ -f .venv/lib/libportaudio.so ]; then
    export LD_LIBRARY_PATH="$PWD/.venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [ "${1:-}" = '--test' ]; then
    shift
    exec "$PYTHON" -m pytest "$@"
fi
exec "$PYTHON" main.py "$@"
