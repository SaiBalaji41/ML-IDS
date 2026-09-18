#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ -x .venv-macos/bin/python ]; then
  IDS_PYTHON=.venv-macos/bin/python
elif [ -x .venv/bin/python ]; then
  IDS_PYTHON=.venv/bin/python
else
  echo 'Set up a Python 3.12 environment and install requirements.txt first. See README.md.'
  exit 1
fi
mkdir -p runtime
export MPLCONFIGDIR="${TMPDIR:-/tmp}/ml-ids-matplotlib"
exec "$IDS_PYTHON" -u dashboard/server.py --port "${IDS_PORT:-8765}"
