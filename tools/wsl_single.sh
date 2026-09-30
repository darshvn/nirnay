#!/bin/bash
# Runs one comparator solve inside WSL (venv /opt/cuopt/venv) and prints the JSON marker line.
# usage: wsl_single.sh SOLVER MPS_PATH TIME_LIMIT METHOD VERBOSE(0|1)
export PATH=/opt/cuopt/venv/bin:$PATH
HERE="$(cd "$(dirname "$0")" && pwd)"
exec /opt/cuopt/venv/bin/python "$HERE/../bench/comparators.py" --single "$@"
