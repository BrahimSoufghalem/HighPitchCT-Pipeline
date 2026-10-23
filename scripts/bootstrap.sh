#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v

echo "Environment ready. Activate it with: source .venv/bin/activate"
