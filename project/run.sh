#!/usr/bin/env bash
# Run the Munder Difflin multi-agent project (uses local .venv)
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
  echo "Creating venv with Python 3.12..."
  python3.12 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
else
  source .venv/bin/activate
fi

if [[ ! -f .env ]]; then
  echo "ERROR: Create .env with UDACITY_OPENAI_API_KEY=your_key"
  exit 1
fi

python -u project_starter.py
