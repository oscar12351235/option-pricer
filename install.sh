#!/usr/bin/env bash
set -euo pipefail

poetry env use python3.12 >/dev/null 2>&1 || true
poetry install --with dev,lint,types
poetry export -f requirements.txt --without-hashes --output requirements.txt >/dev/null 2>&1 || true
poetry run pre-commit install >/dev/null 2>&1 || true

echo "✅ Installed. Try: poetry run pytest"
