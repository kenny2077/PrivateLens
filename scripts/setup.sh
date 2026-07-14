#!/usr/bin/env bash

set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    echo "PrivateLens setup requires uv: https://docs.astral.sh/uv/" >&2
    exit 1
fi

echo "Installing the locked PrivateLens full stack with Python 3.11..."
uv sync --python 3.11 --locked --extra full

mkdir -p "$HOME/.privatelens/thumbnails" "$HOME/.privatelens/models"

echo ""
echo "Setup complete. Next steps:"
echo "  1. Activate:        source .venv/bin/activate"
echo "  2. Inspect setup:   privatelens setup"
echo "  3. Scan a folder:   privatelens scan ~/Pictures"
echo "  4. Index photos:    privatelens index --skip-face --skip-vlm --batch-size 1"
echo "  5. Search:          privatelens search 'driver license'"
