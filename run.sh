#!/bin/bash
# confeTTI local dev runner.
set -euo pipefail
cd "$(dirname "$0")"

echo "== tests =="
python3 -m pytest -q "$@"
