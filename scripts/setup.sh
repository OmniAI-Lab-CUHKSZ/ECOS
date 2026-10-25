#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"
python -m pip install -r requirements/runtime.txt -r requirements/environments.txt
python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt punkt_tab stopwords
python -m pip install --no-deps -e .
