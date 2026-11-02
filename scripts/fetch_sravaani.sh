#!/bin/sh
# Download the SraVaani source files used by lecture_notes/asr.py.
# The repository has no licence file, so its code is not vendored here; this fetches it at a pinned commit.
set -e
COMMIT=075cf52fc6a2b7f1e24ba44d7515614a9d8b0ebe
BASE=https://raw.githubusercontent.com/switchtwitch12345/sravaani-for-mac/$COMMIT
DEST="$(cd "$(dirname "$0")/.." && pwd)/vendor-sravaani"
FILES="sravaani_flow/__init__.py sravaani_flow/audio.py sravaani_flow/cleanup.py sravaani_flow/config.py sravaani_flow/decoding.py sravaani_flow/engine.py sravaani_flow/fetch.py sravaani_flow/languages.py sravaani_flow/translit.py"
mkdir -p "$DEST/sravaani_flow"
for f in $FILES; do
  curl -fsSL "$BASE/$f" -o "$DEST/$f"
  echo "fetched $f"
done
