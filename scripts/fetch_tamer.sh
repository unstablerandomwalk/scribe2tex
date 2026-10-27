#!/usr/bin/env bash
# Pinned upstream TAMER (AAAI 2025) -> third_party/TAMER. The repo ships no licence, so we
# clone it at a fixed commit instead of vendoring it. Includes the released HME100K weights.
set -euo pipefail
TAMER_COMMIT=2213e2a145cb1e63a58d08a493553fcedc362acb
DEST="$(cd "$(dirname "$0")/.." && pwd)/third_party/TAMER"
[ -d "$DEST/.git" ] || git clone -q https://github.com/qingzhenduyu/TAMER.git "$DEST"
git -C "$DEST" checkout -q "$TAMER_COMMIT"
echo "TAMER @ $(git -C "$DEST" rev-parse --short HEAD) -> $DEST"
