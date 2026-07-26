#!/bin/bash
# Round-2 labeling pipeline: dedup -> pre-label -> filter to construction frames.
# Run after extract_frames.py finishes.
set -e
cd "$(dirname "$0")"
PY="../stereo_unet/.venv/bin/python"
BASE="/run/media/dvaghani/Expansion/Yolo"

# 1. Deduplicate the freshly extracted frames (they are already time-sampled,
#    so use a gentler hamming threshold than the every-Nth-frame round 1).
echo "== dedup =="
rm -rf "$BASE/pool_round2/images"
$PY select_frames_flat.py \
    --src "$BASE/frames_round2" \
    --dst "$BASE/pool_round2/images" \
    --min-hamming 14

# 2. Pre-label with YOLO-World (same 33-prompt -> 12-class setup).
echo "== pre-label =="
$PY prelabel_yoloworld.py \
    --images "$BASE/pool_round2/images" \
    --out    "$BASE/pool_round2"

# 3. Keep only construction-relevant frames.
echo "== filter =="
$PY filter_construction.py \
    --pool "$BASE/pool_round2" \
    --dst  "$BASE/label_subset_round2"

echo ""
echo "Done. Correction-ready frames: $BASE/label_subset_round2/images"
