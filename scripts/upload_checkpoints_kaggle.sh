#!/usr/bin/env bash
# Upload key training checkpoints to Kaggle as a private dataset.
# These are the starting points needed for Kaggle training runs.
#
# Usage:
#   bash scripts/upload_checkpoints_kaggle.sh YOUR_KAGGLE_USERNAME

set -e

USERNAME="${1:-}"
if [ -z "$USERNAME" ]; then
    echo "Usage: bash scripts/upload_checkpoints_kaggle.sh YOUR_KAGGLE_USERNAME"
    exit 1
fi

STAGING="/run/media/dvaghani/Expansion/kaggle_staging/stereo-checkpoints"
mkdir -p "$STAGING/kitti_aanet_ft"
mkdir -p "$STAGING/middlebury_aanet_v2"

# Copy only the key checkpoints (best.pt only, skip last.pt)
cp checkpoints/kitti_aanet_ft/best.pt        "$STAGING/kitti_aanet_ft/best.pt"
cp checkpoints/middlebury_aanet_v2/best.pt   "$STAGING/middlebury_aanet_v2/best.pt"

cat > "$STAGING/dataset-metadata.json" <<EOF
{
  "title": "stereo-checkpoints",
  "id": "${USERNAME}/stereo-checkpoints",
  "licenses": [{"name": "other"}]
}
EOF

echo "Uploading checkpoints (~100 MB)..."
kaggle datasets create -p "$STAGING" --dir-mode zip
echo "stereo-checkpoints uploaded ✓"
echo "Available at: kaggle.com/${USERNAME}/stereo-checkpoints"
