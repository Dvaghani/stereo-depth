#!/usr/bin/env bash
# Upload Middlebury 2014 dataset (~5 GB) to Kaggle.
#
# Usage:
#   bash scripts/upload_middlebury_kaggle.sh YOUR_KAGGLE_USERNAME

set -e

USERNAME="${1:-}"
if [ -z "$USERNAME" ]; then
    echo "Usage: bash scripts/upload_middlebury_kaggle.sh YOUR_KAGGLE_USERNAME"
    exit 1
fi

# Data is on the native filesystem (no symlink workarounds needed)
MB_ROOT="/home/dvaghani/PycharmProjects/Depth Map generation/stereo_unet/datasets/middlebury2014"
STAGING="$HOME/kaggle_staging/middlebury2014"
mkdir -p "$STAGING"

chmod 600 ~/.kaggle/kaggle.json

cat > "$STAGING/dataset-metadata.json" <<EOF
{
  "title": "middlebury2014",
  "id": "${USERNAME}/middlebury2014",
  "licenses": [{"name": "other"}]
}
EOF

# Symlink the dataset folder — native filesystem supports symlinks
ln -sfn "$MB_ROOT" "$STAGING/middlebury2014"

echo "Staging : $STAGING"
echo "Uploading Middlebury 2014 (~5 GB)..."
kaggle datasets create -p "$STAGING" --dir-mode zip
echo "Done -> kaggle.com/${USERNAME}/middlebury2014"
