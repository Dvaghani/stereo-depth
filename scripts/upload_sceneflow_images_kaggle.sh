#!/usr/bin/env bash
# Upload FlyingThings3D subset IMAGES only (~35 GB) to Kaggle.
#
# Usage:
#   bash scripts/upload_sceneflow_images_kaggle.sh YOUR_KAGGLE_USERNAME
#
# Notes:
#   - Symlinks live on $HOME (ext4) because the external drive is exFAT and
#     does not support symlinks.
#   - TMPDIR is redirected to the external drive so the zip does not fill /tmp.

set -e

USERNAME="${1:-}"
if [ -z "$USERNAME" ]; then
    echo "Usage: bash scripts/upload_sceneflow_images_kaggle.sh YOUR_KAGGLE_USERNAME"
    exit 1
fi

SF_ROOT="/run/media/dvaghani/Expansion/FlyingThings3D subset/FlyingThings3D_subset"
EXT_DRIVE="/run/media/dvaghani/Expansion"

# Staging on HOME (ext4) so symlinks work
STAGING="$HOME/kaggle_staging/sceneflow-images"
mkdir -p "$STAGING"

# Zip temp on external drive so the ~35 GB zip doesn't fill /tmp
KAGGLE_TMP="$EXT_DRIVE/kaggle_tmp"
mkdir -p "$KAGGLE_TMP"
export TMPDIR="$KAGGLE_TMP"

chmod 600 ~/.kaggle/kaggle.json

cat > "$STAGING/dataset-metadata.json" <<EOF
{
  "title": "sceneflow-images",
  "id": "${USERNAME}/sceneflow-images",
  "licenses": [{"name": "other"}]
}
EOF

# Symlinks on ext4 HOME pointing to data on external drive
ln -sfn "$SF_ROOT/train/image_clean" "$STAGING/train_image_clean"
ln -sfn "$SF_ROOT/val/image_clean"   "$STAGING/val_image_clean"

echo "Staging : $STAGING"
echo "Zip tmp : $KAGGLE_TMP"
echo "Uploading SceneFlow images (~35 GB) — this will take a while..."
kaggle datasets create -p "$STAGING" --dir-mode zip
echo "Done -> kaggle.com/${USERNAME}/sceneflow-images"
