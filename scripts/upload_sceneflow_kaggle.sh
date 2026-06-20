#!/usr/bin/env bash
# Upload FlyingThings3D subset to Kaggle as two private datasets.
#
# Prerequisites:
#   pip install kaggle
#   Place kaggle.json at ~/.kaggle/kaggle.json  (kaggle.com → Settings → API)
#
# Usage:
#   bash scripts/upload_sceneflow_kaggle.sh YOUR_KAGGLE_USERNAME

set -e

USERNAME="${1:-}"
if [ -z "$USERNAME" ]; then
    echo "Usage: bash scripts/upload_sceneflow_kaggle.sh YOUR_KAGGLE_USERNAME"
    exit 1
fi

SF_ROOT="/run/media/dvaghani/Expansion/FlyingThings3D subset/FlyingThings3D_subset"
STAGING="/tmp/kaggle_staging"

echo "=== FlyingThings3D → Kaggle upload ==="
echo "Source : $SF_ROOT"
echo "User   : $USERNAME"
echo

# ── Check kaggle CLI ─────────────────────────────────────────────────────────
if ! command -v kaggle &>/dev/null; then
    echo "Installing kaggle CLI..."
    pip install kaggle -q
fi
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "ERROR: ~/.kaggle/kaggle.json not found."
    echo "  1. Go to kaggle.com → Settings → API → Create New Token"
    echo "  2. Move the downloaded kaggle.json to ~/.kaggle/kaggle.json"
    exit 1
fi
chmod 600 ~/.kaggle/kaggle.json

# ── Dataset 1: Images (~35 GB) ───────────────────────────────────────────────
echo "--- Dataset 1: sceneflow-images (~35 GB) ---"
IMG_STAGE="$STAGING/sceneflow-images"
mkdir -p "$IMG_STAGE"

cat > "$IMG_STAGE/dataset-metadata.json" <<EOF
{
  "title": "sceneflow-images",
  "id": "${USERNAME}/sceneflow-images",
  "licenses": [{"name": "other"}]
}
EOF

# Symlink the actual data folders (avoids copying 35 GB)
ln -sfn "$SF_ROOT/train/image_clean" "$IMG_STAGE/train_image_clean"
ln -sfn "$SF_ROOT/val/image_clean"   "$IMG_STAGE/val_image_clean"

echo "Uploading images... (35 GB — takes a while depending on upload speed)"
kaggle datasets create -p "$IMG_STAGE" --dir-mode zip
echo "sceneflow-images uploaded ✓"
echo

# ── Dataset 2: Disparity (~89 GB) ────────────────────────────────────────────
echo "--- Dataset 2: sceneflow-disparity (~89 GB) ---"
DISP_STAGE="$STAGING/sceneflow-disparity"
mkdir -p "$DISP_STAGE"

cat > "$DISP_STAGE/dataset-metadata.json" <<EOF
{
  "title": "sceneflow-disparity",
  "id": "${USERNAME}/sceneflow-disparity",
  "licenses": [{"name": "other"}]
}
EOF

ln -sfn "$SF_ROOT/train/disparity" "$DISP_STAGE/train_disparity"
ln -sfn "$SF_ROOT/val/disparity"   "$DISP_STAGE/val_disparity"

echo "Uploading disparity... (89 GB — leave overnight)"
kaggle datasets create -p "$DISP_STAGE" --dir-mode zip
echo "sceneflow-disparity uploaded ✓"
echo

echo "=== All done ==="
echo "Datasets available at:"
echo "  kaggle.com/${USERNAME}/sceneflow-images"
echo "  kaggle.com/${USERNAME}/sceneflow-disparity"
echo
echo "In your Kaggle notebook, mount both datasets and set data_root to:"
echo "  /kaggle/input/sceneflow-images and /kaggle/input/sceneflow-disparity"
