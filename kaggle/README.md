# Fine-tuning RAFT-Stereo on Kaggle

## One-time setup

1. **Upload the Middlebury dataset** to Kaggle:
   - Go to kaggle.com → Datasets → New Dataset
   - Upload the contents of `/run/media/dvaghani/Expansion/Dataset/middlebury2014`
     (all 46 scene folders — ~2 GB)
   - Name it `middlebury2014`
   - Tip: zip it first (`cd /run/media/dvaghani/Expansion/Dataset && zip -r middlebury2014.zip middlebury2014`) —
     Kaggle auto-extracts zips on upload.

2. **Upload this script** as a second small dataset (or paste it into the
   notebook directly): `train_raft_middlebury.py`

## Notebook

Create a new Kaggle Notebook with **GPU T4 x2 or P100** accelerator
(Settings → Accelerator), then:

```python
# Cell 1 — get RAFT-Stereo code + pretrained weights
!git clone --depth 1 https://github.com/princeton-vl/RAFT-Stereo /kaggle/working/RAFT-Stereo
!cd /kaggle/working/RAFT-Stereo && mkdir -p models && cd models && \
    wget -q https://www.dropbox.com/s/ftveifyqcomiwaq/models.zip && \
    unzip -o -q models.zip && rm models.zip
!pip install -q opt_einsum

# Cell 2 — train (~30-60 min on P100)
!python /kaggle/input/YOUR-SCRIPT-DATASET/train_raft_middlebury.py \
    --data-root /kaggle/input/middlebury2014 \
    --raft-dir  /kaggle/working/RAFT-Stereo \
    --ckpt      /kaggle/working/RAFT-Stereo/models/raftstereo-realtime.pth \
    --out       /kaggle/working/ckpt \
    --steps 4000 --batch-size 4 --crop 320 640 --lr 2e-5

# Cell 3 — keep the result
# /kaggle/working/ckpt/best.pth appears in the notebook Output tab — download it.
```

## After training

Download `best.pth`, place it at
`stereo_unet/checkpoints/raft_middlebury_ft/best.pth`, then evaluate locally
with the same protocol as the thesis table:

```bash
# (eval_raft.py needs a --ckpt override — ask Claude to add it, or replace
#  the models/raftstereo-realtime.pth file with best.pth)
.venv/bin/python scripts/eval_raft.py --dataset middlebury \
    --data-root /run/media/dvaghani/Expansion/Dataset/middlebury2014 \
    --model realtime --iters 16
```

## Reference numbers (seed-42 val split, thesis protocol)

| Model | Middlebury val D1 |
|---|---|
| AANet fine-tuned + uncertainty (thesis) | 10.13% |
| RAFT-realtime zero-shot, 7 iters | 9.46% |
| RAFT-realtime zero-shot, 16 iters | 7.36% |
| RAFT-realtime fine-tuned (this run) | **target: < 6%** |

## KITTI fine-tune (second run)

Upload KITTI the same way (zip `/run/media/dvaghani/Expansion/Dataset/data_scene_flow`,
name the dataset e.g. `kitti-scene-flow`), paste `train_raft_kitti.py` via
`%%writefile /kaggle/working/train_raft_kitti.py`, then:

```python
!python /kaggle/working/train_raft_kitti.py \
    --data-root /kaggle/input/kitti-scene-flow/data_scene_flow \
    --raft-dir  /kaggle/working/RAFT-Stereo \
    --ckpt      /kaggle/working/RAFT-Stereo/models/raftstereo-realtime.pth \
    --out       /kaggle/working/ckpt_kitti \
    --steps 4000 --batch-size 4 --crop 320 640 --lr 2e-5
```

(Adjust `--data-root` to whatever `!ls /kaggle/input/` shows — the folder that
contains `training/image_2`.) Reference: AANet fine-tuned = 0.72% D1,
RAFT zero-shot = 5.10%. Download `ckpt_kitti/best.pth` →
`stereo_unet/checkpoints/raft_kitti_ft/best.pth`.

## Uncertainty head (phase 3)

Needs the fine-tuned checkpoint as input: upload
`checkpoints/raft_middlebury_ft/best.pth` as a Kaggle dataset (e.g.
`raft-middlebury-ft`). Paste `train_raft_uncertainty.py` via `%%writefile`,
then (Middlebury dataset + RAFT-Stereo clone as before):

```python
!python /kaggle/working/train_raft_uncertainty.py \
    --data-root /kaggle/input/middlebury2014 \
    --raft-dir  /kaggle/working/RAFT-Stereo \
    --ckpt      /kaggle/input/raft-middlebury-ft/best.pth \
    --out       /kaggle/working/ckpt_unc \
    --steps 2000 --batch-size 4 --crop 320 640 --lr 1e-4
```

Base network is frozen — only the 149k-param head trains, so this is the
fastest run (~20-30 min). Watch the sparsification line: D1@100% → D1@50% →
D1@20% must decrease monotonically. Download `ckpt_unc/best.pth` →
`stereo_unet/checkpoints/raft_middlebury_unc/best.pth`.

## Hyperparameters to try if the first run disappoints

- `--lr 1e-5` (more conservative) or `--lr 4e-5`
- `--steps 8000` (Middlebury is tiny — watch for overfitting in the val prints)
- `--crop 360 720` with `--batch-size 3` (bigger context, paper's crop)
