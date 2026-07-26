"""
Pre-label the labeling pool with YOLO-World (open-vocabulary detection).

Detects the 12 SCD classes from text prompts — no training needed. Multiple
prompts per class are supported and collapsed back to the final class id, so
weak classes (scaffolding, suspended-loads, cable ...) get several phrasings.

These labels are a STARTING POINT for human correction in Label Studio, not
final ground truth. The threshold is deliberately low: over-propose for the
rare classes, let the human reject false positives (faster than drawing).

Usage:
    python prelabel_yoloworld.py \
        --images "/run/media/dvaghani/Expansion/Yolo/labeling_pool/images" \
        --out    "/run/media/dvaghani/Expansion/Yolo/labeling_pool" \
        [--limit 20]        # small test batch; omit for the full pool
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLOWorld

# Final taxonomy (index = YOLO class id).
CLASS_NAMES = [
    "container", "pole", "scaffolding", "crane", "person", "machinery",
    "suspended-loads", "vehicle", "building", "bridge", "barrier", "cable",
]

# Prompt -> class id.  Several prompts may map to one class; the rare
# classes (2 scaffolding, 6 suspended-loads, 10 barrier, 11 cable, 0 container,
# 1 pole) get extra phrasings because YOLO-World misses them with one prompt.
PROMPTS = [
    # container (0)
    ("shipping container", 0), ("construction site container", 0),
    ("dumpster skip container", 0),
    # pole (1)
    ("utility pole", 1), ("street lamp post", 1), ("vertical metal post", 1),
    # scaffolding (2)
    ("scaffolding", 2), ("metal scaffolding on building facade", 2),
    ("construction scaffold structure", 2),
    # crane (3)
    ("construction crane", 3), ("tower crane", 3), ("mobile crane", 3),
    # person (4)
    ("person", 4), ("construction worker", 4),
    # machinery (5)
    ("construction machinery", 5), ("excavator", 5), ("bulldozer", 5),
    ("wheel loader", 5),
    # suspended-loads (6)
    ("load hanging from crane hook", 6), ("suspended construction load", 6),
    ("material lifted by crane", 6),
    # vehicle (7)
    ("car", 7), ("truck", 7), ("van", 7),
    # building (8)
    ("building", 8), ("building under construction", 8),
    # bridge (9)
    ("bridge", 9),
    # barrier (10)
    ("construction barrier fence", 10), ("safety barrier", 10),
    ("temporary fence", 10),
    # cable (11)
    ("hanging cable", 11), ("power line wire", 11), ("suspended cable", 11),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--out",    required=True)
    p.add_argument("--model",  default="yolov8x-worldv2.pt")
    p.add_argument("--conf",   type=float, default=0.12)
    p.add_argument("--limit",  type=int, default=None)
    args = p.parse_args()

    out = Path(args.out)
    lbl_dir = out / "labels";  lbl_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out / "vis";     vis_dir.mkdir(parents=True, exist_ok=True)

    prompts    = [t for t, _ in PROMPTS]
    prompt2cls = [c for _, c in PROMPTS]

    model = YOLOWorld(args.model)
    model.set_classes(prompts)

    images = sorted(Path(args.images).glob("*.jpg"))
    if args.limit:
        images = images[:: max(1, len(images) // args.limit)][: args.limit]
    print(f"Pre-labeling {len(images)} images with {args.model} "
          f"({len(prompts)} prompts -> 12 classes)")

    per_class = [0] * len(CLASS_NAMES)
    n_boxes = 0
    for i, img_path in enumerate(images):
        res = model.predict(str(img_path), conf=args.conf, verbose=False)[0]
        H, W = res.orig_shape

        lines = []
        vis = cv2.imread(str(img_path))
        for box in res.boxes:
            cls = prompt2cls[int(box.cls)]          # collapse prompt -> class
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
            bw, bh = (x2 - x1) / W, (y2 - y1) / H
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            per_class[cls] += 1
            # draw with the collapsed class name
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 220, 60), 2)
            cv2.putText(vis, f"{CLASS_NAMES[cls]} {float(box.conf):.2f}",
                        (int(x1), max(0, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 60), 1)
        (lbl_dir / f"{img_path.stem}.txt").write_text("\n".join(lines))
        cv2.imwrite(str(vis_dir / img_path.name), vis)
        n_boxes += len(lines)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(images)}")

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    (out / "dataset.yaml").write_text(
        f"path: {out}\ntrain: images\nval: images  # split later\n"
        f"nc: {len(CLASS_NAMES)}\nnames:\n{names}\n")

    print(f"\n{n_boxes} boxes over {len(images)} images "
          f"({n_boxes/max(len(images),1):.1f}/image)")
    print("Per-class box counts:")
    for i, n in enumerate(CLASS_NAMES):
        print(f"  {i:2d} {n:16s} {per_class[i]}")
    print(f"\nlabels → {lbl_dir}\nvis → {vis_dir}\ndataset.yaml written")


if __name__ == "__main__":
    main()
