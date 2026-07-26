"""
Re-label the weak classes (container, crane, pole) with Grounding DINO via
autodistill — a stronger open-vocabulary detector than YOLO-World, used here
ONLY for the classes the baseline showed were too noisy to train on.

Building/vehicle/scaffolding/person/machinery already work from YOLO-World —
this script does not touch them.

Usage:
    python prelabel_groundingdino.py \
        --images "/run/media/dvaghani/Expansion/Yolo/construction_dataset/images/train" \
        --out    "/run/media/dvaghani/Expansion/Yolo/gdino_relabel" \
        --limit 20        # test batch first
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("TORCH_HOME", "/run/media/dvaghani/Expansion/model_cache/torch")
os.environ.setdefault("HF_HOME", "/run/media/dvaghani/Expansion/model_cache/huggingface")

import cv2
from autodistill.detection import CaptionOntology
from autodistill_grounding_dino import GroundingDINO

# Only the 3 classes the YOLO-World baseline could not learn (mAP50 < 0.21).
# class name -> final class id (must match construction_dataset taxonomy)
ONTOLOGY = {
    "shipping container, metal storage container": 0,
    "tower crane, construction crane with long arm": 3,
    "utility pole, thin vertical post": 1,
}
CLASS_NAMES = {0: "container", 1: "pole", 3: "crane"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--out",    required=True)
    p.add_argument("--box-threshold", type=float, default=0.30)
    p.add_argument("--text-threshold", type=float, default=0.25)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    out = Path(args.out)
    lbl_dir = out / "labels"; lbl_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out / "vis";    vis_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Grounding DINO (downloads ~700MB on first run)...")
    ontology = CaptionOntology({k: str(v) for k, v in ONTOLOGY.items()})
    model = GroundingDINO(ontology=ontology,
                          box_threshold=args.box_threshold,
                          text_threshold=args.text_threshold)

    prompts = list(ONTOLOGY.keys())
    id_for_prompt = list(ONTOLOGY.values())

    images = sorted(Path(args.images).glob("*.jpg"))
    if args.limit:
        images = images[:: max(1, len(images) // args.limit)][: args.limit]
    print(f"Labeling {len(images)} images for: {list(CLASS_NAMES.values())}")

    per_class = {c: 0 for c in CLASS_NAMES}
    for i, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        detections = model.predict(str(img_path))
        lines = []
        vis = img.copy()
        for xyxy, _, conf, cls_idx, _, _ in zip(
                detections.xyxy, detections.mask if detections.mask is not None else [None]*len(detections),
                detections.confidence, detections.class_id,
                detections.tracker_id if detections.tracker_id is not None else [None]*len(detections),
                detections.data.get("class_name", [None]*len(detections)) if detections.data else [None]*len(detections)):
            cls_id = id_for_prompt[cls_idx]
            x1, y1, x2, y2 = xyxy
            cx, cy = (x1+x2)/2/W, (y1+y2)/2/H
            bw, bh = (x2-x1)/W, (y2-y1)/H
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            per_class[cls_id] += 1
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 60), 2)
            cv2.putText(vis, f"{CLASS_NAMES[cls_id]} {conf:.2f}",
                        (int(x1), max(0, int(y1)-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 60), 1)

        (lbl_dir / f"{img_path.stem}.txt").write_text("\n".join(lines))
        cv2.imwrite(str(vis_dir / img_path.name), vis)
        if (i+1) % 20 == 0:
            print(f"  {i+1}/{len(images)}")

    print("\nBoxes found:")
    for cid, name in CLASS_NAMES.items():
        print(f"  {name:12s} {per_class[cid]}")
    print(f"\nlabels -> {lbl_dir}\nvis    -> {vis_dir}")


if __name__ == "__main__":
    main()
