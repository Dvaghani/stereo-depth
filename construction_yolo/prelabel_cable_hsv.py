"""Pre-label orange construction cables by HSV colour, emit labelme polygons.

Hand-tracing cable polygons is slow. These cables are bright orange against sky
and foliage, which separates cleanly in HSV — so threshold for orange, thin the
mask to a centreline, and emit a polygon per connected span. The output is
labelme JSON, so it opens directly for correction alongside the TTPLA data,
which uses the same format.

This produces CANDIDATES, not labels. Expect to delete false positives (traffic
cones, rust, warning signs) and fix broken spans. Same pre-label-then-correct
workflow as prelabel_yoloworld.py.

Usage:
    python construction_yolo/prelabel_cable_hsv.py \
        --src /run/media/dvaghani/Expansion/capture_raw \
        --out /run/media/dvaghani/Expansion/Yolo/cable_prelabel \
        --preview
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# Orange in OpenCV HSV (hue 0-179). Calibrated against actual cable pixels in
# overcast dusk light, which read around H=9-22, S=35-90, V=50-130 — far less
# saturated than daylight orange. Demanding S>=90 rejected most real cable.
HSV_LOW = np.array([3, 35, 50])
HSV_HIGH = np.array([30, 255, 255])

MIN_AREA = 60           # px; a 3px-wide 60px span is only ~180px of area
MIN_LENGTH = 50         # px, a cable span is long; blobs are not
MAX_THICKNESS = 40      # px, rejects large orange objects (cones, signs)
MIN_ELONGATION = 3.0    # length/thickness


def cable_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)
    # Bridge gaps along a span. No MORPH_OPEN here: an opening with any kernel
    # wider than the cable erases it, which is precisely the failure mode —
    # rely on the shape filter to reject noise instead.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return mask


def contour_to_polygon(cnt, epsilon_frac=0.004):
    peri = cv2.arcLength(cnt, closed=True)
    approx = cv2.approxPolyDP(cnt, epsilon_frac * peri, closed=True)
    return [[float(p[0][0]), float(p[0][1])] for p in approx]


def is_cable_like(cnt):
    """Long and thin, not blobby. Rejects cones/signs/rust patches."""
    area = cv2.contourArea(cnt)
    if area < MIN_AREA:
        return False, "area"
    rect = cv2.minAreaRect(cnt)
    (w, h) = rect[1]
    if w == 0 or h == 0:
        return False, "degenerate"
    length, thickness = max(w, h), min(w, h)
    if length < MIN_LENGTH:
        return False, "short"
    if thickness > MAX_THICKNESS:
        return False, "thick"
    if length / thickness < MIN_ELONGATION:
        return False, "not elongated"
    return True, ""


def process(img_path: Path, out_dir: Path, preview: bool):
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        print("  skip (unreadable): %s" % img_path.name)
        return 0
    h, w = bgr.shape[:2]

    mask = cable_mask(bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shapes, rejected = [], 0
    for cnt in contours:
        ok, _ = is_cable_like(cnt)
        if not ok:
            rejected += 1
            continue
        pts = contour_to_polygon(cnt)
        if len(pts) < 3:
            continue
        shapes.append({
            "label": "cable",
            "points": pts,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {},
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    # labelme expects the image beside its json
    img_out = out_dir / img_path.name
    if not img_out.exists():
        cv2.imwrite(str(img_out), bgr)

    doc = {
        "version": "4.2.7",
        "flags": {},
        "shapes": shapes,
        "imagePath": img_path.name,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w,
    }
    (out_dir / (img_path.stem + ".json")).write_text(json.dumps(doc, indent=2))

    if preview:
        vis = bgr.copy()
        cv2.drawContours(vis, [np.array(s["points"], dtype=np.int32) for s in shapes],
                         -1, (0, 0, 255), 2)
        prev_dir = out_dir / "_preview"
        prev_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(prev_dir / img_path.name), vis)

    print("  %-46s %3d candidates (%d rejected)" % (img_path.name, len(shapes), rejected))
    return len(shapes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--preview", action="store_true",
                   help="write _preview/ images with candidates drawn on")
    args = p.parse_args()

    src = Path(args.src)
    files = sorted([f for f in src.iterdir()
                    if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if not files:
        raise SystemExit("no images under %s" % src)

    print("pre-labelling %d images -> %s\n" % (len(files), args.out))
    total = sum(process(f, Path(args.out), args.preview) for f in files)
    print("\n%d cable candidates across %d images (%.1f per image)"
          % (total, len(files), total / len(files)))
    print("\nNext: open in labelme to correct —")
    print("  labelme %s --labels cable" % args.out)
    print("Delete false positives, fix broken spans, then convert to YOLO-seg.")


if __name__ == "__main__":
    main()
