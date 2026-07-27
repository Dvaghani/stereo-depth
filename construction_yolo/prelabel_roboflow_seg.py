"""Auto-label images with a promptable Roboflow segmentation workflow.

Replaces prelabel_yoloworld.py for the classes it handles well. The workflow is
open-vocabulary (Grounding-DINO-style detector feeding SAM), so it takes a text
prompt rather than a fixed class list and produces genuine masks instead of the
loose boxes YOLO-World gave us. On a DJI frame already carrying 7 crude
scaffolding pseudo-boxes it returned 2 clean masks at 0.78/0.81 confidence that
correctly excluded the building between the two scaffold towers.

Masks come back as COCO RLE, which is decoded and traced to polygons. Output is
labelme JSON so it opens for correction alongside the cable and TTPLA data.

This is a LABELLING tool, not a deployment path: it is a hosted API, costs
credits per call, and cannot run on the Jetson. The point is distilling it into
our own model.

Usage:
    python construction_yolo/prelabel_roboflow_seg.py \
        --src  /run/media/dvaghani/Expansion/Yolo/construction_dataset/images/train \
        --out  /run/media/dvaghani/Expansion/Yolo/prelabel_scaffolding \
        --classes scaffolding --limit 50 --preview
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as maskutil

ENDPOINT = ("https://serverless.roboflow.com/infer/workflows/"
            "{workspace}/{workflow}")
# key lives in the environment so it stays out of the repo
API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")

MIN_CONF = 0.30
MIN_AREA_FRAC = 0.0005      # drop specks
POLY_EPS = 0.002            # approxPolyDP tolerance, fraction of perimeter


def call_workflow(img_path: Path, classes: str, workspace: str, workflow: str,
                  timeout: int = 180):
    b64 = base64.b64encode(img_path.read_bytes()).decode()
    body = json.dumps({
        "api_key": API_KEY,
        "inputs": {"image": {"type": "base64", "value": b64}, "classes": classes},
    }).encode()
    req = urllib.request.Request(
        ENDPOINT.format(workspace=workspace, workflow=workflow),
        data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def extract_polygons(result, min_conf: float, min_area_frac: float):
    """-> list of (label, confidence, [[x,y],...]) in pixel coordinates."""
    out = result.get("outputs", result)
    if isinstance(out, list):
        out = out[0] if out else {}
    preds = out.get("predictions", {})
    if isinstance(preds, dict):
        preds = preds.get("predictions", [])

    shapes = []
    for d in preds:
        conf = float(d.get("confidence", 0.0))
        if conf < min_conf:
            continue
        rle = d.get("rle_mask")
        if not rle:
            continue
        rle = dict(rle)
        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode()
        m = maskutil.decode(rle).astype(np.uint8)
        if m.mean() < min_area_frac:
            continue
        # one detection can yield several disconnected regions (a scaffold seen
        # through a gap); keep each as its own polygon rather than merging
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area_frac * m.size:
                continue
            approx = cv2.approxPolyDP(cnt, POLY_EPS * cv2.arcLength(cnt, True), True)
            pts = [[float(p[0][0]), float(p[0][1])] for p in approx]
            if len(pts) >= 3:
                shapes.append((d.get("class", "object"), conf, pts))
    return shapes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--classes", required=True,
                   help="text prompt, e.g. 'scaffolding' or 'shipping container'")
    p.add_argument("--workspace", default="dhruvits-workspace")
    p.add_argument("--workflow", default="general-segmentation-api")
    p.add_argument("--limit", type=int, default=50,
                   help="stop after N images — start small, each call costs credits")
    p.add_argument("--min-conf", type=float, default=MIN_CONF)
    p.add_argument("--sleep", type=float, default=0.3, help="pause between calls")
    p.add_argument("--preview", action="store_true")
    args = p.parse_args()

    if not API_KEY:
        raise SystemExit("set ROBOFLOW_API_KEY in the environment first")

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prev = out.parent / (out.name + "_preview")
    if args.preview:
        prev.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in src.iterdir()
                   if f.suffix.lower() in (".jpg", ".jpeg", ".png"))[:args.limit]
    print("labelling %d images with prompt %r\n" % (len(files), args.classes))

    n_shapes = n_empty = n_fail = 0
    for i, f in enumerate(files, 1):
        # skip work already done, so the script is resumable
        if (out / (f.stem + ".json")).exists():
            continue
        try:
            res = call_workflow(f, args.classes, args.workspace, args.workflow)
        except Exception as exc:
            n_fail += 1
            print("  [%3d/%d] %-40s FAILED: %s" % (i, len(files), f.name, str(exc)[:60]))
            continue

        shapes = extract_polygons(res, args.min_conf, MIN_AREA_FRAC)
        img = cv2.imread(str(f))
        h, w = img.shape[:2]

        doc = {
            "version": "4.2.7", "flags": {},
            "shapes": [{"label": lab, "points": pts, "group_id": None,
                        "shape_type": "polygon", "flags": {}}
                       for lab, _, pts in shapes],
            "imagePath": f.name, "imageData": None,
            "imageHeight": h, "imageWidth": w,
        }
        (out / (f.stem + ".json")).write_text(json.dumps(doc, indent=2))
        if not (out / f.name).exists():
            cv2.imwrite(str(out / f.name), img)

        if args.preview and shapes:
            vis = img.copy()
            cv2.drawContours(vis, [np.array(p, np.int32) for _, _, p in shapes],
                             -1, (255, 0, 255), 3)
            cv2.imwrite(str(prev / f.name), vis)

        n_shapes += len(shapes)
        n_empty += (len(shapes) == 0)
        confs = [c for _, c, _ in shapes]
        print("  [%3d/%d] %-40s %2d polys  conf %s"
              % (i, len(files), f.name, len(shapes),
                 ("%.2f-%.2f" % (min(confs), max(confs))) if confs else "-"))
        time.sleep(args.sleep)

    print("\n%d polygons over %d images (%.1f/image); %d images empty, %d failed"
          % (n_shapes, len(files), n_shapes / max(len(files), 1), n_empty, n_fail))
    print("correct in labelme:\n  labelme %s --labels %s" % (out, args.classes))


if __name__ == "__main__":
    main()
