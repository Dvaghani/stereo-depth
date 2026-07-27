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
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
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
                  timeout: int = 180, retries: int = 4):
    """The hosted endpoint drops requests intermittently — Cloudflare 524s,
    connection resets, occasional SSL record errors. All are transient, so retry
    with exponential backoff rather than losing the image."""
    b64 = base64.b64encode(img_path.read_bytes()).decode()
    body = json.dumps({
        "api_key": API_KEY,
        "inputs": {"image": {"type": "base64", "value": b64}, "classes": classes},
    }).encode()
    url = ENDPOINT.format(workspace=workspace, workflow=workflow)

    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except Exception as exc:
            last = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)      # 1s, 2s, 4s
    raise last


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
    p.add_argument("--workers", type=int, default=6,
                   help="parallel requests; the work is network-bound so this is "
                        "the main speed lever")
    p.add_argument("--sleep", type=float, default=0.0,
                   help="unused with --workers, kept for compatibility")
    p.add_argument("--preview", action="store_true")
    args = p.parse_args()

    if not API_KEY:
        raise SystemExit("set ROBOFLOW_API_KEY in the environment first")

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prev = out.parent / (out.name + "_preview")
    if args.preview:
        prev.mkdir(parents=True, exist_ok=True)

    files = [f for f in sorted(src.iterdir())
             if f.suffix.lower() in (".jpg", ".jpeg", ".png")][:args.limit]
    todo = [f for f in files if not (out / (f.stem + ".json")).exists()]
    print("labelling %d images with prompt %r (%d already done, %d workers)\n"
          % (len(todo), args.classes, len(files) - len(todo), args.workers))

    counters = {"shapes": 0, "empty": 0, "fail": 0, "done": 0}
    lock = threading.Lock()

    def process(f: Path):
        try:
            res = call_workflow(f, args.classes, args.workspace, args.workflow)
        except Exception as exc:
            with lock:
                counters["fail"] += 1
                counters["done"] += 1
                print("  [%4d/%d] %-42s FAILED: %s"
                      % (counters["done"], len(todo), f.name, str(exc)[:50]))
            return

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

        confs = [c for _, c, _ in shapes]
        with lock:
            counters["shapes"] += len(shapes)
            counters["empty"] += (len(shapes) == 0)
            counters["done"] += 1
            if counters["done"] % 10 == 0 or shapes:
                print("  [%4d/%d] %-42s %2d polys  conf %s"
                      % (counters["done"], len(todo), f.name, len(shapes),
                         ("%.2f-%.2f" % (min(confs), max(confs))) if confs else "-"))

    # network-bound, so threads overlap the waiting rather than the CPU work
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(process, todo))

    n = max(len(todo), 1)
    print("\n%d polygons over %d images (%.1f/image); %d empty, %d failed after retries"
          % (counters["shapes"], len(todo), counters["shapes"] / n,
             counters["empty"], counters["fail"]))
    print("correct in labelme:\n  labelme %s --labels %s" % (out, args.classes))


if __name__ == "__main__":
    main()
