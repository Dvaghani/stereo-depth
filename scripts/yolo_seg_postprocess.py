"""Decode raw YOLO11-seg outputs into boxes + masks. Pure numpy/OpenCV, no
TensorRT or pycuda import — so this is importable and unit-testable on the
desktop, unlike the scripts that actually run inference on the Jetson (which
need `tensorrt`, only installed there). jetson_yolo_seg_infer.py and
jetson_end_to_end_bench.py both import this rather than duplicating the logic,
so a fix here fixes both.

Output layout (from the ONNX export): output0 is (4+nc+32, 8400) — 4 box
coords, nc class scores, 32 mask coefficients per anchor. output1 is
(32, 160, 160) — mask prototypes; a detection's mask is
sigmoid(coeffs @ prototypes).
"""
import numpy as np
import cv2

CLASS_NAMES = ["container", "pole", "scaffolding", "crane", "person", "machinery",
               "vehicle", "building", "suspended-load", "barrier", "cable"]


def decode(output0, output1, conf_thres=0.25, iou_thres=0.45, img_size=640,
          class_names=CLASS_NAMES):
    """output0: (4+nc+32, n_anchors), output1: (32, H, W) — both already
    squeezed from the batch dimension. Returns a list of dicts with
    box (xyxy, in img_size-space), conf, class_id, mask_coeff, mask (bool)."""
    nc = len(class_names)
    boxes_raw = output0[:4].T                  # (n_anchors, 4) cx,cy,w,h
    cls_scores = output0[4:4 + nc].T           # (n_anchors, nc)
    mask_coeffs = output0[4 + nc:].T           # (n_anchors, 32)

    class_ids = np.argmax(cls_scores, axis=1)
    confs = cls_scores[np.arange(len(class_ids)), class_ids]
    keep = confs > conf_thres
    if not keep.any():
        return []

    boxes_raw = boxes_raw[keep]
    confs = confs[keep]
    class_ids = class_ids[keep]
    mask_coeffs = mask_coeffs[keep]

    cx, cy, w, h = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]
    xyxy = np.stack([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], axis=1)

    detections = []
    for c in np.unique(class_ids):
        idx = np.where(class_ids == c)[0]
        rects = [[xyxy[i][0], xyxy[i][1], xyxy[i][2] - xyxy[i][0], xyxy[i][3] - xyxy[i][1]]
                 for i in idx]
        scores = [float(confs[i]) for i in idx]
        keep_idx = cv2.dnn.NMSBoxes(rects, scores, conf_thres, iou_thres)
        if len(keep_idx) == 0:
            continue
        for ki in np.array(keep_idx).flatten():
            i = idx[ki]
            detections.append({
                "box": xyxy[i],
                "conf": float(confs[i]),
                "class_id": int(c),
                "mask_coeff": mask_coeffs[i],
            })

    protos = output1.reshape(output1.shape[0], -1)
    for det in detections:
        m = 1.0 / (1.0 + np.exp(-(det["mask_coeff"] @ protos)))
        m = m.reshape(output1.shape[1], output1.shape[2])
        m = cv2.resize(m, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        det["mask"] = m > 0.5

    return detections
