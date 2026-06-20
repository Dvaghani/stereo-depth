# Detect connected USB cameras and grab one frame from each.

from __future__ import annotations

import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    print("ERROR: OpenCV not installed. Run: pip install opencv-python")
    sys.exit(1)


OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "camera_check"
MAX_INDEX_TO_PROBE = 8
REQUEST_W = 1920
REQUEST_H = 1080
MIN_REAL_PIXELS = 640 * 360


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    real_cams = []
    sub_devices = []

    for i in range(MAX_INDEX_TO_PROBE):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUEST_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUEST_H)

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        frame = None
        for _ in range(3):
            ret, frame = cap.read()
            if ret and frame is not None:
                break

        if frame is None:
            print(f"  index {i}: opened but no frame (likely a V4L2 sub-device)")
            cap.release()
            continue

        h, w = frame.shape[:2]
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join(chr((fourcc >> (8 * j)) & 0xFF) for j in range(4))
        fps = cap.get(cv2.CAP_PROP_FPS)


        is_valid_fourcc = fourcc != 0
        is_real = is_valid_fourcc and (w * h) >= MIN_REAL_PIXELS
        kind = "REAL CAMERA" if is_real else "(sub-device, skipping)"

        if is_real:
            out_path = OUT_DIR / f"cam_{i}.jpg"
            cv2.imwrite(str(out_path), frame)
            real_cams.append(i)
        else:
            sub_devices.append(i)

        print(f"  index {i}: {w}x{h} @ {fps:.0f}fps, fourcc={fourcc_str}  {kind}")
        cap.release()

    print(f"\nReal cameras at indices: {real_cams}")
    print(f"V4L2 sub-devices (ignore): {sub_devices}")
    print(f"Snapshots: {OUT_DIR}")

    if len(real_cams) == 2:
        print(f"\n  LEFT  = index {real_cams[0]}  (cam_{real_cams[0]}.jpg)")
        print(f"  RIGHT = index {real_cams[1]}  (cam_{real_cams[1]}.jpg)")
        print("\nOpen the two snapshots and verify which is physically left vs right")
        print("of your rig. If they're swapped, just swap the indices in the")
        print("capture script later.")
        print("\nNext step: stereo calibration with a checkerboard.")
    elif len(real_cams) == 3:
        print("\n3 real cameras detected (likely laptop webcam + 2 Brios).")
        print("Open all three snapshots and identify which 2 are the Brios.")
    elif len(real_cams) < 2:
        print("\nWARNING: fewer than 2 real cameras found.")
        print("Check `v4l2-ctl --list-devices` and `lsusb` to debug.")
    else:
        print(f"\nMore real cameras than expected ({len(real_cams)}). Identify")
        print("the two Brios from the snapshots.")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
