# Baseline capture protocol

A fixed, repeatable capture session to be run **once per physical baseline**
(280 / 160 / 110 mm). Written for the 280 mm session; the distances, target and
scene list must be reproduced **identically** at every other baseline, or the
comparison is uncontrolled and the resulting table proves nothing.

## Why this document exists

Remounting the cameras invalidates the stereo calibration — unclamping and
re-clamping cannot reproduce `R` and `T` to sub-pixel accuracy. So data for a
given baseline can only be collected while that baseline is physically mounted
and freshly calibrated. There is no going back to add a missing station later
without redoing the whole calibration. Everything needed must be captured in
one session.

Archived data does *not* expire: `stereo_calib.npz` plus the raw pairs stay a
self-consistent dataset forever, and `capture_stereo.py` saves `left_raw.png` /
`right_raw.png` alongside the rectified images, so anything captured can be
re-rectified later. Only the ability to capture *new* data at that baseline is
lost.

---

## 0. Target preparation — do this before the session

**The existing 25 mm checkerboard on cardboard is the target. No new print is
needed.** This is worth stating explicitly because it is easy to assume
otherwise: the target here is *not* for calibration (that is already done and
verified), it is only a flat surface at a known distance. What the measurement
needs is the median disparity over a planar region, and that works on any
textured surface — detectable corners were only ever a convenience.

Corner auto-detection does stop working with distance. `findChessboardCorners`
needs roughly 12 px per square, and at the 280 mm rig's rectified focal length
(1232.6 px at 1920 wide) a 25 mm square subtends:

| distance | px per square | auto-detect |
|---|---:|---|
| 0.75 m | 41.1 | yes |
| 1.00 m | 30.8 | yes |
| 2.00 m | 15.4 | yes |
| 3.00 m | 10.3 | marginal |
| 4.00 m | 7.7 | no |
| 6.00 m | 5.1 | no |

So corner-based analysis is available out to ~2.6 m and a hand-placed region of
interest is used beyond it. Both yield the same quantity — median disparity over
a flat patch — so the error table is continuous across the whole range.

**One free improvement:** tape a newspaper page or magazine spread to the
cardboard. Plain brown cardboard is nearly uniform and matches poorly at range,
whereas newsprint is dense high-contrast texture that matches everywhere. The
existing ~600 mm backing is 123 px wide at 6 m, which is thousands of pixels to
take a median over — ample, provided they are matchable.

Keep the board **flat** on its rigid backing. Planarity is the assumption the
whole measurement rests on.

---

## 1. Pre-flight

Do not touch, bump, or re-clamp the rig at any point from here on.

```bash
cd "/home/dvaghani/PycharmProjects/Depth Map generation/stereo_unet"

# Confirm the calibration still describes the rig (expect < 1 px)
./.venv/bin/python scripts/check_calib.py \
    --calib outputs/calibration_280mm/stereo_calib.npz
```

- Under 1 px: proceed.
- 1–3 px: usable but degraded; prefer to recalibrate.
- Over 3 px: something moved. Recalibrate with the checkerboard — do **not**
  reach for `quick_recalib.py`, which only re-estimates rotation from natural
  features and is less accurate than a fresh checkerboard solve.

Focus must stay pinned at the calibration value (`focus.txt` = 0). Never press
`[` or `]` during capture: it changes the intrinsics and silently invalidates
rectification for everything captured afterwards.

---

## 2. Part A — quantitative accuracy stations

This produces the measured depth-error table. It is the only part that cannot be
approximated or reconstructed later.

**Stations:** 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0 m — eight in total.
Spacing is deliberately tighter at the near end, where the `Z^2` error curve
changes fastest and a linear spacing would under-sample it.

**Per station: 3 captures.** Depth noise is per-frame; three captures let the
error bar be a measurement rather than a guess.

**Setup at each station:**

1. Put the board's centre at the **same height as the cameras**. This is what
   makes a horizontal tape reading equal `Z` directly (see below).
2. Position the board **fronto-parallel** — its face square to the camera
   front, not angled. A tilted board puts its own surface at a range of true
   depths, which contaminates the measurement.
3. **Centre** it in the frame. Lens distortion residual is largest at the
   edges, and this measures depth accuracy, not distortion.
4. Tape-measure **horizontally**, parallel to the floor, from the **front face
   of the camera body** to the board face.
5. Confirm both halves of the rectified preview show the whole board before
   pressing SPACE.

### What "distance" means here

Stereo `Z` is not the straight-line distance to the object. It is the distance
**along the optical axis** — perpendicular from the image plane to the point.
This is precisely why the target must be flat and square to the camera: when it
is, every point on its face shares one `Z`, so "measure to the top, middle or
bottom?" has no answer to get wrong. Set it up correctly and the question
disappears.

Equal heights matter for the same reason. If the board sits higher or lower
than the cameras, the tape spans a hypotenuse and reads long — at 1 m with a
30 cm height difference that is a 4 % error, larger than the precision being
measured.

**Run one invocation per station** so the captures are self-labelling —
otherwise you end up with 24 timestamped folders and no record of which is
which:

```bash
./.venv/bin/python scripts/capture_stereo.py \
    --calib outputs/calibration_280mm/stereo_calib.npz \
    --out-root outputs/accuracy_280mm/z0.75m
# SPACE three times, then Q. Repeat with z1.0m, z1.5m, ... z6.0m
```

### On the tape measure

Stereo depth is measured from the camera's optical centre, which sits a
centimetre or so *inside* the lens — not at the body's front face. This
introduces a constant offset, worth ~1–2 % at 0.75 m and negligible by 6 m.
Two ways to handle it, both fine as long as the choice is stated in the thesis:

- Report raw error, and note the systematic offset in the text.
- Fit `Z_stereo = Z_tape + c` across all stations and report both raw and
  offset-corrected error. The fitted `c` *is* the optical-centre offset, which
  is a defensible thing to solve for rather than guess.

Better still, for stations where corners are detectable, `solvePnP` on the
checkerboard gives a ground-truth distance measured from the **same origin** as
the stereo depth, removing the reference-point ambiguity completely. This
requires analysis code that does not exist yet — capture first, write it after.

---

## 3. Part B — qualitative scenes for thesis figures

Five to eight scenes, chosen for what they demonstrate rather than for looking
good. Capture 1–2 pairs each; no distance measurement needed.

| scene | what it demonstrates |
|---|---|
| Person at 2 m, 4 m, 6 m | the primary safety-relevant class, across the range |
| Cable / rope strung at 2–3 m | thin-structure preservation — the known weak point (a cable is a few px wide, and section 3 of the deployment doc shows low resolution destroys thin structures first) |
| Cluttered scene, objects at many depths | depth discontinuities and edge quality |
| Large machinery or a vehicle, if accessible | the class the detector is actually for |

---

## 4. Part C — failure modes

Honest failure documentation is worth more in a thesis than another clean
result, and reviewers ask for it. Capture 1 pair each.

| scene | expected failure |
|---|---|
| Blank wall / smooth floor | no texture to match — holes or garbage disparity |
| Railing, fence, brick, tiling | repetitive texture — classic false-match ambiguity |
| Backlit window | dynamic range exceeded, one view blows out |
| Glass or polished metal | specular, view-dependent appearance breaks the matching assumption |

---

## 5. Recording

Keep a plain-text log beside the captures. Without it the folders are just
timestamps, and reproducing the session at 110 mm becomes guesswork:

```
outputs/accuracy_280mm/session_notes.txt

date, time, operator
calibration used + its check_calib residual
board: square size (measured, not nominal), inner corners, paper, backing
camera height above floor, rig position in room, target position
lighting (daylight / overhead / mixed), curtains open or closed
per station: nominal distance, tape reading, capture folder names
anything that went wrong
```

Camera height and room position matter more than they look: at 110 mm the
comparison is only valid if the scene geometry is reproduced, and "roughly the
same place" is not reproducible three weeks later.

---

## 6. After the session

```bash
# Archive the calibration alongside the data it belongs to
cp -r outputs/calibration_280mm outputs/calibration_280mm_ARCHIVE

# Sanity-check one capture from the near and far ends
./.venv/bin/python scripts/verify_rectification.py \
    --calib outputs/calibration_280mm/stereo_calib.npz \
    --left  outputs/accuracy_280mm/z1.0m/capture_280mm_*/left_raw.png \
    --right outputs/accuracy_280mm/z1.0m/capture_280mm_*/right_raw.png \
    --max-disp 400
```

Only after this is verified should the rig be unclamped.

Geometric predictions to compare the measured results against are produced by
`scripts/baseline_stats.py` — no capture required, since they follow from the
calibration alone.
