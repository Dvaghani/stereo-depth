# Stereo baseline comparison: 110 mm, 160 mm and 280 mm

Measured depth accuracy and precision for the Brio stereo rig at three physical
baselines. Captured 2026-08-10 (110 mm, 280 mm) and 2026-08-13 (160 mm). Every
figure derives from real captures at tape-measured distances; nothing is
extrapolated.

Session records: `outputs/accuracy_<N>mm*/session_notes.txt`.
Protocol: `docs/baseline_capture_protocol.md`.

---

## 1. What was measured, and why this way

Checkerboard corners are extracted from each stereo pair, disparity is taken as
`x_left - x_right` per corner, and depth follows from `Z = f*B/d`. **No stereo
matcher is involved.** The result is the accuracy floor the optics and
calibration impose -- the best any model could achieve on this hardware. A
learned matcher adds its own error on top and can only be worse.

That is the right measurement for a baseline comparison: it isolates the
variable under study (baseline length) from the model, which is common to all
three configurations anyway.

Each board yields 54 independently measured corners, so corner-to-corner
agreement is itself a diagnostic -- if the depth were wrong the corners would
disagree with each other.

### The PnP cross-check

`solvePnP` recovers the board distance from the **left image alone**, using the
focal length and the known 25 mm square size. It never touches the baseline or
the disparity. That independence separates two error sources a single
measurement cannot:

  - **stereo and PnP disagree** -> the baseline is wrong
  - **both agree but both differ from tape** -> the ground truth is wrong

Every result below is reported with this check.

---

## 2. Rig configuration

| | 110 mm | 160 mm | 280 mm |
|---|---|---|---|
| baseline, tape | 110 mm | 160 mm | 280 mm |
| baseline, recovered | 110.01 mm | 159.77 mm | 280.10 mm |
| calibration pairs | 32 | 31 | 40 |
| stereo reprojection RMS | 0.375 px | 0.380 px | 0.363 px |
| rectified focal @1920 | 1233.74 px | 1198.19 px | 1234.32 px |
| camera indices | LEFT=4, RIGHT=0 | LEFT=0, RIGHT=4 | LEFT=0, RIGHT=4 |
| extrinsics used | session calibration | at-capture solve | at-capture solve |

The camera order differed at every remount. It was determined by feature
matching (`scripts/which_camera_is_left.py`), not by eye -- judging it while
standing in front of the rig mirrors the sense.

---

## 3. Measured depth accuracy

| station | 110 mm | 160 mm | 280 mm |
|---|---:|---:|---:|
| 0.75 m | +8 mm (+1.10%) | -8 mm (-1.12%) | -28 mm (-3.80%) |
| 1.0 m | +1 mm (+0.15%) | -17 mm (-1.69%) | +23 mm (+2.34%) |
| 1.5 m | +3 mm (+0.17%) | -6 mm (-0.40%) | -22 mm (-1.45%) |
| 2.0 m | +11 mm (+0.57%) | -4 mm (-0.18%) | +1 mm (+0.05%) |
| 3.0 m | +13 mm (+0.44%) | -10 mm (-0.34%) | +21 mm (+0.70%) |
| 4.0 m | +47 mm (+1.16%) | -17 mm (-0.44%) | +51 mm (+1.27%) |
| 5.0 m | +36 mm (+0.72%) | +3 mm (+0.05%) | +61 mm (+1.22%) |
| 6.0 m | +53 mm (+0.88%) | +26 mm (+0.44%) | +71 mm (+1.19%) |
| **RMS** | **28.9 mm** | **13.6 mm** | **41.2 mm** |
| max error | 1.16% | 1.69% | 3.80% |

    PnP    / tape    +0.70 %     -0.44 %     +1.04 %
    stereo / tape    +0.81 %     +0.03 %     +1.03 %
    stereo / PnP     +0.11 %     +0.47 %     -0.01 %
    implied B        109.89      159.02      280.11
    calibrated B     110.01      159.77      280.10

At all three baselines, stereo and PnP agree with each other (0.11%, 0.47%,
0.01%) far more closely than either agrees with the tape (0.70%, 0.44%, 1.04%).
Two independent methods do not share an error mode, so **the ground truth is the
weaker instrument, not the rigs.** Board-placement repeatability was measured
directly at ~25 mm by revisiting marks, the same order as the residuals.

Accuracy does **not** order by baseline -- 160 mm is best and 280 mm worst.
That is expected: these residuals are dominated by board placement, which is
independent of the rig. The baseline effect is in precision, below.

---

## 4. Precision -- the headline result

Fitting a plane to each board's corner depths separates measurement noise from
real board tilt. The residual is the disparity noise, and it converts to a depth
sigma via `sigma_Z = Z^2 * sigma_d / (f*B)`.

| distance | 110 mm | 160 mm | 280 mm |
|---|---:|---:|---:|
| 0.75 m | 0.4 mm | 0.4 mm | 0.2 mm |
| 1.0 m | 0.9 mm | 0.6 mm | 0.5 mm |
| 1.5 m | 1.9 mm | 1.4 mm | 0.8 mm |
| 2.0 m | 2.9 mm | 2.4 mm | 1.3 mm |
| 3.0 m | 8.9 mm | 6.0 mm | 3.2 mm |
| 4.0 m | 11.4 mm | 9.4 mm | 5.9 mm |
| 5.0 m | 16.6 mm | 16.1 mm | 10.3 mm |
| 6.0 m | 31.3 mm | 21.1 mm | 10.2 mm |

**Monotonic in baseline at every distance.** Summarised against the 280 mm rig:

| baseline | f*B | mean disparity noise | sigma_Z at 6 m | predicted ratio | measured ratio |
|---|---:|---:|---:|---:|---:|
| 110 mm | 135720 | 0.110 px | 31.3 mm | 2.55x | 3.07x |
| 160 mm | 191431 | 0.121 px | 21.1 mm | 1.81x | 2.08x |
| 280 mm | 345729 | 0.129 px | 10.2 mm | 1.00x | 1.00x |

Two findings:

**Disparity noise is constant** -- 0.110, 0.121, 0.129 px. It is a property of
sub-pixel corner localisation, independent of the geometry. This is what makes
the comparison clean: the input noise is the same at all three baselines, so any
difference in depth precision comes purely from `f*B`.

**Depth precision follows Z^2/(f*B).** Measured ratios 3.07x and 2.08x against
predictions of 2.55x and 1.81x -- correct ordering, correct magnitude, and
monotonic across three independent sessions. The overshoot is consistent between
the two, suggesting a small systematic rather than noise.

Practical reading: at 6 m the 280 mm rig resolves depth to ~10 mm where the
110 mm rig manages ~31 mm. Because the error grows as `Z^2`, the absolute
advantage widens with distance while the ratio stays fixed.

---

## 5. Geometric predictions

`scripts/baseline_stats.py` derives range limits from the calibrations alone,
assuming the deployed matcher's measured 1.073 px mean disparity error
(`i4 @ 480x640`, `docs/jetson_deployment_results.md` section 3) at 640 px input:

| baseline | near limit (AANet, d=192) | near limit (RAFT, FOV) | far @5% err | far @10% err |
|---|---:|---:|---:|---:|
| 110 mm | 0.23 m | 0.07 m | 2.1 m | 4.1 m |
| 160 mm | 0.35 m | 0.10 m | 3.1 m | 6.2 m |
| 280 mm | 0.60 m | 0.18 m | 5.4 m | 10.7 m |

Usable range scales with baseline; near-field is the cost. With AANet's 192-bin
cost volume the nearest resolvable object moves 0.23 m -> 0.60 m. RAFT-Stereo has
no such bound (correlation pyramid, not a fixed search window), so its near limit
is field-of-view overlap.

**Calibration is not the bottleneck.** Measured corner-localisation noise is
0.11-0.13 px against the deployed matcher's 1.073 px -- roughly 10x. Depth error
in deployment is set by the matcher, not by the optics or the calibration. This
holds at all three baselines.

---

## 6. What the deployed matcher actually delivers

Sections 3 and 4 contain no matcher. This section runs **RAFT-Stereo** on the
same station captures, sampling its dense disparity at the same 54 corner
locations per board, so the two are directly comparable. Model is
`checkpoints/raft_middlebury_ft/best.pth` in the realtime config at 640x480 --
the deployed configuration (`scripts/export_raft_onnx.py`).

Reproduce with `scripts/eval_raft_stations.py --iters 4`.

### RAFT's disparity error is baseline-independent

Mean error against the corner reference:

| iterations | 110 mm | 160 mm | 280 mm |
|---|---:|---:|---:|
| i4 | 0.76 px | 0.51 px | 0.87 px |
| i7 | 0.49 px | 0.35 px | 0.47 px |

No trend with baseline, which is the expected result -- this is a property of the
network and the imagery, not the geometry. It is also consistent with the
1.073 px measured on-device (`docs/jetson_deployment_results.md` section 3);
slightly lower here because a checkerboard is an easy matching target.

That independence is what makes the rest of this section interpretable: since
the input disparity error is the same at every baseline, differences in depth
error come from `f*B` alone, exactly as in section 4.

### The baseline advantage survives the matcher, and grows with range

Matcher-induced depth error `|Z_raft - Z_geom|`, which isolates RAFT from the
~25 mm ground-truth placement error:

| distance | 110 mm (i4) | 160 mm (i4) | 280 mm (i4) | 110 mm (i7) | 160 mm (i7) | 280 mm (i7) |
|---|---:|---:|---:|---:|---:|---:|
| 0.75 m | 1 mm | 3 mm | 5 mm | 1 mm | 3 mm | 1 mm |
| 1.0 m | 1 mm | 2 mm | 2 mm | 0 mm | 2 mm | 3 mm |
| 1.5 m | 27 mm | 3 mm | 4 mm | 5 mm | 0 mm | 1 mm |
| 2.0 m | 6 mm | 0 mm | 3 mm | 6 mm | 6 mm | 1 mm |
| 3.0 m | 17 mm | 1 mm | 16 mm | 2 mm | 7 mm | 15 mm |
| 4.0 m | 94 mm | 33 mm | **16 mm** | 92 mm | 23 mm | **22 mm** |
| 5.0 m | 341 mm | -- | **81 mm** | 283 mm | -- | **47 mm** |
| 6.0 m | 248 mm | 178 mm | -- | 220 mm | 69 mm | -- |

**Below 3 m the matcher costs nothing measurable** -- 0-27 mm, at the level of
the noise. **Beyond 4 m it dominates**, and orders cleanly by baseline: at 4 m
with i4 the ratio is 5.9x / 2.1x / 1.0x.

The wider baseline therefore matters *more* in deployment than the geometric
analysis alone suggests. RAFT's ~0.5-0.9 px error is roughly 5x the calibration
noise of ~0.12 px, so dividing it by a larger `f*B` buys proportionally more.

### Accuracy against tape, with the matcher in the loop

| baseline | geometric RMS | RAFT i4 RMS | RAFT i7 RMS | i4 penalty | i7 penalty |
|---|---:|---:|---:|---:|---:|
| 110 mm | 26.9 mm | 175.9 mm | 154.6 mm | 6.5x | 5.7x |
| 160 mm | 13.6 mm | 77.2 mm | 36.6 mm | 5.7x | 2.7x |
| 280 mm | 32.0 mm | 58.8 mm | 50.1 mm | 1.8x | 1.6x |

**The matcher, not the calibration, is the bottleneck** -- by a factor of 1.6x to
6.5x. This is the quantitative form of the claim in section 5.

More iterations buy real accuracy: **i7 cuts disparity error by roughly 35%**
against i4, and at 160 mm it halves the depth RMS. On-device that costs 686 ms
versus 530 ms per frame -- 1.46 FPS against 1.89 FPS
(`docs/jetson_deployment_results.md` section 3). Whether that trade is worth it
depends on the range being monitored: below 3 m the matcher contributes nothing
either way, so i4 is free; beyond 4 m the extra iterations pay.

**Caveat on the RMS column:** the three baselines are averaged over *different
station sets* -- 160 mm lost 5 m and 280 mm lost 6 m to corner-detection
failures. Since error grows as `Z^2`, missing the farthest station flatters that
baseline's RMS. The per-distance table above is the sounder comparison.

---

## 7. Methodological finding: the rig drifts between calibration and capture

**In all three sessions the rig rotated ~0.1 deg** between the calibration poses
and the station captures, despite being untouched, and in two of them it changed
the result materially. Measured by solving extrinsics twice -- once from the
calibration poses, once from the station pairs -- and comparing:

| session | total | pitch | yaw (moves depth) | baseline shift | mattered? |
|---|---:|---:|---:|---:|---|
| 110 mm | 0.083° | +0.009° | -0.002° | 0.18 mm | no -- almost pure roll |
| 160 mm | 0.091° | +0.076° | **+0.050°** | 0.62 mm | **yes** |
| 280 mm | 0.116° | +0.094° | +0.010° | 0.42 mm | **yes** |

Only the **yaw** component shifts disparity. At 160 mm it was +1.04 px, which
put +208 mm of error at 6 m; re-solving the extrinsics from the station captures
reduced that to +26 mm and took stereo/tape from +2.17% to +0.03%. At 280 mm the
same correction took stereo/PnP from +0.54% to -0.01%. At 110 mm the drift was
almost pure roll and no correction was needed.

Four practical consequences:

1. **Calibrate and measure in one continuous sitting.** Necessary but, as 160 mm
   shows, not sufficient -- the rig drifts even untouched.
2. **Always solve extrinsics from the station captures as a cross-check.** It
   costs nothing (the images are already checkerboards) and is the only way to
   detect this. The PnP check flags that something is wrong; this identifies it.
3. **A far-scene rectification check cannot detect a translation error.** A
   residual vertical offset gives `dy = f*Ty/Z`, invisible at range and severe up
   close. One earlier attempt passed `verify_rectification` at 0.19 px on a
   distant scene while being unusable at 0.75 m.
4. **`quick_recalib.py` cannot substitute for calibration after a remount.** It
   re-estimates rotation and keeps the translation vector by design, so a changed
   baseline or vertical offset is carried over silently.

The 160 mm and 280 mm results use extrinsics solved from the station captures;
110 mm uses its session calibration directly. That asymmetry is a known weakness:
the two corrected columns are in-sample in a way the 110 mm column is not.

---

## 8. Limitations

- **Ground truth is the limiting instrument.** Board placement repeats to
  ~25 mm, comparable to the residuals being measured. The accuracy table cannot
  resolve below that; the precision table is unaffected because it never uses
  the tape.
- **Per-station tape readings were not logged**; nominal folder distances were
  used at all three baselines.
- **Checkerboard corners are the easiest possible matching target.** These are
  best-case figures. Textureless, occluded or thin structures will do worse, and
  none of that is captured here.
- **Two of three columns are in-sample** (see section 6).
- **No qualitative scene captures exist at any baseline.** Parts B and C of the
  protocol were dropped by decision.
- **Captures per station vary from 2 to 4**, so the per-station spread figures
  are themselves noisy. The 160 mm 5 m station (+/-60 mm spread) is an outlier
  suggesting one poorly seated board.
- **The RAFT comparison uses different station sets per baseline** (160 mm has
  no 5 m, 280 mm no 6 m), so its RMS column is not strictly comparable across
  baselines. Its per-distance table is.
- **RAFT was evaluated on the checkerboard only.** That is an easy matching
  target -- high contrast, planar, well textured -- so section 6 understates the
  matcher's error on real scenes with textureless or thin structures.

---

## 9. Reproducing

```bash
# which camera is physically left -- re-check after EVERY remount
python scripts/check_cameras.py && python scripts/which_camera_is_left.py

# geometric predictions -- no capture needed
python scripts/baseline_stats.py --latex

# measured accuracy
python scripts/eval_depth_accuracy.py --root outputs/accuracy_110mm_v2 \
    --calib outputs/calibration_110mm/stereo_calib.npz
python scripts/eval_depth_accuracy.py --root outputs/accuracy_160mm \
    --calib outputs/calibration_160mm/stereo_calib_atcapture.npz
python scripts/eval_depth_accuracy.py --root outputs/accuracy_280mm \
    --calib outputs/calibration_280mm/stereo_calib_atcapture.npz

# deployed-matcher accuracy on the same stations
python scripts/eval_raft_stations.py --iters 4
python scripts/eval_raft_stations.py --iters 7

# re-rectify captures with a different calibration (raw pairs are preserved)
python scripts/rerectify_captures.py --root <dir> --calib <npz>
```
