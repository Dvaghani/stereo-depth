"""
Build a small, portable INT8 calibration set for the YOLO11s-seg detector.

Why this exists
---------------
INT8 quantisation runs at *engine build time*: TensorRT pushes a fixed set of
frames through the network, records the activation range at every layer, and
bakes those scales into the engine. It therefore needs a saved, reproducible
set of images — a live camera cannot be used, because the same input must
produce the same scales every time the engine is rebuilt.

Random noise produces poor scales, and so does an unrepresentative sample. The
set must cover the deployment distribution, *including the rare classes* — if
`cable` never appears during calibration, the activation ranges that matter for
detecting cables are never observed and INT8 will degrade that class hardest.
So this samples per-class rather than uniformly at random.

~300 frames is the usual sweet spot: TensorRT's entropy calibrator converges
well before that, and it stays small enough to copy to the Orin on a USB stick.

Output layout (self-contained, copy the whole folder to the target board):

    <out>/
        images/val/*.jpg|png
        labels/val/*.txt
        dataset.yaml        ← points at itself with relative paths

Then on the Orin:

    yolo export model=best.pt format=engine int8=True \\
        data=<out>/dataset.yaml imgsz=640

Usage:
    python scripts/build_int8_calib_set.py \\
        --dataset /run/media/dvaghani/Expansion/Yolo/unified_seg_dataset \\
        --out outputs/int8_calib_yolo --count 300
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, required=True,
                   help="root of the YOLO dataset (must contain dataset.yaml)")
    p.add_argument("--out", type=Path, required=True,
                   help="output directory for the calibration set")
    p.add_argument("--count", type=int, default=300,
                   help="target number of frames (default 300)")
    p.add_argument("--split", default="val",
                   help="split to sample from (default val). Use train only if "
                        "val is too small — calibrating on training frames is "
                        "acceptable since no gradients or metrics are involved.")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed, so the set is reproducible")
    p.add_argument("--boost-min", type=int, default=0,
                   help="ensure every class reaches at least this many frames, "
                        "topping up from --boost-from when the chosen split is "
                        "too thin. Rare classes (cable) are exactly the ones "
                        "INT8 degrades worst, so leaving them at a handful of "
                        "frames defeats the purpose. 0 disables.")
    p.add_argument("--boost-from", default="train",
                   help="split to draw top-up frames from (default train). "
                        "Using train frames is fine here: calibration computes "
                        "activation ranges, not gradients or metrics.")
    p.add_argument("--symlink", action="store_true",
                   help="symlink instead of copying (smaller, but NOT portable "
                        "to another machine — do not use if copying to the Orin)")
    return p.parse_args()


def label_for(img_path, images_root, labels_root):
    """Map images/<split>/foo.jpg -> labels/<split>/foo.txt"""
    rel = img_path.relative_to(images_root)
    return (labels_root / rel).with_suffix(".txt")


def classes_in(label_path):
    """Class ids present in a YOLO label file (first token per line)."""
    if not label_path.exists():
        return set()
    ids = set()
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                ids.add(int(float(line.split()[0])))
            except (ValueError, IndexError):
                continue
    return ids


def stratified_sample(by_class, all_images, count, rng):
    """Round-robin over classes rarest-first so every class is represented
    before any class gets a second slot. Falls back to filling any shortfall
    with random unpicked frames."""
    picked = []
    seen = set()
    # rarest class first — it has the fewest chances to be picked incidentally
    order = sorted(by_class, key=lambda c: len(by_class[c]))
    pools = {c: rng.sample(by_class[c], len(by_class[c])) for c in order}
    cursor = {c: 0 for c in order}

    while len(picked) < count:
        progressed = False
        for c in order:
            if len(picked) >= count:
                break
            pool, i = pools[c], cursor[c]
            while i < len(pool) and pool[i] in seen:
                i += 1
            cursor[c] = i
            if i < len(pool):
                img = pool[i]
                cursor[c] = i + 1
                picked.append(img)
                seen.add(img)
                progressed = True
        if not progressed:
            break  # every class pool exhausted

    if len(picked) < count:
        rest = [p for p in all_images if p not in seen]
        rng.shuffle(rest)
        picked.extend(rest[: count - len(picked)])

    return picked


def boost_rare_classes(picked, by_class, names, args, rng,
                       images_root, labels_root):
    """Top up any class below --boost-min using frames from --boost-from.

    Scans the donor split in random order and stops as soon as every deficit is
    filled, rather than indexing the whole split — the donor split can be tens
    of thousands of images on slow external storage, and a class present in a
    few percent of frames is found within a few hundred reads.
    """
    have = defaultdict(int)
    for img in picked:
        for cid in classes_in(label_for(img, images_root, labels_root)):
            have[cid] += 1

    deficit = {cid: args.boost_min - have.get(cid, 0)
               for cid in names if have.get(cid, 0) < args.boost_min}
    if not deficit:
        print("\nevery class already has >= %d frames, no top-up needed"
              % args.boost_min)
        return picked

    print("\ntopping up from '%s' split (target >= %d/class): %s"
          % (args.boost_from, args.boost_min,
             ", ".join("%s +%d" % (names[c], n) for c, n in sorted(deficit.items()))))

    donor_dir = images_root / args.boost_from
    if not donor_dir.is_dir():
        print("  WARNING: donor split %s not found, skipping top-up" % donor_dir)
        return picked

    donors = [p for p in donor_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    rng.shuffle(donors)

    already = set(p.name for p in picked)
    added, scanned = [], 0
    for img in donors:
        if not deficit:
            break
        scanned += 1
        if img.name in already:
            continue
        cids = classes_in(label_for(img, images_root, labels_root))
        # only take frames that actually help a class still in deficit
        useful = cids & set(deficit)
        if not useful:
            continue
        added.append(img)
        already.add(img.name)
        for cid in cids:
            if cid in deficit:
                deficit[cid] -= 1
                if deficit[cid] <= 0:
                    del deficit[cid]

    print("  scanned %d donor frames, added %d" % (scanned, len(added)))
    if deficit:
        print("  WARNING: still short after exhausting donor split: %s"
              % ", ".join("%s (needs %d more)" % (names[c], n)
                          for c, n in sorted(deficit.items())))
    return picked + added


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    meta = yaml.safe_load((args.dataset / "dataset.yaml").read_text())
    names = meta.get("names", {})
    if isinstance(names, list):
        names = dict(enumerate(names))

    images_root = args.dataset / "images"
    labels_root = args.dataset / "labels"
    split_dir = images_root / args.split
    if not split_dir.is_dir():
        raise SystemExit("no such split: %s" % split_dir)

    all_images = sorted(p for p in split_dir.rglob("*")
                        if p.suffix.lower() in IMAGE_EXTS)
    if not all_images:
        raise SystemExit("no images found under %s" % split_dir)
    print("scanning %d images in %s ..." % (len(all_images), split_dir))

    by_class = defaultdict(list)
    for img in all_images:
        for cid in classes_in(label_for(img, images_root, labels_root)):
            by_class[cid].append(img)

    if not by_class:
        raise SystemExit("no labels found — checked %s" % (labels_root / args.split))

    count = min(args.count, len(all_images))
    if count < args.count:
        print("NOTE: split only has %d images, using all of them" % count)
    picked = stratified_sample(by_class, all_images, count, rng)

    if args.boost_min:
        picked = boost_rare_classes(picked, by_class, names, args, rng,
                                    images_root, labels_root)

    out_img = args.out / "images" / "val"
    out_lbl = args.out / "labels" / "val"
    for d in (out_img, out_lbl):
        d.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    for img in picked:
        dst_img = out_img / img.name
        src_lbl = label_for(img, images_root, labels_root)
        dst_lbl = out_lbl / src_lbl.name
        if args.symlink:
            for src, dst in ((img, dst_img), (src_lbl, dst_lbl)):
                if src.exists():
                    if dst.is_symlink() or dst.exists():
                        dst.unlink()
                    dst.symlink_to(src.resolve())
        else:
            shutil.copy2(img, dst_img)
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)
        total_bytes += img.stat().st_size

    # relative paths so the folder works wherever it is copied
    (args.out / "dataset.yaml").write_text(yaml.safe_dump({
        "path": ".",
        "train": "images/val",   # unused by calibration; present so loaders don't complain
        "val": "images/val",
        "nc": meta.get("nc", len(names)),
        "names": names,
    }, sort_keys=False))

    # report coverage — the point of stratifying is that this has no zeros
    got = defaultdict(int)
    for img in picked:
        for cid in classes_in(label_for(img, images_root, labels_root)):
            got[cid] += 1

    print("\nwrote %d frames to %s (%.1f MB)"
          % (len(picked), args.out, total_bytes / 1e6))
    print("\nclass coverage in the calibration set:")
    missing = []
    for cid in sorted(names):
        n = got.get(cid, 0)
        flag = "  <-- ABSENT" if n == 0 else ""
        if n == 0:
            missing.append(names[cid])
        print("  %-16s %4d frames%s" % (names[cid], n, flag))

    if missing:
        print("\nWARNING: %s absent from the calibration set. INT8 will not "
              "observe the activation ranges those classes rely on and may "
              "degrade them disproportionately." % ", ".join(missing))
    else:
        print("\nall %d classes represented" % len(names))


if __name__ == "__main__":
    main()
