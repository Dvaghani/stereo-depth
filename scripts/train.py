"""
Train StereoUNet or AANetWrapper on KITTI 2015 (default) or Middlebury 2014.

Examples:
    # U-Net (original)
    python scripts/train.py --config configs/kitti.yaml

    # AANet — fine-tune backbone from pretrained KITTI weights
    python scripts/train.py --config configs/kitti_aanet.yaml

    # AANet — train only the uncertainty head (backbone frozen)
    python scripts/train.py --config configs/kitti_aanet_uncertainty.yaml
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Make `src` importable when running from the project root.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from src.models import StereoUNet
from src.models.stereo_unet import StereoUNetConfig
from src.models.aanet import AANetWrapper
from src.datasets import KITTI2015Stereo, Middlebury2014Stereo, SceneFlowStereo, StereoTransform
from src.utils.losses import multi_scale_loss
from src.utils.metrics import compute_kitti_metrics


def build_dataset(name: str, root: str, training: bool, crop, middlebury_variant: str = "imperfect",
                  downsample: int = 2, color_jitter: float = 0.4, max_disp: int = 192,
                  sceneflow_split: str = "train"):
    transform = StereoTransform(
        crop_size=tuple(crop) if training else None,
        color_jitter=color_jitter if training else 0.0,
        training=training,
    )
    name = name.lower()
    if name == "kitti":
        return KITTI2015Stereo(root, split="training", transform=transform)
    if name == "middlebury":
        return Middlebury2014Stereo(root, transform=transform, downsample=downsample,
                                    variant=middlebury_variant)
    if name == "sceneflow":
        return SceneFlowStereo(root, split=sceneflow_split, transform=transform,
                               max_disp=float(max_disp))
    raise ValueError(f"Unknown dataset: {name}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None,
                   help="Optional YAML config (overrides CLI args).")
    p.add_argument("--model", choices=["unet", "aanet"], default="unet",
                   help="Model architecture. 'aanet' uses AANetWrapper; "
                        "'unet' uses the original StereoUNet.")
    p.add_argument("--dataset", choices=["kitti", "middlebury", "sceneflow"], default="kitti")
    p.add_argument("--data-root", type=str, required=False)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-disp", type=int, default=192)
    p.add_argument("--crop", type=int, nargs=2, default=[256, 512])
    p.add_argument("--amp", action="store_true", help="Enable mixed-precision training.")
    p.add_argument("--ckpt-dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume full training state (model + optim + sched + epoch).")
    p.add_argument("--init-from", type=str, default=None,
                   help="Initialize model weights from a checkpoint, but start "
                        "training fresh (epoch 0, new optim/sched). Used for "
                        "fine-tuning: new modules in the model that aren't in "
                        "the checkpoint will be left at random init.")
    p.add_argument("--predict-uncertainty", action="store_true",
                   help="Add the Laplace-uncertainty head and switch the loss "
                        "to Laplace NLL. Use with --init-from to fine-tune an "
                        "existing disparity backbone.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--middlebury-variant", choices=["imperfect", "perfect", "both"],
                   default="imperfect",
                   help="Which Middlebury calibration variant to use. Default "
                        "'imperfect' matches the practical-calibration setup of "
                        "real cameras (e.g. Logitech Brio).")
    p.add_argument("--downsample", type=int, default=2,
                   help="Middlebury load-time downsample factor. 2 = half-res "
                        "(deployment); 4 = quarter-res (smaller disparities).")
    p.add_argument("--color-jitter", type=float, default=0.4,
                   help="Asymmetric color-jitter strength for training augmentation.")
    return p.parse_args()


def maybe_load_config(args):
    if not args.config:
        return args
    try:
        import yaml  # type: ignore
    except ImportError:
        print("PyYAML not installed; ignoring --config.", file=sys.stderr)
        return args
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for k, v in (cfg or {}).items():
        if hasattr(args, k):
            setattr(args, k, v)
    return args


def main():
    args = maybe_load_config(parse_args())
    if not args.data_root:
        raise SystemExit("--data-root is required (or set it in the YAML config).")
    device = torch.device(args.device)
    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    val_transform = StereoTransform(crop_size=tuple(args.crop), training=False)
    if args.dataset.lower() == "sceneflow":
        # SceneFlow has a pre-defined train/val split — use it directly.
        train_set = build_dataset(args.dataset, args.data_root, training=True,
                                  crop=args.crop, color_jitter=args.color_jitter,
                                  max_disp=args.max_disp, sceneflow_split="train")
        val_set   = build_dataset(args.dataset, args.data_root, training=False,
                                  crop=args.crop, color_jitter=0.0,
                                  max_disp=args.max_disp, sceneflow_split="val")
        val_set.transform = val_transform
    else:
        full = build_dataset(args.dataset, args.data_root, training=True, crop=args.crop,
                             middlebury_variant=args.middlebury_variant,
                             downsample=args.downsample, color_jitter=args.color_jitter)
        n_val = max(1, int(len(full) * args.val_split))
        n_train = len(full) - n_val
        train_set, val_set = torch.utils.data.random_split(
            full, [n_train, n_val], generator=torch.Generator().manual_seed(42)
        )
        val_set.dataset.transform = val_transform

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model_type = getattr(args, "model", "unet")   # YAML may set this as "model: aanet"

    if model_type == "aanet":
        model = AANetWrapper(
            max_disp=args.max_disp,
            predict_uncertainty=args.predict_uncertainty,
        ).to(device)
        cfg_dict = {"model": "aanet", "max_disp": args.max_disp,
                    "predict_uncertainty": args.predict_uncertainty,
                    "downsample": args.downsample}
        print(f"Model: AANetWrapper  params={sum(p.numel() for p in model.parameters()):,}")
    else:
        cfg = StereoUNetConfig(
            max_disp=args.max_disp,
            use_cuda_extension=True,
            predict_uncertainty=args.predict_uncertainty,
        )
        model = StereoUNet(cfg).to(device)
        cfg_dict = cfg.__dict__
        print(f"Model: StereoUNet  params={sum(p.numel() for p in model.parameters()):,}")

    # ── Weight initialisation ─────────────────────────────────────────────────
    start_epoch = 0
    best_d1 = float("inf")

    if args.resume and Path(args.resume).exists():
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        start_epoch = state["epoch"] + 1
        best_d1 = state.get("best_d1", best_d1)
        print(f"Resumed from {args.resume} @ epoch {start_epoch}")

    elif args.init_from and Path(args.init_from).exists():
        raw = torch.load(args.init_from, map_location=device)
        sd  = raw["model"] if isinstance(raw, dict) and "model" in raw else raw

        if model_type == "aanet":
            # Pretrained AANet weights are flat (no "backbone." prefix).
            # Our wrapper stores AANet as self.backbone, so remap the keys.
            # Keys that already start with "backbone." (i.e. from a wrapper
            # checkpoint) are left untouched.
            if not any(k.startswith("backbone.") for k in sd):
                sd = {"backbone." + k: v for k, v in sd.items()}

        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"  init-from: {len(missing)} param(s) at random init "
                  f"(first 3: {missing[:3]})")
        if unexpected:
            print(f"  init-from: ignoring {len(unexpected)} unexpected key(s) "
                  f"(first 3: {unexpected[:3]})")
        print(f"Initialized weights from {args.init_from}; starting at epoch 0")

    # ── Backbone freeze (AANet uncertainty-head fine-tune only) ───────────────
    # When training only the uncertainty head we freeze everything except the
    # head itself so the pretrained disparity backbone doesn't drift.
    freeze_backbone = (
        model_type == "aanet"
        and args.predict_uncertainty
        and args.init_from is not None
    )
    if freeze_backbone:
        frozen = 0
        for name, p in model.named_parameters():
            if "uncertainty_head" not in name:
                p.requires_grad_(False)
                frozen += p.numel()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Backbone FROZEN — {frozen:,} params frozen, {trainable:,} trainable "
              f"(uncertainty head only)")

    # Use all available GPUs — must be AFTER weight loading and backbone freeze.
    # Skip DataParallel when backbone is frozen: only 9k params are trainable,
    # multi-GPU adds overhead and causes grad_fn issues with dict outputs.
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1 and not freeze_backbone:
        print(f"Using {n_gpus} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    if args.resume and Path(args.resume).exists():
        # Reload optim/sched state (model was loaded above)
        state = torch.load(args.resume, map_location=device)
        optim.load_state_dict(state["optim"])
        sched.load_state_dict(state["sched"])

    # ----------------------------- training -------------------------------
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for it, batch in enumerate(train_loader):
            L = batch["left"].to(device, non_blocking=True)
            R = batch["right"].to(device, non_blocking=True)
            D = batch["disparity"].to(device, non_blocking=True)
            V = batch["valid"].to(device, non_blocking=True)
            # Mask out GT pixels whose disparity exceeds the model's max_disp:
            # the model architecturally cannot predict them, so including them
            # in the loss only contributes unrecoverable gradient. This is
            # critical for Middlebury, where some scenes have GT disparities
            # up to ~400 px at downsample=2 while max_disp is typically 192-384.
            V = V * (D < float(args.max_disp)).float()

            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                out = model(L, R)
                loss = multi_scale_loss(
                    out["disparity"], out["disparity_low"], D, V,
                    log_b_full=out.get("log_b"),
                    log_b_low=out.get("log_b_low"),
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()

            running += loss.item()
            if (it + 1) % args.log_every == 0:
                print(f"  epoch {epoch:03d} it {it+1:04d}/{len(train_loader)} "
                      f"loss {running/(it+1):.4f}")
        sched.step()

        # ---------------------- validation --------------------------------
        torch.cuda.empty_cache()
        model.eval()
        agg = {"EPE": 0.0, "D1-all": 0.0, "bad-1": 0.0, "bad-2": 0.0, "bad-3": 0.0}
        n = 0
        with torch.no_grad():
            for batch in val_loader:
                L = batch["left"].to(device); R = batch["right"].to(device)
                D = batch["disparity"].to(device); V = batch["valid"].to(device)
                # Match the training mask: the model architecturally cannot
                # predict disparities >= max_disp, so don't grade it on them.
                V = V * (D < float(args.max_disp)).float()
                pred = model(L, R)["disparity"]
                m = compute_kitti_metrics(pred, D, V)
                for k in agg:
                    agg[k] += m[k]
                n += 1
        for k in agg:
            agg[k] /= max(n, 1)
        dt = time.time() - t0
        print(f"[epoch {epoch:03d}] train_loss={running/max(1,len(train_loader)):.4f} "
              f"EPE={agg['EPE']:.3f} D1-all={agg['D1-all']:.2f}% "
              f"bad-3={agg['bad-3']:.2f}% lr={sched.get_last_lr()[0]:.2e} ({dt:.1f}s)")

        # Save checkpoint — unwrap DataParallel so checkpoints are portable
        save_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        is_best = agg["D1-all"] < best_d1
        if is_best:
            best_d1 = agg["D1-all"]
        torch.save(
            {
                "epoch": epoch,
                "model": save_model.state_dict(),
                "optim": optim.state_dict(),
                "sched": sched.state_dict(),
                "best_d1": best_d1,
                "config": cfg_dict,
            },
            ckpt_dir / ("best.pt" if is_best else "last.pt"),
        )

    print(f"Training done. Best D1-all = {best_d1:.2f}%")


if __name__ == "__main__":
    main()
