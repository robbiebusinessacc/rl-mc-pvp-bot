"""Train PerceptionCNN on streamed synthetic batches.

CLI:
    python3 -m pvpbot.perception.train --steps 2000 --device auto

Loss: per-slot weighted MSE over PERCEPTION_LAYOUT (aim errors weighted
highest). Eval reports per-slot MAE in real units (degrees for angles,
HP for hearts, px/tick for screen velocity); aim-error MAE is additionally
reported masked to frames where the enemy is actually visible, since
invisible frames carry aim labels of exactly 0 by convention.

Checkpoints: torch.save({"model": state_dict, "meta": {...}}, path).
Metrics: one JSON object per line (JSONL).
"""
import argparse
import json
import os
import time
from typing import Callable, Dict, Optional

import numpy as np
import torch

from pvpbot.models import PerceptionCNN
from pvpbot.spec import FRAME_SHAPE, PERCEPTION_DIM, PERCEPTION_LAYOUT
from pvpbot.perception.synth import (
    FRAME_H, FRAME_W, SynthConfig, generate_batch, generate_stacked_batch,
)

# ---------------------------------------------------------------------------
# Per-slot loss weights: aim errors dominate (they drive the policy's camera
# control), screen velocity is upweighted because its normalized magnitude is
# tiny, reserved slot is ignored.
# ---------------------------------------------------------------------------
SLOT_WEIGHTS: Dict[str, float] = {
    "aim_err_yaw": 6.0,
    "aim_err_pitch": 6.0,
    # raised 2.0 -> 6.0: live matches showed the deployed model predicted
    # a near-constant bbox_height on real frames (no distance signal at
    # all -- perceived dist read ~5-6 blocks whether the enemy was at 2
    # or 8), which broke the policy's spacing and in_reach entirely
    "bbox_height": 6.0,
    "visible": 1.0,
    "self_pitch": 1.5,
    "self_hp": 1.0,
    "hurt_flash": 1.0,
    "rel_screen_vx": 3.0,
    "rel_screen_vy": 3.0,
    "self_speed": 2.0,
    "enemy_on_ground": 1.0,
    "reserved": 0.0,
}


def loss_weight_vector() -> np.ndarray:
    w = np.zeros(PERCEPTION_DIM, dtype=np.float32)
    for name, idx in PERCEPTION_LAYOUT.items():
        w[idx] = SLOT_WEIGHTS[name]
    return w


def select_device(name: str = "auto") -> torch.device:
    """auto: prefer MPS (Apple GPU) then CPU. Never requires CUDA."""
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def frames_to_tensor(frames_u8: np.ndarray, device: torch.device) -> torch.Tensor:
    """uint8 (N,1,H,W) -> float32 in [0,1] on device. The one true normalization."""
    t = torch.from_numpy(np.ascontiguousarray(frames_u8))
    return t.to(device=device, dtype=torch.float32).div_(255.0)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------
def save_checkpoint(model: torch.nn.Module, path: str, step: int,
                    extra_meta: Optional[dict] = None) -> None:
    meta = {
        "kind": "perception",
        "perception_dim": PERCEPTION_DIM,
        "frame_shape": list(FRAME_SHAPE),
        "step": int(step),
        "normalize": "uint8/255",
        "loss_weights": {k: float(v) for k, v in SLOT_WEIGHTS.items()},
    }
    if extra_meta:
        meta.update(extra_meta)
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    torch.save({"model": model.state_dict(), "meta": meta}, path)


def load_checkpoint(path: str, map_location: str = "cpu"):
    """Returns (model on map_location in eval mode, meta dict)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    in_ch = ckpt["model"]["net.0.weight"].shape[1]
    model = PerceptionCNN(stack=max(1, in_ch // 3))
    model.load_state_dict(ckpt["model"])
    model.to(map_location).eval()
    return model, ckpt.get("meta", {})


# ---------------------------------------------------------------------------
# Eval: per-slot MAE in real units.
# ---------------------------------------------------------------------------
# multiply normalized |err| by this to get real units
_SLOT_UNITS = {
    "aim_err_yaw": ("deg", 180.0),
    "aim_err_pitch": ("deg", 90.0),
    "bbox_height": ("frac", 1.0),
    "visible": ("p", 1.0),
    "self_pitch": ("deg", 90.0),
    "self_hp": ("hp", 20.0),
    "hurt_flash": ("p", 1.0),
    "rel_screen_vx": ("px", float(FRAME_W)),
    "rel_screen_vy": ("px", float(FRAME_H)),
    "self_speed": ("bpt", 1.0),
    "enemy_on_ground": ("p", 1.0),
    "reserved": ("", 1.0),
}


@torch.no_grad()
def evaluate(model: torch.nn.Module, device: torch.device, n: int = 512,
             seed: int = 12345, cfg: Optional[SynthConfig] = None,
             batch: int = 128) -> Dict[str, float]:
    """Per-slot MAE on a fresh deterministic synthetic eval set.

    Keys look like "mae/aim_err_yaw_deg". Aim errors and bbox also get
    "_vis"-suffixed variants restricted to visible-enemy frames (the metric
    that matters for aiming).
    """
    model.eval()
    frames, labels = generate_batch(n, seed=seed, cfg=cfg)
    preds = np.empty_like(labels)
    for i in range(0, n, batch):
        x = frames_to_tensor(frames[i:i + batch], device)
        preds[i:i + batch] = model(x).cpu().numpy()
    err = np.abs(preds - labels)

    out: Dict[str, float] = {}
    for name, idx in PERCEPTION_LAYOUT.items():
        unit, scale = _SLOT_UNITS[name]
        key = "mae/{}_{}".format(name, unit) if unit else "mae/{}".format(name)
        out[key] = float(err[:, idx].mean() * scale)
    vis = labels[:, PERCEPTION_LAYOUT["visible"]] > 0.5
    if vis.any():
        for name in ("aim_err_yaw", "aim_err_pitch", "bbox_height"):
            unit, scale = _SLOT_UNITS[name]
            key = "mae/{}_{}_vis".format(name, unit)
            out[key] = float(err[vis, PERCEPTION_LAYOUT[name]].mean() * scale)
    out["eval_n"] = float(n)
    return out


# ---------------------------------------------------------------------------
# Training loop.
# ---------------------------------------------------------------------------
def train(
    steps: int,
    batch_size: int = 96,
    lr: float = 1e-3,
    device: str = "auto",
    seed: int = 0,
    out: Optional[str] = None,
    metrics_path: Optional[str] = None,
    eval_every: int = 200,
    eval_size: int = 384,
    cfg: Optional[SynthConfig] = None,
    model: Optional[torch.nn.Module] = None,
    log: Optional[Callable[[str], None]] = print,
    real_data: Optional[str] = None,
    real_frac: float = 0.5,
    init_ckpt: Optional[str] = None,
) -> Dict[str, float]:
    """Train PerceptionCNN on streamed synthetic batches. Returns final eval.

    With real_data (an NPZ from tools/realdata/build_dataset.py: frames u8,
    labels f32, mask f32) each step uses a real batch with probability
    real_frac; real loss is additionally masked per slot. The last 15% of
    real rows (by time) are held out and reported as mae_real/* metrics.
    """
    dev = select_device(device)
    torch.manual_seed(seed)
    if model is None:
        model = PerceptionCNN()
    if init_ckpt:
        model, meta0 = load_checkpoint(init_ckpt)
        if log:
            log("initialized from {} (step {})".format(
                init_ckpt, meta0.get("step", "?")))
    model.to(dev).train()

    real = None
    if real_data:
        z = np.load(real_data)
        rf = z["frames"]
        if rf.ndim == 3:  # legacy grayscale (N,H,W) -> replicate channels
            rf = np.repeat(rf[:, None], 3, axis=1)
        elif rf.ndim == 4 and rf.shape[-1] == 3:  # (N,H,W,3) -> (N,3,H,W)
            rf = rf.transpose(0, 3, 1, 2)
        elif rf.ndim == 5 and rf.shape[-1] == 3:
            # temporal windows (N,S,H,W,3) -> (N,S*3,H,W); the model must be
            # built with stack=S (v10 motion-aware perception)
            n, s = rf.shape[:2]
            rf = rf.transpose(0, 1, 4, 2, 3).reshape(n, s * 3, *rf.shape[2:4])
            if getattr(model, "stack", 1) != s:
                old = model
                model = PerceptionCNN(stack=s)
                if old is not None:
                    sd = old.state_dict()
                    w = sd["net.0.weight"]
                    if w.shape[1] * s == model.net[0].in_channels:
                        sd["net.0.weight"] = w.repeat(1, s, 1, 1) / float(s)
                    model.load_state_dict(sd)
                model.to(dev).train()
                if log:
                    log("stacked real data (S=%d): model widened, conv1 "
                        "warm-started tiled/%d" % (s, s))
        z = {"frames": rf, "labels": z["labels"], "mask": z["mask"]}
        n_tr = int(len(z["frames"]) * 0.85)
        real = {
            "frames": z["frames"][:n_tr], "labels": z["labels"][:n_tr],
            "mask": z["mask"][:n_tr],
            "val_frames": z["frames"][n_tr:], "val_labels": z["labels"][n_tr:],
            "val_mask": z["mask"][n_tr:],
        }
        if log:
            log("real data: {} train / {} val rows".format(
                n_tr, len(z["frames"]) - n_tr))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    w = torch.from_numpy(loss_weight_vector()).to(dev)
    w_norm = w / w.sum()
    rng = np.random.default_rng(seed)
    if cfg is None:
        cfg = SynthConfig()

    mf = None
    if metrics_path:
        d = os.path.dirname(os.path.abspath(metrics_path))
        if d:
            os.makedirs(d, exist_ok=True)
        mf = open(metrics_path, "a")

    def emit(record: dict) -> None:
        if mf:
            mf.write(json.dumps(record) + "\n")
            mf.flush()

    t0 = time.time()
    try:
        for step in range(1, steps + 1):
            use_real = real is not None and rng.random() < real_frac
            if use_real:
                idx = rng.integers(0, len(real["frames"]), batch_size)
                x = frames_to_tensor(real["frames"][idx], dev)
                y = torch.from_numpy(real["labels"][idx]).to(dev)
                m = torch.from_numpy(real["mask"][idx]).to(dev)
                pred = model(x)
                loss = (((pred - y) ** 2) * w_norm * m).sum(dim=1).mean()
            else:
                if getattr(model, "stack", 1) > 1:
                    # motion-consistent synth windows: infinite true-motion
                    # supervision (real rounds alone memorize context)
                    frames, labels = generate_stacked_batch(
                        batch_size, model.stack, cfg=cfg, rng=rng)
                else:
                    frames, labels = generate_batch(batch_size, cfg=cfg, rng=rng)
                x = frames_to_tensor(frames, dev)
                y = torch.from_numpy(labels).to(dev)
                pred = model(x)
                loss = (((pred - y) ** 2) * w_norm).sum(dim=1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step % eval_every == 0 or step == steps:
                metrics = evaluate(model, dev, n=eval_size,
                                   seed=10_000 + seed, cfg=cfg)
                model.train()
                rec = {
                    "step": step,
                    "loss": float(loss.item()),
                    "elapsed_s": round(time.time() - t0, 2),
                }
                rec.update({k: round(v, 5) for k, v in metrics.items()})
                emit(rec)
                if log:
                    log(
                        "step {:5d}  loss {:.5f}  yaw_vis {:.2f} deg  "
                        "pitch_vis {:.2f} deg  hp {:.2f}  [{:.0f}s]".format(
                            step, loss.item(),
                            metrics.get("mae/aim_err_yaw_deg_vis", float("nan")),
                            metrics.get("mae/aim_err_pitch_deg_vis", float("nan")),
                            metrics.get("mae/self_hp_hp", float("nan")),
                            time.time() - t0,
                        )
                    )
    finally:
        if mf:
            mf.close()

    final = evaluate(model, dev, n=max(eval_size, 512),
                     seed=20_000 + seed, cfg=cfg)
    if real is not None and len(real["val_frames"]):
        model.eval()
        with torch.no_grad():
            preds = []
            vf = real["val_frames"]
            for i in range(0, len(vf), 256):
                preds.append(model(frames_to_tensor(vf[i:i + 256], dev)).cpu().numpy())
        preds = np.concatenate(preds)
        err = np.abs(preds - real["val_labels"]) * real["val_mask"]
        vis = real["val_labels"][:, PERCEPTION_LAYOUT["visible"]] > 0.5
        for name, idx in PERCEPTION_LAYOUT.items():
            unit, scale = _SLOT_UNITS[name]
            sel = vis if name in ("aim_err_yaw", "aim_err_pitch",
                                  "bbox_height", "rel_screen_vx",
                                  "rel_screen_vy") else np.ones(len(err), bool)
            if sel.any():
                final["mae_real/%s_%s" % (name, unit)] = float(
                    err[sel, idx].mean() * scale)
        model.train()
    if out:
        save_checkpoint(model, out, step=steps, extra_meta={
            "seed": seed, "lr": lr, "batch_size": batch_size,
            "real_data": real_data, "real_frac": real_frac if real_data else 0,
            "final_eval": {k: round(v, 5) for k, v in final.items()},
        })
        if log:
            log("saved checkpoint -> {}".format(out))
    return final


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Train PerceptionCNN on synthetic frames")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto",
                    help="auto|cpu|mps (auto prefers mps)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/perception/perception.pt")
    ap.add_argument("--metrics", default="runs/perception/metrics.jsonl")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-size", type=int, default=384)
    ap.add_argument("--real-data", default=None,
                    help="NPZ from tools/realdata/build_dataset.py")
    ap.add_argument("--real-frac", type=float, default=0.5)
    ap.add_argument("--init", default=None,
                    help="checkpoint to fine-tune from")
    args = ap.parse_args(argv)

    dev = select_device(args.device)
    print("device: {}".format(dev))
    final = train(
        steps=args.steps, batch_size=args.batch_size, lr=args.lr,
        device=args.device, seed=args.seed, out=args.out,
        metrics_path=args.metrics, eval_every=args.eval_every,
        eval_size=args.eval_size, real_data=args.real_data,
        real_frac=args.real_frac, init_ckpt=args.init,
    )
    print("final eval:")
    for k in sorted(final):
        print("  {:32s} {:.4f}".format(k, final[k]))


if __name__ == "__main__":
    main()
