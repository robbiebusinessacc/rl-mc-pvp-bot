"""FrameEncoder: checkpoint wrapper for single-frame perception inference.

    enc = FrameEncoder("runs/perception/perception.pt")   # cpu by default
    p = enc.encode(frame_u8_gray)                          # (PERCEPTION_DIM,)

Handles uint8->float normalization (must match train.frames_to_tensor:
uint8/255). Optional torch.jit.script export for speed; falls back to eager
gracefully. `benchmark()` reports single-frame CPU latency in ms — the
deployment budget is <5 ms/frame.

CLI:
    python3 -m pvpbot.perception.infer --ckpt runs/perception/perception.pt --bench
"""
import argparse
import time
from typing import Optional

import numpy as np
import torch

from pvpbot.models import PerceptionCNN
from pvpbot.spec import FRAME_SHAPE, PERCEPTION_DIM
from pvpbot.perception.synth import generate_batch
from pvpbot.perception.train import load_checkpoint, select_device


class FrameEncoder:
    """Loads a perception checkpoint and encodes grayscale frames."""

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        model: Optional[torch.nn.Module] = None,
        device: str = "cpu",
        jit: bool = False,
    ):
        self.device = select_device(device)
        self.meta: dict = {}
        if model is not None:
            self.model = model
        elif ckpt_path is not None:
            self.model, self.meta = load_checkpoint(ckpt_path, map_location="cpu")
        else:
            # random-init encoder: useful for wiring/latency tests
            self.model = PerceptionCNN()
        self.model.to(self.device).eval()
        self.jitted = False
        if jit:
            try:
                self.model = torch.jit.script(self.model)
                # burn in / optimize
                self.model(torch.zeros(1, *FRAME_SHAPE, device=self.device))
                self.jitted = True
            except Exception:
                # optional fast path only; eager model still loaded and valid
                self.jitted = False

    def _to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        arr = np.asarray(frame)
        if arr.ndim == 2:              # grayscale -> replicate channels
            arr = np.repeat(arr[None], FRAME_SHAPE[0], axis=0)
        if arr.ndim == 4 and arr.shape[-1] == FRAME_SHAPE[0]:
            # (S, H, W, C) temporal stack -> (S*C, H, W)
            arr = np.concatenate([np.moveaxis(f, -1, 0) for f in arr], axis=0)
        if arr.ndim == 3 and arr.shape[-1] == FRAME_SHAPE[0]:
            arr = np.moveaxis(arr, -1, 0)   # (H, W, C) -> (C, H, W)
        if arr.ndim == 3:
            arr = arr[None]            # (1, C, H, W)
        want_c = self.model.net[0].in_channels if hasattr(self.model, "net") else FRAME_SHAPE[0]
        if arr.ndim == 3 and arr.shape[0] != 1:
            arr = arr[None]
        if arr.shape[1] == FRAME_SHAPE[0] and want_c == 2 * FRAME_SHAPE[0]:
            # single frame into a stack-2 model: replicate (degenerate pair)
            arr = np.concatenate([arr, arr], axis=1)
        elif arr.shape[1] == 2 * FRAME_SHAPE[0] and want_c == FRAME_SHAPE[0]:
            # pair into a single-frame model: keep the CURRENT frame
            arr = arr[:, FRAME_SHAPE[0]:]
        if arr.shape[1] != want_c or arr.shape[2:] != FRAME_SHAPE[1:]:
            raise ValueError(
                "frame shape {} does not match model input {}x{}x{}".format(
                    arr.shape, want_c, FRAME_SHAPE[1], FRAME_SHAPE[2]))
        t = torch.from_numpy(np.ascontiguousarray(arr))
        t = t.to(device=self.device, dtype=torch.float32)
        if arr.dtype == np.uint8:
            t = t.div_(255.0)
        return t

    @torch.no_grad()
    def encode(self, frame_u8_gray: np.ndarray) -> np.ndarray:
        """(H, W) | (1, H, W) uint8 (or pre-normalized float) -> (PERCEPTION_DIM,) f32."""
        out = self.model(self._to_tensor(frame_u8_gray))
        return out.squeeze(0).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_batch(self, frames: np.ndarray) -> np.ndarray:
        """(N, 1, H, W) -> (N, PERCEPTION_DIM) f32."""
        arr = np.asarray(frames)
        t = torch.from_numpy(np.ascontiguousarray(arr)).to(
            device=self.device, dtype=torch.float32)
        if arr.dtype == np.uint8:
            t = t.div_(255.0)
        return self.model(t).cpu().numpy().astype(np.float32)

    def benchmark(self, n: int = 200, warmup: int = 20) -> float:
        """Mean single-frame encode latency in milliseconds."""
        frames, _ = generate_batch(max(n, 1) + warmup, seed=0)
        singles = [frames[i, 0] for i in range(frames.shape[0])]
        for i in range(warmup):
            self.encode(singles[i])
        t0 = time.perf_counter()
        for i in range(warmup, warmup + n):
            self.encode(singles[i])
        return (time.perf_counter() - t0) / n * 1000.0


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Perception inference benchmark")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint path (omit for random-init model)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--jit", action="store_true")
    ap.add_argument("--bench", type=int, nargs="?", const=200, default=200,
                    help="number of benchmark frames")
    args = ap.parse_args(argv)

    enc = FrameEncoder(ckpt_path=args.ckpt, device=args.device, jit=args.jit)
    print("device: {}  jitted: {}  meta step: {}".format(
        enc.device, enc.jitted, enc.meta.get("step", "n/a")))
    frames, _ = generate_batch(1, seed=0)
    p = enc.encode(frames[0, 0])
    assert p.shape == (PERCEPTION_DIM,)
    ms = enc.benchmark(n=args.bench)
    print("single-frame latency: {:.3f} ms  (budget < 5 ms)".format(ms))


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Module-level convenience for the deploy loop (pvpbot.deploy.loop probes for
# a callable named ``encode_frame``). Lazily builds one FrameEncoder from
# $PVPBOT_PERCEPTION_CKPT (default runs/perception/perception.pt); returns a
# zero estimate with a one-time warning if the checkpoint is missing.
# ---------------------------------------------------------------------------
_DEFAULT_ENCODER = None
_DIST_ENCODER = None
_WARNED_MISSING = False


def _dist_encoder():
    """Optional second checkpoint used ONLY for the bbox_height slot.

    $PVPBOT_DIST_CKPT names a model whose distance regression is better
    (combat-domain fine-tune) but whose aim calibration is worse -- the
    fine-tune shifted the pitch zero-point ~15 deg, observed live. Slot
    merge takes the best of both. Returns None when unset/missing."""
    global _DIST_ENCODER
    if _DIST_ENCODER is None:
        import os

        path = os.environ.get(
            "PVPBOT_DIST_CKPT", "runs/perception/perception_dist.pt")
        if not path or not os.path.exists(path):
            _DIST_ENCODER = False
        else:
            try:
                _DIST_ENCODER = FrameEncoder(path)
            except Exception:
                _DIST_ENCODER = False
    return _DIST_ENCODER or None


def encode_frame(frame_u8_gray: np.ndarray) -> np.ndarray:
    global _DEFAULT_ENCODER, _WARNED_MISSING
    if _DEFAULT_ENCODER is None:
        import os

        path = os.environ.get(
            "PVPBOT_PERCEPTION_CKPT", "runs/perception/perception.pt"
        )
        if not os.path.exists(path):
            if not _WARNED_MISSING:
                _WARNED_MISSING = True
                print(
                    "perception: no checkpoint at %r -- returning zeros "
                    "(train one: python3 -m pvpbot.perception.train)" % path
                )
            from pvpbot.spec import PERCEPTION_DIM

            return np.zeros(PERCEPTION_DIM, dtype=np.float32)
        try:
            _DEFAULT_ENCODER = FrameEncoder(path)
        except Exception as exc:  # stale architecture, corrupt file, ...
            if not _WARNED_MISSING:
                _WARNED_MISSING = True
                print(
                    "perception: checkpoint %r unusable (%s) -- returning "
                    "zeros; retrain with the current FRAME_SHAPE" % (path, exc)
                )
            from pvpbot.spec import PERCEPTION_DIM

            return np.zeros(PERCEPTION_DIM, dtype=np.float32)
    est = _DEFAULT_ENCODER.encode(frame_u8_gray)
    de = _dist_encoder()
    if de is not None:
        from pvpbot.spec import PERCEPTION_LAYOUT

        est = np.array(est, dtype=np.float32)
        alt = de.encode(frame_u8_gray)
        # bbox_height ONLY: the combat fine-tune halved distance error.
        # Never mix AIM slots across models -- each model's angular
        # estimates are internally coherent but disagree slightly about
        # where the enemy is; per-axis mixing made the aim assist
        # corkscrew (visible collapsed 57% -> 9% live).
        est[PERCEPTION_LAYOUT["bbox_height"]] = alt[
            PERCEPTION_LAYOUT["bbox_height"]]
    return est
