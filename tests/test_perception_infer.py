"""FrameEncoder: checkpoint round-trip, normalization, jit fallback, latency."""
import os

import numpy as np
import torch

from pvpbot.models import PerceptionCNN
from pvpbot.spec import FRAME_SHAPE, PERCEPTION_DIM
from pvpbot.perception.infer import FrameEncoder
from pvpbot.perception.synth import generate_batch
from pvpbot.perception.train import save_checkpoint


def _ckpt(tmp_path) -> str:
    torch.manual_seed(1)
    path = os.path.join(str(tmp_path), "enc.pt")
    save_checkpoint(PerceptionCNN(), path, step=7)
    return path


def test_encode_roundtrip_matches_model(tmp_path):
    path = _ckpt(tmp_path)
    enc = FrameEncoder(path, device="cpu")
    frames, _ = generate_batch(4, seed=5)
    # synth is true RGB now: the (H, W) grayscale path is still accepted
    # (replicates channels) but is no longer numerically identical to the
    # color frame, so the roundtrip is checked on the full (C, H, W) input
    p = enc.encode(frames[0])
    assert p.shape == (PERCEPTION_DIM,) and p.dtype == np.float32
    assert np.isfinite(p).all()
    gray = enc.encode(frames[0, 0])              # (H, W) path stays valid
    assert gray.shape == (PERCEPTION_DIM,) and np.isfinite(gray).all()
    # matches a manual normalized forward pass
    with torch.no_grad():
        x = torch.from_numpy(frames[:1]).float() / 255.0
        ref = enc.model(x).squeeze(0).numpy()
    assert np.allclose(p, ref, atol=1e-5)
    # batch API agrees with single-frame API
    pb = enc.encode_batch(frames)
    assert pb.shape == (4, PERCEPTION_DIM)
    assert np.allclose(pb[0], p, atol=1e-5)


def test_encode_rejects_bad_shape(tmp_path):
    enc = FrameEncoder(_ckpt(tmp_path), device="cpu")
    try:
        enc.encode(np.zeros((32, 32), dtype=np.uint8))
        raise AssertionError("expected ValueError for wrong frame shape")
    except ValueError:
        pass


def test_jit_path_works_or_falls_back(tmp_path):
    path = _ckpt(tmp_path)
    eager = FrameEncoder(path, device="cpu", jit=False)
    jitted = FrameEncoder(path, device="cpu", jit=True)
    frames, _ = generate_batch(2, seed=6)
    a = eager.encode(frames[0, 0])
    b = jitted.encode(frames[0, 0])
    assert np.allclose(a, b, atol=1e-4)   # jit or graceful eager fallback


def test_single_frame_latency_sane(tmp_path):
    enc = FrameEncoder(_ckpt(tmp_path), device="cpu", jit=True)
    ms = enc.benchmark(n=50, warmup=10)
    # deploy budget is <5 ms; keep a loose CI bound so slow machines pass
    assert ms < 50.0, "single-frame CPU latency {:.2f} ms".format(ms)


def test_random_init_encoder_for_wiring():
    enc = FrameEncoder(device="cpu")            # no checkpoint: random init
    frame = np.zeros(FRAME_SHAPE[1:], dtype=np.uint8)
    assert enc.encode(frame).shape == (PERCEPTION_DIM,)
