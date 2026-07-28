"""On-disk format for recorded play episodes (human or synthetic).

One ``.npz`` file per episode with three entries:

  obs      (T, OBS_DIM)          float32 -- per pvpbot.spec.OBS_LAYOUT
  actions  (T, NUM_ACTION_HEADS) int64   -- per pvpbot.spec.ACTION_HEADS
  meta     0-d unicode array holding a JSON object. Required keys:
           "tick_rate" (int), "source" (str), "map" (str). Also carries
           "format_version"; writers may add extra keys.

Row t pairs the observation the player saw at tick t with the action they
took at tick t. No pickled objects are stored, so files load with
``allow_pickle=False`` (the numpy default).

Human mouse input arrives as continuous degrees-per-tick deltas; the action
space wants indices into ``CAMERA_BINS``. ``mouse_to_bins`` /
``bins_to_mouse`` convert between the two.
"""
import json
import os
from glob import glob
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch

from pvpbot.spec import (
    ACTION_HEADS,
    ACTION_HEAD_SIZES,
    CAMERA_BINS,
    NUM_ACTION_HEADS,
    OBS_DIM,
    TICK_RATE,
)

FORMAT_VERSION = 1
RECORDING_SUFFIX = ".npz"
_REQUIRED_META = ("tick_rate", "source", "map")
# Sanity bound on normalized observation magnitudes (they are ~[-2, 2] by
# design; anything huge means the writer fed unnormalized data).
_OBS_ABS_MAX = 1e3


# ---------------------------------------------------------------------------
# Mouse delta <-> camera bin quantization
# ---------------------------------------------------------------------------

def mouse_to_bins(
    dx_degrees: Union[float, np.ndarray], dy_degrees: Union[float, np.ndarray]
) -> Tuple[Union[int, np.ndarray], Union[int, np.ndarray]]:
    """Quantize continuous mouse deltas (degrees/tick) to CAMERA_BINS indices.

    Picks the nearest bin by absolute distance (ties resolve to the lower
    index); values beyond the extreme bins clamp to the first/last index.
    Accepts scalars or arrays. Scalars return plain ints, arrays return
    int64 arrays of the same shape.
    """
    bins = np.asarray(CAMERA_BINS, dtype=np.float64)

    def _quant(v):
        arr = np.asarray(v, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise ValueError("mouse deltas must be finite")
        idx = np.argmin(np.abs(arr[..., None] - bins), axis=-1).astype(np.int64)
        return int(idx) if np.isscalar(v) or arr.ndim == 0 else idx

    return _quant(dx_degrees), _quant(dy_degrees)


def bins_to_mouse(
    yaw_idx: Union[int, np.ndarray], pitch_idx: Union[int, np.ndarray]
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Inverse of :func:`mouse_to_bins`: bin indices -> degrees/tick deltas.

    Exact inverse on bin centers: ``mouse_to_bins(*bins_to_mouse(i, j)) ==
    (i, j)`` for all valid indices.
    """
    bins = np.asarray(CAMERA_BINS, dtype=np.float64)
    n = len(CAMERA_BINS)

    def _lookup(idx):
        arr = np.asarray(idx)
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError("bin indices must be integers, got %s" % arr.dtype)
        if arr.min() < 0 or arr.max() >= n:
            raise ValueError("bin index out of range [0, %d)" % n)
        out = bins[arr]
        return float(out) if arr.ndim == 0 else out

    return _lookup(yaw_idx), _lookup(pitch_idx)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class RecordingWriter:
    """Accumulates (obs, action) ticks for one episode, then writes one NPZ.

    Usage::

        with RecordingWriter("ep000.npz", source="human", map_name="flat") as w:
            for obs, action in episode:
                w.append(obs, action)
        # finalized on clean exit; or call w.finalize() explicitly

    ``append`` validates every tick up front so a corrupt frame fails at
    record time, not at training time.
    """

    def __init__(
        self,
        path: Union[str, "os.PathLike[str]"],
        tick_rate: int = TICK_RATE,
        source: str = "unknown",
        map_name: str = "unknown",
        extra_meta: Optional[Dict[str, object]] = None,
    ):
        self.path = os.fspath(path)
        self.meta: Dict[str, object] = {
            "format_version": FORMAT_VERSION,
            "tick_rate": int(tick_rate),
            "source": str(source),
            "map": str(map_name),
        }
        if extra_meta:
            for k, v in extra_meta.items():
                if k not in self.meta:
                    self.meta[k] = v
        self._obs: List[np.ndarray] = []
        self._actions: List[np.ndarray] = []
        self._finalized = False

    def __len__(self) -> int:
        return len(self._obs)

    def append(self, obs: np.ndarray, action: np.ndarray) -> None:
        """Add one tick: obs (OBS_DIM,) float-like, action (NUM_ACTION_HEADS,) ints."""
        if self._finalized:
            raise RuntimeError("RecordingWriter already finalized: %s" % self.path)
        o = np.asarray(obs, dtype=np.float32).reshape(-1)
        if o.shape != (OBS_DIM,):
            raise ValueError(
                "obs must have %d elements, got shape %s" % (OBS_DIM, np.shape(obs))
            )
        if not np.all(np.isfinite(o)):
            raise ValueError("obs contains non-finite values")
        a = np.asarray(action)
        if not np.issubdtype(a.dtype, np.integer):
            raise ValueError("action must be integer-typed, got %s" % a.dtype)
        a = a.astype(np.int64).reshape(-1)
        if a.shape != (NUM_ACTION_HEADS,):
            raise ValueError(
                "action must have %d heads, got shape %s"
                % (NUM_ACTION_HEADS, np.shape(action))
            )
        for k, (name, size) in enumerate(ACTION_HEADS):
            if not (0 <= a[k] < size):
                raise ValueError(
                    "action head '%s' value %d outside [0, %d)" % (name, a[k], size)
                )
        self._obs.append(o.copy())
        self._actions.append(a.copy())

    def finalize(self) -> str:
        """Write the NPZ and return its path. A writer finalizes exactly once."""
        if self._finalized:
            raise RuntimeError("RecordingWriter already finalized: %s" % self.path)
        if not self._obs:
            raise ValueError("cannot finalize an empty recording: %s" % self.path)
        obs = np.stack(self._obs).astype(np.float32)
        actions = np.stack(self._actions).astype(np.int64)
        parent = os.path.dirname(os.path.abspath(self.path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        np.savez_compressed(
            self.path,
            obs=obs,
            actions=actions,
            meta=np.asarray(json.dumps(self.meta)),
        )
        self._finalized = True
        return self.path

    def __enter__(self) -> "RecordingWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None and not self._finalized and self._obs:
            self.finalize()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class Recording:
    """One validated episode: obs (T, OBS_DIM) f32, actions (T, heads) i64."""

    def __init__(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        meta: Dict[str, object],
        path: Optional[str] = None,
    ):
        self.obs = obs
        self.actions = actions
        self.meta = meta
        self.path = path

    @property
    def num_ticks(self) -> int:
        return int(self.obs.shape[0])

    def __len__(self) -> int:
        return self.num_ticks

    def torch_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """(obs (T, OBS_DIM) float32, actions (T, NUM_ACTION_HEADS) int64)."""
        return (
            torch.from_numpy(np.ascontiguousarray(self.obs)),
            torch.from_numpy(np.ascontiguousarray(self.actions)),
        )


def _validate_arrays(obs: np.ndarray, actions: np.ndarray, path: str) -> None:
    if obs.ndim != 2 or obs.shape[1] != OBS_DIM:
        raise ValueError("%s: obs shape %s, expected (T, %d)" % (path, obs.shape, OBS_DIM))
    if actions.ndim != 2 or actions.shape[1] != NUM_ACTION_HEADS:
        raise ValueError(
            "%s: actions shape %s, expected (T, %d)"
            % (path, actions.shape, NUM_ACTION_HEADS)
        )
    if obs.shape[0] != actions.shape[0]:
        raise ValueError(
            "%s: obs has %d ticks but actions has %d"
            % (path, obs.shape[0], actions.shape[0])
        )
    if obs.shape[0] < 1:
        raise ValueError("%s: empty recording" % path)
    if not np.all(np.isfinite(obs)):
        raise ValueError("%s: obs contains non-finite values" % path)
    if np.abs(obs).max() > _OBS_ABS_MAX:
        raise ValueError(
            "%s: obs magnitude %.3g exceeds sanity bound %g (unnormalized data?)"
            % (path, float(np.abs(obs).max()), _OBS_ABS_MAX)
        )
    if not np.issubdtype(actions.dtype, np.integer):
        raise ValueError("%s: actions dtype %s is not integer" % (path, actions.dtype))
    for k, (name, size) in enumerate(ACTION_HEADS):
        col = actions[:, k]
        if col.min() < 0 or col.max() >= size:
            raise ValueError(
                "%s: action head '%s' outside [0, %d) (min=%d max=%d)"
                % (path, name, size, int(col.min()), int(col.max()))
            )


def load_recording(path: Union[str, "os.PathLike[str]"]) -> Recording:
    """Load and validate one episode NPZ. Raises ValueError on any violation."""
    path = os.fspath(path)
    with np.load(path, allow_pickle=False) as data:
        for key in ("obs", "actions", "meta"):
            if key not in data.files:
                raise ValueError("%s: missing '%s' entry" % (path, key))
        obs = np.asarray(data["obs"], dtype=np.float32)
        actions = np.asarray(data["actions"]).astype(np.int64, casting="same_kind")
        raw_meta = str(data["meta"][()])
    try:
        meta = json.loads(raw_meta)
    except json.JSONDecodeError as e:
        raise ValueError("%s: meta is not valid JSON: %s" % (path, e))
    if not isinstance(meta, dict):
        raise ValueError("%s: meta must be a JSON object" % path)
    for key in _REQUIRED_META:
        if key not in meta:
            raise ValueError("%s: meta missing required key '%s'" % (path, key))
    _validate_arrays(obs, actions, path)
    return Recording(obs=obs, actions=actions, meta=meta, path=path)


def list_recordings(
    directory: Union[str, "os.PathLike[str]"], pattern: str = "*" + RECORDING_SUFFIX
) -> List[str]:
    """Sorted recording paths under ``directory`` (non-recursive)."""
    return sorted(glob(os.path.join(os.fspath(directory), pattern)))


def iter_recordings(
    directory: Union[str, "os.PathLike[str]"], pattern: str = "*" + RECORDING_SUFFIX
) -> Iterator[Recording]:
    """Yield validated recordings from ``directory`` in sorted path order."""
    for p in list_recordings(directory, pattern):
        yield load_recording(p)
