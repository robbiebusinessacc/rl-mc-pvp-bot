"""M2 milestone metric: crosshair-on-hitbox time from a flight recording.

MILESTONES.md M2: crosshair-on-hitbox >= 25% of ENGAGED ticks live.
"Engaged" = enemy perceived visible and within 4 blocks. "On hitbox" =
the perceived aim error lies inside the angle the 0.6 x 1.8 box subtends
at the perceived distance (the same geometry the ray hit-check uses).

Usage: python3 tools/realdata/measure_m2.py [pvpbot-flight.jsonl]
"""
import json
import math
import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_THIS)))

from pvpbot.perception.synth import dist_from_bbox_height  # noqa: E402
from pvpbot.spec import OBS_LAYOUT, PERCEPTION_LAYOUT  # noqa: E402


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "pvpbot-flight.jsonl"
    rows = [json.loads(l) for l in open(path)]
    P = np.array([r["percep"] for r in rows], dtype=np.float32)
    O = np.array([r["obs"] for r in rows], dtype=np.float32)
    L, OL = PERCEPTION_LAYOUT, OBS_LAYOUT

    vis = P[:, L["visible"]] > 0.5
    dist = O[:, OL["dist"][0]] * 8.0
    yaw = np.abs(P[:, L["aim_err_yaw"]]) * 180.0
    pitch = np.abs(P[:, L["aim_err_pitch"]]) * 90.0

    engaged = vis & (dist < 4.0)
    d = np.maximum(dist, 0.5)
    on_box = (yaw < np.degrees(np.arctan2(0.35, d))) & (
        pitch < np.degrees(np.arctan2(1.05, d)))

    n_eng = int(engaged.sum())
    pct = 100.0 * float((engaged & on_box).sum()) / max(n_eng, 1)
    print("ticks: %d | engaged: %d (%.0f%%) | visible: %.0f%%"
          % (len(rows), n_eng, 100.0 * n_eng / max(len(rows), 1),
             100.0 * vis.mean()))
    print("crosshair-on-hitbox while engaged: %.1f%%  (M2 bar: 25%%)" % pct)
    print("median |yaw err| while engaged: %.1f deg"
          % float(np.median(yaw[engaged])) if n_eng else "no engaged ticks")
    print("M2: %s" % ("PASS" if pct >= 25.0 else "not yet"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
