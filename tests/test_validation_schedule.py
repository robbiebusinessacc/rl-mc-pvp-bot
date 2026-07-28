"""Schedule JSON schema validation + mechanic coverage (tools/validation)."""
import copy
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VALIDATION_DIR = REPO / "tools" / "validation"
for p in (str(REPO), str(VALIDATION_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import harness  # noqa: E402

BASIC = VALIDATION_DIR / "schedules" / "basic.json"


def load_basic():
    with open(BASIC) as f:
        return json.load(f)


def test_basic_schedule_is_valid():
    sched = load_basic()
    assert harness.validate_schedule(sched) == []


def test_basic_schedule_covers_all_required_mechanics():
    sched = load_basic()
    mechanics = {seg["mechanic"] for seg in sched["segments"]}
    missing = harness.REQUIRED_MECHANICS - mechanics
    assert not missing, "schedule missing mechanics: %s" % sorted(missing)


def test_basic_schedule_uses_both_bots_and_attack_reaches():
    sched = load_basic()
    bots_used = {seg["bot"] for seg in sched["segments"]}
    assert bots_used == set(sched["bots"]) == {"A", "B"}
    # attack window must exist and have the attack input held
    attack_segs = [s for s in sched["segments"] if s["mechanic"] == "attack"]
    assert attack_segs and all(s["inputs"].get("attack") for s in attack_segs)


def test_turn_segment_totals_180_degrees():
    sched = load_basic()
    turns = [s for s in sched["segments"] if s["mechanic"] == "turn_180"]
    assert turns
    seg = turns[0]
    total = (seg["tick_end"] - seg["tick_start"]) * seg["inputs"]["yaw_delta_deg"]
    assert abs(abs(total) - 180.0) < 1e-9


def test_overlapping_segments_rejected():
    sched = load_basic()
    bad = copy.deepcopy(sched)
    bad["segments"].append(
        {
            "id": "overlap_probe",
            "bot": "A",
            "tick_start": 0,
            "tick_end": 10,
            "mechanic": "walk",
            "inputs": {"forward": True},
        }
    )
    errs = harness.validate_schedule(bad)
    assert any("overlap" in e for e in errs)


def test_unknown_input_key_rejected():
    sched = load_basic()
    bad = copy.deepcopy(sched)
    bad["segments"][0]["inputs"] = {"forward": True, "teleport": True}
    errs = harness.validate_schedule(bad)
    assert any("unknown input keys" in e for e in errs)


def test_missing_mechanic_coverage_rejected():
    sched = load_basic()
    bad = copy.deepcopy(sched)
    for seg in bad["segments"]:
        if seg["mechanic"] == "sprint_jump":
            seg["mechanic"] = "coast"
    errs = harness.validate_schedule(bad)
    assert any("does not cover mechanics" in e for e in errs)


def test_bad_tick_range_rejected():
    sched = load_basic()
    bad = copy.deepcopy(sched)
    bad["segments"][0]["tick_end"] = bad["segments"][0]["tick_start"]
    errs = harness.validate_schedule(bad)
    assert any("tick_start < tick_end" in e for e in errs)


def test_idle_ticks_default_to_no_inputs():
    sched = load_basic()
    # carve a hole: drop the disengage segment; those ticks must become idle
    sched = copy.deepcopy(sched)
    sched["segments"] = [s for s in sched["segments"] if s["id"] != "disengage"]
    track = harness.inputs_track(sched)
    assert track[415][0] == {}  # bot A side, formerly "disengage"
