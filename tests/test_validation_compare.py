"""End-to-end harness proof: fixtures from the stub -> compare.py metrics.

perfect.jsonl (stub trajectory re-encoded as a Minecraft-frame recording)
must round-trip to ~0 RMSE -- this exercises the input->action mapping,
the yaw-convention conversion and the y/ground_y shift. perturbed.jsonl
(seeded velocity noise, boosted in one segment) must show nonzero RMSE
with the boosted segment identified as worst.
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VALIDATION_DIR = REPO / "tools" / "validation"
for p in (str(REPO), str(VALIDATION_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import compare  # noqa: E402
import harness  # noqa: E402
import make_fixture  # noqa: E402

SCHEDULE = str(VALIDATION_DIR / "schedules" / "basic.json")
BOOST_SEGMENT = "sprint_jump"


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory):
    out = tmp_path_factory.mktemp("val_fixtures")
    summary = make_fixture.generate_fixtures(
        SCHEDULE, str(out), seed=0, noise_vel=0.02,
        boost_segment=BOOST_SEGMENT, boost=8.0,
    )
    return summary


def test_fixture_combat_actually_happened(fixtures):
    # the schedule's open-loop approach must land hits in the stub,
    # otherwise the attack/knockback mechanics are not exercised at all
    assert len(fixtures["hit_ticks"]) >= 2
    assert fixtures["final_hp"]["B"] < 20.0
    assert fixtures["final_hp"]["A"] == 20.0
    # 1.8 hurt-time gating in the stub: hits at least 11 ticks apart
    ticks = fixtures["hit_ticks"]
    assert all(b - a >= 11 for a, b in zip(ticks, ticks[1:]))


def test_fixture_rows_are_mc_frame(fixtures):
    rows, hurt = harness.read_recording(fixtures["perfect_path"])
    assert hurt and all(ev["victim"] == "B" for ev in hurt)
    # feet y sits on the superflat surface (ground_y=4), not the sim's 0
    grounded_y = [r["pos"][1] for r in rows.values() if r["onGround"]]
    assert grounded_y and all(abs(y - 4.0) < 1e-9 for y in grounded_y)
    # 840 tick rows: 420 ticks x 2 bots
    assert len(rows) == fixtures["num_ticks"] * 2


def test_perfect_fixture_zero_rmse(fixtures):
    rep = compare.run_compare(fixtures["perfect_path"], SCHEDULE, env_kind="stub")
    assert rep["env"] == "stub"
    for seg in rep["segments"]:
        assert seg["n_ticks_compared"] == seg["tick_end"] - seg["tick_start"]
        assert seg["pos_rmse"] == pytest.approx(0.0, abs=1e-9), seg["id"]
        assert seg["vel_rmse"] == pytest.approx(0.0, abs=1e-9), seg["id"]
        assert seg["yaw_rmse_deg"] == pytest.approx(0.0, abs=1e-9), seg["id"]
    assert rep["combined_pos_rmse"] == pytest.approx(0.0, abs=1e-9)
    for bot in ("A", "B"):
        assert rep["overall_per_bot"][bot]["pos_rmse"] == pytest.approx(0.0, abs=1e-9)
    assert rep["num_hurt_events_recorded"] == rep["num_hurt_events_sim"] == len(
        fixtures["hit_ticks"]
    )


def test_perturbed_fixture_nonzero_rmse_and_worst_segment(fixtures):
    rep = compare.run_compare(fixtures["perturbed_path"], SCHEDULE, env_kind="stub")
    scored = [s for s in rep["segments"] if s["pos_rmse"] is not None]
    assert scored
    for seg in scored:
        assert seg["pos_rmse"] > 1e-4, seg["id"]
        assert seg["vel_rmse"] > 1e-4, seg["id"]
    assert rep["combined_pos_rmse"] > 1e-3
    # deterministic worst-segment identification: the boosted segment
    assert rep["worst_segment_pos"] == BOOST_SEGMENT
    assert rep["worst_segment_vel"] == BOOST_SEGMENT
    # and the reported worst must equal the argmax of the reported numbers
    assert rep["worst_segment_vel"] == max(scored, key=lambda s: s["vel_rmse"])["id"]
    assert rep["worst_segment_pos"] == max(scored, key=lambda s: s["pos_rmse"])["id"]
    # max-divergence tick of the boosted segment lies inside that segment
    boosted = next(s for s in scored if s["id"] == BOOST_SEGMENT)
    assert boosted["tick_start"] <= boosted["max_pos_err_tick"] < boosted["tick_end"]


def test_report_is_json_serializable_and_formats(fixtures):
    rep = compare.run_compare(fixtures["perturbed_path"], SCHEDULE, env_kind="stub")
    encoded = json.dumps(rep)  # raises on numpy leftovers
    assert BOOST_SEGMENT in encoded
    text = compare.format_report(rep)
    assert "worst segment by vel RMSE: %s" % BOOST_SEGMENT in text
    assert "sim env  : stub" in text


def test_tick_offset_misalignment_detected(fixtures):
    # shifting a perfect recording by one tick must produce nonzero error
    # (this is the failure mode --tick-offset exists to diagnose)
    rep = compare.run_compare(
        fixtures["perfect_path"], SCHEDULE, env_kind="stub", tick_offset=1
    )
    moving = next(s for s in rep["segments"] if s["id"] == "walk_out")
    assert moving["pos_rmse"] > 1e-3


def test_env_auto_falls_back_to_stub():
    # pvpbot.sim.env does not exist yet on this branch; auto must fall back
    env, kind = harness.make_env("auto")
    try:
        from pvpbot.sim.env import DuelVecEnv  # noqa: F401

        assert kind == "real"
    except ImportError:
        assert kind == "stub"
    assert env.num_envs == 1


def test_cli_smoke(fixtures, tmp_path, capsys):
    json_out = tmp_path / "report.json"
    rc = compare.main(
        [
            "--recording", fixtures["perfect_path"],
            "--schedule", SCHEDULE,
            "--env", "stub",
            "--json-out", str(json_out),
            "--fail-above-pos-rmse", "0.001",
        ]
    )
    assert rc == 0
    assert json.loads(json_out.read_text())["combined_pos_rmse"] < 1e-9
    out = capsys.readouterr().out
    assert "physics validation report" in out
