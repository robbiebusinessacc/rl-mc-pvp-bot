"""Synthetic demos: end-to-end integration test of the recording format."""
import numpy as np

from pvpbot.bc.recording import iter_recordings
from pvpbot.bc.synth_demos import generate_demos

ATTACK, FORWARD = 4, 0


def test_generate_and_load_round_trip(tmp_path):
    paths = generate_demos(str(tmp_path), episodes=2, seed=7, max_ticks=120)
    assert len(paths) == 4  # both sides of 2 episodes
    recs = list(iter_recordings(tmp_path))  # loader validates everything
    assert len(recs) == 4
    for rec in recs:
        assert rec.num_ticks >= 2
        assert rec.meta["source"] == "synthetic-scripted-v1"
        assert rec.meta["map"] == "stub-flat"
        assert rec.meta["tick_rate"] == 20


def test_demos_have_human_like_label_skew(tmp_path):
    generate_demos(str(tmp_path), episodes=3, seed=0, max_ticks=240)
    actions = np.concatenate([r.actions for r in iter_recordings(tmp_path)])
    attack_rate = float((actions[:, ATTACK] == 1).mean())
    # attack must exist but be rare -- the imbalance BC has to survive
    assert 0.01 < attack_rate < 0.4
    # movement dominated by not-backing-up (hold-W-ish play)
    assert float((actions[:, FORWARD] != 0).mean()) > 0.6


def test_demos_deterministic_under_seed(tmp_path):
    d1, d2 = tmp_path / "a", tmp_path / "b"
    generate_demos(str(d1), episodes=2, seed=13, max_ticks=100)
    generate_demos(str(d2), episodes=2, seed=13, max_ticks=100)
    recs1 = list(iter_recordings(d1))
    recs2 = list(iter_recordings(d2))
    assert len(recs1) == len(recs2)
    for a, b in zip(recs1, recs2):
        np.testing.assert_array_equal(a.obs, b.obs)
        np.testing.assert_array_equal(a.actions, b.actions)


def test_demos_one_side_flag(tmp_path):
    paths = generate_demos(
        str(tmp_path), episodes=2, seed=1, max_ticks=60, record_both_sides=False
    )
    assert len(paths) == 2
