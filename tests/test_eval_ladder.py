"""Ladder CLI end-to-end (scripted tiers only, plus a checkpoint run)."""
import json
import os

import torch

from pvpbot.eval import ladder
from pvpbot.models import PolicyNet
from pvpbot.spec import ACTION_HEAD_SIZES, OBS_DIM

TIERS = ["T0-Idle", "T1-Aimbot", "T2-Chaser", "T3-Strafer", "T4-Pro",
         "P1-Easy", "P2-Medium", "P3-Hard", "P4-Hacker"]


def test_ladder_cli_scripted_tiers_only(tmp_path):
    out = str(tmp_path / "reports")
    rc = ladder.main(["--matches", "8", "--out", out, "--seed", "3", "--env", "stub"])
    assert rc == 0  # sanity checks passed

    with open(os.path.join(out, "report.json")) as f:
        report = json.load(f)
    assert sorted(report["contestants"]) == sorted(TIERS)
    assert report["matches_per_pair"] == 8

    # rating table: sorted by Elo, all tiers present
    names_ranked = [row["name"] for row in report["ratings"]]
    assert sorted(names_ranked) == sorted(TIERS)
    elos = [row["elo"] for row in report["ratings"]]
    assert elos == sorted(elos, reverse=True)
    assert names_ranked[0] == "T4-Pro" and names_ranked[-1] == "T0-Idle"

    # self-consistency proof
    sanity = report["sanity"]
    assert sanity["ok"] is True
    assert sanity["t4_beats_t2"] is True
    for name in TIERS[1:]:
        if name == "T1-Aimbot":
            assert sanity["T1-Aimbot_never_loses_to_t0"] is True
        else:
            assert sanity["%s_beats_t0" % name] is True

    # win matrix complete and antisymmetric
    for a in TIERS:
        for b in TIERS:
            if a == b:
                continue
            cell = report["pairs"][a][b]
            mirror = report["pairs"][b][a]
            assert cell["wins"] == mirror["losses"]
            assert cell["draws"] == mirror["draws"]
            assert cell["wins"] + cell["losses"] + cell["draws"] == 8

    # stats present for every contestant
    for name in TIERS:
        s = report["stats"][name]
        for key in ("hits_landed", "hits_taken", "avg_combo",
                    "mean_aim_err_deg", "mean_episode_len"):
            assert key in s

    # ratings serialization embedded
    assert report["elo_json"]["type"] == "elo"
    assert report["trueskill_json"]["type"] == "trueskill-lite"

    # markdown report renders with all sections
    with open(os.path.join(out, "report.md")) as f:
        md = f.read()
    assert "## Ratings" in md and "## Win matrix" in md
    assert "## Per-contestant stats" in md and "## Sanity" in md
    for name in TIERS:
        assert name in md


def test_ladder_with_checkpoint(tmp_path):
    ckpt = str(tmp_path / "ckpt_000001.pt")
    net = PolicyNet()
    torch.save({"model": net.state_dict(),
                "meta": {"obs_dim": OBS_DIM,
                         "action_heads": ACTION_HEAD_SIZES, "step": 1}}, ckpt)
    out = str(tmp_path / "reports")
    ladder.main(["--checkpoints", ckpt, "--matches", "4", "--env", "stub",
                 "--out", out, "--seed", "1", "--max-steps", "150"])
    with open(os.path.join(out, "report.json")) as f:
        report = json.load(f)
    assert "ckpt_000001" in report["contestants"]
    assert len(report["contestants"]) == len(TIERS) + 1
    assert "ckpt_000001" in report["pairs"]["T0-Idle"]


def test_build_contestants_missing_checkpoint():
    import pytest
    with pytest.raises(FileNotFoundError):
        ladder.build_contestants(["nope/does_not_exist_*.pt".replace("*", "x")])


def test_render_markdown_is_pure():
    report = {
        "matches_per_pair": 1, "seed": 0,
        "ratings": [{"name": "x", "elo": 1000.0, "ts_mu": 25.0,
                     "ts_sigma": 8.3, "ts_conservative": 0.0,
                     "wins": 1, "losses": 0, "draws": 0}],
        "pairs": {"x": {}},
        "stats": {"x": {"hits_landed": 1.0, "hits_taken": 0.0,
                        "avg_combo": 1.0, "mean_aim_err_deg": 2.0,
                        "mean_episode_len": 10.0}},
        "sanity": {"ok": True},
    }
    md = ladder.render_markdown(report)
    assert "| 1 | x | 1000 |" in md
