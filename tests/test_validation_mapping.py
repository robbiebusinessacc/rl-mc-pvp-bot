"""Input -> ACTION_HEADS mapping and coordinate conventions (hand-checked)."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
VALIDATION_DIR = REPO / "tools" / "validation"
for p in (str(REPO), str(VALIDATION_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import harness  # noqa: E402
from pvpbot.spec import ACTION_HEADS, CAMERA_BINS, NUM_ACTION_HEADS  # noqa: E402


def act(**inputs):
    return harness.inputs_to_action(inputs).tolist()


def test_head_order_matches_spec():
    assert [name for name, _ in ACTION_HEADS] == [
        "forward", "strafe", "jump", "sprint", "attack", "yaw", "pitch",
    ]
    assert harness.IDLE_ACTION.shape == (NUM_ACTION_HEADS,)


def test_idle_action():
    # none/none/no/no/no + zero camera deltas (bin 5 is 0.0 in CAMERA_BINS)
    assert CAMERA_BINS[5] == 0.0
    assert act() == [1, 1, 0, 0, 0, 5, 5]


def test_movement_head_values():
    assert act(forward=True) == [2, 1, 0, 0, 0, 5, 5]
    assert act(back=True)[0] == 0
    assert act(forward=True, back=True)[0] == 1  # cancel
    assert act(left=True)[1] == 0   # spec: strafe 0=left
    assert act(right=True)[1] == 2  # spec: strafe 2=right
    assert act(left=True, right=True)[1] == 1


def test_flag_heads():
    assert act(jump=True)[2] == 1
    assert act(sprint=True)[3] == 1
    assert act(attack=True)[4] == 1
    full = act(forward=True, sprint=True, jump=True, attack=True)
    assert full == [2, 1, 1, 1, 1, 5, 5]


@pytest.mark.parametrize(
    "delta,expected_idx",
    [
        (0.0, 5),
        (1.0, 6),
        (-1.0, 4),
        (3.0, 7),
        (7.0, 8),
        (15.0, 9),
        (30.0, 10),
        (-30.0, 0),
        (-15.0, 1),
        (100.0, 10),   # clips to extreme bin
        (-100.0, 0),
        (0.4, 5),      # nearest wins
        (-0.6, 4),     # closer to -1.0 than 0.0
        (2.0, 6),      # exact tie between 1.0 and 3.0 -> lowest index
        (-2.0, 3),     # tie between -3.0 and -1.0 -> lowest index (bin -3.0)
        (28.0, 10),
    ],
)
def test_yaw_binning_hand_checked(delta, expected_idx):
    assert harness.nearest_camera_bin(delta) == expected_idx
    a = harness.inputs_to_action({"yaw_delta_deg": delta})
    assert a[5] == expected_idx


def test_pitch_binning_uses_same_bins():
    a = harness.inputs_to_action({"pitch_delta_deg": -7.0})
    assert a[6] == 2  # CAMERA_BINS[2] == -7.0
    assert CAMERA_BINS[2] == -7.0


def test_unknown_input_key_raises():
    with pytest.raises(ValueError):
        harness.inputs_to_action({"fly": True})


# ---------------------------------------------------------------------------
# Yaw convention: sim_yaw = mc_yaw + 90 must make the forward vectors agree.
# MC forward = (-sin(mc), cos(mc)) in (x, z); sim forward = (cos(sim), sin(sim)).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mc,sim", [(0, 90), (90, 180), (180, 270), (270, 0), (45, 135)])
def test_mc_to_sim_yaw_hand_checked(mc, sim):
    assert harness.mc_yaw_to_sim(mc) == pytest.approx(sim)
    assert harness.sim_yaw_to_mc(sim) == pytest.approx(mc % 360.0)


@pytest.mark.parametrize("mc_yaw", [0.0, 33.0, 90.0, 123.4, 180.0, 270.0, 359.0])
def test_forward_vectors_agree_under_conversion(mc_yaw):
    mc = np.deg2rad(mc_yaw)
    mc_fwd = np.array([-np.sin(mc), np.cos(mc)])           # MC (x, z)
    sim = np.deg2rad(harness.mc_yaw_to_sim(mc_yaw))
    sim_fwd = np.array([np.cos(sim), np.sin(sim)])         # stub (x, z)
    np.testing.assert_allclose(mc_fwd, sim_fwd, atol=1e-12)


def test_left_strafe_stays_left_under_conversion():
    # MC yaw 0 faces +Z; MC-left is +X (east). In the stub at the converted
    # yaw (90), strafe head "left" (index 0 -> -1) must also produce +X:
    # ax = fwd*cos - strafe*sin = -(-1)*sin(90) = +1.
    sim = np.deg2rad(harness.mc_yaw_to_sim(0.0))
    strafe = -1.0  # left
    ax = -strafe * np.sin(sim)
    az = strafe * np.cos(sim)
    assert ax == pytest.approx(1.0)
    assert az == pytest.approx(0.0, abs=1e-12)


def test_pos_conversion_shifts_only_y():
    p = harness.mc_pos_to_sim([1.5, 4.0, -2.5], ground_y=4.0)
    np.testing.assert_allclose(p, [1.5, 0.0, -2.5])
    back = harness.sim_pos_to_mc(p, ground_y=4.0)
    np.testing.assert_allclose(back, [1.5, 4.0, -2.5])


def test_build_action_track_shape_and_content():
    sched = harness.load_schedule(str(VALIDATION_DIR / "schedules" / "basic.json"))
    track = harness.build_action_track(sched)
    T = harness.schedule_num_ticks(sched)
    assert track.shape == (T, 2, NUM_ACTION_HEADS)
    assert track.dtype == np.int64
    # tick 0: A walks forward, B walks forward
    assert track[0, 0].tolist() == [2, 1, 0, 0, 0, 5, 5]
    assert track[0, 1].tolist() == [2, 1, 0, 0, 0, 5, 5]
    # turn_180 ticks: forward + max positive yaw bin
    assert track[230, 0].tolist() == [2, 1, 0, 0, 0, 10, 5]
    # sprint_jump ticks
    assert track[150, 0].tolist() == [2, 1, 1, 1, 0, 5, 5]
    # attack window: attack only
    assert track[385, 0].tolist() == [1, 1, 0, 0, 1, 5, 5]
    # B idle in hold segment
    assert track[100, 1].tolist() == [1, 1, 0, 0, 0, 5, 5]
