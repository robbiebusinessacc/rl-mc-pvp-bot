"""Deploy module: tick loop — dry run, pacing, kill-switch, flight recorder."""
import json
import logging

import numpy as np
import pytest

from pvpbot.deploy.capture import FRAME_H, FRAME_W, MockFrameSource
from pvpbot.deploy.input_inject import Calibration, MockInputSink
from pvpbot.deploy.loop import (
    NEUTRAL_ACTION,
    STAGES,
    TICK_PERIOD,
    MatchState,
    TickLoop,
    TorchPolicy,
)
from pvpbot.spec import ACTION_HEAD_SIZES, CAMERA_BINS, NUM_ACTION_HEADS, OBS_DIM


class FakeClock:
    """Deterministic time source: every clock() reading costs `work` seconds,
    sleep() advances exactly the requested amount."""

    def __init__(self, work: float = 0.0005):
        self.t = 0.0
        self.work = work
        self.slept = 0.0

    def clock(self) -> float:
        self.t += self.work
        return self.t

    def sleep(self, dt: float) -> None:
        assert dt >= 0.0
        self.t += dt
        self.slept += dt


AGGRESSIVE = np.array([2, 1, 1, 1, 1, CAMERA_BINS.index(30.0), CAMERA_BINS.index(0.0)],
                      dtype=np.int64)


def make_loop(tmp_path, policy=None, clock=None, ticks_frames=None, **kw):
    clock = clock or FakeClock()
    frames = (
        MockFrameSource.noise(seed=0)
        if ticks_frames is None
        else MockFrameSource(
            [np.zeros((FRAME_H, FRAME_W), np.uint8)] * ticks_frames
        )
    )
    sink = MockInputSink(time_fn=clock.clock)
    kw.setdefault("recorder_path", str(tmp_path / "flight.jsonl"))
    kw.setdefault("kill_switch_path", str(tmp_path / "stop-sentinel"))
    loop = TickLoop(
        frame_source=frames,
        input_sink=sink,
        policy=policy if policy is not None else (lambda obs: AGGRESSIVE.copy()),
        calibration=Calibration(px_per_degree_x=10.0, px_per_degree_y=10.0),
        clock=clock.clock,
        sleep=clock.sleep,
        **kw,
    )
    return loop, sink, clock


# ---------------------------------------------------------------------------
# Dry run: actions flow, pacing, latency stats
# ---------------------------------------------------------------------------

def test_dry_run_40_ticks_actions_reach_sink(tmp_path):
    loop, sink, _ = make_loop(tmp_path)
    result = loop.run(40)

    assert result.ticks_run == 40
    assert result.state is MatchState.STOPPED
    assert result.stop_reason == "completed 40 ticks"

    # movement keys: pressed once at tick 0 (edge-triggered), released at end
    key_events = sink.events_of("key")
    downs = [e.data for e in key_events if e.data[1]]
    assert ("w", True) in downs and ("space", True) in downs and ("sprint", True) in downs
    assert sink.held_keys == set()  # release_all ran on stop

    # attack: click down+up every tick
    clicks = sink.events_of("click")
    assert len(clicks) == 80
    assert [e.data[0] for e in clicks[:2]] == [True, False]

    # camera: +30 deg/tick at 10 px/deg -> 300 px/tick, 40 ticks
    moves = sink.events_of("move")
    assert len(moves) == 40
    assert sink.total_mouse_delta() == (300 * 40, 0)


def test_dry_run_tick_pacing_with_fake_clock(tmp_path):
    clock = FakeClock(work=0.0005)
    loop, _, _ = make_loop(tmp_path, clock=clock)
    start = clock.t
    result = loop.run(40)
    elapsed = clock.t - start

    assert result.ticks_run == 40
    # 40 ticks at 50 ms — the loop must pace to the deadline grid
    assert elapsed == pytest.approx(40 * TICK_PERIOD, rel=0.10)
    assert clock.slept > 0.8 * 40 * TICK_PERIOD  # mostly idle at these latencies
    assert result.stats.overruns == 0


def test_latency_stats_and_table(tmp_path):
    loop, _, _ = make_loop(tmp_path)
    result = loop.run(10)
    summary = result.stats.summary()
    for stage in STAGES + ("tick",):
        assert stage in summary
        assert len(result.stats.samples_ms[stage]) == 10
        assert summary[stage]["max"] >= summary[stage]["p50"] >= 0.0
    table = result.stats.format_table()
    for stage in STAGES:
        assert stage in table
    assert "budget" in table


def test_overrun_detection_warns(tmp_path, caplog):
    clock = FakeClock(work=0.02)  # 6 clock reads/tick -> ~120 ms of "work"
    loop, _, _ = make_loop(tmp_path, clock=clock)
    with caplog.at_level(logging.WARNING, logger="pvpbot.deploy.loop"):
        result = loop.run(5)
    assert result.stats.overruns == 5
    assert any("exceeded budget" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Kill-switch and stop callback
# ---------------------------------------------------------------------------

def test_kill_switch_file_stops_loop(tmp_path):
    sentinel = tmp_path / "stop-sentinel"
    calls = {"n": 0}

    def policy(obs):
        calls["n"] += 1
        if calls["n"] == 10:
            sentinel.write_text("stop")
        return AGGRESSIVE.copy()

    loop, sink, _ = make_loop(tmp_path, policy=policy)
    result = loop.run(40)

    assert calls["n"] == 10           # tick 10 never ran the policy
    assert result.ticks_run == 10
    assert "kill-switch" in result.stop_reason
    assert str(sentinel) in result.stop_reason
    assert result.state is MatchState.STOPPED
    assert sink.held_keys == set() and sink.mouse_down is False


def test_stop_check_callback_stops_loop(tmp_path):
    ticks = {"n": 0}

    def policy(obs):
        ticks["n"] += 1
        return AGGRESSIVE.copy()

    loop, _, _ = make_loop(tmp_path, policy=policy,
                           stop_check=lambda: ticks["n"] >= 7)
    result = loop.run(40)
    assert result.ticks_run == 7
    assert "stop_check" in result.stop_reason


def test_frame_exhaustion_stops_loop(tmp_path):
    loop, _, _ = make_loop(tmp_path, ticks_frames=10)
    result = loop.run(40)
    assert result.ticks_run == 10
    assert result.stop_reason == "frame source exhausted"


# ---------------------------------------------------------------------------
# Flight recorder
# ---------------------------------------------------------------------------

def test_flight_recorder_valid_jsonl(tmp_path):
    path = tmp_path / "flight.jsonl"
    loop, _, _ = make_loop(tmp_path, recorder_path=str(path))
    loop.run(40)

    lines = path.read_text().strip().split("\n")
    assert len(lines) == 40
    for i, line in enumerate(lines):
        rec = json.loads(line)  # every line is standalone valid JSON
        assert rec["tick"] == i
        assert rec["state"] == "fighting"
        assert len(rec["obs"]) == OBS_DIM
        assert len(rec["percep"]) == 12
        assert len(rec["action"]) == NUM_ACTION_HEADS
        for head, size in enumerate(ACTION_HEAD_SIZES):
            assert 0 <= rec["action"][head] < size
        assert set(rec["latency_ms"]) == {"capture", "encode", "policy",
                                          "inject", "tick"}
        assert rec["latency_ms"]["tick"] >= 0
    # timestamps strictly increase
    ts = [json.loads(l)["t"] for l in lines]
    assert ts == sorted(ts) and ts[0] < ts[-1]


def test_recorder_disabled(tmp_path):
    loop, _, _ = make_loop(tmp_path, recorder_path=None)
    result = loop.run(5)
    assert result.ticks_run == 5
    assert loop.recorder is None


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

def test_idle_state_sends_neutral_inputs_only(tmp_path):
    loop, sink, _ = make_loop(tmp_path, auto_fight=False)
    result = loop.run(10)
    assert result.ticks_run == 10
    assert result.state is MatchState.STOPPED
    # neutral action: no keys, no clicks, no mouse motion
    assert sink.events_of("click") == []
    assert sink.events_of("move") == []
    assert all(not e.data[1] for e in sink.events_of("key"))  # only releases (none held)


def test_start_fight_transition(tmp_path):
    loop, sink, _ = make_loop(tmp_path, auto_fight=False)
    loop.start_fight()
    assert loop.state is MatchState.FIGHTING
    result = loop.run(5)
    assert len(sink.events_of("click")) == 10  # policy in control
    assert result.state is MatchState.STOPPED


def test_obs_carries_prev_action(tmp_path):
    """Fallback assembler must expose the previous EFFECTIVE action.

    The move-latch (live mirror of sim act_hold_ticks) filters what reaches
    the keys, so obs must carry that same filtered action -- the policy is
    trained on effective prev_action, and dead-reckoning from raw intents
    would report motion the client never performed.
    """
    from pvpbot.spec import OBS_LAYOUT

    seen = []

    def policy(obs):
        s, e = OBS_LAYOUT["prev_action"]
        seen.append(obs[s:e].copy())
        return AGGRESSIVE.copy()

    loop, _, _ = make_loop(tmp_path, policy=policy)
    loop.run(6)
    sizes = np.asarray(ACTION_HEAD_SIZES, np.float64)
    # first tick: neutral prev_action
    np.testing.assert_allclose(seen[0], NEUTRAL_ACTION / sizes, atol=1e-6)
    # tick 2: movement/sprint heads still latched at neutral, rest aggressive
    latched = AGGRESSIVE.copy()
    latched[[0, 1, 3]] = NEUTRAL_ACTION[[0, 1, 3]]
    np.testing.assert_allclose(seen[1], latched / sizes, atol=1e-6)
    # after persisting act-hold ticks the movement heads commit
    np.testing.assert_allclose(seen[4], AGGRESSIVE / sizes, atol=1e-6)


# ---------------------------------------------------------------------------
# Real PolicyNet path (still offline: random init, mock I/O)
# ---------------------------------------------------------------------------

def test_loop_with_real_policynet(tmp_path):
    policy = TorchPolicy()  # random init
    loop, sink, _ = make_loop(tmp_path, policy=policy)
    result = loop.run(5)
    assert result.ticks_run == 5
    assert len(result.stats.samples_ms["policy"]) == 5
    # sampled actions must be within head ranges (checked via recorder)
    lines = (tmp_path / "flight.jsonl").read_text().strip().split("\n")
    for line in lines:
        action = json.loads(line)["action"]
        for head, size in enumerate(ACTION_HEAD_SIZES):
            assert 0 <= action[head] < size


def test_torch_policy_checkpoint_roundtrip(tmp_path):
    import torch

    from pvpbot.models import PolicyNet

    net = PolicyNet()
    ckpt = tmp_path / "policy.pt"
    torch.save(
        {"model": net.state_dict(),
         "meta": {"obs_dim": OBS_DIM, "action_heads": ACTION_HEAD_SIZES, "step": 0}},
        str(ckpt),
    )
    policy = TorchPolicy(checkpoint=str(ckpt), deterministic=True)
    assert policy.meta["step"] == 0
    action = policy(np.zeros(OBS_DIM, dtype=np.float32))
    assert action.shape == (NUM_ACTION_HEADS,)
    # deterministic: same obs + fresh state -> same action
    policy.reset()
    action2 = policy(np.zeros(OBS_DIM, dtype=np.float32))
    np.testing.assert_array_equal(action, action2)


def test_torch_policy_rejects_bad_obs_dim(tmp_path):
    import torch

    from pvpbot.models import PolicyNet

    ckpt = tmp_path / "bad.pt"
    torch.save({"model": PolicyNet().state_dict(), "meta": {"obs_dim": 99}}, str(ckpt))
    with pytest.raises(ValueError):
        TorchPolicy(checkpoint=str(ckpt))
