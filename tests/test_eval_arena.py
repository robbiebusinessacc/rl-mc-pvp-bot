"""Arena: stat accounting on a constructed episode, checkpoint contestants."""
import numpy as np
import pytest
import torch

from pvpbot.eval.arena import (
    CheckpointContestant,
    FnContestant,
    as_contestant,
    run_match,
)
from pvpbot.eval.scripted import Chaser, Idle
from pvpbot.models import PolicyNet
from pvpbot.sim.stub import DuelVecEnv
from pvpbot.spec import ACTION_HEAD_SIZES, NUM_ACTION_HEADS, OBS_DIM, OBS_LAYOUT

SELF_HP = OBS_LAYOUT["self_hp"][0]
ENEMY_HP = OBS_LAYOUT["enemy_hp"][0]
AIM_YAW = OBS_LAYOUT["aim_err_yaw"][0]


# ---------------------------------------------------------------------------
# A fully scripted fake env: 1 duel, 7 ticks, choreographed hp drops.
#   tick:      1     2     3     4     5     6     7(done, A wins)
#   hp_a:     20    20    20    19    19    19    dead-B
#   hp_b:     20    19    18    18    18    17    --
# A (side 0) deals at ticks 2,3,6 plus the killing blow; takes 1 at tick 4.
# Expected (A): hits=4, taken=1, combos [2, 2] -> avg 2.0
# Expected (B): hits=1 (no killing-blow credit), taken=4, combos [1] -> 1.0
# ---------------------------------------------------------------------------
HP_SCRIPT = [  # (hp_a, hp_b) observed at reset and after each step
    (20, 20),  # reset
    (20, 20),  # step 1
    (20, 19),  # step 2: A hits
    (20, 18),  # step 3: A hits
    (19, 18),  # step 4: B hits
    (19, 18),  # step 5
    (19, 17),  # step 6: A hits
]
AIM_A, AIM_B = 0.1, 0.2  # constant aim_err_yaw obs -> 18 deg and 36 deg


class FakeEnv:
    MAX_TICKS = 50

    def __init__(self, num_envs: int, seed: int = 0):
        assert num_envs == 1
        self.num_envs = num_envs
        self.t = 0

    def _obs(self, hp_a, hp_b):
        obs = np.zeros((1, 2, OBS_DIM), dtype=np.float32)
        obs[0, 0, SELF_HP] = hp_a / 20.0
        obs[0, 0, ENEMY_HP] = hp_b / 20.0
        obs[0, 0, AIM_YAW] = AIM_A
        obs[0, 1, SELF_HP] = hp_b / 20.0
        obs[0, 1, ENEMY_HP] = hp_a / 20.0
        obs[0, 1, AIM_YAW] = AIM_B
        return obs

    def reset(self):
        self.t = 0
        return self._obs(*HP_SCRIPT[0])

    def step(self, actions):
        assert actions.shape == (1, 2, NUM_ACTION_HEADS)
        self.t += 1
        win = np.zeros((1, 2), dtype=np.float32)
        rew = np.zeros((1, 2), dtype=np.float32)
        if self.t < len(HP_SCRIPT):
            done = np.zeros(1, dtype=bool)
            obs = self._obs(*HP_SCRIPT[self.t])
        else:  # tick 7: B dies, fresh obs for the next episode
            done = np.ones(1, dtype=bool)
            win[0, 0] = 1.0
            obs = self._obs(20, 20)
        return obs, rew, done, {"win": win}


def _noop_contestant(name):
    def fn(obs):
        a = np.zeros((obs.shape[0], NUM_ACTION_HEADS), dtype=np.int64)
        a[:, 0] = 1
        a[:, 1] = 1
        a[:, 5] = 5
        a[:, 6] = 5
        return a
    return FnContestant(fn, name=name)


def test_arena_stats_on_constructed_episode():
    res = run_match(_noop_contestant("A"), _noop_contestant("B"),
                    num_duels=1, env_cls=FakeEnv)
    assert (res.wins_a, res.wins_b, res.draws) == (1, 0, 0)
    assert res.mean_episode_len == 7.0
    assert res.stats_a.hits_landed == 4.0      # 3 observed + killing blow
    assert res.stats_a.hits_taken == 1.0
    assert res.stats_a.avg_combo == 2.0        # combos [2, 2]
    assert res.stats_b.hits_landed == 1.0
    assert res.stats_b.hits_taken == 4.0
    assert res.stats_b.avg_combo == 1.0        # combos [1]
    assert abs(res.stats_a.mean_aim_err_deg - 18.0) < 1e-4
    assert abs(res.stats_b.mean_aim_err_deg - 36.0) < 1e-4


def test_arena_counts_every_duel_once():
    res = run_match(Chaser(seed=1), Idle(seed=2), num_duels=9, seed=4,
                    max_steps=120)
    assert res.wins_a + res.wins_b + res.draws == 9
    assert res.num_duels == 9
    assert res.name_a == "T2-Chaser" and res.name_b == "T0-Idle"


def test_arena_step_cap_records_draws():
    res = run_match(Idle(seed=1), Idle(seed=2), num_duels=4, seed=0,
                    max_steps=30)
    assert (res.wins_a, res.wins_b, res.draws) == (0, 0, 4)
    assert res.mean_episode_len == 30.0


def test_score_a_and_serialization():
    res = run_match(Chaser(seed=1), Idle(seed=2), num_duels=6, seed=4,
                    max_steps=400)
    d = res.to_dict()
    assert d["wins_a"] == res.wins_a and "stats_a" in d
    assert 0.0 <= res.score_a() <= 1.0


def _save_ckpt(path, step=123):
    net = PolicyNet()
    torch.save({
        "model": net.state_dict(),
        "meta": {"obs_dim": OBS_DIM, "action_heads": ACTION_HEAD_SIZES,
                 "step": step},
    }, path)
    return net


def test_checkpoint_contestant_loads_and_acts(tmp_path):
    path = str(tmp_path / "ckpt_000123.pt")
    _save_ckpt(path)
    c = CheckpointContestant(path)
    assert c.name == "ckpt_000123"
    assert c.meta["step"] == 123
    c.begin(3)
    obs = np.random.default_rng(0).normal(size=(3, OBS_DIM)).astype(np.float32)
    acts = c.act(obs)
    assert acts.shape == (3, NUM_ACTION_HEADS) and acts.dtype == np.int64
    for h, size in enumerate(ACTION_HEAD_SIZES):
        assert (acts[:, h] >= 0).all() and (acts[:, h] < size).all()
    # recurrent state: advances every act, resets on done
    state_before = c._state.clone()
    c.act(obs)
    assert not torch.equal(c._state, state_before)
    c.on_done(np.array([True, False, False]))
    assert torch.all(c._state[0] == 0.0)
    assert not torch.all(c._state[1] == 0.0)


def test_checkpoint_contestant_in_a_match(tmp_path):
    path = str(tmp_path / "ckpt_1.pt")
    _save_ckpt(path)
    res = run_match(CheckpointContestant(path), Idle(seed=3), num_duels=4,
                    seed=2, max_steps=60, env_cls=DuelVecEnv)
    assert res.wins_a + res.wins_b + res.draws == 4


def test_checkpoint_rejects_wrong_format(tmp_path):
    path = str(tmp_path / "bad.pt")
    torch.save({"weights": {}}, path)
    with pytest.raises(ValueError):
        CheckpointContestant(path)


def test_as_contestant_coercions(tmp_path):
    assert as_contestant(Idle()).name == "T0-Idle"
    c = as_contestant(lambda obs: np.zeros((obs.shape[0], NUM_ACTION_HEADS),
                                           dtype=np.int64))
    assert hasattr(c, "act") and hasattr(c, "begin")
    path = str(tmp_path / "ckpt_2.pt")
    _save_ckpt(path)
    assert as_contestant(path).name == "ckpt_2"
    with pytest.raises(TypeError):
        as_contestant(42)
