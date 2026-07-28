"""Synthetic frame generator: determinism, shapes, label/pixel consistency."""
import dataclasses
import math

import numpy as np

from pvpbot.spec import FRAME_SHAPE, PERCEPTION_DIM, PERCEPTION_LAYOUT
from pvpbot.perception.synth import (
    EYE_HEIGHT,
    FRAME_H,
    FRAME_W,
    PLAYER_HALF_H,
    PX_PER_DEG,
    SceneParams,
    SynthConfig,
    bbox_height_frac,
    dist_from_bbox_height,
    generate_batch,
    render_scene,
)

L = PERCEPTION_LAYOUT


def test_generate_batch_shapes_and_determinism():
    f1, l1 = generate_batch(64, seed=123)
    f2, l2 = generate_batch(64, seed=123)
    assert f1.shape == (64, FRAME_SHAPE[0], FRAME_H, FRAME_W) and f1.dtype == np.uint8
    assert (FRAME_SHAPE[1], FRAME_SHAPE[2]) == (FRAME_H, FRAME_W)
    assert l1.shape == (64, PERCEPTION_DIM) and l1.dtype == np.float32
    assert np.array_equal(f1, f2) and np.array_equal(l1, l2)
    f3, _ = generate_batch(64, seed=124)
    assert not np.array_equal(f1, f3)
    assert np.isfinite(l1).all()


def test_label_ranges():
    _, labels = generate_batch(256, seed=7)
    vis = labels[:, L["visible"]]
    assert set(np.unique(vis)).issubset({0.0, 1.0})
    assert 0.2 < vis.mean() < 0.9
    assert np.abs(labels[:, L["aim_err_yaw"]]).max() <= 0.5      # < 90 deg
    assert np.abs(labels[:, L["aim_err_pitch"]]).max() <= 1.0
    assert labels[:, L["self_hp"]].min() >= 0.0
    assert labels[:, L["self_hp"]].max() <= 1.0
    assert (labels[:, L["reserved"]] == 0).all()
    # invisible frames carry zeroed enemy labels
    inv = vis < 0.5
    for name in ("aim_err_yaw", "aim_err_pitch", "bbox_height", "hurt_flash",
                 "rel_screen_vx", "rel_screen_vy", "enemy_on_ground"):
        assert (labels[inv, L[name]] == 0).all(), name


def test_bbox_dist_roundtrip():
    for d in (1.5, 2.0, 3.0, 5.0, 8.0):
        assert abs(dist_from_bbox_height(bbox_height_frac(d)) - d) < 1e-3 * d


def _target_diff(params, cfg, seed=7):
    """Pixel diff between the scene with and without the enemy.
    Frames are (H, W, 3) now; geometry checks use the max over channels."""
    fA, lab = render_scene(params, np.random.default_rng(seed), cfg)
    fB, _ = render_scene(
        dataclasses.replace(params, draw_target=False),
        np.random.default_rng(seed), cfg)
    diff = np.abs(fA.astype(np.int32) - fB.astype(np.int32)).max(axis=-1)
    return fA, lab, diff


def test_target_appears_at_commanded_screen_position():
    """Pixel mass of the rendered enemy sits where the aim errors say it is."""
    cfg = SynthConfig(frame_noise=0.0)
    for yaw_err, pitch, dist in ((12.0, 5.0, 3.0), (-20.0, 0.0, 4.0),
                                 (0.0, 15.0, 2.0), (25.0, -10.0, 6.0)):
        p = SceneParams(self_pitch_deg=pitch, yaw_err_deg=yaw_err, dist=dist,
                        self_hp=14.0, n_distractors=2)
        _, lab, diff = _target_diff(p, cfg)
        ys, xs = np.nonzero(diff)
        assert len(xs) > 20, "target left no pixels"
        exp_x = FRAME_W / 2.0 + yaw_err * PX_PER_DEG
        ang = math.degrees(math.atan2(EYE_HEIGHT - PLAYER_HALF_H, dist))
        exp_y = FRAME_H / 2.0 + (ang - pitch) * PX_PER_DEG
        mid_x = (xs.min() + xs.max()) / 2.0
        mid_y = (ys.min() + ys.max()) / 2.0
        assert abs(mid_x - exp_x) <= 3.0, (yaw_err, mid_x, exp_x)
        assert abs(mid_y - exp_y) <= 4.0, (pitch, mid_y, exp_y)
        # vertical extent of the drawn enemy matches the bbox_height label
        # (shadow adds up to ~3 rows below the feet)
        h_label_px = lab[L["bbox_height"]] * FRAME_H
        h_drawn = ys.max() - ys.min() + 1
        assert abs(h_drawn - h_label_px) <= max(4.0, 0.2 * h_label_px)
        # and the labels reflect the commanded geometry
        assert abs(lab[L["aim_err_yaw"]] * 180.0 - yaw_err) < 1e-4
        assert lab[L["visible"]] == 1.0


def test_offscreen_and_occluded_are_invisible():
    cfg = SynthConfig(frame_noise=0.0)
    off = SceneParams(yaw_err_deg=65.0, dist=4.0)   # beyond FOV_X/2 = 48 deg
    _, lab = render_scene(off, np.random.default_rng(3), cfg)
    assert lab[L["visible"]] == 0.0 and lab[L["aim_err_yaw"]] == 0.0
    occ = SceneParams(yaw_err_deg=0.0, dist=4.0, occluded=True)
    _, lab = render_scene(occ, np.random.default_rng(3), cfg)
    assert lab[L["visible"]] == 0.0 and lab[L["bbox_height"]] == 0.0


def test_hurt_flash_purples_diamond_armor():
    """Source-exact hurt chain: cyan diamond-armor texels, then the
    RGBA(1,0,0,0.3) hurt blend (RendererLivingEntity, MCP-919 1.8.9).
    On a cyan base the observable cue is DESATURATION toward purple --
    R-B rises sharply, green drops -- NOT a naive red excess (that cue
    cannot fire on diamond armor; v12's blind hurt head proved it)."""
    cfg = SynthConfig(frame_noise=0.0)
    base = SceneParams(yaw_err_deg=0.0, dist=3.0, hurt_flash=False)
    f0, _ = render_scene(base, np.random.default_rng(11), cfg)
    f1, lab1 = render_scene(
        dataclasses.replace(base, hurt_flash=True),
        np.random.default_rng(11), cfg)
    assert lab1[L["hurt_flash"]] == 1.0
    rb0 = (f0[..., 0].astype(np.int32) - f0[..., 2].astype(np.int32)).min()
    rb1 = (f1[..., 0].astype(np.int32) - f1[..., 2].astype(np.int32)).min()
    assert rb1 > rb0 + 60          # blue body desaturates hard
    g_drop = (f0[..., 1].astype(np.int32) - f1[..., 1].astype(np.int32)).max()
    assert g_drop > 25             # green falls with the blend


def test_hearts_strip_tracks_hp():
    cfg = SynthConfig(frame_noise=0.0)
    lit = []
    for hp in (2.0, 10.0, 20.0):
        f, lab = render_scene(
            SceneParams(yaw_err_deg=60.0, self_hp=hp, dist=6.0),
            np.random.default_rng(5), cfg)
        assert abs(lab[L["self_hp"]] - hp / 20.0) < 1e-6
        strip = f[FRAME_H - 6:FRAME_H - 4, 8:40]
        lit.append(int((strip > 200).sum()))
    assert lit[0] < lit[1] < lit[2]


def test_self_pitch_moves_horizon():
    cfg = SynthConfig(frame_noise=0.0, max_distractors=0)
    p_down = SceneParams(self_pitch_deg=25.0, yaw_err_deg=65.0, self_hp=20.0)
    p_up = SceneParams(self_pitch_deg=-25.0, yaw_err_deg=65.0, self_hp=20.0)
    f_down, lab_down = render_scene(p_down, np.random.default_rng(9), cfg)
    f_up, lab_up = render_scene(p_up, np.random.default_rng(9), cfg)
    assert lab_down[L["self_pitch"]] == 25.0 / 90.0
    assert lab_up[L["self_pitch"]] == -25.0 / 90.0
    # looking down -> mostly textured ground; looking up -> mostly flat sky.
    # Compare row-to-row variation in the upper half (HUD-free zone).
    var_down = np.diff(f_down[:32].astype(np.int32), axis=0).astype(np.float64)
    var_up = np.diff(f_up[:32].astype(np.int32), axis=0).astype(np.float64)
    assert np.abs(var_down).mean() > np.abs(var_up).mean()
