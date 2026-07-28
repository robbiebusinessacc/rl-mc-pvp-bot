"""Procedural synthetic Minecraft-ish frame generator. Pure NumPy.

Produces (frame, label) pairs matching FRAME_SHAPE / PERCEPTION_LAYOUT from
pvpbot.spec. This is the pretraining set for PerceptionCNN until real-game
capture data exists; domain-randomization knobs live in SynthConfig.

Scene model (crude, but geometrically self-consistent):
  * Pinhole-ish camera with a linear angle->pixel mapping, square pixels:
    PX_PER_DEG = FRAME_W / FOV_X_DEG (~1.19 px/deg, FOV ~96 x 54 deg).
  * Sky/ground split at the horizon row, which moves with self_pitch
    (Minecraft convention: pitch > 0 means looking DOWN, so the horizon
    moves UP the screen).
  * Ground is blocky value-noise texture; own movement speed adds vertical
    motion blur to the ground (the single-frame stand-in for optic flow).
  * The enemy is a humanoid (head + torso + arms + legs rectangles,
    armor-ish light shade with speckle) placed by world geometry:
    distance -> angular size (perspective scaling), aim errors -> screen
    position, jump height -> feet/shadow gap (enemy_on_ground cue).
  * hurt_flash brightens the enemy (grayscale stand-in for the red tint).
  * Screen-space enemy velocity is rendered as a translucent motion ghost
    displaced by one tick of motion (single-frame stand-in for frame pairs).
  * Self HP is a hearts-bar strip at the bottom of the HUD.
  * Distractor rectangles + per-pixel noise for robustness.

Label conventions (all in PERCEPTION_LAYOUT order, normalized as spec.py):
  * aim_err_yaw   > 0  => enemy is to the RIGHT of the crosshair (deg/180).
  * aim_err_pitch > 0  => enemy is BELOW the crosshair, i.e. the bot must
                          pitch down (increase pitch) to center it (deg/90).
  * bbox_height: enemy bbox height in px / FRAME_H.
  * rel_screen_vx/vy: enemy screen motion in px/tick divided by FRAME_W /
    FRAME_H respectively.
  * self_speed: blocks/tick, raw (sprint ~0.28).
  * When visible == 0 all enemy-derived labels (aim errors, bbox_height,
    hurt_flash, rel_screen_v*, enemy_on_ground) are 0.

Deterministic given a seed / Generator.
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from pvpbot.spec import FRAME_SHAPE, PERCEPTION_DIM, PERCEPTION_LAYOUT

# ---------------------------------------------------------------------------
# Camera / world constants (shared with adapter.py for inversion).
# ---------------------------------------------------------------------------
FRAME_H = FRAME_SHAPE[1]
FRAME_W = FRAME_SHAPE[2]
FOV_X_DEG = 96.0
PX_PER_DEG = FRAME_W / FOV_X_DEG          # ~1.1875 px/deg, both axes
FOV_Y_DEG = FRAME_H / PX_PER_DEG          # ~53.9 deg
EYE_HEIGHT = 1.62                          # camera height above feet, blocks
PLAYER_HEIGHT = 1.8
PLAYER_HALF_H = PLAYER_HEIGHT / 2.0        # 0.9
PLAYER_HALF_W = 0.3                        # 0.6-block-wide hitbox

_L = PERCEPTION_LAYOUT  # shorthand


def bbox_height_frac(dist: float) -> float:
    """Angular height of a 1.8-block player at `dist`, as fraction of FRAME_H."""
    dist = max(float(dist), 1e-3)
    ang_deg = math.degrees(2.0 * math.atan(PLAYER_HALF_H / dist))
    return ang_deg * PX_PER_DEG / FRAME_H


def dist_from_bbox_height(frac: float) -> float:
    """Inverse of bbox_height_frac. Clamped to a sane range."""
    frac = float(np.clip(frac, 0.02, 1.5))
    ang_deg = frac * FRAME_H / PX_PER_DEG
    d = PLAYER_HALF_H / math.tan(math.radians(ang_deg) / 2.0)
    return float(np.clip(d, 0.3, 30.0))


# ---------------------------------------------------------------------------
# Config: domain-randomization knobs.
# ---------------------------------------------------------------------------
@dataclass
class SynthConfig:
    # brightness ranges (0..1 grayscale)
    sky_lo: float = 0.55
    sky_hi: float = 0.88
    ground_lo: float = 0.22
    ground_hi: float = 0.50
    ground_noise: float = 0.10       # texture amplitude
    texture_cell: int = 3            # blocky-noise cell size, px
    target_lo: float = 0.55          # armor-ish light shades
    target_hi: float = 0.82
    armor_speckle: float = 0.06
    hurt_boost: float = 0.22         # grayscale stand-in for red flash
    ghost_alpha: float = 0.35
    frame_noise: float = 0.03        # per-pixel gaussian, post-render
    # scene sampling
    dist_lo: float = 1.5
    dist_hi: float = 8.0
    yaw_err_max: float = 70.0        # > FOV_X/2 so some targets are off-screen
    pitch_lo: float = -30.0
    pitch_hi: float = 40.0
    p_airborne: float = 0.3
    p_hurt: float = 0.25
    p_occluded: float = 0.08
    p_moving_self: float = 0.7
    max_self_speed: float = 0.3      # blocks/tick
    speed_blur_gain: float = 14.0    # blur kernel ~ 1 + 2*int(speed*gain)
    max_screen_vx: float = 4.0       # px/tick
    max_screen_vy: float = 2.5
    max_distractors: int = 4
    # visibility rule
    min_onscreen_frac: float = 0.30


# ---------------------------------------------------------------------------
# Scene parameters: the ground truth a single frame is rendered from.
# ---------------------------------------------------------------------------
@dataclass
class SceneParams:
    self_pitch_deg: float = 0.0      # >0 looking down
    yaw_err_deg: float = 0.0         # >0 enemy right of crosshair
    dist: float = 4.0                # blocks, camera -> enemy
    enemy_y: float = 0.0             # enemy feet height above ground (jump)
    self_hp: float = 20.0
    hurt_flash: bool = False
    screen_vx_px: float = 0.0        # enemy screen velocity, px/tick
    screen_vy_px: float = 0.0
    self_speed: float = 0.0          # blocks/tick
    occluded: bool = False
    n_distractors: int = 0
    draw_target: bool = True         # test hook: render scene minus enemy


def sample_scene(rng: np.random.Generator, cfg: SynthConfig) -> SceneParams:
    moving = rng.random() < cfg.p_moving_self
    return SceneParams(
        self_pitch_deg=float(rng.uniform(cfg.pitch_lo, cfg.pitch_hi)),
        yaw_err_deg=float(rng.uniform(-cfg.yaw_err_max, cfg.yaw_err_max)),
        dist=float(rng.uniform(cfg.dist_lo, cfg.dist_hi)),
        enemy_y=float(rng.uniform(0.2, 1.1)) if rng.random() < cfg.p_airborne else 0.0,
        self_hp=float(rng.integers(0, 21)),
        hurt_flash=bool(rng.random() < cfg.p_hurt),
        screen_vx_px=float(rng.uniform(-cfg.max_screen_vx, cfg.max_screen_vx)),
        screen_vy_px=float(rng.uniform(-cfg.max_screen_vy, cfg.max_screen_vy)),
        self_speed=float(rng.uniform(0.05, cfg.max_self_speed)) if moving else 0.0,
        occluded=bool(rng.random() < cfg.p_occluded),
        n_distractors=int(rng.integers(0, cfg.max_distractors + 1)),
    )


# ---------------------------------------------------------------------------
# Drawing helpers.
# ---------------------------------------------------------------------------
def _blend_rect(img: np.ndarray, x0: float, x1: float, y0: float, y1: float,
                val: float, alpha: float = 1.0) -> None:
    xi0, xi1 = max(int(round(x0)), 0), min(int(round(x1)), FRAME_W)
    yi0, yi1 = max(int(round(y0)), 0), min(int(round(y1)), FRAME_H)
    if xi1 <= xi0 or yi1 <= yi0:
        return
    if alpha >= 1.0:
        img[yi0:yi1, xi0:xi1] = val
    else:
        img[yi0:yi1, xi0:xi1] *= (1.0 - alpha)
        img[yi0:yi1, xi0:xi1] += alpha * val


def _draw_humanoid(img: np.ndarray, x_c: float, y_c: float, h_px: float,
                   w_px: float, base: float, rng: np.random.Generator,
                   cfg: SynthConfig, alpha: float = 1.0,
                   mask: Optional[np.ndarray] = None) -> None:
    """Head + torso + arms + legs rectangles centered on (x_c, y_c).
    When ``mask`` is given, every body rect is also painted 1.0 into it
    (single pass -- keeps the rng stream identical either way)."""
    top = y_c - h_px / 2.0
    head_h = 0.28 * h_px
    torso_h = 0.42 * h_px

    def rect(x0, x1, y0, y1, shade):
        _blend_rect(img, x0, x1, y0, y1, shade, alpha)
        if mask is not None:
            _blend_rect(mask, x0, x1, y0, y1, 1.0, 1.0)

    # head (slightly narrower, darker: skin, not armor)
    rect(x_c - 0.38 * w_px, x_c + 0.38 * w_px, top, top + head_h,
         base * 0.82)
    # torso (armor plate)
    rect(x_c - 0.36 * w_px, x_c + 0.36 * w_px,
         top + head_h, top + head_h + torso_h, base)
    # arms flanking the torso, a touch darker
    rect(x_c - 0.5 * w_px, x_c - 0.36 * w_px,
         top + head_h, top + head_h + 0.85 * torso_h, base * 0.9)
    rect(x_c + 0.36 * w_px, x_c + 0.5 * w_px,
         top + head_h, top + head_h + 0.85 * torso_h, base * 0.9)
    # two legs with a gap
    leg_top = top + head_h + torso_h
    rect(x_c - 0.34 * w_px, x_c - 0.06 * w_px, leg_top, top + h_px,
         base * 0.75)
    rect(x_c + 0.06 * w_px, x_c + 0.34 * w_px, leg_top, top + h_px,
         base * 0.75)
    # armor speckle over the torso region
    if alpha >= 1.0 and cfg.armor_speckle > 0:
        xi0 = max(int(round(x_c - 0.36 * w_px)), 0)
        xi1 = min(int(round(x_c + 0.36 * w_px)), FRAME_W)
        yi0 = max(int(round(top + head_h)), 0)
        yi1 = min(int(round(top + head_h + torso_h)), FRAME_H)
        if xi1 > xi0 and yi1 > yi0:
            img[yi0:yi1, xi0:xi1] += rng.normal(
                0.0, cfg.armor_speckle, (yi1 - yi0, xi1 - xi0)
            ).astype(np.float32)


def _draw_hearts(img: np.ndarray, hp: float) -> None:
    """10-cell hearts strip, half-heart granularity, bottom-left HUD."""
    hp = float(np.clip(hp, 0.0, 20.0))
    x0, y0 = 8, FRAME_H - 6
    img[y0 - 1:y0 + 3, x0 - 2:x0 + 32] = 0.08   # dark backing strip
    for i in range(10):
        cell_hp = hp - 2.0 * i
        cx = x0 + 3 * i
        if cell_hp >= 2.0:
            img[y0:y0 + 2, cx:cx + 2] = 0.95     # full heart
        elif cell_hp >= 1.0:
            img[y0:y0 + 2, cx:cx + 1] = 0.95     # half heart
        else:
            img[y0:y0 + 2, cx:cx + 2] = 0.25     # empty container


def _tint_hearts_red(rgb: np.ndarray, gray: np.ndarray) -> None:
    """Recolor the bright heart cells red in the color frame (real 1.8
    hearts are red; the luminance render keeps their geometry)."""
    x0, y0 = 8, FRAME_H - 6
    strip = gray[y0:y0 + 2, x0 - 2:x0 + 32]
    lit = strip > 0.6                            # full/half heart pixels
    region = rgb[y0:y0 + 2, x0 - 2:x0 + 32]
    # bright red, G/B kept at container level so full-vs-empty contrast
    # stays positive in every channel (real hearts are vivid, not dark)
    region[..., 0][lit] = 0.92
    region[..., 1][lit] = 0.28
    region[..., 2][lit] = 0.28


def _vertical_box_blur(img: np.ndarray, k: int) -> np.ndarray:
    """Odd-k vertical box blur via cumsum. Returns a new array."""
    m = k // 2
    padded = np.pad(img, ((m, m), (0, 0)), mode="edge")
    cs = np.vstack([np.zeros((1, img.shape[1]), np.float32),
                    np.cumsum(padded, axis=0, dtype=np.float32)])
    return (cs[k:] - cs[:-k]) / float(k)


# ---------------------------------------------------------------------------
# Renderer.
# ---------------------------------------------------------------------------
def render_scene(
    params: SceneParams,
    rng: np.random.Generator,
    cfg: Optional[SynthConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Render one frame. Returns (frame uint8 (FRAME_H, FRAME_W, 3), label f32 (12,)).

    RNG draw order is stable for a given (params.n_distractors, occluded,
    draw_target) so `draw_target=False` re-renders the identical background.
    """
    if cfg is None:
        cfg = SynthConfig()
    chan_gain = rng.uniform(0.96, 1.04, (1, 1, 3)).astype(np.float32)
    img = np.empty((FRAME_H, FRAME_W), dtype=np.float32)
    p = params.self_pitch_deg

    # --- sky + ground split by horizon (pitch>0 = looking down = horizon up)
    y_h = FRAME_H / 2.0 - p * PX_PER_DEG
    sky_base = float(rng.uniform(cfg.sky_lo, cfg.sky_hi))
    ground_base = float(rng.uniform(cfg.ground_lo, cfg.ground_hi))
    grad = np.linspace(0.05, -0.05, FRAME_H, dtype=np.float32)[:, None]
    img[:] = sky_base + grad
    cell = max(cfg.texture_cell, 1)
    th, tw = FRAME_H // cell + 1, FRAME_W // cell + 1
    tex = rng.uniform(-1.0, 1.0, (th, tw)).astype(np.float32)
    tex = np.repeat(np.repeat(tex, cell, axis=0), cell, axis=1)[:FRAME_H, :FRAME_W]
    gmask = np.arange(FRAME_H, dtype=np.float32) >= y_h
    if gmask.any():
        ground = ground_base + cfg.ground_noise * tex
        img[gmask] = ground[gmask]

    # --- own-motion blur on the ground (optic-flow stand-in for self_speed)
    k = 1 + 2 * int(round(params.self_speed * cfg.speed_blur_gain))
    if k > 1 and gmask.any():
        blurred = _vertical_box_blur(img, k)
        img[gmask] = blurred[gmask]

    # --- distractor rectangles (drawn under the target)
    for _ in range(params.n_distractors):
        dx = int(rng.integers(0, FRAME_W))
        dw = int(rng.integers(3, 20))
        dy = int(rng.integers(0, FRAME_H - 8))
        dh = int(rng.integers(3, 16))
        shade = float(rng.uniform(0.05, 0.95))
        _blend_rect(img, dx, dx + dw, dy, dy + dh, shade)

    # --- enemy geometry (down-positive angles)
    dist = max(params.dist, 0.5)
    center_h = params.enemy_y + PLAYER_HALF_H
    ang_center_down = math.degrees(math.atan2(EYE_HEIGHT - center_h, dist))
    pitch_err_deg = ang_center_down - p
    x_c = FRAME_W / 2.0 + params.yaw_err_deg * PX_PER_DEG
    y_c = FRAME_H / 2.0 + pitch_err_deg * PX_PER_DEG
    h_px = bbox_height_frac(dist) * FRAME_H
    w_px = math.degrees(2.0 * math.atan(PLAYER_HALF_W / dist)) * PX_PER_DEG

    # on-screen fraction of the bbox
    bx0, bx1 = x_c - w_px / 2.0, x_c + w_px / 2.0
    by0, by1 = y_c - h_px / 2.0, y_c + h_px / 2.0
    ow = max(0.0, min(bx1, FRAME_W) - max(bx0, 0.0))
    oh = max(0.0, min(by1, FRAME_H) - max(by0, 0.0))
    onscreen = (ow * oh) / max(w_px * h_px, 1e-6)
    visible = (
        params.draw_target
        and not params.occluded
        and onscreen >= cfg.min_onscreen_frac
    )

    body_mask = None
    if params.draw_target and onscreen > 0.0:
        base = float(rng.uniform(cfg.target_lo, cfg.target_hi))
        # ground shadow under the enemy (feet/shadow gap encodes on_ground)
        ang_ground_down = math.degrees(math.atan2(EYE_HEIGHT, dist))
        y_shadow = FRAME_H / 2.0 + (ang_ground_down - p) * PX_PER_DEG
        _blend_rect(img, x_c - 0.45 * w_px, x_c + 0.45 * w_px,
                    y_shadow - 1.0, y_shadow + 1.0, 0.12, alpha=0.7)
        # one-tick motion ghost (screen-velocity cue)
        if abs(params.screen_vx_px) + abs(params.screen_vy_px) > 1e-6:
            _draw_humanoid(img, x_c - params.screen_vx_px,
                           y_c - params.screen_vy_px, h_px, w_px,
                           base * 0.92, rng, cfg, alpha=cfg.ghost_alpha)
        body_mask = np.zeros_like(img)
        _draw_humanoid(img, x_c, y_c, h_px, w_px, base, rng, cfg,
                       alpha=1.0, mask=body_mask)

    # --- occluder: a big rectangle slammed over the enemy
    if params.occluded:
        pad = float(rng.uniform(2.0, 5.0))
        shade = float(rng.uniform(0.15, 0.85))
        _blend_rect(img, bx0 - pad, bx1 + pad, by0 - pad, by1 + pad, shade)

    # --- HUD hearts + sensor noise
    _draw_hearts(img, params.self_hp)
    if cfg.frame_noise > 0:
        img += rng.normal(0.0, cfg.frame_noise,
                          (FRAME_H, FRAME_W)).astype(np.float32)

    # color output. Two source-grounded color facts matter here:
    #  1. the opponent wears DIAMOND armor -- cyan-blue texels, so the
    #     enemy body gets an armor tint (not gray);
    #  2. RendererLivingEntity (MCP-919 1.8.9) blends RGBA(1, 0, 0, 0.3)
    #     over the entity texels while hurtTime > 0. On a CYAN base that
    #     yields a desaturated purple -- red-minus-blue stays NEGATIVE.
    #     v12's failure taught this: a red-excess cue trained on gray
    #     bodies cannot fire on diamond armor. Render the true chain:
    #     armor color first, hurt blend second.
    gray = np.clip(img, 0.0, 1.0)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    if body_mask is not None:
        m = np.clip(body_mask, 0.0, 1.0)
        # diamond-armor tint, jittered (lighting/biome variation)
        col = np.array([0.50, 0.78, 0.96], dtype=np.float32) \
            * rng.uniform(0.9, 1.1, 3).astype(np.float32)
        for c in range(3):
            rgb[..., c] = (1.0 - m) * rgb[..., c] + m * np.clip(
                gray * col[c] * 1.25, 0.0, 1.0)
        if params.hurt_flash:
            # exact hurt blend over the already-colored texels
            rgb[..., 0] = (1.0 - m) * rgb[..., 0] + m * (
                0.7 * rgb[..., 0] + 0.3)
            rgb[..., 1] = (1.0 - m) * rgb[..., 1] + m * (0.7 * rgb[..., 1])
            rgb[..., 2] = (1.0 - m) * rgb[..., 2] + m * (0.7 * rgb[..., 2])
    # red HUD hearts (real 1.8 hearts are red; gray hearts starved the
    # hp head of its channel cue)
    _tint_hearts_red(rgb, gray)
    # mild per-channel gain jitter so "any color difference" alone is not
    # a trivially fake separator (real capture has color casts); sampled
    # at render entry to keep same-seed A/B renders comparable
    rgb *= chan_gain
    frame = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

    # --- label
    lab = np.zeros(PERCEPTION_DIM, dtype=np.float32)
    lab[_L["self_pitch"]] = p / 90.0
    lab[_L["self_hp"]] = params.self_hp / 20.0
    lab[_L["self_speed"]] = params.self_speed
    if visible:
        lab[_L["aim_err_yaw"]] = params.yaw_err_deg / 180.0
        lab[_L["aim_err_pitch"]] = pitch_err_deg / 90.0
        lab[_L["bbox_height"]] = h_px / FRAME_H
        lab[_L["visible"]] = 1.0
        lab[_L["hurt_flash"]] = 1.0 if params.hurt_flash else 0.0
        lab[_L["rel_screen_vx"]] = params.screen_vx_px / FRAME_W
        lab[_L["rel_screen_vy"]] = params.screen_vy_px / FRAME_H
        lab[_L["enemy_on_ground"]] = 1.0 if params.enemy_y <= 1e-6 else 0.0
    return frame, lab


def generate_frame(
    rng: np.random.Generator, cfg: Optional[SynthConfig] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a random scene and render it."""
    if cfg is None:
        cfg = SynthConfig()
    return render_scene(sample_scene(rng, cfg), rng, cfg)


def generate_stacked_batch(
    n: int,
    stack: int,
    seed: Optional[int] = None,
    cfg: Optional[SynthConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Motion-consistent temporal windows: (n, stack*C, H, W) + labels at t.

    Each window renders the SAME sampled scene at t-(stack-1) .. t, moving
    the enemy backwards along its labeled screen velocity (yaw via
    PX_PER_DEG, height via the px-per-block at the scene's distance). Every
    sub-frame re-renders with an identical child RNG so noise, distractors
    and textures are frozen -- the only inter-frame change is true motion,
    which is exactly the signal a single frame cannot carry (v10:
    single-frame nets plateau ~20 deg aim error on fast strafers).
    """
    if rng is None:
        rng = np.random.default_rng(0 if seed is None else seed)
    if cfg is None:
        cfg = SynthConfig()
    c = FRAME_SHAPE[0]
    frames = np.empty((n, stack * c, FRAME_H, FRAME_W), dtype=np.uint8)
    labels = np.empty((n, PERCEPTION_DIM), dtype=np.float32)
    for i in range(n):
        scene = sample_scene(rng, cfg)
        sub_seed = int(rng.integers(0, 2 ** 31 - 1))
        frac = bbox_height_frac(scene.dist)
        blocks_per_px = 1.8 / max(frac * FRAME_H, 1e-6)
        lab = None
        for k in range(stack):
            back = stack - 1 - k
            s = SceneParams(**{**scene.__dict__})
            s.yaw_err_deg = scene.yaw_err_deg - back * (
                scene.screen_vx_px / PX_PER_DEG)
            s.enemy_y = max(0.0, scene.enemy_y + back
                            * scene.screen_vy_px * blocks_per_px)
            f, l = render_scene(s, np.random.default_rng(sub_seed), cfg)
            frames[i, k * c:(k + 1) * c] = f.transpose(2, 0, 1)
            if back == 0:
                lab = l
        labels[i] = lab
    return frames, labels


def generate_batch(
    n: int,
    seed: Optional[int] = None,
    cfg: Optional[SynthConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """(frames uint8 (n, C, H, W), labels f32 (n, PERCEPTION_DIM)).

    Pass either a seed (fresh deterministic stream) or an existing Generator
    (streamed training batches).
    """
    if rng is None:
        rng = np.random.default_rng(0 if seed is None else seed)
    if cfg is None:
        cfg = SynthConfig()
    c = FRAME_SHAPE[0]
    frames = np.empty((n, c, FRAME_H, FRAME_W), dtype=np.uint8)
    labels = np.empty((n, PERCEPTION_DIM), dtype=np.float32)
    for i in range(n):
        f, l = generate_frame(rng, cfg)
        # scene is luminance-based but the output is true RGB now: the
        # hurt overlay carries the source-exact red channel signature
        frames[i] = f.transpose(2, 0, 1)
        labels[i] = l
    return frames, labels
