"""S6 smoke — 3-tier tower reward (iter 33, LATCHED SHAPING / STRICT SPARSE).

Iter 33 STRUCTURAL CHANGE: cube_0 SHAPING terms rebound to LATCHED-gated
variants. The latch is a per-env boolean stored in module-level
`_LATCH_BUFFERS`; it starts False at episode start (when
`env.episode_length_buf <= 1`), flips True when cube_1 is TRULY stacked on
cube_2 at any point during the episode, and STAYS True for the rest of the
episode. Gives the policy ~150-200 frames of cube_0 shaping signal per
stage-1-success episode (7-10x the ~21-frame strict instantaneous gate).

The iter-32 shortcut (lift cube_1 transiently to unlock cube_0 shaping) is
BLOCKED: the latch requires a true stack, not just a lift. Sparse bonuses
(`cube_0_stacked_bonus`, `full_tower_bonus`) STAY on the strict
instantaneous gate, so the policy cannot collect +200/+500 events without
re-stacking cube_1.

  | term                          | iter-32 binding                       | iter-33 binding                          |
  |-------------------------------|---------------------------------------|------------------------------------------|
  | reaching_cube_0               | mdp.cube_0_ee_distance_loose_gated    | mdp.cube_0_ee_distance_latched_gated     |
  | lifting_cube_0                | mdp.cube_0_is_lifted_loose_gated      | mdp.cube_0_is_lifted_latched_gated       |
  | cube_0_goal_tracking_coarse   | mdp.cube_0_goal_distance_loose_staged | mdp.cube_0_goal_distance_latched_staged  |
  | cube_0_goal_tracking_fine     | mdp.cube_0_goal_distance_loose_staged | mdp.cube_0_goal_distance_latched_staged  |
  | cube_0_stacked_bonus          | mdp.cube_0_stacked_bonus_gated        | UNCHANGED (STRICT)                       |
  | full_tower_bonus              | mdp.three_tier_tower_bonus            | UNCHANGED (STRICT)                       |

  Weights UNCHANGED (symmetric across stages, iter-31/32 values). Term count: 13.

Verifies:
  1. Env builds with exactly 13-term RewardsCfg in the expected order.
  2. Weight table unchanged from iter 31/32 (symmetric design).
  3. iter-33 PRIMARY: cube_0 SHAPING bindings point at the new LATCHED helpers
     (suffix `_latched_gated` / `_latched_staged`); SPARSE bonuses still strict.
  4. Rewards finite + non-constant over 30 steps at 128 envs.
  5. Reset behavior: at `episode_length_buf <= 1` the latch buffer is False
     for all envs → cube_0 shaping returns 0.
  6. Latch TRIGGER: teleport cube_1 ONTO cube_2 (cube_0 lifted too) → latch
     indicator becomes 1.0 → cube_0 shaping FIRES.
  7. Latch PERSISTENCE: teleport cube_1 OFF cube_2 (back near table) but
     KEEP cube_0 lifted → latch STAYS True → cube_0 shaping STILL FIRES,
     while `cube_0_stacked_bonus_gated` (strict) returns 0 (cube_1 not on
     cube_2 right now).
  8. Latch RESET on episode end: manually zero `episode_length_buf` for half
     the envs → next call to the latch helper resets those envs' latch to
     False → cube_0 shaping returns 0 for them again.
  9. Composer-sum: per-term `info["detailed_reward"]` sums to total reward
     when present; else passthrough.
"""
import argparse
import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args([])
args.headless = True
args.enable_cameras = False
sim_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch              # noqa: E402
import isaaclab_tasks    # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _to_np(t):
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


cfg = parse_env_cfg("Triton-Franka-StackCube", device="cuda:0", num_envs=128, use_fabric=True)
env = gym.make("Triton-Franka-StackCube", cfg=cfg)
obs, info = env.reset(seed=0)

unwrapped = env.unwrapped
rew_terms = list(unwrapped.reward_manager.active_terms)
EXPECTED_TERMS = [
    # stage 1: cube_1 -> cube_2 (UNCHANGED)
    "reaching_cube_1",
    "lifting_cube_1",
    "cube_1_goal_tracking_coarse",
    "cube_1_goal_tracking_fine",
    "cube_1_stacked_bonus",
    # stage 2: cube_0 -> cube_1 (iter 33: shaping rebound to LATCHED helpers)
    "reaching_cube_0",
    "lifting_cube_0",
    "cube_0_goal_tracking_coarse",
    "cube_0_goal_tracking_fine",
    "cube_0_stacked_bonus",
    # full-tower alignment (STRICT, unchanged)
    "full_tower_bonus",
    # regularizers
    "action_rate",
    "joint_vel",
]
assert rew_terms == EXPECTED_TERMS, f"reward term mismatch:\n  actual={rew_terms}\n  expected={EXPECTED_TERMS}"
assert len(rew_terms) == 13, f"iter 33: expected exactly 13 terms (UNCHANGED from iter 31/32), got {len(rew_terms)}"
print(f"S6 OK: 13 reward terms in expected order: {rew_terms}")

# -----------------------------------------------------------------------------
# Weight table (UNCHANGED from iter 31/32 — symmetric across stages).
# -----------------------------------------------------------------------------
def _w(name):
    return float(unwrapped.reward_manager.get_term_cfg(name).weight)

ITER33_EXPECTED = {
    # stage 1 (cube_1) — unchanged
    "reaching_cube_1":             0.02,
    "lifting_cube_1":              0.1,
    "cube_1_goal_tracking_coarse": 1.0,
    "cube_1_goal_tracking_fine":   0.5,
    "cube_1_stacked_bonus":      200.0,
    # stage 2 (cube_0) — MIRROR stage 1 (weights unchanged; bindings changed)
    "reaching_cube_0":             0.02,
    "lifting_cube_0":              0.1,
    "cube_0_goal_tracking_coarse": 1.0,
    "cube_0_goal_tracking_fine":   0.5,
    "cube_0_stacked_bonus":      200.0,
    # full tower — strict, unchanged
    "full_tower_bonus":          500.0,
    # regularizers
    "action_rate":               -2e-6,
    "joint_vel":                 -2e-6,
}
for name, expected in ITER33_EXPECTED.items():
    actual = _w(name)
    if abs(expected) > 1e-3:
        assert abs(actual - expected) < 1e-6, (
            f"iter-33 weight mismatch for {name}: expected {expected}, got {actual}"
        )
    else:
        assert abs(actual - expected) < 1e-9, (
            f"iter-33 weight mismatch for {name}: expected {expected}, got {actual}"
        )
# Symmetric equality checks (unchanged from iter 31/32).
assert _w("cube_1_stacked_bonus") == _w("cube_0_stacked_bonus") == 200.0
assert _w("cube_1_goal_tracking_coarse") == _w("cube_0_goal_tracking_coarse") == 1.0
assert _w("cube_1_goal_tracking_fine") == _w("cube_0_goal_tracking_fine") == 0.5
assert _w("full_tower_bonus") == 500.0
print(
    "iter-33 weight table OK: weights identical to iter 31/32 "
    "(symmetric stage shaping); structural change is binding only"
)

# -----------------------------------------------------------------------------
# iter-33 PRIMARY: binding check. cube_0 SHAPING -> LATCHED helpers; SPARSE STRICT.
# -----------------------------------------------------------------------------
reach0_term = unwrapped.reward_manager.get_term_cfg("reaching_cube_0")
lift0_term  = unwrapped.reward_manager.get_term_cfg("lifting_cube_0")
coarse_c0_term = unwrapped.reward_manager.get_term_cfg("cube_0_goal_tracking_coarse")
fine_c0_term   = unwrapped.reward_manager.get_term_cfg("cube_0_goal_tracking_fine")
c0_bonus_term  = unwrapped.reward_manager.get_term_cfg("cube_0_stacked_bonus")
c1_bonus_term  = unwrapped.reward_manager.get_term_cfg("cube_1_stacked_bonus")
tower_term     = unwrapped.reward_manager.get_term_cfg("full_tower_bonus")

assert reach0_term.func.__name__ == "cube_0_ee_distance_latched_gated", (
    f"iter-33: reaching_cube_0 must bind to cube_0_ee_distance_latched_gated, "
    f"got {reach0_term.func.__name__}"
)
assert lift0_term.func.__name__ == "cube_0_is_lifted_latched_gated", (
    f"iter-33: lifting_cube_0 must bind to cube_0_is_lifted_latched_gated, "
    f"got {lift0_term.func.__name__}"
)
assert coarse_c0_term.func.__name__ == "cube_0_goal_distance_latched_staged", (
    f"iter-33: cube_0_goal_tracking_coarse must bind to cube_0_goal_distance_latched_staged, "
    f"got {coarse_c0_term.func.__name__}"
)
assert fine_c0_term.func.__name__ == "cube_0_goal_distance_latched_staged", (
    f"iter-33: cube_0_goal_tracking_fine must bind to cube_0_goal_distance_latched_staged, "
    f"got {fine_c0_term.func.__name__}"
)
# SPARSE STILL STRICT
assert c0_bonus_term.func.__name__ == "cube_0_stacked_bonus_gated", (
    f"iter-33: cube_0_stacked_bonus must STAY on cube_0_stacked_bonus_gated (STRICT), "
    f"got {c0_bonus_term.func.__name__}"
)
assert tower_term.func.__name__ == "three_tier_tower_bonus", (
    f"iter-33: full_tower_bonus must STAY on three_tier_tower_bonus (STRICT), "
    f"got {tower_term.func.__name__}"
)
assert c1_bonus_term.func.__name__ == "cube_1_stacked_on_cube_2_bonus", (
    f"cube_1_stacked_bonus binding mismatch, got {c1_bonus_term.func.__name__}"
)
assert abs(float(coarse_c0_term.params["std"]) - 0.3) < 1e-9
assert abs(float(fine_c0_term.params["std"]) - 0.05) < 1e-9
print(
    "iter-33 bindings OK: cube_0 SHAPING -> latched helpers (reach_latched_gated, "
    "lift_latched_gated, goal_distance_latched_staged std=0.3 and 0.05); "
    "cube_0_stacked_bonus + full_tower_bonus stay STRICT-gated"
)

# -----------------------------------------------------------------------------
# Reward finiteness + non-constancy + composer sanity (when info has detail).
# -----------------------------------------------------------------------------
step_means = []
term_sums = {}
saw_detailed = False
all_finite = True

for _ in range(30):
    a = env.action_space.sample()
    a = torch.as_tensor(a, device="cuda:0")
    obs, r, term, trunc, info = env.step(a)

    r_np = _to_np(r).reshape(-1)
    if not np.all(np.isfinite(r_np)):
        all_finite = False
    step_means.append(float(r_np.mean()))

    detailed = info.get("detailed_reward")
    if detailed is not None:
        saw_detailed = True
        per_term_np = {k: _to_np(v).reshape(-1) for k, v in detailed.items()}
        composed = np.zeros_like(r_np)
        for v in per_term_np.values():
            composed = composed + v
        assert np.allclose(composed, r_np, atol=1e-5), (
            "composer mismatch (some env): "
            f"max|sum(terms) - reward|={np.max(np.abs(composed - r_np)):.6f} "
            f"terms={list(per_term_np)}"
        )
        for k, v in per_term_np.items():
            term_sums[k] = term_sums.get(k, 0.0) + float(v.mean())

assert all_finite, "NaN or Inf in reward at 128 envs over 30 steps"
arr = np.asarray(step_means)
assert arr.std() > 0, f"reward (mean across envs) is constant over 30 steps (mean={arr.mean()})"

# -----------------------------------------------------------------------------
# Per-function sanity: each helper returns a finite (num_envs,) tensor at 128 envs.
# -----------------------------------------------------------------------------
from isaaclab_tasks.manager_based.manipulation.stack_cube.mdp import rewards as r_mod  # noqa: E402

num_envs = unwrapped.num_envs

for fn_name, kw in [
    ("cube_1_ee_distance", {"std": 0.1}),
    ("cube_1_is_lifted", {"minimal_height": 0.04}),
    ("cube_1_goal_distance", {"std": 0.3, "minimal_height": 0.04}),
    ("cube_1_goal_distance", {"std": 0.05, "minimal_height": 0.04}),
    ("cube_1_stacked_on_cube_2_bonus", {"xy_threshold": 0.02, "z_threshold": 0.01}),
    # iter-33 new LATCHED helpers
    ("cube_1_was_stacked_latched_indicator", {"xy_threshold": 0.02, "z_threshold": 0.01}),
    ("cube_0_ee_distance_latched_gated", {"std": 0.1}),
    ("cube_0_is_lifted_latched_gated", {"minimal_height": 0.04}),
    ("cube_0_goal_distance_latched_staged", {"std": 0.3, "minimal_height": 0.04}),
    ("cube_0_goal_distance_latched_staged", {"std": 0.05, "minimal_height": 0.04}),
    # strict helpers still present (kept for rollback)
    ("cube_0_goal_distance_staged", {"std": 0.3, "minimal_height": 0.04, "xy_threshold": 0.02, "z_threshold": 0.01}),
    ("cube_0_stacked_bonus_gated", {"xy_threshold": 0.02, "z_threshold": 0.01}),
    ("three_tier_tower_bonus", {"xy_threshold": 0.02, "z_threshold": 0.01}),
    ("cube_1_on_cube_2_indicator", {"xy_threshold": 0.02, "z_threshold": 0.01}),
]:
    fn = getattr(r_mod, fn_name)
    v = fn(unwrapped, **kw)
    assert v.shape == (num_envs,), f"{fn_name} bad shape {tuple(v.shape)}"
    assert torch.isfinite(v).all(), f"{fn_name} not finite"
print("per-function shape/finiteness: OK (includes iter-33 latched helpers)")

# -----------------------------------------------------------------------------
# iter-33 PRIMARY: latched-gate behavior — the bootstrap window check.
#
# Force a clean state via env.reset() so episode_length_buf is well-defined.
# After reset, env.step() increments episode_length_buf to 1 BEFORE reward
# computation; for the smoke we call helpers directly, so we manually mirror
# that: force episode_length_buf == 1 to make the latch reset condition apply.
# -----------------------------------------------------------------------------
obs, info = env.reset(seed=42)
# Force per-step counter to 1 (the value at the first reward computation).
unwrapped.episode_length_buf[:] = 1

# Clear any stale latch state from earlier 30-step sample loop. Direct fresh
# reads from `_get_latch_buffer` will re-pull, then the reset branch clears.
# We just need to invoke the indicator once to trigger the just_reset path.
v_latch_post_reset = r_mod.cube_1_was_stacked_latched_indicator(unwrapped)

cube_0 = unwrapped.scene["cube_0"]
cube_1 = unwrapped.scene["cube_1"]
cube_2 = unwrapped.scene["cube_2"]

# State A — fresh reset. cube_1 on the table (NOT stacked on cube_2); latch is False.
# (Verify by reading the indicator AFTER the just_reset clear path.)
unwrapped.episode_length_buf[:] = 1
# Also force cube_1 to be off cube_2 so the now_stacked check fails this call:
# (use scene root_pos_w — same as cube_1.data.root_pos_w underlying buffer).
# Save reset state first.
saved_p0_A = cube_0.data.root_pos_w[:, :3].clone()
saved_p1_A = cube_1.data.root_pos_w[:, :3].clone()
saved_p2_A = cube_2.data.root_pos_w[:, :3].clone()
# Place cube_1 off-stack (lateral offset from cube_2) to ensure now_stacked == 0
cube_1.data.root_pos_w[:, 0] = saved_p2_A[:, 0] + 0.30
cube_1.data.root_pos_w[:, 1] = saved_p2_A[:, 1] + 0.30
cube_1.data.root_pos_w[:, 2] = saved_p2_A[:, 2]  # at table top, NOT lifted

v_latch_A = r_mod.cube_1_was_stacked_latched_indicator(unwrapped)
assert torch.all(v_latch_A == 0.0), (
    f"iter-33 state A: latch should be False at reset (episode_length_buf==1, cube_1 not on cube_2), "
    f"max={float(v_latch_A.max()):.6f}"
)
v_reach0_A = r_mod.cube_0_ee_distance_latched_gated(unwrapped, std=0.1)
v_lift0_A  = r_mod.cube_0_is_lifted_latched_gated(unwrapped, minimal_height=0.04)
v_coarse_A = r_mod.cube_0_goal_distance_latched_staged(unwrapped, std=0.3, minimal_height=0.04)
v_fine_A   = r_mod.cube_0_goal_distance_latched_staged(unwrapped, std=0.05, minimal_height=0.04)
assert torch.all(v_reach0_A == 0.0), (
    f"iter-33 state A: cube_0_ee_distance_latched_gated must be 0 at reset (latch off), "
    f"max={float(v_reach0_A.max()):.6f}"
)
assert torch.all(v_lift0_A == 0.0), (
    f"iter-33 state A: cube_0_is_lifted_latched_gated must be 0 at reset (latch off), "
    f"max={float(v_lift0_A.max()):.6f}"
)
assert torch.all(v_coarse_A == 0.0), (
    f"iter-33 state A: cube_0_goal_distance_latched_staged (coarse) must be 0 at reset, "
    f"max={float(v_coarse_A.max()):.6f}"
)
assert torch.all(v_fine_A == 0.0), (
    f"iter-33 state A: cube_0_goal_distance_latched_staged (fine) must be 0 at reset, "
    f"max={float(v_fine_A.max()):.6f}"
)
print("iter-33 state A (reset, latch False): all 4 cube_0 latched shaping helpers == 0: OK")

# -----------------------------------------------------------------------------
# State B — TRIGGER: teleport cube_1 ONTO cube_2 (true stack), cube_0 lifted.
#   Bump episode_length_buf > 1 first so the latch DOES NOT reset; then call
#   the indicator to fire `now_stacked` and SET the latch True for all envs.
# -----------------------------------------------------------------------------
unwrapped.episode_length_buf[:] = 5  # past the reset edge

CUBE_SIZE = r_mod.CUBE_SIZE
# Perfect stack: cube_1 directly on cube_2.
cube_1.data.root_pos_w[:, 0] = saved_p2_A[:, 0]
cube_1.data.root_pos_w[:, 1] = saved_p2_A[:, 1]
cube_1.data.root_pos_w[:, 2] = saved_p2_A[:, 2] + CUBE_SIZE
# cube_0 also lifted, sit 0.10 lateral from cube_1.
cube_0.data.root_pos_w[:, 0] = cube_1.data.root_pos_w[:, 0] + 0.10
cube_0.data.root_pos_w[:, 1] = cube_1.data.root_pos_w[:, 1]
cube_0.data.root_pos_w[:, 2] = 0.20  # well above 0.04

# Fire the indicator — this should SET the latch via the now_stacked branch.
v_latch_B = r_mod.cube_1_was_stacked_latched_indicator(unwrapped)
assert torch.all(v_latch_B == 1.0), (
    f"iter-33 state B: latch should become True after teleporting cube_1 onto cube_2, "
    f"min={float(v_latch_B.min()):.6f}"
)
# cube_0 shaping must FIRE now.
v_reach0_B = r_mod.cube_0_ee_distance_latched_gated(unwrapped, std=0.1)
v_lift0_B  = r_mod.cube_0_is_lifted_latched_gated(unwrapped, minimal_height=0.04)
v_coarse_B = r_mod.cube_0_goal_distance_latched_staged(unwrapped, std=0.3, minimal_height=0.04)
v_fine_B   = r_mod.cube_0_goal_distance_latched_staged(unwrapped, std=0.05, minimal_height=0.04)
assert float(v_reach0_B.max()) > 0.0, (
    f"iter-33 state B: cube_0_ee_distance_latched_gated should FIRE (latch True), "
    f"max={float(v_reach0_B.max()):.6f}"
)
assert torch.all(v_lift0_B == 1.0), (
    f"iter-33 state B: cube_0_is_lifted_latched_gated should be 1 (cube_0 lifted, latch True), "
    f"min={float(v_lift0_B.min()):.6f}"
)
assert torch.all(v_coarse_B > 0.0), (
    f"iter-33 state B: cube_0_goal_distance_latched_staged (coarse) should fire, "
    f"min={float(v_coarse_B.min()):.6f}"
)
assert torch.all(v_fine_B > 0.0), (
    f"iter-33 state B: cube_0_goal_distance_latched_staged (fine) should fire, "
    f"min={float(v_fine_B.min()):.6f}"
)
print(
    "iter-33 state B (cube_1 STACKED on cube_2, cube_0 lifted): "
    "latch=True, all 4 cube_0 latched shaping helpers FIRE: OK"
)

# -----------------------------------------------------------------------------
# State C — PERSISTENCE: teleport cube_1 OFF cube_2 (back to table, NOT lifted).
#   cube_0 kept lifted. Latch was True in state B — it must STAY True even
#   though cube_1 is no longer on cube_2. cube_0 shaping STILL fires; STRICT
#   sparse `cube_0_stacked_bonus_gated` returns 0 (no current stack).
# -----------------------------------------------------------------------------
unwrapped.episode_length_buf[:] = 6  # still > 1, no reset
# Move cube_1 OFF cube_2 (lateral offset + back to table top z).
cube_1.data.root_pos_w[:, 0] = saved_p2_A[:, 0] + 0.30
cube_1.data.root_pos_w[:, 1] = saved_p2_A[:, 1] + 0.30
cube_1.data.root_pos_w[:, 2] = saved_p2_A[:, 2]  # at table top — NOT stacked
# cube_0 stays lifted (we don't touch it; preserved from state B).
# Re-firing the indicator: now_stacked == 0 BUT latch was already True.
v_latch_C = r_mod.cube_1_was_stacked_latched_indicator(unwrapped)
assert torch.all(v_latch_C == 1.0), (
    f"iter-33 state C: latch must STAY True after knock-off (persistence), "
    f"min={float(v_latch_C.min()):.6f}"
)
# Sanity: instantaneous (non-latched) cube_1_on_cube_2 is now 0.
v_c1_on_c2_C = r_mod.cube_1_on_cube_2_indicator(unwrapped, xy_threshold=0.02, z_threshold=0.01)
assert torch.all(v_c1_on_c2_C == 0.0), (
    f"iter-33 state C: instantaneous cube_1_on_cube_2_indicator should be 0 (knock-off), "
    f"max={float(v_c1_on_c2_C.max()):.6f}"
)
# cube_0 shaping STILL fires under the latch.
v_reach0_C = r_mod.cube_0_ee_distance_latched_gated(unwrapped, std=0.1)
v_lift0_C  = r_mod.cube_0_is_lifted_latched_gated(unwrapped, minimal_height=0.04)
v_coarse_C = r_mod.cube_0_goal_distance_latched_staged(unwrapped, std=0.3, minimal_height=0.04)
v_fine_C   = r_mod.cube_0_goal_distance_latched_staged(unwrapped, std=0.05, minimal_height=0.04)
assert float(v_reach0_C.max()) > 0.0, (
    f"iter-33 state C: cube_0_ee_distance_latched_gated should STILL fire (latch sticky), "
    f"max={float(v_reach0_C.max()):.6f}"
)
assert torch.all(v_lift0_C == 1.0), (
    f"iter-33 state C: cube_0_is_lifted_latched_gated should STILL be 1, "
    f"min={float(v_lift0_C.min()):.6f}"
)
assert torch.all(v_coarse_C > 0.0), (
    f"iter-33 state C: cube_0_goal_distance_latched_staged (coarse) should STILL fire, "
    f"min={float(v_coarse_C.min()):.6f}"
)
assert torch.all(v_fine_C > 0.0), (
    f"iter-33 state C: cube_0_goal_distance_latched_staged (fine) should STILL fire, "
    f"min={float(v_fine_C.min()):.6f}"
)
# STRICT sparse bonus stays 0 — proves shortcut blocked.
v_c0_strict_bonus_C = r_mod.cube_0_stacked_bonus_gated(unwrapped, xy_threshold=0.02, z_threshold=0.01)
v_tower_strict_C    = r_mod.three_tier_tower_bonus(unwrapped, xy_threshold=0.02, z_threshold=0.01)
assert torch.all(v_c0_strict_bonus_C == 0.0), (
    f"iter-33 state C: cube_0_stacked_bonus_gated must be 0 (strict gate, cube_1 not on cube_2), "
    f"max={float(v_c0_strict_bonus_C.max()):.6f}"
)
assert torch.all(v_tower_strict_C == 0.0), (
    f"iter-33 state C: three_tier_tower_bonus must be 0 (strict gate), "
    f"max={float(v_tower_strict_C.max()):.6f}"
)
print(
    "iter-33 state C (cube_1 knocked OFF cube_2, cube_0 still lifted):\n"
    "  latch STAYS True (sticky persistence)\n"
    "  cube_0 latched shaping STILL fires (gives policy the ~150-frame window)\n"
    "  cube_0_stacked_bonus_gated + full_tower_bonus STAY 0 (strict shortcut block)"
)

# -----------------------------------------------------------------------------
# State D — RESET: zero `episode_length_buf` for half the envs → next latch
# call must reset their latch back to False. The OTHER half keeps latch True.
# Scene state mirrors state C (cube_1 off cube_2), so once latch resets the
# cube_0 shaping for those envs returns 0 again.
# -----------------------------------------------------------------------------
half = num_envs // 2
unwrapped.episode_length_buf[:half] = 0   # 0 -> the just_reset edge fires (<= 1)
unwrapped.episode_length_buf[half:] = 7   # well past reset edge

v_latch_D = r_mod.cube_1_was_stacked_latched_indicator(unwrapped)
assert torch.all(v_latch_D[:half] == 0.0), (
    f"iter-33 state D: first half should have latch RESET to False, "
    f"max[:half]={float(v_latch_D[:half].max()):.6f}"
)
assert torch.all(v_latch_D[half:] == 1.0), (
    f"iter-33 state D: second half should KEEP latch True, "
    f"min[half:]={float(v_latch_D[half:].min()):.6f}"
)
v_reach0_D = r_mod.cube_0_ee_distance_latched_gated(unwrapped, std=0.1)
v_lift0_D  = r_mod.cube_0_is_lifted_latched_gated(unwrapped, minimal_height=0.04)
v_coarse_D = r_mod.cube_0_goal_distance_latched_staged(unwrapped, std=0.3, minimal_height=0.04)
v_fine_D   = r_mod.cube_0_goal_distance_latched_staged(unwrapped, std=0.05, minimal_height=0.04)
assert torch.all(v_reach0_D[:half] == 0.0), (
    f"iter-33 state D: first half cube_0 reaching must be 0 (latch reset), "
    f"max[:half]={float(v_reach0_D[:half].max()):.6f}"
)
assert torch.all(v_lift0_D[:half] == 0.0), (
    f"iter-33 state D: first half cube_0 lifting must be 0, "
    f"max[:half]={float(v_lift0_D[:half].max()):.6f}"
)
assert torch.all(v_coarse_D[:half] == 0.0), (
    f"iter-33 state D: first half coarse must be 0, "
    f"max[:half]={float(v_coarse_D[:half].max()):.6f}"
)
assert torch.all(v_fine_D[:half] == 0.0), (
    f"iter-33 state D: first half fine must be 0, "
    f"max[:half]={float(v_fine_D[:half].max()):.6f}"
)
# Second half still has latch True + cube_0 lifted → still fires.
assert float(v_reach0_D[half:].max()) > 0.0, (
    f"iter-33 state D: second half reaching should STILL fire, "
    f"max[half:]={float(v_reach0_D[half:].max()):.6f}"
)
assert torch.all(v_lift0_D[half:] == 1.0), (
    f"iter-33 state D: second half lifting should STILL fire, "
    f"min[half:]={float(v_lift0_D[half:].min()):.6f}"
)
print(
    "iter-33 state D (half envs episode_length_buf=0, half=7): "
    "latch RESETS for first half, STAYS for second half; "
    "cube_0 shaping correctly partitioned across envs"
)

# Restore cube positions for cleanliness.
cube_0.data.root_pos_w[:, :3] = saved_p0_A
cube_1.data.root_pos_w[:, :3] = saved_p1_A
cube_2.data.root_pos_w[:, :3] = saved_p2_A

# -----------------------------------------------------------------------------
# State E — PERFECT TOWER (post-latch-set): all gates open; weighted bonuses
# match iter-31/32 contributions.
# -----------------------------------------------------------------------------
unwrapped.episode_length_buf[:] = 10  # well past reset edge
saved_p0_e = cube_0.data.root_pos_w[:, :3].clone()
saved_p1_e = cube_1.data.root_pos_w[:, :3].clone()
saved_p2_e = cube_2.data.root_pos_w[:, :3].clone()

base_xy = saved_p2_e[:, :2]
base_z = saved_p2_e[:, 2]
cube_2.data.root_pos_w[:, 0] = base_xy[:, 0]
cube_2.data.root_pos_w[:, 1] = base_xy[:, 1]
cube_2.data.root_pos_w[:, 2] = base_z
cube_1.data.root_pos_w[:, 0] = base_xy[:, 0]
cube_1.data.root_pos_w[:, 1] = base_xy[:, 1]
cube_1.data.root_pos_w[:, 2] = base_z + CUBE_SIZE
cube_0.data.root_pos_w[:, 0] = base_xy[:, 0]
cube_0.data.root_pos_w[:, 1] = base_xy[:, 1]
cube_0.data.root_pos_w[:, 2] = base_z + 2.0 * CUBE_SIZE

# Fire the indicator to set the latch (perfect tower satisfies the stack check)
_ = r_mod.cube_1_was_stacked_latched_indicator(unwrapped)

v_tower    = r_mod.three_tier_tower_bonus(unwrapped, xy_threshold=0.02, z_threshold=0.01)
v_c1_bonus = r_mod.cube_1_stacked_on_cube_2_bonus(unwrapped, xy_threshold=0.02, z_threshold=0.01)
v_c0_gated = r_mod.cube_0_stacked_bonus_gated(unwrapped, xy_threshold=0.02, z_threshold=0.01)
assert torch.all(v_tower == 1.0), f"perfect tower: tower_bonus should be 1, mean={float(v_tower.mean()):.6f}"
assert torch.all(v_c1_bonus == 1.0), f"perfect tower: cube_1_bonus should be 1, mean={float(v_c1_bonus.mean()):.6f}"
assert torch.all(v_c0_gated == 1.0), f"perfect tower: cube_0_stacked_bonus_gated should be 1, mean={float(v_c0_gated.mean()):.6f}"

v_lift0_E  = r_mod.cube_0_is_lifted_latched_gated(unwrapped, minimal_height=0.04)
v_coarse_E = r_mod.cube_0_goal_distance_latched_staged(unwrapped, std=0.3, minimal_height=0.04)
v_fine_E   = r_mod.cube_0_goal_distance_latched_staged(unwrapped, std=0.05, minimal_height=0.04)
assert torch.all(v_lift0_E == 1.0), (
    f"perfect tower: cube_0_is_lifted_latched_gated should be 1, min={float(v_lift0_E.min()):.6f}"
)
assert torch.all(v_coarse_E > 0.99), (
    f"perfect tower: cube_0_goal_distance_latched_staged (coarse) should be ~1, "
    f"min={float(v_coarse_E.min()):.6f}"
)
assert torch.all(v_fine_E > 0.99), (
    f"perfect tower: cube_0_goal_distance_latched_staged (fine) should be ~1, "
    f"min={float(v_fine_E.min()):.6f}"
)

# Weighted contributions match iter-31/32 values (unchanged in iter 33).
c1_bonus_contribution = float(c1_bonus_term.weight) * float(v_c1_bonus.mean())
assert abs(c1_bonus_contribution - 200.0) < 1e-3, (
    f"perfect tower: cube_1_stacked_bonus contrib should be 200, got {c1_bonus_contribution:.6f}"
)
c0_bonus_contribution = float(c0_bonus_term.weight) * float(v_c0_gated.mean())
assert abs(c0_bonus_contribution - 200.0) < 1e-3, (
    f"perfect tower: cube_0_stacked_bonus contrib should be 200, got {c0_bonus_contribution:.6f}"
)
tower_contribution = float(tower_term.weight) * float(v_tower.mean())
assert abs(tower_contribution - 500.0) < 1e-3, (
    f"perfect tower: full_tower_bonus contrib should be 500, got {tower_contribution:.6f}"
)
lift0_contribution = float(lift0_term.weight) * float(v_lift0_E.mean())
assert abs(lift0_contribution - 0.1) < 1e-4, (
    f"perfect tower: lifting_cube_0 (latched) contrib should be 0.1, got {lift0_contribution:.6f}"
)
coarse_c0_contribution = float(coarse_c0_term.weight) * float(v_coarse_E.mean())
assert coarse_c0_contribution > 0.98, (
    f"perfect tower: cube_0_goal_tracking_coarse contrib should be ~1, got {coarse_c0_contribution:.6f}"
)
fine_c0_contribution = float(fine_c0_term.weight) * float(v_fine_E.mean())
assert fine_c0_contribution > 0.49, (
    f"perfect tower: cube_0_goal_tracking_fine contrib should be ~0.5, got {fine_c0_contribution:.6f}"
)

print(
    f"iter-33 perfect-tower contributions: "
    f"full_tower_bonus={tower_contribution:.1f}, "
    f"cube_0_stacked_bonus={c0_bonus_contribution:.1f} (STRICT-gated), "
    f"cube_1_stacked_bonus={c1_bonus_contribution:.1f}, "
    f"lifting_cube_0={lift0_contribution:.3f} (latched), "
    f"cube_0_goal_tracking_coarse={coarse_c0_contribution:.3f} (latched), "
    f"cube_0_goal_tracking_fine={fine_c0_contribution:.3f} (latched)"
)

# Restore for cleanup.
cube_0.data.root_pos_w[:, :3] = saved_p0_e
cube_1.data.root_pos_w[:, :3] = saved_p1_e
cube_2.data.root_pos_w[:, :3] = saved_p2_e

if saw_detailed:
    print("per-term episodic mean (across 128 envs, summed over 30 steps):")
    for k, v in sorted(term_sums.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {k}: {v:+.4f}")

print(
    f"S6 OK: 30 steps x 128 envs, reward mean={arr.mean():.3f} std={arr.std():.3f} "
    f"composer={'sum-verified' if saw_detailed else 'passthrough'} "
    f"latch-gates=OK (reset/trigger/persistence/per-env-reset) "
    f"iter-33 LATCHED-SHAPING/STRICT-SPARSE: cube_0 shaping unlocks for rest of episode "
    f"once cube_1 has been TRULY stacked at any point; sparse +200/+500 stay strict-gated"
)
env.close()
sim_app.close()
