"""
app/grader.py
Orbit — AI Space Mission Architect

Deterministic scoring for completed missions and dense step rewards.

Scoring philosophy:
  - All scoring is 100% deterministic — same inputs always produce the same score.
  - Scores are broken into weighted components so agents get meaningful signal.
  - Step rewards are dense (given every step) to help RL agents learn faster.
  - Terminal reward (on SubmitMission) is the full grade_mission() score.

Score component weights for grade_mission():
  Altitude accuracy   : 30%  ← most important — did you reach the right orbit?
  Fuel efficiency     : 30%  ← are you as good as a real engineer?
  Eccentricity        : 20%  ← is the orbit circular (as required)?
  Inclination         : 15%  ← are you in the right orbital plane?
  Step efficiency     : 5%   ← did you do it without wasting actions?
  Total               : 100%
"""

from __future__ import annotations

from typing import Dict

from physics import fuel_efficiency_ratio, proximity_score
from tasks import get_task


# ─────────────────────────────────────────────────────────────────────────────
# Component Weights
# (must sum to 1.0 — verified in the module-level assert below)
# ─────────────────────────────────────────────────────────────────────────────

WEIGHTS = {
    "altitude":    0.30,
    "efficiency":  0.30,
    "eccentricity":0.20,
    "inclination": 0.15,
    "steps":       0.05,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, \
    f"Score weights must sum to 1.0, got {sum(WEIGHTS.values())}"


# ─────────────────────────────────────────────────────────────────────────────
# Terminal Grader — called when the agent submits or times out
# ─────────────────────────────────────────────────────────────────────────────

def grade_mission(task_id: str, state: dict) -> Dict:
    """
    Grade a completed (or submitted) mission.

    Computes a score in [0.0, 1.0] from five weighted components.
    Also checks whether all success_criteria are met for mission_success flag.

    Args:
        task_id: Mission identifier string.
        state:   Full environment state dict (from State.model_dump()).
                 Required keys: current_orbit, delta_v_used, step_index.

    Returns:
        Dict with keys:
          score            (float 0.0–1.0)  — weighted total
          component_scores (dict)           — per-component weighted scores
          raw_scores       (dict)           — per-component 0–1 before weighting
          mission_success  (bool)           — True if all criteria met
          delta_v_used     (float)          — m/s spent by agent
          delta_v_optimal  (float)          — theoretical optimum m/s
          efficiency_ratio (float)          — optimal / actual (capped 1.0)
          steps_used       (int)
          errors           (dict)           — absolute errors vs target
    """
    task           = get_task(task_id)
    current_orbit  = state["current_orbit"]
    target_orbit   = task["target_orbit"]
    criteria       = task["success_criteria"]
    delta_v_used   = state["delta_v_used"]
    theoretical_dv = task["theoretical_delta_v"]
    steps_used     = state["step_index"]
    max_steps      = task["max_steps"]

    # Support both dict and Pydantic model for current_orbit
    if hasattr(current_orbit, "altitude_km"):
        alt = current_orbit.altitude_km
        ecc = current_orbit.eccentricity
        inc = current_orbit.inclination_deg
    else:
        alt = current_orbit["altitude_km"]
        ecc = current_orbit["eccentricity"]
        inc = current_orbit["inclination_deg"]

    t_alt = target_orbit["altitude_km"]
    t_ecc = target_orbit["eccentricity"]
    t_inc = target_orbit["inclination_deg"]

    # ── Raw scores (0.0–1.0 each, before weighting) ───────────────────────────

    # 1. Altitude accuracy
    #    Uses log-scale proximity for large-range missions (Moon, Bennu)
    #    so the agent isn't penalised linearly over 120,000,000 km.
    alt_tolerance = criteria["altitude_tolerance_km"]
    alt_raw = _altitude_score(alt, t_alt, alt_tolerance)

    # 2. Fuel efficiency — optimal Δ-v / actual Δ-v (capped at 1.0)
    eff_raw = fuel_efficiency_ratio(delta_v_used, theoretical_dv)

    # 3. Eccentricity accuracy
    ecc_tolerance = criteria["eccentricity_tolerance"]
    ecc_raw = proximity_score(ecc, t_ecc, ecc_tolerance)

    # 4. Inclination accuracy
    inc_tolerance = criteria["inclination_tolerance_deg"]
    inc_raw = proximity_score(inc, t_inc, inc_tolerance)

    # 5. Step efficiency — reward finishing faster than the step limit
    #    Score = 1.0 if done in 1 step, 0.0 if used all steps
    step_raw = max(0.0, 1.0 - (steps_used / max(max_steps, 1)))

    raw_scores = {
        "altitude":     round(alt_raw,  4),
        "efficiency":   round(eff_raw,  4),
        "eccentricity": round(ecc_raw,  4),
        "inclination":  round(inc_raw,  4),
        "steps":        round(step_raw, 4),
    }

    # ── Weighted scores ────────────────────────────────────────────────────────
    component_scores = {
        k: round(raw_scores[k] * WEIGHTS[k], 4)
        for k in WEIGHTS
    }

    total_score = sum(component_scores.values())
    total_score = round(max(0.0, min(1.0, total_score)), 4)

    # ── Mission success check ─────────────────────────────────────────────────
    #    All three orbital parameters must be within tolerance simultaneously.
    alt_error = abs(alt - t_alt)
    ecc_error = abs(ecc - t_ecc)
    inc_error = abs(inc - t_inc)

    mission_success = (
        alt_error <= criteria["altitude_tolerance_km"]   and
        ecc_error <= criteria["eccentricity_tolerance"]  and
        inc_error <= criteria["inclination_tolerance_deg"]
    )

    # ── Efficiency ratio (for reporting) ──────────────────────────────────────
    efficiency_ratio = round(
        theoretical_dv / max(delta_v_used, 1.0), 4
    )

    return {
        "score":             total_score,
        "component_scores":  component_scores,
        "raw_scores":        raw_scores,
        "weights":           WEIGHTS,
        "mission_success":   mission_success,
        "delta_v_used":      round(delta_v_used, 2),
        "delta_v_optimal":   theoretical_dv,
        "efficiency_ratio":  efficiency_ratio,
        "steps_used":        steps_used,
        "errors": {
            "altitude_km":      round(alt_error, 2),
            "eccentricity":     round(ecc_error, 6),
            "inclination_deg":  round(inc_error, 4),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dense Step Reward — called after every action inside env.step()
# ─────────────────────────────────────────────────────────────────────────────

def compute_step_reward(
    prev_state: dict,
    action: dict,
    new_state: dict,
) -> float:
    """
    Compute a dense reward signal for a single environment step.

    Dense rewards are critical for RL agents — without them, the agent only
    gets signal at episode end and learning is very slow (sparse reward problem).

    Reward components:
      +0.10  Getting closer to target altitude (proportional to progress)
      -0.05  Moving further from target altitude
      -0.005 Per 1,000 m/s of Δ-v spent (fuel penalty, always applied)
      +0.03  Using RunSimulation (encourages planning before acting)
      +0.05  Using SetFlyby (encourages gravity assist planning)
      -0.30  Exceeding the Δ-v budget (hard penalty)
      -0.01  Per step taken (encourages step efficiency)

    Final reward is clamped to [-0.5, 0.5] for training stability.

    Args:
        prev_state: State dict BEFORE the action was applied.
        action:     Action dict that was executed.
        new_state:  State dict AFTER the action was applied.

    Returns:
        Reward float in [-0.5, 0.5].
    """
    task_id = new_state.get("task_id") or prev_state.get("task_id")
    task    = get_task(task_id)
    target  = task["target_orbit"]
    t_alt   = target["altitude_km"]

    reward = 0.0

    # ── Altitude progress reward ───────────────────────────────────────────────
    prev_orbit = prev_state.get("current_orbit", {})
    new_orbit  = new_state.get("current_orbit", {})

    # Support both Pydantic model and dict
    if hasattr(prev_orbit, "altitude_km"):
        prev_alt = prev_orbit.altitude_km
    else:
        prev_alt = prev_orbit.get("altitude_km", 0.0)

    if hasattr(new_orbit, "altitude_km"):
        new_alt = new_orbit.altitude_km
    else:
        new_alt = new_orbit.get("altitude_km", 0.0)

    prev_alt_error = abs(prev_alt - t_alt)
    new_alt_error  = abs(new_alt  - t_alt)

    # Scale improvement reward by how large the target altitude is
    # (so Moon/Bennu missions get meaningful signal too)
    scale = max(t_alt, 1.0)

    if new_alt_error < prev_alt_error:
        # Progress toward target — reward proportional to how much we improved
        improvement_fraction = (prev_alt_error - new_alt_error) / scale
        reward += min(0.10, improvement_fraction * 2.0)
    elif new_alt_error > prev_alt_error:
        # Moving away from target
        regression_fraction = (new_alt_error - prev_alt_error) / scale
        reward -= min(0.05, regression_fraction * 1.0)

    # ── Fuel penalty (applied to all burn actions) ─────────────────────────────
    action_type = action.get("type", "")
    dv_spent    = action.get("delta_v_ms", 0.0)

    if dv_spent > 0:
        # Penalty: 0.005 per 1,000 m/s spent — small but accumulates
        reward -= (dv_spent / 1_000.0) * 0.005

    # ── Action-specific bonuses ────────────────────────────────────────────────
    if action_type == "run_simulation":
        # Reward planning — agent looked before leaping
        reward += 0.03

    elif action_type == "set_flyby":
        # Reward gravity assist planning (especially valuable for asteroid mission)
        reward += 0.05

    elif action_type == "set_orbit":
        # Small reward for committing to a target plan
        reward += 0.01

    # ── Budget violation penalty ───────────────────────────────────────────────
    delta_v_used   = new_state.get("delta_v_used", 0.0)
    delta_v_budget = task["delta_v_budget"]

    if delta_v_used > delta_v_budget:
        reward -= 0.30  # Hard penalty for going over budget

    # ── Per-step cost (encourages doing things in fewer steps) ────────────────
    reward -= 0.01

    # ── Clamp to training-stable range ────────────────────────────────────────
    reward = max(-0.5, min(0.5, reward))

    return round(reward, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Internal Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _altitude_score(actual_km: float, target_km: float, tolerance_km: float) -> float:
    """
    Altitude proximity score with log-scale adjustment for large-range missions.

    For LEO (target = 400 km, tolerance = 50 km):
        Linear proximity works fine — errors are small relative to target.

    For Lunar/Bennu (target = 384,400 km or 120,000,000 km):
        Linear proximity would give near-zero scores for anything slightly off.
        Log-scale makes the scoring fair across 9 orders of magnitude.

    Strategy:
        If target < 10,000 km  → use linear proximity_score (LEO missions)
        If target >= 10,000 km → use log-ratio score         (deep space missions)

    Args:
        actual_km:    Agent's current altitude in km.
        target_km:    Target altitude in km.
        tolerance_km: Acceptable error range in km.

    Returns:
        Score 0.0–1.0.
    """
    if target_km < 10_000.0:
        # LEO — linear proximity is fine
        return proximity_score(actual_km, target_km, tolerance_km)

    # Deep space — use log-ratio scoring
    # Score = 1.0 if exactly on target, drops as ratio diverges from 1.0
    if actual_km <= 0 or target_km <= 0:
        return 0.0

    import math
    log_actual = math.log10(max(actual_km, 1.0))
    log_target = math.log10(max(target_km, 1.0))
    log_tol    = math.log10(max(target_km + tolerance_km, 1.0)) - log_target

    if log_tol <= 0:
        return 1.0 if actual_km == target_km else 0.0

    log_error = abs(log_actual - log_target)
    return float(max(0.0, 1.0 - log_error / max(log_tol, 1e-9)))