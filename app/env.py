"""
app/env.py
Orbit — AI Space Mission Architect

Core OpenEnv-compliant environment.
Implements the standard reset() / step() / state() interface.

v2.0 Changes:
    - Added ExecuteManeuverAction handler (strategic high-level actions)
    - Observations now include available_maneuvers, mission_analysis, recommendations
    - All original actions still work (backward compatible)
    - Environment calculates delta-v internally for strategic maneuvers
"""

from __future__ import annotations

from typing import Optional

from app.grader import compute_step_reward, grade_mission
from app.models import (
    Action,
    AddBurnAction,
    AvailableManeuver,
    ExecuteManeuverAction,
    MissionAnalysis,
    Observation,
    OrbitalState,
    RunSimulationAction,
    SetFlybyAction,
    SetOrbitAction,
    State,
    StepResult,
    SubmitMissionAction,
)
from app.physics import (
    apply_burn,
    compute_mission_analysis,
    execute_circularize,
    execute_combined_transfer,
    execute_correction_burn,
    execute_gravity_assist,
    execute_hohmann_transfer,
    execute_lunar_orbit_insertion,
    execute_plane_change,
    execute_trans_lunar_injection,
    get_available_maneuvers,
    get_recommendations,
    orbital_velocity,
)
from app.tasks import TASKS, get_task


class OrbitEnvironment:
    """
    OpenEnv-compliant environment for AI space mission planning.

    The agent interacts with this class exclusively through:
        reset(task_id)  → Observation
        step(action)    → StepResult
        state()         → State

    Supports both strategic maneuvers (ExecuteManeuverAction) and
    legacy low-level burns (AddBurnAction).
    """

    def __init__(self) -> None:
        self._state: Optional[State] = None
        self._task:  Optional[dict]  = None

    # ─────────────────────────────────────────────────────────────────────────
    # reset()
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, task_id: str) -> Observation:
        """
        Reset the environment and start a new episode for the given task.

        Args:
            task_id: One of 'leo_satellite', 'lunar_orbit', 'asteroid_rendezvous'

        Returns:
            Initial Observation with available maneuvers and mission analysis.

        Raises:
            ValueError: If task_id is not recognised.
        """
        if task_id not in TASKS:
            raise ValueError(
                f"Unknown task_id: '{task_id}'. "
                f"Available: {list(TASKS.keys())}"
            )

        self._task = get_task(task_id)
        start  = self._task["start_orbit"]
        target = self._task["target_orbit"]

        current_orbit = OrbitalState(
            altitude_km     = start["altitude_km"],
            eccentricity    = start["eccentricity"],
            inclination_deg = start["inclination_deg"],
            true_anomaly_deg= start.get("true_anomaly_deg", 0.0),
            velocity_ms     = start.get("velocity_ms", 0.0),
        )

        target_orbit = OrbitalState(
            altitude_km     = target["altitude_km"],
            eccentricity    = target["eccentricity"],
            inclination_deg = target["inclination_deg"],
            true_anomaly_deg= target.get("true_anomaly_deg", 0.0),
            velocity_ms     = target.get("velocity_ms", 0.0),
        )

        self._state = State(
            task_id         = task_id,
            current_orbit   = current_orbit,
            target_orbit    = target_orbit,
            delta_v_used    = 0.0,
            delta_v_budget  = self._task["delta_v_budget"],
            trajectory      = [current_orbit],
            burns           = [],
            flybys          = [],
            maneuvers       = [],
            step_index      = 0,
            max_steps       = self._task["max_steps"],
            episode_done    = False,
            mission_success = False,
            current_score   = 0.0,
            action_history  = [],
        )

        return self._build_observation(last_action_result="Mission initialised. Ready for actions.")

    # ─────────────────────────────────────────────────────────────────────────
    # step()
    # ─────────────────────────────────────────────────────────────────────────

    def step(self, action: Action) -> StepResult:
        """
        Execute one action and advance the environment by one step.

        Args:
            action: Any Action subtype including ExecuteManeuverAction.

        Returns:
            StepResult with (observation, reward, done, info).

        Raises:
            RuntimeError: If reset() has not been called, or the episode is done.
        """
        if self._state is None:
            raise RuntimeError("Environment not initialised. Call reset(task_id) first.")
        if self._state.episode_done:
            raise RuntimeError(
                "Episode already finished. Call reset(task_id) to start a new episode."
            )

        prev_state_dict = self._state.model_dump()

        self._state.step_index += 1
        action_dict = action.model_dump()
        self._state.action_history.append(action_dict)

        reward, done, info, last_msg = self._dispatch(action, action_dict, prev_state_dict)

        if not done and self._state.step_index >= self._state.max_steps:
            done, reward, info = self._handle_timeout(reward, info)

        if done:
            self._state.episode_done = True

        observation = self._build_observation(last_action_result=last_msg)
        if done:
            observation.mission_status = (
                "completed" if self._state.mission_success else "failed"
            )

        return StepResult(
            observation = observation,
            reward      = round(max(-1.0, min(1.0, reward)), 4),
            done        = done,
            info        = info,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # state()
    # ─────────────────────────────────────────────────────────────────────────

    def state(self) -> State:
        """Return a deep copy of the full internal state."""
        if self._state is None:
            raise RuntimeError("Environment not initialised. Call reset(task_id) first.")
        return self._state.model_copy(deep=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Action Dispatch
    # ─────────────────────────────────────────────────────────────────────────

    def _dispatch(
        self,
        action: Action,
        action_dict: dict,
        prev_state_dict: dict,
    ) -> tuple[float, bool, dict, str]:
        """Route the action to the correct handler."""

        if isinstance(action, ExecuteManeuverAction):
            return self._handle_execute_maneuver(action, action_dict, prev_state_dict)

        elif isinstance(action, SetOrbitAction):
            return self._handle_set_orbit(action)

        elif isinstance(action, AddBurnAction):
            return self._handle_add_burn(action, action_dict, prev_state_dict)

        elif isinstance(action, SetFlybyAction):
            return self._handle_set_flyby(action)

        elif isinstance(action, RunSimulationAction):
            return self._handle_run_simulation()

        elif isinstance(action, SubmitMissionAction):
            return self._handle_submit_mission()

        else:
            return -0.05, False, {"error": f"Unknown action type: {type(action)}"}, "Unknown action."

    # ─────────────────────────────────────────────────────────────────────────
    # NEW: Execute Maneuver Handler (Strategic Actions)
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_execute_maneuver(
        self,
        action: ExecuteManeuverAction,
        action_dict: dict,
        prev_state_dict: dict,
    ) -> tuple:
        """
        Execute a high-level strategic maneuver.
        The environment calculates the required delta-v internally.
        """
        maneuver = action.maneuver.value  # enum → string
        current_orbit_dict = self._state.current_orbit.model_dump()
        target_orbit_dict = self._task["target_orbit"]

        # ── Route to the correct maneuver execution function ──
        try:
            new_orbit_dict, dv_consumed = self._execute_maneuver_physics(
                maneuver, current_orbit_dict, target_orbit_dict, action
            )
        except ValueError as e:
            msg = f"Maneuver '{maneuver}' failed: {str(e)}"
            info = {"action_type": "execute_maneuver", "maneuver": maneuver,
                    "rejected": True, "message": msg}
            return -0.05, False, info, msg

        # ── Check fuel budget ──
        if dv_consumed > 0 and (self._state.delta_v_used + dv_consumed > self._state.delta_v_budget):
            remaining = self._state.delta_v_budget - self._state.delta_v_used
            msg = (
                f"Maneuver '{maneuver}' REJECTED — needs {dv_consumed:.1f} m/s "
                f"but only {remaining:.1f} m/s remaining."
            )
            info = {"action_type": "execute_maneuver", "maneuver": maneuver,
                    "rejected": True, "delta_v_needed": round(dv_consumed, 2),
                    "delta_v_remaining": round(remaining, 2), "message": msg}
            return -0.10, False, info, msg

        # ── Apply the maneuver ──
        self._state.current_orbit = OrbitalState(**new_orbit_dict)
        self._state.delta_v_used += dv_consumed
        self._state.trajectory.append(self._state.current_orbit)

        # Log the maneuver
        self._state.maneuvers.append({
            "step":         self._state.step_index,
            "maneuver":     maneuver,
            "delta_v_ms":   round(dv_consumed, 2),
            "result_altitude_km": round(self._state.current_orbit.altitude_km, 2),
            "result_inclination_deg": round(self._state.current_orbit.inclination_deg, 4),
            "result_eccentricity": round(self._state.current_orbit.eccentricity, 6),
        })

        # ── Compute step reward ──
        new_state_dict = self._state.model_dump()
        reward = compute_step_reward(prev_state_dict, action_dict, new_state_dict)

        remaining = self._state.delta_v_budget - self._state.delta_v_used
        fuel_note = "(FREE — no fuel consumed)" if dv_consumed == 0 else f"Δv consumed: {dv_consumed:.1f} m/s"

        msg = (
            f"Maneuver '{maneuver}' executed successfully. {fuel_note}. "
            f"New orbit: alt={self._state.current_orbit.altitude_km:.1f} km, "
            f"e={self._state.current_orbit.eccentricity:.4f}, "
            f"inc={self._state.current_orbit.inclination_deg:.1f}°. "
            f"Budget remaining: {remaining:.1f} m/s."
        )
        info = {
            "action_type":          "execute_maneuver",
            "maneuver":             maneuver,
            "delta_v_consumed_ms":  round(dv_consumed, 2),
            "delta_v_remaining_ms": round(remaining, 2),
            "new_altitude_km":      round(self._state.current_orbit.altitude_km, 2),
            "new_eccentricity":     round(self._state.current_orbit.eccentricity, 6),
            "new_inclination_deg":  round(self._state.current_orbit.inclination_deg, 4),
            "message":              msg,
        }
        return reward, False, info, msg

    def _execute_maneuver_physics(
        self,
        maneuver: str,
        current_orbit: dict,
        target_orbit: dict,
        action: ExecuteManeuverAction,
    ) -> tuple[dict, float]:
        """
        Execute the physics for a given maneuver type.
        Returns (new_orbit_dict, delta_v_consumed).
        """
        if maneuver == "hohmann_transfer":
            target_alt = action.target_altitude_km
            if target_alt is None:
                target_alt = target_orbit["altitude_km"]
            return execute_hohmann_transfer(current_orbit, target_alt)

        elif maneuver == "plane_change":
            target_inc = action.target_inclination_deg
            if target_inc is None:
                target_inc = target_orbit["inclination_deg"]
            return execute_plane_change(current_orbit, target_inc)

        elif maneuver == "circularize":
            return execute_circularize(current_orbit)

        elif maneuver == "trans_lunar_injection":
            if self._state.task_id != "lunar_orbit":
                raise ValueError("Trans-Lunar Injection is only available for the lunar_orbit mission.")
            return execute_trans_lunar_injection(current_orbit)

        elif maneuver == "lunar_orbit_insertion":
            if self._state.task_id != "lunar_orbit":
                raise ValueError("Lunar Orbit Insertion is only available for the lunar_orbit mission.")
            target_alt = action.target_altitude_km if action.target_altitude_km else 100.0
            return execute_lunar_orbit_insertion(current_orbit, target_alt)

        elif maneuver == "gravity_assist":
            body = action.body
            if body is None:
                raise ValueError("Gravity assist requires a 'body' parameter (moon, venus, or earth).")
            available = self._task.get("available_flybys", [])
            if body not in available:
                raise ValueError(
                    f"Gravity assist around '{body}' not available for this mission. "
                    f"Available: {available}"
                )
            # Log as flyby too (for backward compatibility)
            self._state.flybys.append({
                "body": body,
                "periapsis_km": 500.0,
                "step": self._state.step_index,
            })
            return execute_gravity_assist(current_orbit, body)

        elif maneuver == "combined_transfer":
            target_alt = action.target_altitude_km
            target_inc = action.target_inclination_deg
            if target_alt is None:
                target_alt = target_orbit["altitude_km"]
            if target_inc is None:
                target_inc = target_orbit["inclination_deg"]
            target_ecc = target_orbit.get("eccentricity", None)
            return execute_combined_transfer(current_orbit, target_alt, target_inc, target_ecc)

        elif maneuver == "correction_burn":
            dv = action.delta_v_ms if action.delta_v_ms else 50.0
            return execute_correction_burn(current_orbit, dv)

        else:
            raise ValueError(f"Unknown maneuver type: '{maneuver}'")

    # ─────────────────────────────────────────────────────────────────────────
    # Original Action Handlers (unchanged)
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_set_orbit(self, action: SetOrbitAction) -> tuple:
        """Planning action — agent declares intended orbit. Does not consume fuel."""
        msg = (
            f"Planned orbit: altitude={action.altitude_km:.1f} km, "
            f"e={action.eccentricity:.3f}, inc={action.inclination_deg:.1f}°"
        )
        info = {
            "action_type":    "set_orbit",
            "planned_altitude_km": action.altitude_km,
            "message":        msg,
        }
        return 0.01, False, info, msg

    def _handle_add_burn(
        self,
        action: AddBurnAction,
        action_dict: dict,
        prev_state_dict: dict,
    ) -> tuple:
        """Execute a thruster burn — the legacy low-level physics action."""
        if self._state.delta_v_used + action.delta_v_ms > self._state.delta_v_budget:
            overage = (self._state.delta_v_used + action.delta_v_ms) - self._state.delta_v_budget
            msg = (
                f"Burn REJECTED — would exceed Δ-v budget by {overage:.1f} m/s. "
                f"Remaining budget: {self._state.delta_v_budget - self._state.delta_v_used:.1f} m/s."
            )
            info = {
                "action_type": "add_burn",
                "rejected":    True,
                "overage_ms":  round(overage, 2),
                "message":     msg,
            }
            return -0.20, False, info, msg

        new_orbit_dict = apply_burn(
            current_orbit = self._state.current_orbit.model_dump(),
            delta_v_ms    = action.delta_v_ms,
            prograde      = action.prograde,
            radial        = action.radial,
            normal        = action.normal,
        )

        self._state.current_orbit  = OrbitalState(**new_orbit_dict)
        self._state.delta_v_used  += action.delta_v_ms
        self._state.trajectory.append(self._state.current_orbit)
        self._state.burns.append({
            "step":       self._state.step_index,
            "delta_v_ms": action.delta_v_ms,
            "prograde":   action.prograde,
            "radial":     action.radial,
            "normal":     action.normal,
            "result_altitude_km": self._state.current_orbit.altitude_km,
        })

        new_state_dict = self._state.model_dump()
        reward = compute_step_reward(prev_state_dict, action_dict, new_state_dict)

        remaining = self._state.delta_v_budget - self._state.delta_v_used
        msg = (
            f"Burn executed: Δv={action.delta_v_ms:.1f} m/s "
            f"[prograde={action.prograde:+.2f}, radial={action.radial:+.2f}, normal={action.normal:+.2f}]. "
            f"New altitude: {self._state.current_orbit.altitude_km:.1f} km. "
            f"Budget remaining: {remaining:.1f} m/s."
        )
        info = {
            "action_type":        "add_burn",
            "delta_v_spent_ms":   action.delta_v_ms,
            "delta_v_remaining_ms": round(remaining, 2),
            "new_altitude_km":    round(self._state.current_orbit.altitude_km, 2),
            "new_eccentricity":   round(self._state.current_orbit.eccentricity, 6),
            "new_inclination_deg":round(self._state.current_orbit.inclination_deg, 4),
            "message":            msg,
        }
        return reward, False, info, msg

    def _handle_set_flyby(self, action: SetFlybyAction) -> tuple:
        """Plan a gravity assist maneuver."""
        available = self._task.get("available_flybys", [])
        if action.body not in available:
            msg = (
                f"Flyby of '{action.body}' is not available for mission '{self._state.task_id}'. "
                f"Available bodies: {available if available else 'None'}."
            )
            info = {"action_type": "set_flyby", "rejected": True, "message": msg}
            return -0.05, False, info, msg

        self._state.flybys.append({
            "body":         action.body,
            "periapsis_km": action.periapsis_km,
            "step":         self._state.step_index,
        })

        msg = (
            f"Gravity assist planned: {action.body.capitalize()} flyby "
            f"at {action.periapsis_km:.1f} km periapsis altitude."
        )
        info = {
            "action_type":  "set_flyby",
            "body":         action.body,
            "periapsis_km": action.periapsis_km,
            "total_flybys": len(self._state.flybys),
            "message":      msg,
        }
        return 0.05, False, info, msg

    def _handle_run_simulation(self) -> tuple:
        """Preview the current mission score without submitting."""
        current_score = self._preview_score()
        self._state.current_score = current_score

        target  = self._task["target_orbit"]
        curr    = self._state.current_orbit
        alt_err = abs(curr.altitude_km - target["altitude_km"])
        ecc_err = abs(curr.eccentricity - target["eccentricity"])
        inc_err = abs(curr.inclination_deg - target["inclination_deg"])
        remaining_budget = self._state.delta_v_budget - self._state.delta_v_used
        remaining_steps  = self._state.max_steps - self._state.step_index

        msg = (
            f"Simulation complete. Current score: {current_score:.3f}. "
            f"Altitude error: {alt_err:.1f} km. "
            f"Eccentricity error: {ecc_err:.4f}. "
            f"Inclination error: {inc_err:.2f}°. "
            f"Budget remaining: {remaining_budget:.1f} m/s. "
            f"Steps remaining: {remaining_steps}."
        )
        info = {
            "action_type":        "run_simulation",
            "current_score":      current_score,
            "trajectory_length":  len(self._state.trajectory),
            "altitude_error_km":  round(alt_err, 2),
            "eccentricity_error": round(ecc_err, 6),
            "inclination_error_deg": round(inc_err, 4),
            "delta_v_remaining_ms": round(remaining_budget, 2),
            "steps_remaining":    remaining_steps,
            "message":            msg,
        }
        return 0.02, False, info, msg

    def _handle_submit_mission(self) -> tuple:
        """Final action — grade the mission and end the episode."""
        grade = grade_mission(self._state.task_id, self._state.model_dump())

        self._state.current_score   = grade["score"]
        self._state.mission_success = grade["mission_success"]

        status = "SUCCESS 🎉" if grade["mission_success"] else "INCOMPLETE"
        msg = (
            f"Mission submitted [{status}]. "
            f"Final score: {grade['score']:.4f}. "
            f"Δ-v used: {grade['delta_v_used']:.1f} / {grade['delta_v_optimal']} m/s "
            f"(efficiency: {grade['efficiency_ratio']:.3f}). "
            f"Steps used: {grade['steps_used']} / {self._state.max_steps}."
        )
        info = {
            "action_type":  "submit_mission",
            "grade_result": grade,
            "message":      msg,
        }
        return grade["score"], True, info, msg

    # ─────────────────────────────────────────────────────────────────────────
    # Timeout Handler
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_timeout(self, current_reward: float, current_info: dict) -> tuple:
        """Auto-grade on timeout with 20% penalty."""
        grade = grade_mission(self._state.task_id, self._state.model_dump())

        self._state.current_score   = grade["score"]
        self._state.mission_success = grade["mission_success"]

        timeout_reward = grade["score"] * 0.80

        current_info.update({
            "timeout":      True,
            "grade_result": grade,
            "message": (
                f"Step limit reached ({self._state.max_steps} steps). "
                f"Auto-graded with 20% penalty. "
                f"Score: {grade['score']:.4f} → penalised reward: {timeout_reward:.4f}."
            ),
        })
        return True, timeout_reward, current_info

    # ─────────────────────────────────────────────────────────────────────────
    # Observation Builder (ENRICHED in v2.0)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_observation(self, last_action_result: Optional[str] = None) -> Observation:
        """
        Construct an enriched Observation from the current internal state.
        Includes available maneuvers, mission analysis, and recommendations.
        """
        current_orbit_dict = self._state.current_orbit.model_dump()
        target_orbit_dict  = self._task["target_orbit"]
        delta_v_remaining  = self._state.delta_v_budget - self._state.delta_v_used

        # ── Build available maneuvers list ──
        raw_maneuvers = get_available_maneuvers(
            current_orbit   = current_orbit_dict,
            target_orbit    = target_orbit_dict,
            delta_v_remaining = delta_v_remaining,
            task_id         = self._state.task_id,
        )
        available_maneuvers = [AvailableManeuver(**m) for m in raw_maneuvers]

        # ── Build mission analysis ──
        raw_analysis = compute_mission_analysis(
            current_orbit  = current_orbit_dict,
            target_orbit   = target_orbit_dict,
            delta_v_used   = self._state.delta_v_used,
            delta_v_budget = self._state.delta_v_budget,
            task_id        = self._state.task_id,
        )
        mission_analysis = MissionAnalysis(**raw_analysis)

        # ── Build recommendations ──
        recommendations = get_recommendations(
            current_orbit   = current_orbit_dict,
            target_orbit    = target_orbit_dict,
            delta_v_remaining = delta_v_remaining,
            task_id         = self._state.task_id,
            step_index      = self._state.step_index,
            max_steps       = self._state.max_steps,
        )

        # Include maneuver hints from task config on first observation
        if self._state.step_index == 0:
            hints = self._task.get("maneuver_hints", [])
            recommendations = hints + recommendations

        return Observation(
            task_id             = self._state.task_id,
            task_name           = self._task["name"],
            current_orbit       = self._state.current_orbit,
            target_orbit        = self._state.target_orbit,
            delta_v_used        = self._state.delta_v_used,
            delta_v_budget      = self._state.delta_v_budget,
            trajectory          = self._state.trajectory,
            step_index          = self._state.step_index,
            max_steps           = self._state.max_steps,
            mission_status      = "in_progress",
            last_action_result  = last_action_result,
            available_maneuvers = available_maneuvers,
            mission_analysis    = mission_analysis,
            recommendations     = recommendations,
        )

    def _preview_score(self) -> float:
        """Calculate the current score without ending the episode."""
        result = grade_mission(self._state.task_id, self._state.model_dump())
        return result["score"]