"""
app/env.py
Orbit — AI Space Mission Architect

Core OpenEnv-compliant environment.
Implements the standard reset() / step() / state() interface.

Usage:
    from app.env import OrbitEnvironment
    from app.models import AddBurnAction, SubmitMissionAction

    env = OrbitEnvironment()
    obs = env.reset("leo_satellite")

    action = AddBurnAction(delta_v_ms=9400, prograde=1.0, radial=0.0, normal=0.0)
    result = env.step(action)

    print(result.observation)   # Observation
    print(result.reward)        # float in [-1, 1]
    print(result.done)          # bool
    print(result.info)          # dict with grade info, messages, etc.
"""

from __future__ import annotations

from typing import Optional

from grader import compute_step_reward, grade_mission
from models import (
    Action,
    AddBurnAction,
    Observation,
    OrbitalState,
    RunSimulationAction,
    SetFlybyAction,
    SetOrbitAction,
    State,
    StepResult,
    SubmitMissionAction,
)
from physics import apply_burn, orbital_velocity
from tasks import TASKS, get_task


class OrbitEnvironment:
    """
    OpenEnv-compliant environment for AI space mission planning.

    The agent interacts with this class exclusively through:
        reset(task_id)  → Observation
        step(action)    → StepResult
        state()         → State   (full internal state for debugging)

    Internal state is a Pydantic State model — fully serialisable to JSON,
    which is what the WebSocket server (server.py) depends on.
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

        Always call this before the first step(), and after an episode ends
        (result.done == True) before starting a new one.

        Args:
            task_id: One of 'leo_satellite', 'lunar_orbit', 'asteroid_rendezvous'

        Returns:
            Initial Observation — what the agent sees at the start of the episode.

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

        # Build starting orbital state
        current_orbit = OrbitalState(
            altitude_km     = start["altitude_km"],
            eccentricity    = start["eccentricity"],
            inclination_deg = start["inclination_deg"],
            true_anomaly_deg= start.get("true_anomaly_deg", 0.0),
            velocity_ms     = start.get("velocity_ms", 0.0),
        )

        # Build target orbital state (what the agent is trying to reach)
        target_orbit = OrbitalState(
            altitude_km     = target["altitude_km"],
            eccentricity    = target["eccentricity"],
            inclination_deg = target["inclination_deg"],
            true_anomaly_deg= target.get("true_anomaly_deg", 0.0),
            velocity_ms     = target.get("velocity_ms", 0.0),
        )

        # Initialise full internal state
        self._state = State(
            task_id         = task_id,
            current_orbit   = current_orbit,
            target_orbit    = target_orbit,
            delta_v_used    = 0.0,
            delta_v_budget  = self._task["delta_v_budget"],
            trajectory      = [current_orbit],
            burns           = [],
            flybys          = [],
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
            action: Any Action subtype (SetOrbitAction, AddBurnAction,
                    SetFlybyAction, RunSimulationAction, SubmitMissionAction).

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

        # Snapshot state BEFORE action for reward calculation
        prev_state_dict = self._state.model_dump()

        # Increment step counter and log action
        self._state.step_index += 1
        action_dict = action.model_dump()
        self._state.action_history.append(action_dict)

        # Dispatch to the correct handler
        reward, done, info, last_msg = self._dispatch(action, action_dict, prev_state_dict)

        # Check step limit (only if episode not already done by SubmitMission)
        if not done and self._state.step_index >= self._state.max_steps:
            done, reward, info = self._handle_timeout(reward, info)

        # Mark episode done if needed
        if done:
            self._state.episode_done = True

        # Build final observation
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
        """
        Return a copy of the full internal state.

        Used by server.py for WebSocket state broadcasting,
        and by visualizer.py for trajectory plotting.

        Returns:
            Deep copy of the current State model.

        Raises:
            RuntimeError: If reset() has not been called yet.
        """
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
        """
        Route the action to the correct handler.
        Returns (reward, done, info, last_message).
        """
        if isinstance(action, SetOrbitAction):
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
            # Unknown action type — should never happen with Pydantic validation
            return -0.05, False, {"error": f"Unknown action type: {type(action)}"}, "Unknown action."

    # ─────────────────────────────────────────────────────────────────────────
    # Individual Action Handlers
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_set_orbit(self, action: SetOrbitAction) -> tuple:
        """
        Planning action — agent declares intended orbit. Does not consume fuel.
        Gives a small positive reward to encourage planning.
        """
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
        """
        Execute a thruster burn — the core physics action.
        Updates current_orbit using apply_burn(), tracks Δ-v usage.
        """
        # Check if this burn would exceed the budget
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

        # Apply the burn to compute new orbital parameters
        new_orbit_dict = apply_burn(
            current_orbit = self._state.current_orbit.model_dump(),
            delta_v_ms    = action.delta_v_ms,
            prograde      = action.prograde,
            radial        = action.radial,
            normal        = action.normal,
        )

        # Update state
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

        # Dense step reward — how good was this burn?
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
        """
        Plan a gravity assist maneuver. Only meaningful for asteroid mission.
        Gives a bonus reward to encourage advanced mission planning.
        """
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
        """
        Preview the current mission score without submitting.
        Encourages the agent to evaluate its trajectory before committing.
        """
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
        """
        Final action — grade the mission and end the episode.
        Terminal reward = final score (0.0–1.0).
        """
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

        # Terminal reward is the raw score — agent earns what it deserves
        return grade["score"], True, info, msg

    # ─────────────────────────────────────────────────────────────────────────
    # Timeout Handler
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_timeout(self, current_reward: float, current_info: dict) -> tuple:
        """
        Called when step_index reaches max_steps without SubmitMission.
        Auto-grades and applies a 20% timeout penalty on the terminal reward.
        """
        grade = grade_mission(self._state.task_id, self._state.model_dump())

        self._state.current_score   = grade["score"]
        self._state.mission_success = grade["mission_success"]

        # 20% penalty for not submitting voluntarily
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
    # Internal Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_observation(self, last_action_result: Optional[str] = None) -> Observation:
        """Construct an Observation from the current internal state."""
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
        )

    def _preview_score(self) -> float:
        """Calculate the current score without modifying state or ending the episode."""
        result = grade_mission(self._state.task_id, self._state.model_dump())
        return result["score"]