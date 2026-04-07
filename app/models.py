"""
app/models.py
Orbit — AI Space Mission Architect
All Pydantic v2 models: Action types, Observation, State, StepResult
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    SET_ORBIT       = "set_orbit"
    ADD_BURN        = "add_burn"
    SET_FLYBY       = "set_flyby"
    RUN_SIMULATION  = "run_simulation"
    SUBMIT_MISSION  = "submit_mission"


# ─────────────────────────────────────────────────────────────────────────────
# Action Models
# ─────────────────────────────────────────────────────────────────────────────

class SetOrbitAction(BaseModel):
    """
    Planning action: agent declares its intended orbit parameters.
    Does NOT consume fuel — it's just the agent stating a plan.
    """
    type: Literal[ActionType.SET_ORBIT] = ActionType.SET_ORBIT

    altitude_km: float = Field(
        ...,
        ge=100,
        le=1_000_000,
        description="Target orbit altitude above Earth's surface in km. "
                    "LEO ≈ 200–2000 km | Moon ≈ 384,400 km | Bennu ≈ 120,000,000 km"
    )
    eccentricity: float = Field(
        0.0,
        ge=0.0,
        lt=1.0,
        description="Orbital eccentricity. 0.0 = perfect circle, 0.99 = nearly parabolic. "
                    "ISS ≈ 0.0002 | Moon transfer ≈ 0.97"
    )
    inclination_deg: float = Field(
        0.0,
        ge=-90.0,
        le=90.0,
        description="Orbit inclination in degrees. "
                    "0° = equatorial | 51.6° = ISS | 90° = polar"
    )


class AddBurnAction(BaseModel):
    """
    Execute a thruster burn. This IS the core physics action — consumes Δ-v budget.
    Direction components should be normalized (they are treated as ratios, not magnitudes).
    """
    type: Literal[ActionType.ADD_BURN] = ActionType.ADD_BURN

    delta_v_ms: float = Field(
        ...,
        ge=0.0,
        le=15_000.0,
        description="Magnitude of the burn in m/s. "
                    "LEO launch ≈ 9400 m/s | TLI burn ≈ 3100 m/s | Inclination change ≈ 500–2000 m/s"
    )
    prograde: float = Field(
        1.0,
        ge=-1.0,
        le=1.0,
        description="Prograde component of burn direction. "
                    "+1.0 = forward (raises orbit) | -1.0 = retrograde (lowers orbit)"
    )
    radial: float = Field(
        0.0,
        ge=-1.0,
        le=1.0,
        description="Radial component of burn direction. "
                    "+1.0 = away from Earth | -1.0 = toward Earth. "
                    "Used for eccentricity changes."
    )
    normal: float = Field(
        0.0,
        ge=-1.0,
        le=1.0,
        description="Normal component of burn direction (out-of-plane). "
                    "+1.0 = north | -1.0 = south. "
                    "Used for inclination changes — very Δ-v expensive!"
    )


class SetFlybyAction(BaseModel):
    """
    Plan a gravity assist maneuver. Only valid for Task 3 (Asteroid Rendezvous).
    Gravity assists are free Δ-v — the agent earns a bonus reward for using them.
    """
    type: Literal[ActionType.SET_FLYBY] = ActionType.SET_FLYBY

    body: Literal["moon", "venus", "earth"] = Field(
        ...,
        description="Which celestial body to use for the gravity assist. "
                    "Earth flyby ≈ +1500 m/s | Venus flyby ≈ +2700 m/s | Moon flyby ≈ +200 m/s"
    )
    periapsis_km: float = Field(
        ...,
        ge=100.0,
        le=100_000.0,
        description="Closest approach distance to the flyby body's SURFACE in km. "
                    "Lower = stronger assist but higher risk. "
                    "Safe minimum: Earth 200 km | Venus 300 km | Moon 100 km"
    )


class RunSimulationAction(BaseModel):
    """
    Ask the environment to preview the current trajectory without committing.
    Useful for the agent to evaluate its plan before spending real Δ-v.
    Earns a small positive reward (encourages planning behavior).
    """
    type: Literal[ActionType.RUN_SIMULATION] = ActionType.RUN_SIMULATION


class SubmitMissionAction(BaseModel):
    """
    Final action — submit the current mission state for grading.
    The score from grade_mission() becomes the terminal reward.
    After this, episode_done = True and reset() must be called.
    """
    type: Literal[ActionType.SUBMIT_MISSION] = ActionType.SUBMIT_MISSION


# ─────────────────────────────────────────────────────────────────────────────
# Discriminated Union — the single "Action" type used everywhere
# ─────────────────────────────────────────────────────────────────────────────

Action = Union[
    SetOrbitAction,
    AddBurnAction,
    SetFlybyAction,
    RunSimulationAction,
    SubmitMissionAction,
]


# ─────────────────────────────────────────────────────────────────────────────
# Orbital State
# ─────────────────────────────────────────────────────────────────────────────

class OrbitalState(BaseModel):
    """
    Complete description of a spacecraft's orbit at a single point in time.
    This is the fundamental unit that trajectory lists are made of.
    """
    altitude_km: float = Field(
        ...,
        description="Altitude above Earth's surface in km. "
                    "Ground = 0 | ISS = 408 | Moon = 384,400 | Bennu = ~120,000,000"
    )
    eccentricity: float = Field(
        0.0,
        ge=0.0,
        le=0.99,
        description="Orbital eccentricity. 0.0 = circular, <1.0 = elliptical."
    )
    inclination_deg: float = Field(
        0.0,
        ge=-90.0,
        le=90.0,
        description="Orbital inclination in degrees from equatorial plane."
    )
    true_anomaly_deg: float = Field(
        0.0,
        ge=0.0,
        le=360.0,
        description="Position within the orbit in degrees. 0° = periapsis (closest point)."
    )
    velocity_ms: float = Field(
        0.0,
        ge=0.0,
        description="Orbital velocity in m/s. ISS ≈ 7660 m/s | Moon ≈ 1022 m/s"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Observation  (what the agent SEES after each step)
# ─────────────────────────────────────────────────────────────────────────────

class Observation(BaseModel):
    """
    Returned by reset() and step().
    This is the agent's window into the environment — it sees only this, not State.
    """
    task_id: str = Field(
        ...,
        description="Current mission identifier. One of: 'leo_satellite', 'lunar_orbit', 'asteroid_rendezvous'"
    )
    task_name: str = Field(
        ...,
        description="Human-readable mission name for display / logging."
    )
    current_orbit: OrbitalState = Field(
        ...,
        description="Spacecraft's current orbital parameters."
    )
    target_orbit: OrbitalState = Field(
        ...,
        description="The orbit the agent is trying to reach. "
                    "Score improves as current_orbit approaches target_orbit."
    )
    delta_v_used: float = Field(
        0.0,
        ge=0.0,
        description="Cumulative Δ-v spent so far in m/s. Agent must stay under delta_v_budget."
    )
    delta_v_budget: float = Field(
        15_000.0,
        gt=0.0,
        description="Maximum Δ-v allowed for this mission in m/s. "
                    "LEO = 12,000 | Lunar = 5,000 | Asteroid = 8,000"
    )
    trajectory: List[OrbitalState] = Field(
        default_factory=list,
        description="Ordered history of all orbital states visited this episode."
    )
    step_index: int = Field(
        0,
        ge=0,
        description="How many steps have been taken this episode."
    )
    max_steps: int = Field(
        ...,
        gt=0,
        description="Step limit for this mission. LEO = 10 | Lunar = 15 | Asteroid = 25"
    )
    mission_status: Literal["in_progress", "completed", "failed"] = Field(
        "in_progress",
        description="Current status of the mission."
    )
    last_action_result: Optional[str] = Field(
        None,
        description="Human-readable result of the last action for logging/display."
    )


# ─────────────────────────────────────────────────────────────────────────────
# State  (full internal state — used by env.py, grader.py, server.py)
# ─────────────────────────────────────────────────────────────────────────────

class State(BaseModel):
    """
    Complete internal environment state.
    Observation is a subset of this — agents only see Observation.
    State is used internally by env.py and exposed via env.state() for debugging.
    """
    task_id: str

    current_orbit: OrbitalState
    target_orbit: OrbitalState

    delta_v_used: float = Field(0.0, ge=0.0)
    delta_v_budget: float = Field(15_000.0, gt=0.0)

    trajectory: List[OrbitalState] = Field(default_factory=list)

    burns: List[Dict] = Field(
        default_factory=list,
        description="Log of all burn actions: step, delta_v_ms, prograde, radial, normal"
    )
    flybys: List[Dict] = Field(
        default_factory=list,
        description="Log of all planned flyby maneuvers: body, periapsis_km, step"
    )

    step_index: int = Field(0, ge=0)
    max_steps: int = Field(..., gt=0)

    episode_done: bool = Field(
        False,
        description="True when mission is submitted or step limit reached."
    )
    mission_success: bool = Field(
        False,
        description="True only if all success criteria (altitude, ecc, inc) are within tolerance."
    )
    current_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Last computed score from grade_mission(). Updated on RunSimulation and Submit."
    )
    action_history: List[Dict] = Field(
        default_factory=list,
        description="Complete log of every action taken this episode."
    )


# ─────────────────────────────────────────────────────────────────────────────
# StepResult  (return type of env.step())
# ─────────────────────────────────────────────────────────────────────────────

class StepResult(BaseModel):
    """
    The tuple returned by env.step(action).
    Mirrors the standard (obs, reward, done, info) RL convention.
    """
    observation: Observation = Field(
        ...,
        description="Updated observation after the action was applied."
    )
    reward: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Immediate reward signal. "
                    "Dense step rewards in [-0.5, 0.5] | Terminal reward = final score [0.0, 1.0]"
    )
    done: bool = Field(
        ...,
        description="True if the episode has ended (submitted or timed out)."
    )
    info: Dict = Field(
        default_factory=dict,
        description="Extra metadata: action_type, grade_result, error messages, etc."
    )