"""
app — Orbit: AI Space Mission Architect

OpenEnv-compliant RL environment for AI space mission planning.
Uses real orbital mechanics for deterministic, physics-based grading.

Public API:
    OrbitEnvironment  — main environment class (reset/step/state)
    Action types      — SetOrbit, AddBurn, SetFlyby, RunSimulation, SubmitMission, ExecuteManeuver
    Observation       — what the agent sees after each step
    State             — full internal state (for debugging)
    StepResult        — (observation, reward, done, info) tuple
    grade_mission     — deterministic terminal grader
    TASKS             — task registry
"""

# ── Environment ──
from app.env import OrbitEnvironment

# ── Models ──
from app.models import (
    Action,
    ActionType,
    AddBurnAction,
    AvailableManeuver,
    ExecuteManeuverAction,
    ManeuverType,
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

# ── Grader ──
from app.grader import compute_step_reward, grade_mission

# ── Tasks ──
from app.tasks import TASKS, get_task, get_task_summary, list_tasks

# ── Physics (exposed for advanced users / testing) ──
from app.physics import (
    apply_burn,
    escape_velocity,
    fuel_efficiency_ratio,
    gravity_assist_delta_v,
    hohmann_transfer_delta_v,
    launch_to_leo_delta_v,
    lunar_orbit_insertion_delta_v,
    orbital_period,
    orbital_velocity,
    plane_change_delta_v,
    proximity_score,
    trans_lunar_injection_delta_v,
    vis_viva,
)

__all__ = [
    # Environment
    "OrbitEnvironment",
    # Models
    "Action",
    "ActionType",
    "AddBurnAction",
    "AvailableManeuver",
    "ExecuteManeuverAction",
    "ManeuverType",
    "MissionAnalysis",
    "Observation",
    "OrbitalState",
    "RunSimulationAction",
    "SetFlybyAction",
    "SetOrbitAction",
    "State",
    "StepResult",
    "SubmitMissionAction",
    # Grader
    "grade_mission",
    "compute_step_reward",
    # Tasks
    "TASKS",
    "get_task",
    "get_task_summary",
    "list_tasks",
    # Physics
    "apply_burn",
    "escape_velocity",
    "fuel_efficiency_ratio",
    "gravity_assist_delta_v",
    "hohmann_transfer_delta_v",
    "launch_to_leo_delta_v",
    "lunar_orbit_insertion_delta_v",
    "orbital_period",
    "orbital_velocity",
    "plane_change_delta_v",
    "proximity_score",
    "trans_lunar_injection_delta_v",
    "vis_viva",
]

__version__ = "2.0.0"