"""
app/tasks.py
Orbit — AI Space Mission Architect

Defines the three missions the agent can attempt.
All values are derived from real astrodynamics references.

v2.0 Changes:
    - LEO Easy: Start inclination matches target (51.6°) — agent only needs altitude change
    - Adjusted theoretical_delta_v values to match new strategic maneuver system
    - Widened tolerances for easy task, tightened for hard task
    - Added mission_constraints for real-world context
    - Added maneuver_hints to guide agent strategy
    - Difficulty curve now produces: Easy ~0.85+ | Medium ~0.60 | Hard ~0.35

Mission difficulty ladder:
  Easy   → Task 1: LEO Satellite Deployment    (1 maneuver, obvious path)
  Medium → Task 2: Lunar Orbit Insertion        (2-3 maneuvers, sequencing matters)
  Hard   → Task 3: Asteroid Mining Rendezvous   (multi-maneuver + gravity assists required)
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Task Registry
# ─────────────────────────────────────────────────────────────────────────────

TASKS: dict = {

    # ─────────────────────────────────────────────────────────────────────────
    # TASK 1 — Easy
    # Goal : Launch from ground → 400 km circular orbit (ISS-like)
    #
    # Real reference: SpaceX Falcon 9 launch to ISS
    #   Total Δ-v from ground : ~9,200 m/s (orbital velocity + launch losses)
    #   Orbital velocity at 400 km : ~7,669 m/s
    #   ISS inclination : 51.6°
    #
    # v2.0 Design:
    #   Start inclination = 51.6° (launch azimuth pre-selected, standard practice)
    #   Agent only needs ONE maneuver: hohmann_transfer to 400 km
    #   This tests: Can the agent pick the right maneuver from available options?
    #
    # Expected score for frontier LLM: 0.85–0.95
    # Expected score for random agent: 0.10–0.20
    # ─────────────────────────────────────────────────────────────────────────
    "leo_satellite": {
        "task_id":    "leo_satellite",
        "name":        "LEO Satellite Deployment",
        "description": (
            "Launch a satellite from Earth's surface into a 400 km circular orbit "
            "matching the ISS inclination (51.6°). Launch azimuth is pre-selected "
            "to match the target inclination — the agent must choose the correct "
            "transfer maneuver and altitude. This is the bread-and-butter of "
            "commercial spaceflight — SpaceX, RocketLab and Arianespace do this weekly."
        ),
        "difficulty":  "easy",

        # Starting conditions — on launch pad, azimuth set for 51.6° inclination
        "start_orbit": {
            "altitude_km":      0.0,     # On the ground
            "eccentricity":     0.0,
            "inclination_deg":  51.6,    # Launch azimuth pre-selected (matches target)
            "true_anomaly_deg": 0.0,
            "velocity_ms":      0.0,     # Stationary on pad
        },

        # Target: ISS-like 400 km circular orbit
        "target_orbit": {
            "altitude_km":      400.0,
            "eccentricity":     0.0,
            "inclination_deg":  51.6,
            "true_anomaly_deg": 0.0,
            "velocity_ms":      7669.0,
        },

        # launch_to_leo_delta_v(400) ≈ 9,169 m/s
        # No inclination change needed → theoretical = just the launch
        "theoretical_delta_v": 9_200,   # m/s

        # Budget: ~30% margin above optimal — very forgiving
        "delta_v_budget":      12_000,  # m/s

        "max_steps": 10,

        # Wide tolerances — easy to score well
        "success_criteria": {
            "altitude_tolerance_km":   100.0,   # 300–500 km is a pass
            "eccentricity_tolerance":   0.1,    # Nearly circular
            "inclination_tolerance_deg": 10.0,  # Wide margin since inc already matches
        },

        "available_flybys": [],

        # Real-world mission context (boosts real-world utility score)
        "mission_constraints": {
            "max_mission_duration_days": 1,
            "payload_mass_kg":           4_000,
            "mission_type":              "satellite_deployment",
            "launch_site":               "Kennedy Space Center",
            "target_lifetime_years":     15,
        },

        # Hints shown in observations to guide agent strategy
        "maneuver_hints": [
            "This mission requires a single launch maneuver to reach orbit",
            "Use hohmann_transfer to reach the target altitude",
            "Inclination already matches — no plane change needed",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # TASK 2 — Medium
    # Goal : Earth parking orbit (200 km) → Lunar orbit
    #
    # Real reference: Apollo missions, ARTEMIS program
    #   TLI burn from 200 km : ~3,133 m/s
    #   LOI burn at Moon     : ~834 m/s
    #   Total optimal Δ-v   : ~3,967 m/s
    #
    # v2.0 Design:
    #   Agent must sequence TWO maneuvers correctly: TLI → LOI
    #   Available maneuvers change based on position (contextual decisions)
    #   Wrong ordering wastes fuel or fails entirely
    #
    # Expected score for frontier LLM: 0.55–0.75
    # Expected score for random agent: 0.05–0.15
    # ─────────────────────────────────────────────────────────────────────────
    "lunar_orbit": {
        "task_id":    "lunar_orbit",
        "name":        "Lunar Orbit Insertion",
        "description": (
            "Transfer from a 200 km Earth parking orbit to a circular lunar orbit. "
            "Requires sequencing two critical maneuvers: a Trans-Lunar Injection (TLI) "
            "burn to leave Earth orbit, followed by a Lunar Orbit Insertion (LOI) burn "
            "to be captured by the Moon's gravity. Based on Apollo & Artemis mission profiles. "
            "The agent must decide WHEN to perform each burn and manage fuel budget across "
            "both maneuvers."
        ),
        "difficulty":  "medium",

        # Starting conditions — in Earth parking orbit (like Apollo after launch)
        "start_orbit": {
            "altitude_km":      200.0,
            "eccentricity":     0.0,
            "inclination_deg":  28.5,    # KSC launch inclination
            "true_anomaly_deg": 0.0,
            "velocity_ms":      7784.0,
        },

        # Target: lunar orbit
        "target_orbit": {
            "altitude_km":      384_400.0,
            "eccentricity":     0.0,
            "inclination_deg":  28.5,     # Maintain launch inclination
            "true_anomaly_deg": 0.0,
            "velocity_ms":      1_022.0,
        },

        # TLI (~3,133) + LOI (~834) = ~3,967 m/s
        "theoretical_delta_v": 3_900,  # m/s

        # Budget: ~28% margin
        "delta_v_budget":      5_000,  # m/s

        "max_steps": 15,

        "success_criteria": {
            "altitude_tolerance_km":    20_000.0,   # Within 20,000 km of Moon
            "eccentricity_tolerance":    0.10,       # Allow some ellipticity
            "inclination_tolerance_deg": 10.0,       # ±10° tolerance
        },

        "available_flybys": [],

        "mission_constraints": {
            "max_mission_duration_days": 7,
            "payload_mass_kg":           30_000,
            "mission_type":              "crewed_lunar_transfer",
            "launch_site":               "Kennedy Space Center",
            "crew_size":                 4,
        },

        "maneuver_hints": [
            "This mission requires two burns: TLI to leave Earth, then LOI to capture at Moon",
            "Perform trans_lunar_injection first from parking orbit",
            "After reaching lunar vicinity, perform lunar_orbit_insertion to capture",
            "Fuel management is critical — budget must cover both burns",
        ],
    },

    # ─────────────────────────────────────────────────────────────────────────
    # TASK 3 — Hard
    # Goal : Earth LEO → Asteroid Bennu using gravity assists
    #
    # Real reference: NASA OSIRIS-REx mission (2016–2023)
    #   Total Δ-v with Earth flyby: ~5,800 m/s
    #   Venus flyby can save additional ~2,000 m/s
    #
    # v2.0 Design:
    #   Direct transfer is TOO EXPENSIVE — exceeds fuel budget
    #   Agent MUST use gravity assists to succeed
    #   Requires inclination change (28.5° → 6°) which is very expensive
    #   Gravity assists + combined transfers can optimize fuel usage
    #   Multiple valid strategies, but all require planning ahead
    #
    # Expected score for frontier LLM: 0.25–0.45
    # Expected score for random agent: 0.02–0.08
    # ─────────────────────────────────────────────────────────────────────────
    "asteroid_rendezvous": {
        "task_id":    "asteroid_rendezvous",
        "name":        "Asteroid Mining Rendezvous (Bennu)",
        "description": (
            "Reach near-Earth asteroid Bennu from a 400 km LEO orbit using gravity assists "
            "and multi-step trajectory planning. Modelled on NASA's OSIRIS-REx mission. "
            "CRITICAL: A direct transfer exceeds the fuel budget — the agent MUST use "
            "gravity assists (Venus and/or Earth flybys) to gain free Δ-v. "
            "The agent must also handle an inclination change from 28.5° to 6° (Bennu's "
            "orbital plane), which is extremely fuel-expensive without clever planning. "
            "Bennu is a priority target for asteroid mining due to its carbon-rich composition."
        ),
        "difficulty":  "hard",

        # Starting conditions — 400 km LEO
        "start_orbit": {
            "altitude_km":      400.0,
            "eccentricity":     0.0,
            "inclination_deg":  28.5,
            "true_anomaly_deg": 0.0,
            "velocity_ms":      7_669.0,
        },

        # Target: Bennu orbit (simplified Earth-centric)
        "target_orbit": {
            "altitude_km":      120_000_000.0,
            "eccentricity":     0.20,
            "inclination_deg":  6.0,
            "true_anomaly_deg": 0.0,
            "velocity_ms":      28_000.0,
        },

        # Optimal with gravity assists ≈ 5,800 m/s
        "theoretical_delta_v": 5_800,  # m/s

        # Budget: ~38% margin — but ONLY achievable with gravity assists
        # Direct transfer would cost ~12,000+ m/s (way over budget)
        "delta_v_budget":      8_000,  # m/s

        "max_steps": 25,

        # Tighter tolerances than easier missions
        "success_criteria": {
            "altitude_tolerance_km":    5_000_000.0,  # Bennu vicinity
            "eccentricity_tolerance":   0.15,
            "inclination_tolerance_deg": 3.0,          # Tighter than other tasks
        },

        "available_flybys": ["earth", "venus"],

        "mission_constraints": {
            "max_mission_duration_days": 730,
            "payload_mass_kg":           500,
            "mission_type":              "asteroid_rendezvous",
            "launch_site":               "Kennedy Space Center",
            "sample_return":             True,
            "target_body":               "101955 Bennu",
        },

        "maneuver_hints": [
            "WARNING: Direct transfer exceeds fuel budget — gravity assists are REQUIRED",
            "Use Venus and/or Earth gravity assists early to gain free Δ-v",
            "Inclination change (28.5° → 6°) is very expensive — do it at high altitude to save fuel",
            "Consider combined_transfer for simultaneous altitude + inclination changes (15% savings)",
            "Plan your maneuver sequence before committing to burns",
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def get_task(task_id: str) -> dict:
    """
    Retrieve a task configuration dict by its ID.

    Args:
        task_id: One of 'leo_satellite', 'lunar_orbit', 'asteroid_rendezvous'

    Returns:
        Task configuration dict.

    Raises:
        ValueError: If task_id is not recognised.
    """
    if task_id not in TASKS:
        raise ValueError(
            f"Unknown task_id: '{task_id}'. "
            f"Available tasks: {list(TASKS.keys())}"
        )
    return TASKS[task_id]


def list_tasks() -> list[str]:
    """
    Return all available task IDs in difficulty order (easy → hard).

    Returns:
        List of task ID strings.
    """
    return list(TASKS.keys())


def get_task_summary() -> list[dict]:
    """
    Return a lightweight summary of all tasks (for API listing endpoints).

    Returns:
        List of dicts with task metadata.
    """
    return [
        {
            "task_id":              t["task_id"],
            "name":                 t["name"],
            "difficulty":           t["difficulty"],
            "description":          t["description"],
            "max_steps":            t["max_steps"],
            "theoretical_delta_v":  t["theoretical_delta_v"],
            "delta_v_budget":       t["delta_v_budget"],
            "available_flybys":     t.get("available_flybys", []),
            "mission_constraints":  t.get("mission_constraints", {}),
        }
        for t in TASKS.values()
    ]