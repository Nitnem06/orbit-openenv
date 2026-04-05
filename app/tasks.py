"""
app/tasks.py
Orbit — AI Space Mission Architect

Defines the three missions the agent can attempt.
All values are derived from real astrodynamics references.

Mission difficulty ladder:
  Easy   → Task 1: LEO Satellite Deployment    (1 burn, simple target)
  Medium → Task 2: Lunar Orbit Insertion        (3 burns, long journey)
  Hard   → Task 3: Asteroid Mining Rendezvous   (multi-burn + gravity assists)
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
    #   Total Δ-v from ground : ~9,400 m/s (orbital velocity + launch losses)
    #   Orbital velocity at 400 km : ~7,660 m/s
    #   ISS inclination : 51.6° (chosen to allow crew from Baikonur + KSC)
    #
    # Agent strategy: One large prograde burn (~9,400 m/s) + small inclination correction
    # ─────────────────────────────────────────────────────────────────────────
    "leo_satellite": {
        "task_id":    "leo_satellite",
        "name":        "LEO Satellite Deployment",
        "description": (
            "Launch a satellite from Earth's surface into a 400 km circular orbit "
            "matching the ISS inclination (51.6°). This is the bread-and-butter of "
            "commercial spaceflight — SpaceX, RocketLab and Arianespace do this weekly."
        ),
        "difficulty":  "easy",

        # Starting conditions — spacecraft sitting on the launch pad
        "start_orbit": {
            "altitude_km":      0.0,    # On the ground
            "eccentricity":     0.0,
            "inclination_deg":  0.0,    # Haven't launched yet — no orbit
            "true_anomaly_deg": 0.0,
            "velocity_ms":      0.0,    # Stationary on pad
        },

        # Target: ISS-like 400 km circular orbit
        "target_orbit": {
            "altitude_km":      400.0,  # 400 km above Earth's surface
            "eccentricity":     0.0,    # Perfectly circular
            "inclination_deg":  51.6,   # ISS inclination — covers most launch sites
            "true_anomaly_deg": 0.0,
            "velocity_ms":      7669.0, # Circular orbital velocity at 400 km (from physics.py)
        },

        # Physics reference values
        # launch_to_leo_delta_v(400) = ~9,173 m/s (orbital v + 1,500 m/s launch loss)
        "theoretical_delta_v": 9_400,   # m/s — what a perfect agent should spend
                                         # (slightly above formula to account for inclination burn)

        # Budget: agent has 27% margin above optimal
        # If agent uses more than this → score drops sharply on efficiency component
        "delta_v_budget":      12_000,  # m/s

        "max_steps": 10,  # Easy task — should need only 2-3 actions

        # How close does the agent need to get to call it a success?
        "success_criteria": {
            "altitude_tolerance_km":   50.0,  # Must be within 350–450 km
            "eccentricity_tolerance":   0.05,  # Nearly circular (e < 0.05)
            "inclination_tolerance_deg": 5.0,  # Within 46.6°–56.6°
        },

        "available_flybys": [],  # No gravity assists for LEO mission
    },

    # ─────────────────────────────────────────────────────────────────────────
    # TASK 2 — Medium
    # Goal : Earth parking orbit (200 km) → Lunar orbit (100 km above Moon)
    #
    # Real reference: Apollo missions, ARTEMIS program
    #   TLI burn from 200 km parking orbit : ~3,133 m/s  (trans_lunar_injection_delta_v)
    #   LOI burn into 100 km lunar orbit   : ~834 m/s    (lunar_orbit_insertion_delta_v)
    #   Total optimal Δ-v                  : ~3,967 m/s
    #
    # In our simplified model:
    #   We represent the Moon's position as altitude = 384,400 km (Moon distance from Earth)
    #   The agent must raise its orbit to reach that altitude, then circularize
    #
    # Agent strategy: TLI burn (~3,133 m/s) → coast → LOI burn (~834 m/s)
    # ─────────────────────────────────────────────────────────────────────────
    "lunar_orbit": {
        "task_id":    "lunar_orbit",
        "name":        "Lunar Orbit Insertion",
        "description": (
            "Transfer from a 200 km Earth parking orbit to a 100 km circular lunar orbit. "
            "Requires a Trans-Lunar Injection burn to leave Earth, then a braking burn "
            "to be captured by the Moon's gravity. Based on the Apollo & Artemis mission profiles."
        ),
        "difficulty":  "medium",

        # Starting conditions — already in Earth parking orbit (like Apollo after launch)
        "start_orbit": {
            "altitude_km":      200.0,   # 200 km parking orbit (standard pre-TLI altitude)
            "eccentricity":     0.0,     # Circular
            "inclination_deg":  28.5,    # Kennedy Space Center launch inclination
            "true_anomaly_deg": 0.0,
            "velocity_ms":      7784.0,  # Circular velocity at 200 km
        },

        # Target: 100 km circular orbit around the Moon
        # In our 2D model we represent Moon distance as the altitude value
        "target_orbit": {
            "altitude_km":      384_400.0,  # Moon's mean distance from Earth center (km)
                                             # Agent must reach this altitude to "be at the Moon"
            "eccentricity":     0.0,         # Circular lunar orbit
            "inclination_deg":  28.5,        # Match launch inclination (simplified)
            "true_anomaly_deg": 0.0,
            "velocity_ms":      1_022.0,     # Approximate lunar orbital velocity (Moon orbits Earth)
        },

        # TLI (~3,133 m/s) + LOI (~834 m/s) = ~3,967 m/s
        # We set theoretical to 3,900 m/s (round number, slight efficiency expected)
        "theoretical_delta_v": 3_900,  # m/s

        # Budget: 28% above optimal (mid-course corrections can add ~200–300 m/s)
        "delta_v_budget":      5_000,  # m/s

        "max_steps": 15,  # Medium task — TLI + optional correction + LOI = 3-5 actions

        "success_criteria": {
            "altitude_tolerance_km":    10_000.0,  # Must reach Moon's vicinity (±10,000 km)
                                                    # Large tolerance because our model is simplified
            "eccentricity_tolerance":    0.1,       # Allow slightly elliptical lunar approach
            "inclination_tolerance_deg": 10.0,      # ±10° inclination tolerance
        },

        "available_flybys": [],  # No extra gravity assists — Moon IS the destination
    },

    # ─────────────────────────────────────────────────────────────────────────
    # TASK 3 — Hard
    # Goal : Earth LEO → Asteroid Bennu rendezvous using gravity assists
    #
    # Real reference: NASA OSIRIS-REx mission (2016–2023)
    #   OSIRIS-REx launched: September 2016
    #   Earth gravity assist: September 2017
    #   Bennu arrival: December 2018
    #   Total mission Δ-v (with Earth flyby): ~5,800 m/s
    #
    # Bennu orbital facts:
    #   Semi-major axis : 1.126 AU (~168,000,000 km from Sun)
    #   Eccentricity    : 0.2037 (notably elliptical)
    #   Inclination     : 6.035° to the ecliptic
    #   Close approach  : ~0.005 AU from Earth at closest
    #
    # In our simplified Earth-centric model:
    #   We represent Bennu at a simplified "equivalent" altitude of 120,000,000 km
    #   The agent uses Earth/Venus flybys to gain Δ-v for free
    #
    # Agent strategy: Depart LEO → Earth flyby → (Venus flyby) → Bennu rendezvous
    # ─────────────────────────────────────────────────────────────────────────
    "asteroid_rendezvous": {
        "task_id":    "asteroid_rendezvous",
        "name":        "Asteroid Mining Rendezvous (Bennu)",
        "description": (
            "Reach near-Earth asteroid Bennu from a 400 km LEO orbit using gravity assists. "
            "Modelled on NASA's OSIRIS-REx mission. The agent must plan fuel-efficient burns "
            "and exploit planetary flybys to stretch the Δ-v budget. "
            "Bennu is a priority target for asteroid mining due to its carbon-rich composition."
        ),
        "difficulty":  "hard",

        # Starting conditions — already in a 400 km LEO (post-launch)
        "start_orbit": {
            "altitude_km":      400.0,
            "eccentricity":     0.0,
            "inclination_deg":  28.5,   # KSC launch inclination
            "true_anomaly_deg": 0.0,
            "velocity_ms":      7_669.0,
        },

        # Target: Bennu's orbit (simplified Earth-centric representation)
        "target_orbit": {
            "altitude_km":      120_000_000.0,  # ~0.8 AU (simplified close-approach distance)
            "eccentricity":     0.20,            # Bennu's real eccentricity ≈ 0.2037
            "inclination_deg":  6.0,             # Bennu's real inclination ≈ 6.035°
            "true_anomaly_deg": 0.0,
            "velocity_ms":      28_000.0,        # Bennu's heliocentric speed ≈ 28 km/s
                                                  # (simplified to match our 2D model)
        },

        # OSIRIS-REx used ~5,800 m/s total with Earth gravity assist
        "theoretical_delta_v": 5_800,  # m/s

        # Budget: 38% above optimal (gravity assists help, but timing is hard)
        "delta_v_budget":      8_000,  # m/s

        "max_steps": 25,  # Hard task — many burns + planning steps needed

        "success_criteria": {
            "altitude_tolerance_km":    5_000_000.0,  # ±5,000,000 km (Bennu vicinity)
                                                       # Wide because simplified model
            "eccentricity_tolerance":   0.15,          # Allow e ∈ [0.05, 0.35]
            "inclination_tolerance_deg": 5.0,          # Must match Bennu's plane within 5°
        },

        # Gravity assists the agent can plan (via SetFlybyAction)
        "available_flybys": ["earth", "venus"],
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
    Strips the heavy start/target orbit dicts for brevity.

    Returns:
        List of dicts with: task_id, name, difficulty, max_steps,
        theoretical_delta_v, delta_v_budget
    """
    return [
        {
            "task_id":              t["task_id"],
            "name":                 t["name"],
            "difficulty":           t["difficulty"],
            "max_steps":            t["max_steps"],
            "theoretical_delta_v":  t["theoretical_delta_v"],
            "delta_v_budget":       t["delta_v_budget"],
            "available_flybys":     t.get("available_flybys", []),
        }
        for t in TASKS.values()
    ]