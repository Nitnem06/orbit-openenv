"""
app/physics.py
Orbit — AI Space Mission Architect

All orbital mechanics formulas used by the environment.
Pure functions only — no side effects, no state, fully deterministic.

Key reference values (real physics):
  - Earth radius        : 6,371 km
  - Earth gravity param : 398,600.4418 km³/s²
  - Moon distance       : 384,400 km from Earth center
  - ISS altitude        : ~408 km, velocity ~7,660 m/s
  - Escape velocity LEO : ~11,186 m/s from surface
"""

from __future__ import annotations

import math

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Physical Constants  (all distances in km, velocities in m/s, times in s)
# ─────────────────────────────────────────────────────────────────────────────

# Gravitational parameters (μ = G × M) in km³/s²
MU_EARTH = 398_600.4418   # Earth  — standard, used by every space agency
MU_MOON  =   4_902.8      # Moon   — used for LOI & flyby calculations
MU_SUN   = 132_712_440_018.0  # Sun — used for interplanetary transfers

# Body radii in km
EARTH_RADIUS_KM = 6_371.0    # Mean radius
MOON_RADIUS_KM  = 1_737.4    # Mean radius
VENUS_RADIUS_KM = 6_051.8    # Mean radius
SUN_RADIUS_KM   = 695_700.0

# Key distances in km
MOON_DISTANCE_KM    =    384_400.0   # Earth → Moon (center-to-center, mean)
BENNU_DISTANCE_KM   = 120_000_000.0  # ~0.8 AU simplified, close approach

# Launch loss constant (gravity drag + atmospheric drag during ascent)
# Real value is ~1,500–1,800 m/s depending on trajectory.
LAUNCH_LOSS_MS = 1_500.0

# Minimum safe periapsis altitudes for gravity assists (km above surface)
MIN_PERIAPSIS = {
    "earth": 200.0,
    "moon":  100.0,
    "venus": 300.0,
}

# Gravitational parameters and radii keyed by body name
BODY_MU = {
    "earth": MU_EARTH,
    "moon":  MU_MOON,
    "venus": 324_859.0,   # km³/s²
}
BODY_RADIUS = {
    "earth": EARTH_RADIUS_KM,
    "moon":  MOON_RADIUS_KM,
    "venus": VENUS_RADIUS_KM,
}


# ─────────────────────────────────────────────────────────────────────────────
# Core Orbital Mechanics
# ─────────────────────────────────────────────────────────────────────────────

def orbital_velocity(altitude_km: float, mu: float = MU_EARTH) -> float:
    """
    Circular orbital velocity at a given altitude above Earth (or another body).

    Formula: v = √(μ / r)   where r = body_radius + altitude

    Examples (Earth):
      altitude_km =   200 → ~7,784 m/s  (parking orbit)
      altitude_km =   400 → ~7,669 m/s  (ISS-like)
      altitude_km = 35786 → ~3,075 m/s  (GEO)

    Args:
        altitude_km: Altitude above body surface in km.
        mu:          Gravitational parameter in km³/s² (default = Earth).

    Returns:
        Circular orbital velocity in m/s.
    """
    r = EARTH_RADIUS_KM + altitude_km   # radius from body center (km)
    return float(np.sqrt(mu / r) * 1_000)  # convert km/s → m/s


def orbital_period(altitude_km: float, mu: float = MU_EARTH) -> float:
    """
    Orbital period of a circular orbit at the given altitude.

    Formula: T = 2π √(r³ / μ)

    Examples (Earth):
      altitude_km =   400 → ~5,560 s  (~92 min, ISS)
      altitude_km = 35786 → ~86,164 s (~24 h, GEO)

    Args:
        altitude_km: Altitude above Earth surface in km.
        mu:          Gravitational parameter in km³/s².

    Returns:
        Orbital period in seconds.
    """
    r = EARTH_RADIUS_KM + altitude_km
    return float(2 * np.pi * np.sqrt(r**3 / mu))


def escape_velocity(altitude_km: float, mu: float = MU_EARTH) -> float:
    """
    Escape velocity at a given altitude — minimum speed to leave the body's gravity.

    Formula: v_esc = √(2μ / r)   (note: √2 × v_circular)

    Examples (Earth):
      altitude_km =   0   → ~11,186 m/s  (from surface)
      altitude_km = 200   → ~11,009 m/s  (from parking orbit)
      altitude_km = 400   → ~10,845 m/s

    Args:
        altitude_km: Altitude above Earth surface in km.
        mu:          Gravitational parameter in km³/s².

    Returns:
        Escape velocity in m/s.
    """
    r = EARTH_RADIUS_KM + altitude_km
    return float(np.sqrt(2 * mu / r) * 1_000)


def vis_viva(r_km: float, a_km: float, mu: float = MU_EARTH) -> float:
    """
    Vis-viva equation: velocity at any point on an elliptical orbit.

    Formula: v = √(μ × (2/r − 1/a))

    Used internally for Hohmann transfer calculations.

    Args:
        r_km: Current distance from body center in km.
        a_km: Semi-major axis of the orbit in km.
        mu:   Gravitational parameter in km³/s².

    Returns:
        Velocity in m/s.
    """
    return float(np.sqrt(mu * (2.0 / r_km - 1.0 / a_km)) * 1_000)


# ─────────────────────────────────────────────────────────────────────────────
# Hohmann Transfer
# ─────────────────────────────────────────────────────────────────────────────

def hohmann_transfer_delta_v(
    r1_km: float,
    r2_km: float,
    mu: float = MU_EARTH,
) -> float:
    """
    Total Δ-v for a Hohmann transfer between two CIRCULAR orbits.
    This is the most fuel-efficient 2-burn transfer possible.

    Burn 1: At r1, accelerate into the elliptical transfer orbit.
    Burn 2: At r2, circularize into the target orbit.

    Formula:
      a_transfer = (r1 + r2) / 2
      Δv1 = √(μ × (2/r1 − 1/a)) − √(μ/r1)
      Δv2 = √(μ/r2) − √(μ × (2/r2 − 1/a))
      Total Δv = Δv1 + Δv2

    Examples (Earth, altitudes):
      200 km → 400 km  : ~  56 m/s   (small LEO raise)
      200 km → 35786 km: ~3,905 m/s  (GTO → GEO)
      200 km → 384400 km: ~3,135 m/s  (TLI portion)

    Args:
        r1_km: Starting orbit altitude in km (above Earth surface).
        r2_km: Target orbit altitude in km (above Earth surface).
        mu:    Gravitational parameter in km³/s².

    Returns:
        Total Hohmann transfer Δ-v in m/s.
    """
    # Convert altitudes → radii from Earth center
    r1 = EARTH_RADIUS_KM + r1_km
    r2 = EARTH_RADIUS_KM + r2_km

    # Semi-major axis of the transfer ellipse
    a_transfer = (r1 + r2) / 2.0

    # Circular velocities at r1 and r2
    v1 = np.sqrt(mu / r1)
    v2 = np.sqrt(mu / r2)

    # Velocity at periapsis/apoapsis of the transfer ellipse (vis-viva)
    v_trans_peri = np.sqrt(mu * (2.0 / r1 - 1.0 / a_transfer))
    v_trans_apo  = np.sqrt(mu * (2.0 / r2 - 1.0 / a_transfer))

    dv1 = abs(v_trans_peri - v1)
    dv2 = abs(v2 - v_trans_apo)

    return float((dv1 + dv2) * 1_000)  # km/s → m/s


# ─────────────────────────────────────────────────────────────────────────────
# Mission-Specific Burns
# ─────────────────────────────────────────────────────────────────────────────

def launch_to_leo_delta_v(target_altitude_km: float = 400.0) -> float:
    """
    Total Δ-v required to launch from Earth's surface to a circular LEO orbit.

    Breakdown:
      - Orbital velocity at target altitude      : ~7,660 m/s  (at 400 km)
      - Launch losses (gravity drag + aero drag) : +1,500 m/s  (constant approx)
      Total                                      : ~9,160 m/s

    Real value (SpaceX Falcon 9 to ISS): ~9,400 m/s

    Args:
        target_altitude_km: Target LEO altitude in km.

    Returns:
        Total Δ-v for launch in m/s.
    """
    v_orbital = orbital_velocity(target_altitude_km)
    return float(v_orbital + LAUNCH_LOSS_MS)


def trans_lunar_injection_delta_v(parking_orbit_km: float = 200.0) -> float:
    """
    Δ-v for the Trans-Lunar Injection (TLI) burn.
    Takes spacecraft from a circular Earth parking orbit onto a trajectory to the Moon.

    Method: vis-viva on a Hohmann-like ellipse with apoapsis at Moon distance.

    Real values:
      Apollo TLI from 190 km orbit → ~3,130 m/s

    Args:
        parking_orbit_km: Altitude of the Earth parking orbit in km.

    Returns:
        TLI Δ-v in m/s.
    """
    r_park  = EARTH_RADIUS_KM + parking_orbit_km
    r_moon  = MOON_DISTANCE_KM                     # Moon is ~384,400 km from Earth center

    # Parking orbit circular velocity
    v_park = np.sqrt(MU_EARTH / r_park)

    # Injection velocity (vis-viva at periapsis of transfer ellipse)
    a_transfer = (r_park + r_moon) / 2.0
    v_inject   = np.sqrt(MU_EARTH * (2.0 / r_park - 1.0 / a_transfer))

    return float((v_inject - v_park) * 1_000)


def lunar_orbit_insertion_delta_v(capture_altitude_km: float = 100.0) -> float:
    """
    Δ-v for Lunar Orbit Insertion (LOI) burn.
    Decelerates spacecraft from hyperbolic Moon approach into a circular lunar orbit.

    Real values:
      Apollo LOI from hyperbolic approach → ~850 m/s

    Args:
        capture_altitude_km: Target lunar orbit altitude above Moon's surface in km.

    Returns:
        LOI Δ-v in m/s.
    """
    r_capture = MOON_RADIUS_KM + capture_altitude_km  # radius from Moon center

    # Approach speed relative to the Moon (approximate from TLI trajectory)
    # At arrival, spacecraft is traveling ~800 m/s relative to Moon
    v_approach_ms = 800.0

    # Circular velocity in lunar orbit (vis-viva, circular)
    v_circular_ms = float(np.sqrt(MU_MOON / r_capture) * 1_000)

    return float(abs(v_approach_ms - v_circular_ms))


def gravity_assist_delta_v(
    v_infinity_ms: float,
    periapsis_km: float,
    body: str = "moon",
) -> float:
    """
    Velocity change achievable from a gravity assist (flyby) maneuver.

    A gravity assist bends the spacecraft's trajectory using a planet's gravity,
    changing heliocentric velocity without burning fuel.

    Formula:
      Eccentricity of hyperbolic trajectory: e = 1 + (r_peri × v∞²) / μ
      Turn angle:                            δ = 2 × arcsin(1/e)
      Δv from assist:                        Δv = 2 × v∞ × sin(δ/2)

    Typical values:
      Moon flyby  (500 km periapsis): ~200–400 m/s
      Earth flyby (300 km periapsis): ~1,000–2,000 m/s
      Venus flyby (300 km periapsis): ~2,000–3,000 m/s

    Args:
        v_infinity_ms: Spacecraft speed relative to the body at infinity (m/s).
                       Typical: Moon ≈ 800 m/s | Earth ≈ 3,000 m/s | Venus ≈ 5,000 m/s
        periapsis_km:  Closest approach altitude above body surface in km.
        body:          'moon', 'earth', or 'venus'.

    Returns:
        Δ-v gained from gravity assist in m/s. (Always positive.)
    """
    mu_body     = BODY_MU.get(body, MU_MOON)
    r_body      = BODY_RADIUS.get(body, MOON_RADIUS_KM)
    r_periapsis = r_body + periapsis_km                  # km from body center

    # Convert v_infinity to km/s for consistent μ units
    v_inf_kms = v_infinity_ms / 1_000.0

    # Eccentricity of the hyperbolic flyby trajectory
    e = 1.0 + (r_periapsis * v_inf_kms**2) / mu_body

    # Half-angle of the turn
    turn_angle = 2.0 * np.arcsin(1.0 / e)

    # Δv is the change in velocity magnitude from the turn
    delta_v_kms = 2.0 * v_inf_kms * np.sin(turn_angle / 2.0)

    return float(delta_v_kms * 1_000)  # km/s → m/s


# ─────────────────────────────────────────────────────────────────────────────
# Burn Application — Updates orbital state based on a thruster burn
# ─────────────────────────────────────────────────────────────────────────────

def apply_burn(
    current_orbit: dict,
    delta_v_ms: float,
    prograde: float,
    radial: float,
    normal: float,
) -> dict:
    """
    Apply a thruster burn to the current orbital state and return the new state.

    Direction convention:
      prograde (+1.0) → burns in direction of motion → RAISES the opposite side of the orbit
      prograde (-1.0) → retrograde burn             → LOWERS the opposite side of the orbit
      radial   (+1.0) → burns away from Earth       → changes eccentricity
      normal   (+1.0) → burns perpendicular (north) → changes inclination (EXPENSIVE!)

    Physics model (simplified for hackathon — linear approximations):

      Altitude change:
        Prograde burns change the semi-major axis.
        Using Δa ≈ (2 × a² × Δv_prog) / (μ / a)^0.5 simplified to a linear factor.
        Factor: each 1 m/s prograde ~ 2 km altitude change at LEO.

      Eccentricity change:
        Radial burns change eccentricity.
        Factor: each 1 m/s radial at 400 km ~ 0.00013 eccentricity change.

      Inclination change:
        Normal burns change inclination.
        Formula: Δi ≈ Δv_normal / v_orbital (in radians), converted to degrees.
        This is physically accurate and very expensive (costs ~130 m/s per degree at LEO).

    Args:
        current_orbit: dict with keys: altitude_km, eccentricity, inclination_deg,
                       true_anomaly_deg, velocity_ms
        delta_v_ms:   Burn magnitude in m/s.
        prograde:     Prograde direction component [-1.0, 1.0].
        radial:       Radial direction component [-1.0, 1.0].
        normal:       Normal direction component [-1.0, 1.0].

    Returns:
        New orbital state dict with updated parameters.
    """
    # Normalize direction vector so components don't stack unfairly
    magnitude = math.sqrt(prograde**2 + radial**2 + normal**2)
    if magnitude < 1e-9:
        # Zero vector — no direction, no effect
        return dict(current_orbit)

    p_norm = prograde / magnitude
    r_norm = radial   / magnitude
    n_norm = normal   / magnitude

    # Effective Δ-v in each direction
    dv_prog   = delta_v_ms * p_norm
    dv_radial = delta_v_ms * r_norm
    dv_normal = delta_v_ms * n_norm

    # Current state
    alt      = current_orbit["altitude_km"]
    ecc      = current_orbit["eccentricity"]
    inc      = current_orbit["inclination_deg"]
    anom     = current_orbit.get("true_anomaly_deg", 0.0)
    v_orb    = current_orbit.get("velocity_ms", None)

    # If velocity not stored, calculate it
    if v_orb is None or v_orb <= 0:
        v_orb = orbital_velocity(max(alt, 0.1))

    # ── Prograde: altitude change ─────────────────────────────────────────────
    # At circular orbit: Δa ≈ 2 × a × Δv / v_circular
    # Here a = r = EARTH_RADIUS + alt (simplified: treat as circular at current alt)
    r_current_km = EARTH_RADIUS_KM + max(alt, 0.1)
    v_circ_kms   = math.sqrt(MU_EARTH / r_current_km)   # km/s
    dv_prog_kms  = dv_prog / 1_000.0                     # m/s → km/s

    # Δ semi-major axis in km
    delta_a_km   = 2.0 * r_current_km * dv_prog_kms / v_circ_kms
    new_alt      = alt + delta_a_km
    new_alt      = max(80.0, min(500_000_000.0, new_alt))   # clamp: min 80 km, max far space

    # ── Radial: eccentricity change ───────────────────────────────────────────
    # Simplified: radial burns at true anomaly ≈ 0 change eccentricity
    # Factor derived from Gauss's equations (linearised)
    dv_rad_kms   = dv_radial / 1_000.0
    delta_ecc    = (2.0 * dv_rad_kms * math.sin(math.radians(anom))) / v_circ_kms
    # At anom=0 this is 0 (correct — radial at periapsis doesn't change ecc much)
    # Add a small base effect to make agent actions visible
    delta_ecc   += (dv_radial / 1_000.0) * 0.0001
    new_ecc      = ecc + delta_ecc
    new_ecc      = max(0.0, min(0.99, new_ecc))

    # ── Normal: inclination change ────────────────────────────────────────────
    # Most accurate formula: Δi = Δv_normal / v_circular  (in radians)
    # This correctly makes inclination changes very expensive
    dv_norm_kms  = dv_normal / 1_000.0
    delta_inc_rad = dv_norm_kms / v_circ_kms
    delta_inc_deg = math.degrees(delta_inc_rad)
    new_inc       = inc + delta_inc_deg
    new_inc       = max(-90.0, min(90.0, new_inc))

    # ── New orbital velocity at new altitude ──────────────────────────────────
    new_velocity  = orbital_velocity(new_alt)

    return {
        "altitude_km":      round(new_alt,  4),
        "eccentricity":     round(new_ecc,  6),
        "inclination_deg":  round(new_inc,  4),
        "true_anomaly_deg": round(anom,     4),
        "velocity_ms":      round(new_velocity, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Scoring Helpers — used by grader.py
# ─────────────────────────────────────────────────────────────────────────────

def fuel_efficiency_ratio(delta_v_used_ms: float, theoretical_delta_v_ms: float) -> float:
    """
    How efficient was the agent compared to the theoretical optimum?

    Ratio = theoretical / actual
      = 1.0  → perfect (used exactly the minimum possible Δ-v)
      > 1.0  → impossible (capped at 1.0)
      < 1.0  → inefficient (used more fuel than needed)

    Args:
        delta_v_used_ms:       Actual Δ-v used by agent in m/s.
        theoretical_delta_v_ms: Optimal Δ-v from physics formulas in m/s.

    Returns:
        Efficiency ratio capped at 1.0.
    """
    if delta_v_used_ms <= 0:
        return 0.0
    ratio = theoretical_delta_v_ms / delta_v_used_ms
    return float(min(1.0, ratio))


def proximity_score(actual: float, target: float, tolerance: float) -> float:
    """
    How close is the actual value to the target?

    Score = max(0, 1 − |actual − target| / tolerance)
      = 1.0  → exactly on target
      = 0.5  → halfway between target and tolerance edge
      = 0.0  → at or beyond tolerance

    Used for altitude, eccentricity, and inclination scoring.

    Args:
        actual:    Current value.
        target:    Target value.
        tolerance: Acceptable error range.

    Returns:
        Score between 0.0 and 1.0.
    """
    if tolerance <= 0:
        return 1.0 if actual == target else 0.0
    error = abs(actual - target)
    return float(max(0.0, 1.0 - error / tolerance))