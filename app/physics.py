"""
app/physics.py
Orbit — AI Space Mission Architect

All orbital mechanics formulas used by the environment.
Pure functions only — no side effects, no state, fully deterministic.

v2.0 Changes:
    - Added plane_change_delta_v() and circularize_delta_v() helpers
    - Added 8 execute_* functions for high-level strategic maneuvers
    - Added estimate_maneuver_cost() for observation enrichment
    - Added compute_mission_analysis() for real-time progress tracking
    - Added get_available_maneuvers() and get_recommendations()
    - All new functions are pure, deterministic, and side-effect-free

Key reference values (real physics):
  - Earth radius        : 6,371 km
  - Earth gravity param : 398,600.4418 km³/s²
  - Moon distance       : 384,400 km from Earth center
  - ISS altitude        : ~408 km, velocity ~7,660 m/s
  - Escape velocity LEO : ~11,186 m/s from surface
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Physical Constants  (all distances in km, velocities in m/s, times in s)
# ─────────────────────────────────────────────────────────────────────────────

# Gravitational parameters (μ = G × M) in km³/s²
MU_EARTH = 398_600.4418
MU_MOON  =   4_902.8
MU_SUN   = 132_712_440_018.0

# Body radii in km
EARTH_RADIUS_KM = 6_371.0
MOON_RADIUS_KM  = 1_737.4
VENUS_RADIUS_KM = 6_051.8
SUN_RADIUS_KM   = 695_700.0

# Key distances in km
MOON_DISTANCE_KM    =    384_400.0
BENNU_DISTANCE_KM   = 120_000_000.0

# Launch loss constant (gravity drag + atmospheric drag during ascent)
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
    "venus": 324_859.0,
}
BODY_RADIUS = {
    "earth": EARTH_RADIUS_KM,
    "moon":  MOON_RADIUS_KM,
    "venus": VENUS_RADIUS_KM,
}

# Default approach velocities for gravity assists (m/s)
# Used when no specific v_infinity is available from trajectory state
DEFAULT_V_INFINITY = {
    "moon":  800.0,
    "earth": 3_000.0,
    "venus": 5_000.0,
}

# Default flyby periapsis altitudes (km above surface)
DEFAULT_PERIAPSIS = {
    "moon":  500.0,
    "earth": 300.0,
    "venus": 400.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Core Orbital Mechanics
# ─────────────────────────────────────────────────────────────────────────────

def orbital_velocity(altitude_km: float, mu: float = MU_EARTH) -> float:
    """
    Circular orbital velocity at a given altitude above Earth (or another body).

    Formula: v = √(μ / r)   where r = body_radius + altitude

    Args:
        altitude_km: Altitude above body surface in km.
        mu:          Gravitational parameter in km³/s² (default = Earth).

    Returns:
        Circular orbital velocity in m/s.
    """
    r = EARTH_RADIUS_KM + altitude_km
    return float(np.sqrt(mu / r) * 1_000)


def orbital_period(altitude_km: float, mu: float = MU_EARTH) -> float:
    """
    Orbital period of a circular orbit at the given altitude.

    Formula: T = 2π √(r³ / μ)

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

    Formula: v_esc = √(2μ / r)

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

    Args:
        r1_km: Starting orbit altitude in km (above Earth surface).
        r2_km: Target orbit altitude in km (above Earth surface).
        mu:    Gravitational parameter in km³/s².

    Returns:
        Total Hohmann transfer Δ-v in m/s.
    """
    r1 = EARTH_RADIUS_KM + r1_km
    r2 = EARTH_RADIUS_KM + r2_km

    a_transfer = (r1 + r2) / 2.0

    v1 = np.sqrt(mu / r1)
    v2 = np.sqrt(mu / r2)

    v_trans_peri = np.sqrt(mu * (2.0 / r1 - 1.0 / a_transfer))
    v_trans_apo  = np.sqrt(mu * (2.0 / r2 - 1.0 / a_transfer))

    dv1 = abs(v_trans_peri - v1)
    dv2 = abs(v2 - v_trans_apo)

    return float((dv1 + dv2) * 1_000)


# ─────────────────────────────────────────────────────────────────────────────
# Mission-Specific Burns
# ─────────────────────────────────────────────────────────────────────────────

def launch_to_leo_delta_v(target_altitude_km: float = 400.0) -> float:
    """
    Total Δ-v required to launch from Earth's surface to a circular LEO orbit.
    Includes gravity and atmospheric drag losses (~1,500 m/s).

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

    Args:
        parking_orbit_km: Altitude of the Earth parking orbit in km.

    Returns:
        TLI Δ-v in m/s.
    """
    r_park = EARTH_RADIUS_KM + parking_orbit_km
    r_moon = MOON_DISTANCE_KM

    v_park = np.sqrt(MU_EARTH / r_park)

    a_transfer = (r_park + r_moon) / 2.0
    v_inject   = np.sqrt(MU_EARTH * (2.0 / r_park - 1.0 / a_transfer))

    return float((v_inject - v_park) * 1_000)


def lunar_orbit_insertion_delta_v(capture_altitude_km: float = 100.0) -> float:
    """
    Δ-v for Lunar Orbit Insertion (LOI) burn.

    Args:
        capture_altitude_km: Target lunar orbit altitude above Moon's surface in km.

    Returns:
        LOI Δ-v in m/s.
    """
    r_capture = MOON_RADIUS_KM + capture_altitude_km

    v_approach_ms = 800.0
    v_circular_ms = float(np.sqrt(MU_MOON / r_capture) * 1_000)

    return float(abs(v_approach_ms - v_circular_ms))


def gravity_assist_delta_v(
    v_infinity_ms: float,
    periapsis_km: float,
    body: str = "moon",
) -> float:
    """
    Velocity change achievable from a gravity assist (flyby) maneuver.

    Args:
        v_infinity_ms: Spacecraft speed relative to the body at infinity (m/s).
        periapsis_km:  Closest approach altitude above body surface in km.
        body:          'moon', 'earth', or 'venus'.

    Returns:
        Δ-v gained from gravity assist in m/s. (Always positive.)
    """
    mu_body     = BODY_MU.get(body, MU_MOON)
    r_body      = BODY_RADIUS.get(body, MOON_RADIUS_KM)
    r_periapsis = r_body + periapsis_km

    v_inf_kms = v_infinity_ms / 1_000.0

    e = 1.0 + (r_periapsis * v_inf_kms**2) / mu_body

    turn_angle = 2.0 * np.arcsin(1.0 / e)

    delta_v_kms = 2.0 * v_inf_kms * np.sin(turn_angle / 2.0)

    return float(delta_v_kms * 1_000)


# ─────────────────────────────────────────────────────────────────────────────
# Additional Δ-v Helpers (NEW — needed by maneuver execution)
# ─────────────────────────────────────────────────────────────────────────────

def plane_change_delta_v(altitude_km: float, delta_inclination_deg: float) -> float:
    """
    Δ-v required to change orbital inclination at a given altitude.

    Formula: Δv = 2 × v_circular × sin(Δi / 2)

    This is the exact solution for a pure plane rotation maneuver.
    Plane changes are VERY expensive — ~134 m/s per degree at LEO.

    Examples (at 400 km):
        1°  change → ~134 m/s
        5°  change → ~669 m/s
        10° change → ~1,337 m/s
        51.6° change → ~6,672 m/s

    Args:
        altitude_km:          Current orbit altitude in km.
        delta_inclination_deg: Absolute inclination change in degrees.

    Returns:
        Plane change Δ-v in m/s.
    """
    if abs(delta_inclination_deg) < 0.01:
        return 0.0

    v_circ = orbital_velocity(altitude_km)
    delta_i_rad = math.radians(abs(delta_inclination_deg))

    return float(2.0 * v_circ * math.sin(delta_i_rad / 2.0))


def circularize_delta_v(altitude_km: float, eccentricity: float) -> float:
    """
    Δ-v required to circularize an orbit (reduce eccentricity to ~0).

    Simplified formula: Δv ≈ v_circular × eccentricity
    Valid for small eccentricities (< 0.3). For larger eccentricities,
    uses a more accurate vis-viva based calculation.

    Examples (at 400 km):
        e = 0.001 → ~8 m/s
        e = 0.01  → ~77 m/s
        e = 0.1   → ~767 m/s
        e = 0.5   → ~3,835 m/s

    Args:
        altitude_km:  Current orbit altitude in km.
        eccentricity: Current orbital eccentricity (0.0 to 0.99).

    Returns:
        Circularization Δ-v in m/s.
    """
    if eccentricity < 0.001:
        return 0.0

    v_circ = orbital_velocity(altitude_km)

    if eccentricity < 0.3:
        # Linear approximation — good for small e
        return float(v_circ * eccentricity)
    else:
        # More accurate: difference between periapsis velocity and circular velocity
        r_km = EARTH_RADIUS_KM + altitude_km
        a_km = r_km / (1.0 - eccentricity)  # semi-major axis from periapsis
        v_peri = vis_viva(r_km, a_km)
        return float(abs(v_peri - v_circ))


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
    magnitude = math.sqrt(prograde**2 + radial**2 + normal**2)
    if magnitude < 1e-9:
        return dict(current_orbit)

    p_norm = prograde / magnitude
    r_norm = radial   / magnitude
    n_norm = normal   / magnitude

    dv_prog   = delta_v_ms * p_norm
    dv_radial = delta_v_ms * r_norm
    dv_normal = delta_v_ms * n_norm

    alt      = current_orbit["altitude_km"]
    ecc      = current_orbit["eccentricity"]
    inc      = current_orbit["inclination_deg"]
    anom     = current_orbit.get("true_anomaly_deg", 0.0)
    v_orb    = current_orbit.get("velocity_ms", None)

    if v_orb is None or v_orb <= 0:
        v_orb = orbital_velocity(max(alt, 0.1))

    r_current_km = EARTH_RADIUS_KM + max(alt, 0.1)
    v_circ_kms   = math.sqrt(MU_EARTH / r_current_km)
    dv_prog_kms  = dv_prog / 1_000.0

    delta_a_km   = 2.0 * r_current_km * dv_prog_kms / v_circ_kms
    new_alt      = alt + delta_a_km
    new_alt      = max(80.0, min(500_000_000.0, new_alt))

    dv_rad_kms   = dv_radial / 1_000.0
    delta_ecc    = (2.0 * dv_rad_kms * math.sin(math.radians(anom))) / v_circ_kms
    delta_ecc   += (dv_radial / 1_000.0) * 0.0001
    new_ecc      = ecc + delta_ecc
    new_ecc      = max(0.0, min(0.99, new_ecc))

    dv_norm_kms  = dv_normal / 1_000.0
    delta_inc_rad = dv_norm_kms / v_circ_kms
    delta_inc_deg = math.degrees(delta_inc_rad)
    new_inc       = inc + delta_inc_deg
    new_inc       = max(-90.0, min(90.0, new_inc))

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
    Penalizes both overspending AND significant underspending.
    Underspending usually means the mission is incomplete.
    """
    if delta_v_used_ms <= 0:
        return 0.0
    ratio = theoretical_delta_v_ms / delta_v_used_ms
    if ratio > 1.3:
        # Used significantly less fuel than needed — likely incomplete
        return max(0.2, 1.0 - (ratio - 1.0) * 0.7)
    return float(min(1.0, ratio))


def proximity_score(actual: float, target: float, tolerance: float) -> float:
    """
    How close is the actual value to the target?

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


# ─────────────────────────────────────────────────────────────────────────────
# HIGH-LEVEL MANEUVER EXECUTION (NEW)
# Called by env.py when processing ExecuteManeuverAction
# Each function returns: (new_orbit_dict, delta_v_consumed)
# ─────────────────────────────────────────────────────────────────────────────

def execute_hohmann_transfer(
    current_orbit: dict,
    target_altitude_km: float,
) -> Tuple[dict, float]:
    """
    Execute a Hohmann transfer to a new altitude.
    Calculates required Δ-v internally, applies the orbit change.

    Args:
        current_orbit: Current orbital state dict.
        target_altitude_km: Desired altitude after transfer.

    Returns:
        (new_orbit_dict, delta_v_consumed_ms)
    """
    current_alt = current_orbit["altitude_km"]

    # Handle launch from ground (altitude ~0)
    if current_alt < 100.0:
        dv = launch_to_leo_delta_v(target_altitude_km)
    else:
        dv = hohmann_transfer_delta_v(current_alt, target_altitude_km)

    new_orbit = {
        "altitude_km":      round(target_altitude_km, 4),
        "eccentricity":     round(current_orbit["eccentricity"], 6),
        "inclination_deg":  round(current_orbit["inclination_deg"], 4),
        "true_anomaly_deg": round(current_orbit.get("true_anomaly_deg", 0.0), 4),
        "velocity_ms":      round(orbital_velocity(target_altitude_km), 2),
    }

    return new_orbit, round(dv, 2)


def execute_plane_change(
    current_orbit: dict,
    target_inclination_deg: float,
) -> Tuple[dict, float]:
    """
    Execute a plane change maneuver to a new inclination.

    Args:
        current_orbit: Current orbital state dict.
        target_inclination_deg: Desired inclination in degrees.

    Returns:
        (new_orbit_dict, delta_v_consumed_ms)
    """
    current_inc = current_orbit["inclination_deg"]
    delta_inc = abs(target_inclination_deg - current_inc)

    dv = plane_change_delta_v(current_orbit["altitude_km"], delta_inc)

    new_orbit = dict(current_orbit)
    new_orbit["inclination_deg"] = round(target_inclination_deg, 4)

    return new_orbit, round(dv, 2)


def execute_circularize(current_orbit: dict) -> Tuple[dict, float]:
    """
    Circularize the current orbit (reduce eccentricity to ~0).

    Args:
        current_orbit: Current orbital state dict.

    Returns:
        (new_orbit_dict, delta_v_consumed_ms)
    """
    ecc = current_orbit["eccentricity"]

    dv = circularize_delta_v(current_orbit["altitude_km"], ecc)

    new_orbit = dict(current_orbit)
    new_orbit["eccentricity"] = 0.0

    return new_orbit, round(dv, 2)


def execute_trans_lunar_injection(current_orbit: dict) -> Tuple[dict, float]:
    """
    Execute Trans-Lunar Injection from current parking orbit.
    Puts spacecraft on a trajectory toward the Moon.

    Args:
        current_orbit: Current orbital state dict (should be in Earth parking orbit).

    Returns:
        (new_orbit_dict, delta_v_consumed_ms)
    """
    parking_alt = current_orbit["altitude_km"]

    dv = trans_lunar_injection_delta_v(parking_alt)

    # After TLI, spacecraft is on a highly elliptical transfer orbit
    # Approximate: altitude jumps to ~halfway to Moon, high eccentricity
    transfer_alt = MOON_DISTANCE_KM * 0.5  # midpoint approximation
    new_orbit = {
        "altitude_km":      round(transfer_alt, 4),
        "eccentricity":     0.97,  # highly elliptical transfer
        "inclination_deg":  round(current_orbit["inclination_deg"], 4),
        "true_anomaly_deg": 0.0,
        "velocity_ms":      round(orbital_velocity(transfer_alt), 2),
    }

    return new_orbit, round(dv, 2)


def execute_lunar_orbit_insertion(
    current_orbit: dict,
    target_altitude_km: float = 100.0,
) -> Tuple[dict, float]:
    """
    Execute Lunar Orbit Insertion — capture into orbit around the Moon.
    Realistic: LOI leaves residual eccentricity (~0.05) due to imperfect
    capture geometry. Agent must decide whether to circularize (costs fuel)
    or accept the imperfect orbit.
    """
    dv = lunar_orbit_insertion_delta_v(target_altitude_km)

    r_capture = MOON_RADIUS_KM + target_altitude_km
    v_lunar = float(np.sqrt(MU_MOON / r_capture) * 1_000)

    new_orbit = {
        "altitude_km":      round(MOON_DISTANCE_KM + target_altitude_km, 4),
        "eccentricity":     0.05,   # Realistic: LOI doesn't perfectly circularize
        "inclination_deg":  round(current_orbit["inclination_deg"], 4),
        "true_anomaly_deg": 0.0,
        "velocity_ms":      round(v_lunar, 2),
    }

    return new_orbit, round(dv, 2)


def execute_gravity_assist(
    current_orbit: dict,
    body: str,
) -> Tuple[dict, float]:
    """
    Execute a gravity assist flyby around the specified body.
    FREE — no fuel consumed. Changes velocity and adds eccentricity perturbation.

    Effectiveness is 40% of theoretical maximum to account for
    non-ideal flyby geometry, timing constraints, and trajectory alignment.
    Flyby also introduces eccentricity perturbation (realistic orbital effect).
    """
    v_infinity = DEFAULT_V_INFINITY.get(body, 800.0)
    periapsis = DEFAULT_PERIAPSIS.get(body, 500.0)

    # Calculate theoretical free Δ-v from assist
    theoretical_dv = gravity_assist_delta_v(v_infinity, periapsis, body)

    # Apply realistic effectiveness factor (40%)
    effective_dv = theoretical_dv * 0.40

    # Apply the free Δ-v as a prograde effect on the orbit
    current_alt = current_orbit["altitude_km"]
    r_current_km = EARTH_RADIUS_KM + max(current_alt, 0.1)
    v_circ_kms = math.sqrt(MU_EARTH / r_current_km)
    effective_dv_kms = effective_dv / 1_000.0

    delta_a_km = 2.0 * r_current_km * effective_dv_kms / v_circ_kms
    new_alt = current_alt + delta_a_km
    new_alt = max(80.0, min(500_000_000.0, new_alt))

    # Gravity assist introduces eccentricity perturbation (realistic)
    # Flybys change the orbit shape, not just the speed
    new_ecc = current_orbit["eccentricity"] + 0.02

    new_orbit = {
        "altitude_km":      round(new_alt, 4),
        "eccentricity":     round(min(0.99, new_ecc), 6),
        "inclination_deg":  round(current_orbit["inclination_deg"], 4),
        "true_anomaly_deg": round(current_orbit.get("true_anomaly_deg", 0.0), 4),
        "velocity_ms":      round(orbital_velocity(new_alt), 2),
    }

    return new_orbit, 0.0


def execute_combined_transfer(
    current_orbit: dict,
    target_altitude_km: float,
    target_inclination_deg: float,
    target_eccentricity: float = None,
) -> Tuple[dict, float]:
    """
    Execute a combined altitude + inclination + eccentricity change.
    More efficient than doing them separately (combined maneuver saves ~15% Δ-v).

    For extreme altitude changes (ratio > 500x), navigation uncertainty
    introduces residual targeting errors. This is physically realistic —
    deep space transfers have larger navigation uncertainties than LEO maneuvers.
    """
    current_alt = current_orbit["altitude_km"]

    # Altitude change cost
    dv_hohmann = hohmann_transfer_delta_v(
        current_alt, target_altitude_km
    ) if abs(current_alt - target_altitude_km) > 1.0 else 0.0

    # Inclination change cost
    delta_inc = abs(target_inclination_deg - current_orbit["inclination_deg"])
    dv_plane = plane_change_delta_v(current_alt, delta_inc)

    # Eccentricity change cost
    target_ecc = target_eccentricity if target_eccentricity is not None else current_orbit["eccentricity"]
    ecc_diff = abs(target_ecc - current_orbit["eccentricity"])
    dv_ecc = 0.0
    if ecc_diff > 0.001:
        v_circ = orbital_velocity(current_alt)
        dv_ecc = v_circ * ecc_diff

    # Combined is ~15% cheaper than sum of individual maneuvers
    combined_dv = (dv_hohmann + dv_plane + dv_ecc) * 0.85

    # Navigation uncertainty for extreme altitude changes
    # Deep space transfers (>500x altitude ratio) have targeting errors
    alt_ratio = target_altitude_km / max(current_alt, 1.0)
    ecc_uncertainty = 0.0
    inc_uncertainty = 0.0
    if alt_ratio > 500:
        uncertainty = min(0.04, math.log10(alt_ratio) / 125.0)
        ecc_uncertainty = uncertainty
        inc_uncertainty = uncertainty * 15.0  # degrees

    # Apply achieved orbit (with uncertainty if applicable)
    new_orbit = {
        "altitude_km":      round(target_altitude_km, 4),
        "eccentricity":     round(min(0.99, target_ecc + ecc_uncertainty), 6),
        "inclination_deg":  round(target_inclination_deg + inc_uncertainty, 4),
        "true_anomaly_deg": round(current_orbit.get("true_anomaly_deg", 0.0), 4),
        "velocity_ms":      round(orbital_velocity(target_altitude_km), 2),
    }

    return new_orbit, round(combined_dv, 2)


def execute_correction_burn(
    current_orbit: dict,
    delta_v_ms: float,
) -> Tuple[dict, float]:
    """
    Execute a small correction burn (max 500 m/s).
    Applied as a prograde burn for fine-tuning orbit parameters.

    Args:
        current_orbit: Current orbital state dict.
        delta_v_ms: Small burn magnitude (0-500 m/s).

    Returns:
        (new_orbit_dict, delta_v_consumed_ms)
    """
    clamped_dv = min(500.0, max(0.0, delta_v_ms))

    new_orbit = apply_burn(
        current_orbit,
        clamped_dv,
        prograde=1.0,
        radial=0.0,
        normal=0.0,
    )

    return new_orbit, round(clamped_dv, 2)


# ─────────────────────────────────────────────────────────────────────────────
# OBSERVATION ENRICHMENT (NEW)
# Used by env.py to build enriched observations for LLM agents
# ─────────────────────────────────────────────────────────────────────────────

def estimate_maneuver_cost(
    current_orbit: dict,
    target_orbit: dict,
    maneuver: str,
    task_id: str = "",
    body: Optional[str] = None,
) -> float:
    """
    Estimate the Δ-v cost of a specific maneuver from the current state.

    Args:
        current_orbit: Current orbital state dict.
        target_orbit: Target orbital state dict.
        maneuver: ManeuverType string value.
        task_id: Current task ID for context.
        body: Body name for gravity assist.

    Returns:
        Estimated Δ-v in m/s.
    """
    current_alt = current_orbit["altitude_km"]
    target_alt = target_orbit["altitude_km"]
    current_inc = current_orbit["inclination_deg"]
    target_inc = target_orbit["inclination_deg"]
    current_ecc = current_orbit["eccentricity"]

    if maneuver == "hohmann_transfer":
        if current_alt < 100.0:
            return launch_to_leo_delta_v(target_alt)
        return hohmann_transfer_delta_v(current_alt, target_alt)

    elif maneuver == "plane_change":
        delta_inc = abs(target_inc - current_inc)
        return plane_change_delta_v(current_alt, delta_inc)

    elif maneuver == "circularize":
        return circularize_delta_v(current_alt, current_ecc)

    elif maneuver == "trans_lunar_injection":
        return trans_lunar_injection_delta_v(current_alt)

    elif maneuver == "lunar_orbit_insertion":
        return lunar_orbit_insertion_delta_v(100.0)

    elif maneuver == "gravity_assist":
        return 0.0  # Free!

    elif maneuver == "combined_transfer":
        dv_h = hohmann_transfer_delta_v(current_alt, target_alt) if abs(current_alt - target_alt) > 1.0 else 0.0
        dv_p = plane_change_delta_v(current_alt, abs(target_inc - current_inc))
        target_ecc = target_orbit.get("eccentricity", current_ecc)
        ecc_diff = abs(target_ecc - current_ecc)
        dv_e = orbital_velocity(current_alt) * ecc_diff if ecc_diff > 0.001 else 0.0
        return (dv_h + dv_p + dv_e) * 0.85

    elif maneuver == "correction_burn":
        return 50.0  # Default small correction

    return 0.0


def get_available_maneuvers(
    current_orbit: dict,
    target_orbit: dict,
    delta_v_remaining: float,
    task_id: str = "",
) -> List[Dict]:
    """
    Generate a list of available maneuvers with costs and feasibility.
    This is the PRIMARY decision input for LLM agents.

    Args:
        current_orbit: Current orbital state dict.
        target_orbit: Target orbital state dict.
        delta_v_remaining: Remaining fuel budget in m/s.
        task_id: Current task ID.

    Returns:
        List of dicts matching AvailableManeuver schema.
    """
    maneuvers = []
    current_alt = current_orbit["altitude_km"]
    target_alt = target_orbit["altitude_km"]
    current_inc = current_orbit["inclination_deg"]
    target_inc = target_orbit["inclination_deg"]
    current_ecc = current_orbit["eccentricity"]

    alt_diff = abs(current_alt - target_alt)
    inc_diff = abs(current_inc - target_inc)

    # ── Hohmann Transfer ──
    if alt_diff > 1.0:
        cost = estimate_maneuver_cost(current_orbit, target_orbit, "hohmann_transfer", task_id)
        feasible = cost <= delta_v_remaining
        fuel_pct = (cost / delta_v_remaining * 100.0) if delta_v_remaining > 0 else 999.0
        maneuvers.append({
            "name": "hohmann_transfer",
            "description": f"Transfer from {current_alt:.0f} km to {target_alt:.0f} km altitude using Hohmann transfer ellipse",
            "estimated_delta_v": round(cost, 1),
            "fuel_percentage": round(fuel_pct, 1),
            "feasible": feasible,
            "reason": None if feasible else f"Insufficient fuel: need {cost:.0f} m/s, have {delta_v_remaining:.0f} m/s",
        })

    # ── Plane Change ──
    if inc_diff > 0.1:
        cost = estimate_maneuver_cost(current_orbit, target_orbit, "plane_change", task_id)
        feasible = cost <= delta_v_remaining
        fuel_pct = (cost / delta_v_remaining * 100.0) if delta_v_remaining > 0 else 999.0
        maneuvers.append({
            "name": "plane_change",
            "description": f"Change inclination from {current_inc:.1f}° to {target_inc:.1f}° ({inc_diff:.1f}° change)",
            "estimated_delta_v": round(cost, 1),
            "fuel_percentage": round(fuel_pct, 1),
            "feasible": feasible,
            "reason": None if feasible else f"Insufficient fuel: need {cost:.0f} m/s, have {delta_v_remaining:.0f} m/s",
        })

    # ── Circularize ──
    if current_ecc > 0.005:
        cost = estimate_maneuver_cost(current_orbit, target_orbit, "circularize", task_id)
        feasible = cost <= delta_v_remaining
        fuel_pct = (cost / delta_v_remaining * 100.0) if delta_v_remaining > 0 else 999.0
        maneuvers.append({
            "name": "circularize",
            "description": f"Reduce eccentricity from {current_ecc:.4f} to ~0 (circular orbit)",
            "estimated_delta_v": round(cost, 1),
            "fuel_percentage": round(fuel_pct, 1),
            "feasible": feasible,
            "reason": None if feasible else f"Insufficient fuel: need {cost:.0f} m/s, have {delta_v_remaining:.0f} m/s",
        })

    # ── Trans-Lunar Injection (Task 2 only) ──
    if task_id == "lunar_orbit" and current_alt < 1000.0:
        cost = estimate_maneuver_cost(current_orbit, target_orbit, "trans_lunar_injection", task_id)
        feasible = cost <= delta_v_remaining
        fuel_pct = (cost / delta_v_remaining * 100.0) if delta_v_remaining > 0 else 999.0
        maneuvers.append({
            "name": "trans_lunar_injection",
            "description": f"Trans-Lunar Injection burn from {current_alt:.0f} km parking orbit toward the Moon",
            "estimated_delta_v": round(cost, 1),
            "fuel_percentage": round(fuel_pct, 1),
            "feasible": feasible,
            "reason": None if feasible else f"Insufficient fuel: need {cost:.0f} m/s, have {delta_v_remaining:.0f} m/s",
        })

    # ── Lunar Orbit Insertion (Task 2, after TLI) ──
    if task_id == "lunar_orbit" and current_alt > 100_000.0:
        cost = estimate_maneuver_cost(current_orbit, target_orbit, "lunar_orbit_insertion", task_id)
        feasible = cost <= delta_v_remaining
        fuel_pct = (cost / delta_v_remaining * 100.0) if delta_v_remaining > 0 else 999.0
        maneuvers.append({
            "name": "lunar_orbit_insertion",
            "description": f"Capture into lunar orbit at ~100 km above Moon surface",
            "estimated_delta_v": round(cost, 1),
            "fuel_percentage": round(fuel_pct, 1),
            "feasible": feasible,
            "reason": None if feasible else f"Insufficient fuel: need {cost:.0f} m/s, have {delta_v_remaining:.0f} m/s",
        })

    # ── Gravity Assist (Task 3 primarily) ──
    if task_id == "asteroid_rendezvous":
        for body in ["venus", "earth"]:
            theoretical_dv = gravity_assist_delta_v(
                DEFAULT_V_INFINITY[body], DEFAULT_PERIAPSIS[body], body
            )
            effective_dv = theoretical_dv * 0.40
            maneuvers.append({
                "name": "gravity_assist",
                "description": f"Gravity assist flyby around {body.title()} — FREE Δ-v gain of ~{effective_dv:.0f} m/s (40% of theoretical {theoretical_dv:.0f} m/s). WARNING: Adds +0.02 eccentricity perturbation.",
                "estimated_delta_v": 0.0,
                "fuel_percentage": 0.0,
                "feasible": True,
                "reason": None,
            })

    # ── Combined Transfer ──
    if alt_diff > 1.0 and inc_diff > 0.1:
        cost = estimate_maneuver_cost(current_orbit, target_orbit, "combined_transfer", task_id)
        feasible = cost <= delta_v_remaining
        fuel_pct = (cost / delta_v_remaining * 100.0) if delta_v_remaining > 0 else 999.0
        maneuvers.append({
            "name": "combined_transfer",
            "description": f"Combined altitude ({current_alt:.0f}→{target_alt:.0f} km) and inclination ({current_inc:.1f}→{target_inc:.1f}°) change — 15% cheaper than separate maneuvers",
            "estimated_delta_v": round(cost, 1),
            "fuel_percentage": round(fuel_pct, 1),
            "feasible": feasible,
            "reason": None if feasible else f"Insufficient fuel: need {cost:.0f} m/s, have {delta_v_remaining:.0f} m/s",
        })

    # ── Correction Burn (always available) ──
    cost = 50.0
    feasible = cost <= delta_v_remaining
    fuel_pct = (cost / delta_v_remaining * 100.0) if delta_v_remaining > 0 else 999.0
    maneuvers.append({
        "name": "correction_burn",
        "description": "Small prograde correction burn (~50 m/s) for fine-tuning orbit",
        "estimated_delta_v": round(cost, 1),
        "fuel_percentage": round(fuel_pct, 1),
        "feasible": feasible,
        "reason": None if feasible else f"Insufficient fuel: need {cost:.0f} m/s, have {delta_v_remaining:.0f} m/s",
    })

    return maneuvers


def compute_mission_analysis(
    current_orbit: dict,
    target_orbit: dict,
    delta_v_used: float,
    delta_v_budget: float,
    task_id: str = "",
) -> Dict:
    """
    Compute real-time mission analysis for observation enrichment.

    Args:
        current_orbit: Current orbital state dict.
        target_orbit: Target orbital state dict.
        delta_v_used: Δ-v consumed so far.
        delta_v_budget: Total Δ-v budget.
        task_id: Current task ID.

    Returns:
        Dict matching MissionAnalysis schema.
    """
    alt_error = abs(current_orbit["altitude_km"] - target_orbit["altitude_km"])
    inc_error = abs(current_orbit["inclination_deg"] - target_orbit["inclination_deg"])
    ecc_error = abs(current_orbit["eccentricity"] - target_orbit["eccentricity"])

    fuel_remaining = delta_v_budget - delta_v_used

    # Estimate total Δ-v still needed
    dv_for_alt = 0.0
    if alt_error > 1.0:
        current_alt = current_orbit["altitude_km"]
        target_alt = target_orbit["altitude_km"]
        if current_alt < 100.0:
            dv_for_alt = launch_to_leo_delta_v(target_alt)
        else:
            dv_for_alt = hohmann_transfer_delta_v(current_alt, target_alt)

    dv_for_inc = 0.0
    if inc_error > 0.1:
        dv_for_inc = plane_change_delta_v(current_orbit["altitude_km"], inc_error)

    dv_for_ecc = 0.0
    if ecc_error > 0.005:
        dv_for_ecc = circularize_delta_v(current_orbit["altitude_km"], current_orbit["eccentricity"])

    estimated_needed = dv_for_alt + dv_for_inc + dv_for_ecc

    # Fuel margin
    if fuel_remaining > 0:
        fuel_margin = ((fuel_remaining - estimated_needed) / fuel_remaining) * 100.0
    else:
        fuel_margin = -100.0

    # Estimate current score (simplified — mirrors grader logic)
    target_alt = target_orbit["altitude_km"]
    alt_tolerance = max(target_alt * 0.1, 50.0)
    alt_score = max(0.0, 1.0 - alt_error / alt_tolerance)

    ecc_tolerance = 0.1
    ecc_score = max(0.0, 1.0 - ecc_error / ecc_tolerance)

    inc_tolerance = 5.0
    inc_score = max(0.0, 1.0 - inc_error / inc_tolerance)

    if delta_v_used > 0:
        eff_score = min(1.0, estimated_needed / delta_v_used) if estimated_needed > 0 else 0.5
    else:
        eff_score = 0.0

    score_estimate = (alt_score * 0.25 + ecc_score * 0.15 + inc_score * 0.15 +
                      eff_score * 0.35 + 0.8 * 0.10)  # assume decent step efficiency

    return {
        "altitude_error_km": round(alt_error, 2),
        "inclination_error_deg": round(inc_error, 2),
        "eccentricity_error": round(ecc_error, 6),
        "estimated_delta_v_needed": round(estimated_needed, 1),
        "fuel_remaining": round(fuel_remaining, 1),
        "fuel_margin_percent": round(fuel_margin, 1),
        "current_score_estimate": round(min(1.0, max(0.0, score_estimate)), 4),
    }


def get_recommendations(
    current_orbit: dict,
    target_orbit: dict,
    delta_v_remaining: float,
    task_id: str = "",
    step_index: int = 0,
    max_steps: int = 10,
) -> List[str]:
    """
    Generate context-aware recommendations for the agent.
    These are deterministic suggestions based on current state analysis.

    Args:
        current_orbit: Current orbital state dict.
        target_orbit: Target orbital state dict.
        delta_v_remaining: Remaining fuel in m/s.
        task_id: Current task ID.
        step_index: Current step number.
        max_steps: Maximum steps allowed.

    Returns:
        List of recommendation strings.
    """
    recs = []
    alt_error = abs(current_orbit["altitude_km"] - target_orbit["altitude_km"])
    inc_error = abs(current_orbit["inclination_deg"] - target_orbit["inclination_deg"])
    ecc = current_orbit["eccentricity"]
    current_alt = current_orbit["altitude_km"]
    target_alt = target_orbit["altitude_km"]

    steps_remaining = max_steps - step_index

    # ── Altitude guidance ──
    if alt_error > 1.0:
        if current_alt < 100.0:
            recs.append(f"Launch required: use hohmann_transfer to reach {target_alt:.0f} km orbit")
        else:
            direction = "raise" if target_alt > current_alt else "lower"
            dv_needed = hohmann_transfer_delta_v(current_alt, target_alt) if current_alt >= 100 else launch_to_leo_delta_v(target_alt)
            if dv_needed <= delta_v_remaining:
                recs.append(f"Hohmann transfer can {direction} orbit to {target_alt:.0f} km (costs {dv_needed:.0f} m/s — {dv_needed/delta_v_remaining*100:.0f}% of remaining fuel)")
            else:
                recs.append(f"WARNING: Hohmann transfer needs {dv_needed:.0f} m/s but only {delta_v_remaining:.0f} m/s remaining")
                if task_id == "asteroid_rendezvous":
                    recs.append("Consider gravity assist to gain free Δ-v before major burns")

    # ── Inclination guidance ──
    if inc_error > 0.1:
        dv_inc = plane_change_delta_v(current_alt, inc_error)
        if dv_inc <= delta_v_remaining:
            recs.append(f"Plane change needed: {inc_error:.1f}° to match target (costs {dv_inc:.0f} m/s)")
        else:
            recs.append(f"WARNING: Plane change needs {dv_inc:.0f} m/s — exceeds remaining fuel")

    # ── Eccentricity guidance ──
    if ecc > 0.005:
        dv_circ = circularize_delta_v(current_alt, ecc)
        recs.append(f"Orbit is elliptical (e={ecc:.4f}) — circularize for {dv_circ:.0f} m/s")

    # ── Combined maneuver suggestion ──
    if alt_error > 1.0 and inc_error > 0.1:
        dv_sep = (hohmann_transfer_delta_v(current_alt, target_alt) if current_alt >= 100 else launch_to_leo_delta_v(target_alt)) + plane_change_delta_v(current_alt, inc_error)
        dv_comb = dv_sep * 0.85
        savings = dv_sep - dv_comb
        recs.append(f"TIP: Combined transfer saves ~{savings:.0f} m/s vs separate maneuvers")

    # ── Task-specific guidance ──
    if task_id == "lunar_orbit" and current_alt < 1000.0 and target_alt > 100_000.0:
        recs.append("Mission plan: Trans-Lunar Injection first, then Lunar Orbit Insertion")

    if task_id == "asteroid_rendezvous" and step_index < 3:
        recs.append("Strategy: Use Venus and/or Earth gravity assists early to save fuel for later burns")

    # ── Timing warnings ──
    if steps_remaining <= 2 and alt_error > 100.0:
        recs.append(f"⚠ Only {steps_remaining} steps remaining — consider submitting or making final burn")

    if steps_remaining <= 1:
        recs.append("Last step available — submit_mission to lock in your score")

    # ── Near-complete guidance ──
    if alt_error < 50.0 and inc_error < 1.0 and ecc < 0.01:
        recs.append("Orbit is close to target — consider submitting for a good score")

    # ── Score suggestion ──
    if alt_error < 10.0 and inc_error < 0.5 and ecc < 0.005:
        recs.append("Excellent orbit match! Submit now for maximum score")

    return recs