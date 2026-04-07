# app/visualizer.py
# Orbital trajectory visualizer for Orbit — AI Space Mission Architect

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional


# ── Constants ──────────────────────────────────────────────────────────────────
EARTH_RADIUS_KM = 6371.0
MOON_DISTANCE_KM = 384400.0
COLORS = {
    "earth":      "#1a6faf",
    "moon":       "#aaaaaa",
    "trajectory": "#f0a500",
    "burn":       "#ff4444",
    "target":     "#44ff88",
    "background": "#0d0d1a",
    "grid":       "#1a1a2e",
    "text":       "#ffffff",
}


# ── Internal Helpers ───────────────────────────────────────────────────────────

def _draw_earth(ax: plt.Axes) -> None:
    """Draw Earth at origin."""
    earth = plt.Circle((0, 0), EARTH_RADIUS_KM, color=COLORS["earth"], zorder=5)
    ax.add_patch(earth)
    ax.text(0, 0, "🌍", fontsize=14, ha="center", va="center", zorder=6)


def _draw_moon(ax: plt.Axes, scale: float = 1.0) -> None:
    """Draw Moon at its approximate position."""
    moon_x = MOON_DISTANCE_KM * scale
    moon = plt.Circle((moon_x, 0), 1737.0, color=COLORS["moon"], zorder=5)
    ax.add_patch(moon)
    ax.text(moon_x, 0, "🌕", fontsize=10, ha="center", va="center", zorder=6)


def _altitude_to_radius(altitude_km: float) -> float:
    """Convert altitude above Earth surface to orbital radius."""
    return EARTH_RADIUS_KM + altitude_km


def _orbit_points(
    radius_km: float,
    eccentricity: float = 0.0,
    inclination_deg: float = 0.0,
    num_points: int = 360,
) -> tuple:
    """
    Generate x, y coordinates for an elliptical orbit.
    Uses a simplified 2D projection (inclination shown as y-scaling).
    """
    theta = np.linspace(0, 2 * np.pi, num_points)

    # Semi-major and semi-minor axes
    a = radius_km
    b = a * np.sqrt(1 - eccentricity ** 2)

    x = a * np.cos(theta)
    y = b * np.sin(theta) * np.cos(np.radians(inclination_deg))

    return x, y


# ── Main Public Functions ──────────────────────────────────────────────────────

def plot_trajectory(
    trajectory: List[Dict],
    task_id: str,
    delta_v_used: float,
    score: float,
    target_orbit: Optional[Dict] = None,
    save_path: Optional[str] = None,
    show: bool = False,
) -> str:
    """
    Plot a spacecraft's orbital trajectory.

    Args:
        trajectory  : List of orbital state dicts from the environment.
                      Each dict should have: altitude_km, eccentricity, inclination_deg
        task_id     : Mission ID string (e.g. 'leo_satellite')
        delta_v_used: Total Δ-v spent in m/s
        score       : Mission score 0.0 – 1.0
        target_orbit: Target orbital parameters dict (optional)
        save_path   : File path to save PNG (e.g. 'output/mission.png')
        show        : Whether to display the plot interactively

    Returns:
        Path where the image was saved (or empty string if not saved)
    """

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["background"])

    # ── Draw Earth ─────────────────────────────────────────────────────────────
    _draw_earth(ax)

    # ── Draw Target Orbit (dashed green) ───────────────────────────────────────
    if target_orbit:
        t_radius = _altitude_to_radius(target_orbit.get("altitude_km", 400))
        t_ecc    = target_orbit.get("eccentricity", 0.0)
        t_inc    = target_orbit.get("inclination_deg", 0.0)
        tx, ty   = _orbit_points(t_radius, t_ecc, t_inc)
        ax.plot(tx, ty,
                color=COLORS["target"],
                linestyle="--",
                linewidth=1.5,
                alpha=0.6,
                label="Target Orbit",
                zorder=3)

    # ── Draw Trajectory Steps ──────────────────────────────────────────────────
    burn_points_x = []
    burn_points_y = []

    for i, state in enumerate(trajectory):
        altitude   = state.get("altitude_km", 400)
        eccentricity = state.get("eccentricity", 0.0)
        inclination  = state.get("inclination_deg", 0.0)
        is_burn      = state.get("is_burn", False)

        radius = _altitude_to_radius(altitude)
        x, y   = _orbit_points(radius, eccentricity, inclination)

        alpha = 0.3 + (0.7 * (i + 1) / max(len(trajectory), 1))
        ax.plot(x, y,
                color=COLORS["trajectory"],
                linewidth=1.2,
                alpha=alpha,
                zorder=4)

        # Mark burn points
        if is_burn:
            burn_x = radius * np.cos(0)
            burn_y = radius * np.sin(0)
            burn_points_x.append(burn_x)
            burn_points_y.append(burn_y)

    # ── Plot Burn Markers ──────────────────────────────────────────────────────
    if burn_points_x:
        ax.scatter(burn_points_x, burn_points_y,
                   color=COLORS["burn"],
                   s=80,
                   zorder=7,
                   label="Burn Points",
                   marker="*")

    # ── Final Orbit (brightest) ────────────────────────────────────────────────
    if trajectory:
        last   = trajectory[-1]
        radius = _altitude_to_radius(last.get("altitude_km", 400))
        ecc    = last.get("eccentricity", 0.0)
        inc    = last.get("inclination_deg", 0.0)
        fx, fy = _orbit_points(radius, ecc, inc)
        ax.plot(fx, fy,
                color=COLORS["trajectory"],
                linewidth=2.5,
                alpha=1.0,
                label="Final Orbit",
                zorder=5)

    # ── Labels & Formatting ────────────────────────────────────────────────────
    task_labels = {
        "leo_satellite":       "Task 1 — LEO Satellite Deployment",
        "lunar_orbit":         "Task 2 — Lunar Orbit Insertion",
        "asteroid_rendezvous": "Task 3 — Asteroid Mining Rendezvous",
    }
    title = task_labels.get(task_id, task_id)

    score_color = (
        "#ff4444" if score < 0.4 else
        "#f0a500" if score < 0.7 else
        "#44ff88"
    )

    ax.set_title(
        f"🚀 {title}\nΔ-v Used: {delta_v_used:,.0f} m/s   |   Score: {score:.3f}",
        color=COLORS["text"],
        fontsize=13,
        pad=15,
    )

    ax.set_xlabel("Distance (km)", color=COLORS["text"], fontsize=10)
    ax.set_ylabel("Distance (km)", color=COLORS["text"], fontsize=10)
    ax.tick_params(colors=COLORS["text"])
    ax.set_aspect("equal")

    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS["grid"])

    ax.grid(True, color=COLORS["grid"], linewidth=0.5, alpha=0.5)

    legend = ax.legend(
        loc="upper right",
        facecolor=COLORS["background"],
        edgecolor=COLORS["grid"],
        labelcolor=COLORS["text"],
        fontsize=9,
    )

    # Score badge
    fig.text(
        0.13, 0.91,
        f"Score: {score:.3f}",
        fontsize=12,
        color=score_color,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3",
                  facecolor=COLORS["background"],
                  edgecolor=score_color,
                  linewidth=1.5),
    )

    plt.tight_layout()

    # ── Save / Show ────────────────────────────────────────────────────────────
    output_path = ""
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=COLORS["background"])
        output_path = save_path
        print(f"[Visualizer] Saved → {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return output_path


def plot_score_summary(
    results: List[Dict],
    save_path: Optional[str] = None,
    show: bool = False,
) -> str:
    """
    Plot a bar chart summarising scores across all missions.

    Args:
        results  : List of dicts with keys: task_id, score, delta_v_used
        save_path: File path to save PNG
        show     : Whether to display interactively

    Returns:
        Path where the image was saved (or empty string if not saved)
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["background"])

    labels = []
    scores = []
    colors = []

    task_short = {
        "leo_satellite":       "LEO Satellite",
        "lunar_orbit":         "Lunar Orbit",
        "asteroid_rendezvous": "Asteroid Rendezvous",
    }

    for r in results:
        labels.append(task_short.get(r["task_id"], r["task_id"]))
        s = r.get("score", 0.0)
        scores.append(s)
        colors.append(
            "#ff4444" if s < 0.4 else
            "#f0a500" if s < 0.7 else
            "#44ff88"
        )

    bars = ax.bar(labels, scores, color=colors, width=0.5, zorder=3)

    # Value labels on bars
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{score:.3f}",
            ha="center", va="bottom",
            color=COLORS["text"],
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score (0.0 – 1.0)", color=COLORS["text"], fontsize=10)
    ax.set_title("🚀 Orbit — Mission Score Summary",
                 color=COLORS["text"], fontsize=13, pad=15)
    ax.tick_params(colors=COLORS["text"])
    ax.set_facecolor(COLORS["background"])
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.5, alpha=0.5, zorder=0)

    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS["grid"])

    plt.tight_layout()

    output_path = ""
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=COLORS["background"])
        output_path = save_path
        print(f"[Visualizer] Saved → {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return output_path