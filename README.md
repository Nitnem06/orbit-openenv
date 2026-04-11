title: Orbit Env
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860

# Key Differences I See

Your current README has:

| Issue | Current | Should Be |
|---|---|---|
| HF metadata header | ✅ Has it (MUST KEEP) | Keep as-is |
| LEO optimal Δ-v | 9,400 | 9,200 |
| Asteroid max steps | 20 | 25 |
| Action space | v1 only (`add_burn`) | v2 `execute_maneuver` + legacy |
| Observation space | Missing `available_maneuvers`, `mission_analysis`, `recommendations` | Add all three |
| Baseline scores | Old GPT-4 scores (0.42 avg) | New hand-crafted (0.89 avg) |
| Grading | Only mentions efficiency formula | Full 5-component breakdown |
| Reward shaping | Missing | Need dense reward table |
| References | "GPT-4 baseline" | Generic "LLM agent" |
| `visualizer.py` in structure | Listed | Does it still exist? |

**Important:** Your HF metadata at the top MUST stay — HF Spaces reads it.

---

## Updated `README.md`

---

# 🚀 Orbit — AI Space Mission Architect

> An OpenEnv environment where an AI agent plans fuel-efficient space missions using real orbital mechanics.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-brightgreen)](https://github.com/openenv-ai/openenv)
[![HF Space](https://img.shields.io/badge/🤗-Live%20Demo-blue)](https://huggingface.co/spaces/Nitnem/orbit-env)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Overview

**Orbit** is a real-world OpenEnv environment that simulates space mission planning. An AI agent acts as a **mission director**, choosing strategic orbital maneuvers — Hohmann transfers, gravity assists, plane changes — to reach target orbits. The environment handles all physics calculations internally.

**All scoring is 100% deterministic** — same actions always produce the same score. No LLM judges. Pure orbital mechanics formulas.

This is the kind of problem solved daily by engineers at **SpaceX, NASA, ISRO, and ESA**.

---

## 🎯 Why Orbit?

- **Real-world utility** — Trajectory optimization is a genuine aerospace engineering problem
- **Deterministic grading** — No LLM-as-judge. Pure physics. Reproducible scores every time
- **Strategic decision-making** — Agent chooses WHICH maneuver, environment calculates HOW MUCH fuel
- **Rich observations** — Available maneuvers, mission analysis, and recommendations guide reasoning
- **Novel** — First astrodynamics environment in OpenEnv
- **Difficulty progression** — Easy (0.99) > Medium (0.88) > Hard (0.81) verified

---

## 🛰️ The Three Missions

| # | Mission | Difficulty | Optimal Δ-v | Budget | Max Steps |
|---|---------|-----------|-------------|--------|-----------|
| 1 | LEO Satellite Deployment | 🟢 Easy | 9,200 m/s | 12,000 m/s | 10 |
| 2 | Lunar Orbit Insertion | 🟡 Medium | 3,900 m/s | 5,000 m/s | 15 |
| 3 | Asteroid Mining Rendezvous | 🔴 Hard | 5,800 m/s | 8,000 m/s | 25 |

### Task 1 — LEO Satellite Deployment (Easy)

Launch a satellite from Earth's surface into a **400 km circular orbit** matching ISS inclination (51.6°). Launch azimuth is pre-selected — the agent only needs to choose the correct transfer maneuver.

- **Strategy:** Single `hohmann_transfer` to target altitude → submit
- **What's tested:** Can the agent pick the right maneuver from available options?
- **Real reference:** SpaceX Falcon 9 → ISS

### Task 2 — Lunar Orbit Insertion (Medium)

Transfer from a **200 km Earth parking orbit** to lunar orbit. Requires sequencing two critical maneuvers correctly and managing fuel across both burns. LOI leaves residual eccentricity — agent must decide whether to fix it.

- **Strategy:** `trans_lunar_injection` → `lunar_orbit_insertion` → (optional: `circularize`) → submit
- **What's tested:** Multi-burn sequencing, fuel management, trade-off decisions
- **Real reference:** Apollo / Artemis missions

### Task 3 — Asteroid Mining Rendezvous (Hard)

Reach near-Earth asteroid **Bennu** from LEO. Direct transfer exceeds the fuel budget — the agent **MUST** use gravity assists. Also requires inclination change (28.5° → 6°) and managing navigation uncertainty on deep-space transfers.

- **Strategy:** `gravity_assist` (Venus) → `gravity_assist` (Earth) → `combined_transfer` → corrections → submit
- **What's tested:** Multi-step planning, gravity assist usage, uncertainty management
- **Real reference:** NASA OSIRIS-REx mission

---

## 📊 Baseline Scores (Hand-Crafted Optimal)

| Mission | Score | Success | Δ-v Used |
|---------|-------|---------|----------|
| LEO Satellite | 0.99 | ✅ | 9,173 / 12,000 m/s |
| Lunar Orbit | 0.88 | ✅ | 3,967 / 5,000 m/s |
| Asteroid Rendezvous | 0.81 | ✅ | 3,742 / 8,000 m/s |
| **Average** | **0.89** | | |

> Difficulty verified: Easy (0.99) > Medium (0.88) > Hard (0.81) ✅

---

## 🎮 Action Space

### Strategic Maneuvers (Recommended for LLM Agents)

The agent chooses **which** maneuver to perform. The environment calculates the required delta-v using physics formulas internally.

| Maneuver | Example | Fuel Cost |
|----------|---------|-----------|
| Hohmann Transfer | `{"type": "execute_maneuver", "maneuver": "hohmann_transfer", "target_altitude_km": 400}` | Calculated |
| Plane Change | `{"type": "execute_maneuver", "maneuver": "plane_change", "target_inclination_deg": 51.6}` | Calculated |
| Circularize | `{"type": "execute_maneuver", "maneuver": "circularize"}` | Calculated |
| Trans-Lunar Injection | `{"type": "execute_maneuver", "maneuver": "trans_lunar_injection"}` | Calculated |
| Lunar Orbit Insertion | `{"type": "execute_maneuver", "maneuver": "lunar_orbit_insertion", "target_altitude_km": 100}` | Calculated |
| Gravity Assist | `{"type": "execute_maneuver", "maneuver": "gravity_assist", "body": "venus"}` | **FREE** |
| Combined Transfer | `{"type": "execute_maneuver", "maneuver": "combined_transfer", "target_altitude_km": 400, "target_inclination_deg": 6.0}` | 15% cheaper |
| Correction Burn | `{"type": "execute_maneuver", "maneuver": "correction_burn", "delta_v_ms": 50}` | Up to 500 m/s |

### Utility Actions

| Action | Example | Cost |
|--------|---------|------|
| Preview Score | `{"type": "run_simulation"}` | Free |
| Submit Mission | `{"type": "submit_mission"}` | Free |

### Legacy Low-Level Burn (Advanced)

```json
{"type": "add_burn", "delta_v_ms": 9400, "prograde": 1.0, "radial": 0.0, "normal": 0.0}
```

---

## 👁️ Observation Space

After each step, the agent receives a rich observation for strategic reasoning:

| Field | Type | Description |
|-------|------|-------------|
| `current_orbit` | object | Altitude, eccentricity, inclination, velocity |
| `target_orbit` | object | Target orbital parameters |
| `available_maneuvers` | array | **What the agent CAN do** — fuel costs, feasibility, descriptions |
| `mission_analysis` | object | **Where the agent stands** — errors, fuel margin, score estimate |
| `recommendations` | array | **Context-aware strategic advice** based on current state |
| `delta_v_used` / `delta_v_budget` | float | Fuel tracking |
| `step_index` / `max_steps` | int | Step tracking |
| `last_action_result` | string | Human-readable result of last action |

### Example: Available Maneuver

```json
{
  "name": "hohmann_transfer",
  "description": "Transfer from 0 km to 400 km altitude",
  "estimated_delta_v": 9173.0,
  "fuel_percentage": 76.4,
  "feasible": true,
  "reason": null
}
```

### Example: Mission Analysis

```json
{
  "altitude_error_km": 400.0,
  "inclination_error_deg": 0.0,
  "eccentricity_error": 0.0,
  "estimated_delta_v_needed": 9173.0,
  "fuel_remaining": 12000.0,
  "fuel_margin_percent": 23.6,
  "current_score_estimate": 0.38
}
```

---

## 📈 Grading System

All grading is **100% deterministic** — pure physics calculations.

| Component | Weight | Description |
|-----------|--------|-------------|
| Altitude accuracy | 30% | How close to target altitude |
| Fuel efficiency | 30% | optimal Δ-v / actual Δ-v (penalizes over AND underspending) |
| Eccentricity accuracy | 20% | How close to target eccentricity |
| Inclination accuracy | 15% | How close to target inclination |
| Step efficiency | 5% | Fewer steps = higher score |

**Score range:** 0.0 to 1.0 — same actions always produce the same score.

---

## 🎁 Reward Shaping

Dense rewards every step (not just at episode end):

| Event | Reward | Purpose |
|-------|--------|---------|
| Closer to target orbit | +0.03 to +0.10 | Encourage progress |
| Further from target | -0.02 to -0.05 | Discourage regression |
| Gravity assist used | +0.08 | Encourage smart fuel saving |
| Correct maneuver for mission | +0.02 to +0.05 | Encourage strategy |
| Fuel spent | -0.005 per 1000 m/s | Penalize waste |
| Budget exceeded | -0.30 | Hard penalty |
| Per step | -0.01 | Encourage efficiency |
| Submit mission | Final score (0.0–1.0) | Terminal reward |
| Timeout | Score × 0.80 | Penalty for using all steps |

---

## 🌐 WebSocket Protocol

```
Connect → ws://localhost:7860/ws
← {"type": "welcome", "message": "...", "available_tasks": [...]}

→ {"type": "reset", "task_id": "leo_satellite"}
← {"type": "observation", "data": {...}}

→ {"type": "step", "action": {"type": "execute_maneuver", "maneuver": "hohmann_transfer", "target_altitude_km": 400}}
← {"type": "step_result", "observation": {...}, "reward": 0.07, "done": false, "info": {...}}

→ {"type": "step", "action": {"type": "submit_mission"}}
← {"type": "step_result", "observation": {...}, "reward": 0.99, "done": true, "info": {...}}
```

### HTTP Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Landing page |
| `/health` | GET | Health check |
| `/tasks` | GET | List missions with metadata |
| `/reset` | POST | HTTP-based reset |

---

## 🏗️ Project Structure

```
orbit-openenv/
├── app/
│   ├── __init__.py      # Package init
│   ├── models.py        # Pydantic v2 models (Action, Observation, State)
│   ├── physics.py       # Orbital mechanics (pure deterministic functions)
│   ├── env.py           # Core environment (reset/step/state)
│   ├── tasks.py         # 3 mission definitions with parameters
│   ├── grader.py        # Deterministic scoring (0.0–1.0)
│   └── server.py        # FastAPI WebSocket server + landing page
├── baseline/
│   └── run_baseline.py  # Alternative baseline LLM agent
├── inference.py         # Primary baseline agent (OpenEnv-compliant logging)
├── Dockerfile           # Container config (python:3.11-slim, port 7860)
├── openenv.yaml         # OpenEnv specification metadata
├── requirements.txt     # Python dependencies
├── pyproject.toml       # Project metadata
└── README.md            # This file
```

---

## ⚙️ Setup & Usage

### Install

```bash
git clone https://github.com/Nitnem06/orbit-openenv.git
cd orbit-openenv
pip install -r requirements.txt
```

### Run Server

```bash
uvicorn app.server:app --host 0.0.0.0 --port 7860
```

### Run Baseline Agent

```bash
export OPENAI_API_KEY=your_key_here
python inference.py
```

### Docker

```bash
docker build -t orbit-env .
docker run -p 7860:7860 orbit-env
```

### Validation

```bash
# Health check
curl http://localhost:7860/health

# List tasks
curl http://localhost:7860/tasks

# OpenEnv validate
openenv validate
```

---

## 🔬 Physics Engine

Real orbital mechanics formulas:

| Function | Formula | Source |
|----------|---------|--------|
| `orbital_velocity()` | v = √(μ/r) | Newton's laws |
| `hohmann_transfer_delta_v()` | Two-impulse transfer | NASA textbook |
| `plane_change_delta_v()` | Δv = 2v·sin(Δi/2) | Orbital mechanics |
| `gravity_assist_delta_v()` | Hyperbolic turn angle | Patched conic |
| `vis_viva()` | v = √(μ(2/r - 1/a)) | Energy conservation |

**Constants:** μ_Earth = 398,600 km³/s² · R_Earth = 6,371 km · Moon = 384,400 km

---

## 🔑 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | LLM API key |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model name |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| FastAPI | WebSocket server |
| Pydantic v2 | Typed data models |
| NumPy | Physics calculations |
| OpenAI API | LLM baseline agent |
| Docker | Containerization |
| Hugging Face Spaces | Deployment |

---

## 👥 Team

| Member | Role |
|--------|------|
| **Nitnem** | Physics engine, models, environment, grader, tasks |
| **Shimul** | Server, inference, deployment, documentation |

---

## 🌍 Live Deployment

| Resource | URL |
|----------|-----|
| HF Space | https://huggingface.co/spaces/Nitnem/orbit-env |
| Live API | https://nitnem-orbit-env.hf.space |
| Health | https://nitnem-orbit-env.hf.space/health |
| Tasks | https://nitnem-orbit-env.hf.space/tasks |
| GitHub | https://github.com/Nitnem06/orbit-openenv |

---