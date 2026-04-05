# 🚀 Orbit — AI Space Mission Architect

> An OpenEnv environment where an AI agent learns to design fuel-efficient space missions using real orbital mechanics.

---

## 📌 Overview

**Orbit** is a real-world OpenEnv environment that simulates space mission planning. An AI agent must design missions by choosing when to fire engines, how to use gravity assists, and how to optimize fuel consumption — all graded using deterministic Newtonian physics and Hohmann transfer equations.

This is the kind of problem solved daily by engineers at **SpaceX, NASA, and RocketLab**.

---

## 🎯 Why Orbit?

- **Real-world utility** — Astrodynamics trajectory optimization is a genuine engineering problem
- **Deterministic grading** — No LLM-as-judge. Pure physics formulas. Reproducible scores
- **Novel** — No OpenEnv environment exists for orbital mechanics
- **Visual** — Trajectory plots make demos compelling
- **Feasible** — Simplified 2-body physics works in pure Python

---

## 🛰️ The Three Missions

| # | Mission | Difficulty | Optimal Δ-v | Max Steps |
|---|---------|-----------|-------------|-----------|
| 1 | LEO Satellite Deployment | 🟢 Easy | 9,400 m/s | 10 |
| 2 | Lunar Orbit Insertion | 🟡 Medium | 3,900 m/s | 15 |
| 3 | Asteroid Mining Rendezvous | 🔴 Hard | 5,800 m/s | 20 |

### Task 1 — LEO Satellite Deployment (Easy)
Get a satellite into a **400km circular Low Earth Orbit**.
The agent must choose launch parameters and perform a single circularization burn efficiently.
- **Grading:** `optimal_delta_v / actual_delta_v`

### Task 2 — Lunar Orbit Insertion (Medium)
Transfer a spacecraft from **Earth to Lunar orbit** using:
- Trans-lunar injection burn
- Mid-course correction
- Lunar capture burn

Miss distance must be under 100km.
- **Grading:** Fuel efficiency + arrival accuracy

### Task 3 — Asteroid Mining Rendezvous (Hard)
Reach a **near-Earth asteroid** using 2 gravity assists (Earth + Venus).
Agent must optimize burn timing, flyby geometry, and plane change maneuvers.
- **Grading:** Total Δ-v + time-of-flight tradeoff

---

## 🏗️ Project Structure
orbit-openenv/
├── app/
│   ├── init.py          # Package init
│   ├── models.py            # Pydantic models (Action, Observation, State)
│   ├── physics.py           # Orbital mechanics formulas
│   ├── env.py               # Core environment (reset/step/state)
│   ├── tasks.py             # 3 mission definitions
│   ├── grader.py            # Deterministic scoring
│   ├── server.py            # FastAPI WebSocket server
│   └── visualizer.py        # Trajectory plotting
├── baseline/
│   └── run_baseline.py      # GPT-4 agent script
├── requirements.txt         # Dependencies
├── Dockerfile               # Container for HF Spaces
├── openenv.yaml             # OpenEnv spec config
└── README.md                # Documentation

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Nitnem06/orbit-openenv.git
cd orbit-openenv
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running Locally

### Start the WebSocket Server
```bash
uvicorn app.server:app --host 0.0.0.0 --port 7860
```

Server will be available at: ws://localhost:7860/ws

### Run the Baseline Agent
```bash
export OPENAI_API_KEY=your_api_key_here   # macOS/Linux
set OPENAI_API_KEY=your_api_key_here      # Windows

python baseline/run_baseline.py
```

---

## 🐳 Docker Deployment

### Build the Image
```bash
docker build -t orbit-openenv .
```

### Run the Container
```bash
docker run -p 7860:7860 orbit-openenv
```

---

## 🌐 WebSocket API

### Connection
ws://localhost:7860/ws

### Message Format

**Reset Environment:**
```json
{"type": "reset", "task_id": "leo_satellite"}
```

**Take a Step:**
```json
{
  "type": "step",
  "action": {
    "type": "add_burn",
    "delta_v_ms": 500,
    "prograde": 1.0,
    "radial": 0.0
  }
}
```

**Get State:**
```json
{"type": "state"}
```

---

## 🎮 Action Space

| Action | Description | Fields |
|--------|-------------|--------|
| `set_orbit` | Set orbital parameters directly | `altitude_km`, `eccentricity`, `inclination_deg` |
| `add_burn` | Apply a propulsive burn | `delta_v_ms`, `prograde`, `radial` |
| `set_flyby` | Configure a gravity assist | `body`, `altitude_km` |
| `run_simulation` | Run physics simulation | — |
| `submit_mission` | Submit for grading | — |

---

## 👁️ Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | string | Current mission ID |
| `current_orbit` | object | Current altitude, eccentricity, inclination |
| `target_orbit` | object | Target orbital parameters |
| `delta_v_used` | float | Total Δ-v spent (m/s) |
| `delta_v_remaining` | float | Remaining fuel budget (m/s) |
| `trajectory` | array | List of visited orbital states |
| `step_index` | integer | Current step number |
| `max_steps` | integer | Maximum steps allowed |

---

## 📊 Reward Function

- **Type:** Deterministic (no LLM-as-judge)
- **Formula:** `reward = optimal_delta_v / actual_delta_v`
- **Range:** `0.0 – 1.0`
- **Partial credit:** Yes — agent gets rewarded for progress even if mission is incomplete

---

## 📈 Baseline Scores

Scores achieved by the GPT-4 baseline agent (`temperature=0`):

| Mission | Score |
|---------|-------|
| LEO Satellite Deployment | TBD |
| Lunar Orbit Insertion | TBD |
| Asteroid Mining Rendezvous | TBD |

---

## 🌍 Deployment

**Live on Hugging Face Spaces:**
https://huggingface.co/spaces/Nitnem06/orbit-mission-architect


**WebSocket URL:**
wss://nitnem06-orbit-mission-architect.hf.space/ws

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| FastAPI | WebSocket server |
| Pydantic | Typed data models |
| NumPy | Physics calculations |
| Matplotlib | Trajectory visualization |
| OpenAI API | GPT-4 baseline agent |
| Docker | Containerization |
| Hugging Face Spaces | Deployment |

---

## 👥 Team

| Person | Role | Branch |
|--------|------|--------|
| Nitnem | Physics & Engine Architect | `p1-physics` |
| Shimul | Interface & Integration Lead | `p2-server` |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.