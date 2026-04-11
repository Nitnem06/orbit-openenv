"""
app/server.py
Orbit — AI Space Mission Architect

FastAPI WebSocket server that wraps the OrbitEnvironment.
Each WebSocket connection gets its own isolated environment instance.

v2.0 Changes:
    - Added ExecuteManeuverAction import
    - Updated UI scoring table to match grader weights (30/30/20/15/5)
    - Updated Quick Start to use execute_maneuver
    - Enriched health endpoint
    - Updated version to 2.0.0

Message Protocol:
    Client → Server:
        {"type": "reset", "task_id": "leo_satellite"}
        {"type": "step", "action": {"type": "execute_maneuver", "maneuver": "hohmann_transfer", ...}}
        {"type": "state"}
        {"type": "list_tasks"}

    Server → Client:
        {"type": "observation", "data": {...}}
        {"type": "step_result", "observation": {...}, "reward": 0.15, "done": false, "info": {...}}
        {"type": "state", "data": {...}}
        {"type": "task_list", "data": [...]}
        {"type": "error", "message": "..."}
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, TypeAdapter, ValidationError

from app.env import OrbitEnvironment
from app.models import (
    Action,
    AddBurnAction,
    ExecuteManeuverAction,
    RunSimulationAction,
    SetFlybyAction,
    SetOrbitAction,
    SubmitMissionAction,
)
from app.tasks import get_task_summary, list_tasks

env_instance: OrbitEnvironment | None = None
START_TIME = time.time()

# ─────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level  = logging.WARNING,
    format = "%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Orbit — AI Space Mission Architect",
    description = "OpenEnv-compliant WebSocket server for orbital mechanics simulation.",
    version     = "2.0.0",
)

# Pydantic v2 TypeAdapter for parsing Action union
action_adapter = TypeAdapter(Action)


# ─────────────────────────────────────────────────────────────────────────────
# HTTP Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Landing page — professional UI for the HuggingFace Space."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ORBIT — AI Space Mission Architect</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }

            body {
                font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
                background: #0a0e1a;
                color: #e0e0e0;
                min-height: 100vh;
                overflow-x: hidden;
            }

            /* ── Animated star background ── */
            body::before {
                content: '';
                position: fixed;
                top: 0; left: 0; right: 0; bottom: 0;
                background:
                    radial-gradient(1px 1px at 10% 20%, rgba(255,255,255,0.6), transparent),
                    radial-gradient(1px 1px at 30% 65%, rgba(255,255,255,0.4), transparent),
                    radial-gradient(1.5px 1.5px at 50% 10%, rgba(255,255,255,0.7), transparent),
                    radial-gradient(1px 1px at 70% 40%, rgba(255,255,255,0.5), transparent),
                    radial-gradient(1px 1px at 85% 75%, rgba(255,255,255,0.4), transparent),
                    radial-gradient(1.5px 1.5px at 15% 80%, rgba(255,255,255,0.6), transparent),
                    radial-gradient(1px 1px at 60% 90%, rgba(255,255,255,0.3), transparent),
                    radial-gradient(1px 1px at 90% 15%, rgba(255,255,255,0.5), transparent),
                    radial-gradient(1px 1px at 40% 35%, rgba(255,255,255,0.4), transparent),
                    radial-gradient(1.5px 1.5px at 25% 50%, rgba(255,255,255,0.3), transparent);
                background-size: 200% 200%;
                animation: starDrift 120s linear infinite;
                z-index: 0;
                pointer-events: none;
            }
            @keyframes starDrift {
                0% { background-position: 0% 0%; }
                100% { background-position: 100% 100%; }
            }

            .container {
                max-width: 920px;
                margin: 0 auto;
                padding: 40px 24px;
                position: relative;
                z-index: 1;
            }

            /* ── Header ── */
            .header {
                text-align: center;
                margin-bottom: 48px;
            }
            .header-icon {
                font-size: 3.2em;
                margin-bottom: 8px;
                display: block;
            }
            .header h1 {
                font-size: 2.8em;
                font-weight: 800;
                letter-spacing: 6px;
                background: linear-gradient(135deg, #00d4ff 0%, #7b2ff7 50%, #ff6bcb 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 8px;
            }
            .header .subtitle {
                font-size: 1.15em;
                color: #8892b0;
                font-weight: 400;
            }
            .badges {
                margin-top: 18px;
                display: flex;
                justify-content: center;
                gap: 10px;
                flex-wrap: wrap;
            }
            .badge {
                display: inline-block;
                background: #1a1f35;
                border: 1px solid #2a3050;
                color: #8892b0;
                padding: 5px 14px;
                border-radius: 20px;
                font-size: 0.82em;
                font-weight: 500;
            }
            .badge.primary {
                border-color: #7b2ff7;
                color: #00d4ff;
            }

            /* ── Status Bar ── */
            .status-bar {
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 8px;
                margin-top: 20px;
            }
            .status-dot {
                width: 9px; height: 9px;
                background: #34d399;
                border-radius: 50%;
                box-shadow: 0 0 8px #34d399;
                animation: pulse 2s ease-in-out infinite;
            }
            @keyframes pulse {
                0%, 100% { opacity: 1; transform: scale(1); }
                50% { opacity: 0.5; transform: scale(0.85); }
            }
            .status-text { color: #34d399; font-size: 0.88em; font-weight: 500; }

            /* ── Description ── */
            .description {
                text-align: center;
                color: #8892b0;
                max-width: 700px;
                margin: 0 auto 48px;
                line-height: 1.7;
                font-size: 0.98em;
            }

            /* ── Section Titles ── */
            .section-title {
                font-size: 1.2em;
                font-weight: 700;
                color: #fff;
                margin-bottom: 20px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .section-title .icon { font-size: 1.2em; }
            .section-title::after {
                content: '';
                flex: 1;
                height: 1px;
                background: linear-gradient(90deg, #1e293b, transparent);
            }

            /* ── Mission Cards ── */
            .missions { display: grid; gap: 14px; margin-bottom: 44px; }
            .mission-card {
                background: linear-gradient(135deg, #111827 0%, #0f172a 100%);
                border: 1px solid #1e293b;
                border-radius: 14px;
                padding: 22px 24px;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }
            .mission-card::before {
                content: '';
                position: absolute;
                top: 0; left: 0; right: 0;
                height: 2px;
                background: linear-gradient(90deg, transparent, #7b2ff7, transparent);
                opacity: 0;
                transition: opacity 0.3s;
            }
            .mission-card:hover {
                border-color: #7b2ff7;
                transform: translateY(-2px);
                box-shadow: 0 8px 30px rgba(123, 47, 247, 0.1);
            }
            .mission-card:hover::before { opacity: 1; }
            .mission-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 8px;
            }
            .mission-name {
                font-size: 1.05em;
                font-weight: 600;
                color: #fff;
            }
            .difficulty {
                padding: 3px 12px;
                border-radius: 12px;
                font-size: 0.75em;
                font-weight: 700;
                letter-spacing: 0.5px;
                text-transform: uppercase;
            }
            .easy { background: #064e3b; color: #34d399; }
            .medium { background: #713f12; color: #fbbf24; }
            .hard { background: #7f1d1d; color: #f87171; }
            .mission-desc {
                color: #8892b0;
                font-size: 0.92em;
                margin-bottom: 12px;
                line-height: 1.5;
            }
            .mission-stats {
                display: flex;
                gap: 20px;
                flex-wrap: wrap;
                color: #64748b;
                font-size: 0.83em;
            }
            .mission-stats span {
                display: flex;
                align-items: center;
                gap: 4px;
            }
            .baseline-score {
                margin-top: 8px;
                font-size: 0.85em;
                color: #7b2ff7;
                font-weight: 600;
            }

            /* ── API Endpoints ── */
            .api-section { margin-bottom: 44px; }
            .endpoint-list {
                background: #111827;
                border: 1px solid #1e293b;
                border-radius: 12px;
                overflow: hidden;
            }
            .endpoint-row {
                display: flex;
                align-items: center;
                gap: 14px;
                padding: 14px 20px;
                border-bottom: 1px solid #1e293b;
                font-size: 0.93em;
                transition: background 0.2s;
            }
            .endpoint-row:last-child { border-bottom: none; }
            .endpoint-row:hover { background: #1a1f35; }
            .method {
                padding: 3px 10px;
                border-radius: 5px;
                font-weight: 700;
                font-size: 0.75em;
                min-width: 42px;
                text-align: center;
                letter-spacing: 0.5px;
                font-family: 'Fira Code', 'Cascadia Code', monospace;
            }
            .method.ws { background: #7b2ff7; color: #fff; }
            .method.get { background: #064e3b; color: #34d399; }
            .method.post { background: #713f12; color: #fbbf24; }
            .endpoint-path {
                color: #00d4ff;
                font-family: 'Fira Code', 'Cascadia Code', monospace;
                font-size: 0.95em;
            }
            .endpoint-desc { color: #64748b; }

            /* ── Code Block ── */
            .code-block {
                background: #111827;
                border: 1px solid #1e293b;
                border-radius: 12px;
                padding: 20px;
                font-family: 'Fira Code', 'Cascadia Code', monospace;
                font-size: 0.84em;
                color: #e0e0e0;
                overflow-x: auto;
                margin-top: 16px;
                margin-bottom: 44px;
                white-space: pre;
                line-height: 1.65;
                position: relative;
            }
            .code-block .comment { color: #6b7280; }
            .code-block .keyword { color: #7b2ff7; }
            .code-block .string { color: #34d399; }
            .code-block .func { color: #fbbf24; }

            /* ── Scoring Table ── */
            .score-table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 44px;
                font-size: 0.92em;
            }
            .score-table th {
                text-align: left;
                padding: 12px 16px;
                background: #111827;
                color: #8892b0;
                font-weight: 600;
                font-size: 0.85em;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                border-bottom: 2px solid #1e293b;
            }
            .score-table td {
                padding: 12px 16px;
                border-bottom: 1px solid #1e293b;
                color: #e0e0e0;
            }
            .score-table tr:hover td { background: #111827; }
            .score-weight {
                color: #7b2ff7;
                font-weight: 700;
            }

            /* ── Footer ── */
            .footer {
                text-align: center;
                padding-top: 32px;
                border-top: 1px solid #1e293b;
                color: #64748b;
                font-size: 0.85em;
                line-height: 1.8;
            }
            .footer a {
                color: #7b2ff7;
                text-decoration: none;
                font-weight: 500;
                transition: color 0.2s;
            }
            .footer a:hover { color: #00d4ff; }
            .footer .separator { margin: 0 8px; color: #2a3050; }

            /* ── Responsive ── */
            @media (max-width: 600px) {
                .header h1 { font-size: 2em; letter-spacing: 3px; }
                .container { padding: 24px 16px; }
                .mission-stats { flex-direction: column; gap: 6px; }
                .endpoint-row { flex-wrap: wrap; gap: 8px; }
            }
        </style>
    </head>
    <body>
        <div class="container">

            <!-- Header -->
            <div class="header">
                <span class="header-icon">🛰️</span>
                <h1>ORBIT</h1>
                <div class="subtitle">AI Space Mission Architect</div>
                <div class="badges">
                    <span class="badge primary">OpenEnv Environment</span>
                    <span class="badge">Strategic Maneuvers</span>
                    <span class="badge">Orbital Mechanics</span>
                    <span class="badge">Deterministic Grading</span>
                </div>
                <div class="status-bar">
                    <div class="status-dot"></div>
                    <span class="status-text">Environment Online — v2.0.0</span>
                </div>
            </div>

            <!-- Description -->
            <p class="description">
                An RL environment where AI agents plan space missions by choosing strategic orbital
                maneuvers — Hohmann transfers, gravity assists, plane changes — while managing fuel
                budgets and navigating trade-offs between efficiency, accuracy, and mission constraints.
                Enriched observations provide available maneuvers, mission analysis, and recommendations
                to enable strategic reasoning. All scoring uses deterministic physics — no LLM judges.
            </p>

            <!-- Missions -->
            <div class="section-title">
                <span class="icon">🚀</span> Available Missions
            </div>
            <div class="missions">
                <div class="mission-card">
                    <div class="mission-header">
                        <span class="mission-name">LEO Satellite Deployment</span>
                        <span class="difficulty easy">Easy</span>
                    </div>
                    <div class="mission-desc">
                        Launch a satellite to a 400 km circular orbit at ISS inclination (51.6°).
                        Single strategic maneuver with generous fuel margins. Based on SpaceX Falcon 9 missions.
                    </div>
                    <div class="mission-stats">
                        <span>🎯 400 km altitude</span>
                        <span>📐 51.6° inclination</span>
                        <span>⛽ 12,000 m/s budget</span>
                        <span>🔄 Max 10 steps</span>
                    </div>
                    <div class="baseline-score">Baseline Score: 0.99</div>
                </div>
                <div class="mission-card">
                    <div class="mission-header">
                        <span class="mission-name">Lunar Orbit Insertion</span>
                        <span class="difficulty medium">Medium</span>
                    </div>
                    <div class="mission-desc">
                        Transfer from Earth parking orbit to lunar orbit. Two-burn sequence:
                        Trans-Lunar Injection → Lunar Orbit Insertion. Residual eccentricity
                        requires trade-off decisions. Based on Apollo & Artemis.
                    </div>
                    <div class="mission-stats">
                        <span>🎯 384,400 km distance</span>
                        <span>📐 28.5° inclination</span>
                        <span>⛽ 5,000 m/s budget</span>
                        <span>🔄 Max 15 steps</span>
                    </div>
                    <div class="baseline-score">Baseline Score: 0.88</div>
                </div>
                <div class="mission-card">
                    <div class="mission-header">
                        <span class="mission-name">Asteroid Mining Rendezvous (Bennu)</span>
                        <span class="difficulty hard">Hard</span>
                    </div>
                    <div class="mission-desc">
                        Reach asteroid Bennu using gravity assists, plane changes, and multi-step
                        planning. Direct transfer exceeds fuel budget — agent MUST use gravity assists.
                        Navigation uncertainty on deep-space transfers. Based on OSIRIS-REx.
                    </div>
                    <div class="mission-stats">
                        <span>🎯 ~0.8 AU distance</span>
                        <span>📐 6° inclination</span>
                        <span>⛽ 8,000 m/s budget</span>
                        <span>🔄 Max 25 steps</span>
                    </div>
                    <div class="baseline-score">Baseline Score: 0.81</div>
                </div>
            </div>

            <!-- API Endpoints -->
            <div class="section-title">
                <span class="icon">🔌</span> API Endpoints
            </div>
            <div class="api-section">
                <div class="endpoint-list">
                    <div class="endpoint-row">
                        <span class="method ws">WS</span>
                        <span class="endpoint-path">/ws</span>
                        <span class="endpoint-desc">Main environment interface (WebSocket)</span>
                    </div>
                    <div class="endpoint-row">
                        <span class="method post">POST</span>
                        <span class="endpoint-path">/reset</span>
                        <span class="endpoint-desc">Reset environment with task_id</span>
                    </div>
                    <div class="endpoint-row">
                        <span class="method post">POST</span>
                        <span class="endpoint-path">/step</span>
                        <span class="endpoint-desc">Execute one action step</span>
                    </div>
                    <div class="endpoint-row">
                        <span class="method get">GET</span>
                        <span class="endpoint-path">/tasks</span>
                        <span class="endpoint-desc">List all available missions</span>
                    </div>
                    <div class="endpoint-row">
                        <span class="method get">GET</span>
                        <span class="endpoint-path">/health</span>
                        <span class="endpoint-desc">Health check</span>
                    </div>
                </div>
            </div>

            <!-- Scoring -->
            <div class="section-title">
                <span class="icon">📊</span> Scoring System
            </div>
            <table class="score-table">
                <thead>
                    <tr>
                        <th>Component</th>
                        <th>Weight</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Altitude Accuracy</td>
                        <td><span class="score-weight">30%</span></td>
                        <td>How close to target altitude</td>
                    </tr>
                    <tr>
                        <td>Fuel Efficiency</td>
                        <td><span class="score-weight">30%</span></td>
                        <td>optimal_Δv / actual_Δv (penalizes over and underspending)</td>
                    </tr>
                    <tr>
                        <td>Eccentricity Accuracy</td>
                        <td><span class="score-weight">20%</span></td>
                        <td>How close to target eccentricity</td>
                    </tr>
                    <tr>
                        <td>Inclination Accuracy</td>
                        <td><span class="score-weight">15%</span></td>
                        <td>How close to target inclination</td>
                    </tr>
                    <tr>
                        <td>Step Efficiency</td>
                        <td><span class="score-weight">5%</span></td>
                        <td>Fewer steps = higher score</td>
                    </tr>
                </tbody>
            </table>

            <!-- Quick Start -->
            <div class="section-title">
                <span class="icon">⚡</span> Quick Start
            </div>
            <div class="code-block"><span class="keyword">import</span> asyncio, websockets, json

<span class="keyword">async def</span> <span class="func">run_mission</span>():
    <span class="keyword">async with</span> websockets.<span class="func">connect</span>(<span class="string">"wss://nitnem-orbit-env.hf.space/ws"</span>) <span class="keyword">as</span> ws:
        welcome = <span class="keyword">await</span> ws.<span class="func">recv</span>()

        <span class="comment"># Start a mission</span>
        <span class="keyword">await</span> ws.<span class="func">send</span>(json.<span class="func">dumps</span>({
            <span class="string">"type"</span>: <span class="string">"reset"</span>,
            <span class="string">"task_id"</span>: <span class="string">"leo_satellite"</span>
        }))
        obs = json.<span class="func">loads</span>(<span class="keyword">await</span> ws.<span class="func">recv</span>())

        <span class="comment"># Execute a strategic maneuver</span>
        <span class="keyword">await</span> ws.<span class="func">send</span>(json.<span class="func">dumps</span>({
            <span class="string">"type"</span>: <span class="string">"step"</span>,
            <span class="string">"action"</span>: {
                <span class="string">"type"</span>: <span class="string">"execute_maneuver"</span>,
                <span class="string">"maneuver"</span>: <span class="string">"hohmann_transfer"</span>,
                <span class="string">"target_altitude_km"</span>: 400
            }
        }))
        result = json.<span class="func">loads</span>(<span class="keyword">await</span> ws.<span class="func">recv</span>())

        <span class="comment"># Submit mission for grading</span>
        <span class="keyword">await</span> ws.<span class="func">send</span>(json.<span class="func">dumps</span>({
            <span class="string">"type"</span>: <span class="string">"step"</span>,
            <span class="string">"action"</span>: {<span class="string">"type"</span>: <span class="string">"submit_mission"</span>}
        }))
        final = json.<span class="func">loads</span>(<span class="keyword">await</span> ws.<span class="func">recv</span>())
        <span class="func">print</span>(f<span class="string">"Score: {final['reward']}"</span>)

asyncio.<span class="func">run</span>(<span class="func">run_mission</span>())</div>

            <!-- Footer -->
            <div class="footer">
                <p>Built for the <strong>OpenEnv Hackathon</strong> · Deterministic Physics-Based Grading</p>
                <p style="margin-top: 6px;">
                    <a href="https://github.com/Nitnem06/orbit-openenv" target="_blank">GitHub</a>
                    <span class="separator">·</span>
                    <a href="/tasks">Tasks API</a>
                    <span class="separator">·</span>
                    <a href="/health">Health Check</a>
                    <span class="separator">·</span>
                    <a href="/docs">API Docs</a>
                </p>
                <p style="margin-top: 10px; color: #4a5568; font-size: 0.8em;">
                    v2.0.0 · Orbit Environment © 2025
                </p>
            </div>
        </div>
    </body>
    </html>
    """


@app.get("/tasks")
async def get_tasks() -> JSONResponse:
    """List all available missions with their metadata."""
    return JSONResponse({
        "tasks": get_task_summary()
    })


@app.get("/health")
async def health() -> JSONResponse:
    """Health check for deployment monitoring."""
    return JSONResponse({
        "status": "healthy",
        "environment": "orbit-mission-architect",
        "version": "2.0.0",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "tasks_available": len(list_tasks()),
    })


@app.post("/reset")
async def http_reset(request: Request):
    global env_instance

    # SAFE JSON parsing (handles empty body)
    try:
        body = await request.json()
    except Exception:
        body = {}

    # Create new environment
    env_instance = OrbitEnvironment()

    # Get task_id safely
    task_id = body.get("task_id") if isinstance(body, dict) else None
    if not task_id:
        task_id = list_tasks()[0]  # default task

    observation = env_instance.reset(task_id)

    return {
        "observation": observation.model_dump(),
        "done": False
    }


class StepRequest(BaseModel):
    action: Dict[str, Any]


@app.post("/step")
async def http_step(req: StepRequest):
    global env_instance

    if env_instance is None:
        return JSONResponse(
            {"error": "Environment not initialized. Call /reset first."},
            status_code=400
        )

    try:
        action = parse_action(req.action)
        result = env_instance.step(action)

        return {
            "observation": result.observation.model_dump(),
            "reward": result.reward,
            "done": result.done,
            "info": result.info,
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/close")
async def http_close():
    global env_instance

    if env_instance:
        try:
            env_instance.close()
        except Exception:
            pass

    env_instance = None

    return {"status": "closed"}


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket Helpers
# ─────────────────────────────────────────────────────────────────────────────

async def send_json(websocket: WebSocket, data: Dict[str, Any]) -> None:
    """Send a JSON message to the client."""
    await websocket.send_text(json.dumps(data, default=str))


async def send_error(websocket: WebSocket, message: str) -> None:
    """Send an error message to the client."""
    logger.warning(f"Sending error to client: {message}")
    await send_json(websocket, {
        "type":    "error",
        "message": message,
    })


def parse_action(action_data: Dict) -> Action:
    """
    Parse a raw action dict into the correct Pydantic Action subtype.
    Uses Pydantic v2 TypeAdapter with discriminated union on 'type' field.

    Args:
        action_data: Raw dict from WebSocket message.

    Returns:
        Validated Action instance.

    Raises:
        ValidationError: If the action data is invalid.
        ValueError:      If action type is missing or unknown.
    """
    if "type" not in action_data:
        raise ValueError("Action must have a 'type' field.")
    return action_adapter.validate_python(action_data)


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket Message Handlers
# ─────────────────────────────────────────────────────────────────────────────

async def handle_reset(
    websocket: WebSocket,
    env: OrbitEnvironment,
    message: Dict,
) -> None:
    """Handle reset message — start a new episode."""
    task_id = message.get("task_id")
    if not task_id:
        await send_error(websocket, "reset message must include 'task_id'.")
        return

    try:
        observation = env.reset(task_id)
        logger.info(f"Environment reset for task: {task_id}")
        await send_json(websocket, {
            "type": "observation",
            "data": observation.model_dump(),
        })
    except ValueError as e:
        await send_error(websocket, str(e))


async def handle_step(
    websocket: WebSocket,
    env: OrbitEnvironment,
    message: Dict,
) -> None:
    """Handle step message — execute one action."""
    action_data = message.get("action")
    if not action_data:
        await send_error(websocket, "step message must include 'action'.")
        return

    try:
        action = parse_action(action_data)
    except (ValidationError, ValueError) as e:
        await send_error(websocket, f"Invalid action: {e}")
        return

    try:
        result = env.step(action)
        logger.info(
            f"Step {result.observation.step_index}: "
            f"action={action_data.get('type')} | "
            f"reward={result.reward} | "
            f"done={result.done}"
        )
        await send_json(websocket, {
            "type":        "step_result",
            "observation": result.observation.model_dump(),
            "reward":      result.reward,
            "done":        result.done,
            "info":        result.info,
        })
    except RuntimeError as e:
        await send_error(websocket, str(e))


async def handle_state(
    websocket: WebSocket,
    env: OrbitEnvironment,
) -> None:
    """Handle state message — return full internal state."""
    try:
        state = env.state()
        await send_json(websocket, {
            "type": "state",
            "data": state.model_dump(),
        })
    except RuntimeError as e:
        await send_error(websocket, str(e))


async def handle_list_tasks(websocket: WebSocket) -> None:
    """Handle list_tasks message — return all available missions."""
    await send_json(websocket, {
        "type": "task_list",
        "data": get_task_summary(),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Main WebSocket Endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    Main WebSocket endpoint. Each connection gets its own OrbitEnvironment.

    Accepts JSON messages with a 'type' field:
        'reset'      → Start new episode
        'step'       → Execute action
        'state'      → Get internal state
        'list_tasks' → Get all available missions

    Connection lifecycle:
        1. Client connects → server accepts
        2. Client sends messages → server responds
        3. Client disconnects → server cleans up gracefully
    """
    await websocket.accept()
    client = websocket.client
    logger.info(f"Client connected: {client}")

    # Each connection gets its own isolated environment instance
    env = OrbitEnvironment()

    # Send welcome message
    await send_json(websocket, {
        "type":    "welcome",
        "message": "Connected to Orbit — AI Space Mission Architect v2.0",
        "tasks":   list_tasks(),
        "hint":    "Send {\"type\": \"reset\", \"task_id\": \"leo_satellite\"} to start.",
    })

    try:
        while True:
            # Wait for next message from client
            raw = await websocket.receive_text()

            # Parse JSON
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await send_error(websocket, "Invalid JSON. Please send valid JSON messages.")
                continue

            # Route by message type
            msg_type = message.get("type")

            if msg_type == "reset":
                await handle_reset(websocket, env, message)

            elif msg_type == "step":
                await handle_step(websocket, env, message)

            elif msg_type == "state":
                await handle_state(websocket, env)

            elif msg_type == "list_tasks":
                await handle_list_tasks(websocket)

            else:
                await send_error(
                    websocket,
                    f"Unknown message type: '{msg_type}'. "
                    f"Valid types: reset, step, state, list_tasks."
                )

    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {client}")

    except Exception as e:
        logger.error(f"Unexpected error for client {client}: {e}")
        try:
            await send_error(websocket, f"Server error: {str(e)}")
        except Exception:
            pass  # Client already disconnected