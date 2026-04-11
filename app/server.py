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
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ORBIT — Mission Control</title>

<script src="https://cdn.tailwindcss.com"></script>

<style>
body {
    background: radial-gradient(circle at 20% 20%, #1a1f4d, #0a0e1a 50%),
                radial-gradient(circle at 80% 30%, #2a0845, transparent 40%);
}
.glow {
    box-shadow: 0 0 10px #00d4ff, 0 0 20px #7b2ff7;
}
.mono {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}
</style>
</head>

<body class="text-gray-200">

<!-- HEADER -->
<div class="p-4 border-b border-gray-800 flex justify-between items-center">
    <h1 class="text-2xl font-bold text-cyan-400">ORBIT: Mission Control</h1>
    <span id="status" class="px-3 py-1 rounded bg-red-500 text-sm">🔴 Disconnected</span>
</div>

<!-- MISSION SETUP -->
<div class="p-4 flex flex-wrap gap-4 items-center border-b border-gray-800">
    <select id="taskSelect" class="bg-gray-900 p-2 rounded">
        <option value="leo_satellite">LEO Satellite</option>
        <option value="lunar_orbit">Lunar Orbit</option>
        <option value="asteroid_rendezvous">Asteroid Rendezvous</option>
    </select>

    <button onclick="connectWS()" class="bg-purple-600 px-4 py-2 rounded glow">Connect Server</button>
    <button onclick="resetMission()" class="bg-cyan-500 px-4 py-2 rounded glow">Initialize Mission</button>
</div>

<!-- DASHBOARD -->
<div class="grid grid-cols-1 md:grid-cols-3 gap-4 p-4">

    <!-- ORBIT TABLE -->
    <div class="bg-gray-900 p-4 rounded">
        <h2 class="mb-2 text-lg text-cyan-400">Orbital Parameters</h2>
        <table class="w-full text-sm mono">
            <thead>
                <tr class="text-purple-400">
                    <th></th><th>Current</th><th>Target</th>
                </tr>
            </thead>
            <tbody id="orbitTable">
                <tr><td>Altitude</td><td>-</td><td>-</td></tr>
                <tr><td>Inclination</td><td>-</td><td>-</td></tr>
                <tr><td>Eccentricity</td><td>-</td><td>-</td></tr>
            </tbody>
        </table>
    </div>

    <!-- STATUS -->
    <div class="bg-gray-900 p-4 rounded">
        <h2 class="mb-2 text-lg text-cyan-400">Mission Status</h2>

        <div class="mb-3">
            <p class="text-sm">Fuel</p>
            <div class="w-full bg-gray-800 rounded h-4">
                <div id="fuelBar" class="bg-cyan-400 h-4 rounded" style="width:0%"></div>
            </div>
        </div>

        <p class="mono">Steps: <span id="steps">-</span></p>
        <p class="mono">Score: <span id="score">-</span></p>
    </div>

    <!-- INTEL -->
    <div class="bg-gray-900 p-4 rounded">
        <h2 class="mb-2 text-lg text-cyan-400">Mission Intel</h2>
        <ul id="intel" class="text-sm space-y-1 overflow-y-auto max-h-48"></ul>
    </div>

</div>

<!-- COMMAND CENTER -->
<div class="p-4 border-t border-gray-800">
    <h2 class="text-lg text-cyan-400 mb-2">Command Center</h2>
    <div id="actions" class="flex flex-wrap gap-2"></div>

    <div class="mt-4 flex gap-2">
        <input id="altitudeInput" type="number" placeholder="target_altitude_km" class="bg-gray-900 p-2 rounded w-40">
        <input id="inclinationInput" type="number" placeholder="target_inclination_deg" class="bg-gray-900 p-2 rounded w-40">
        <button onclick="submitMission()" class="bg-purple-600 px-4 py-2 rounded">Submit Mission</button>
    </div>
</div>

<!-- TERMINAL -->
<div class="p-4">
    <h2 class="text-lg text-cyan-400 mb-2">Terminal</h2>
    <div id="terminal" class="bg-black p-3 h-48 overflow-y-auto text-green-400 mono text-xs rounded"></div>
</div>

<script>
let ws;

function log(msg){
    const term = document.getElementById("terminal");
    term.innerHTML += msg + "<br>";
    term.scrollTop = term.scrollHeight;
}

function connectWS(){
    ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws');

    ws.onopen = () => {
        document.getElementById("status").innerText = "🟢 Connected";
        document.getElementById("status").classList.replace("bg-red-500","bg-green-500");
        log("Connected");
    };

    ws.onmessage = (e) => {
        let msg = JSON.parse(e.data);
        log(JSON.stringify(msg));

        if(msg.type === "observation" || msg.type === "step_result"){
            let data = msg.data || msg.observation;

            // orbit table
            document.getElementById("orbitTable").innerHTML = `
                <tr><td>Altitude</td><td>${data.current_orbit.altitude_km}</td><td>${data.target_orbit.altitude_km}</td></tr>
                <tr><td>Inclination</td><td>${data.current_orbit.inclination_deg}</td><td>${data.target_orbit.inclination_deg}</td></tr>
                <tr><td>Eccentricity</td><td>${data.current_orbit.eccentricity}</td><td>${data.target_orbit.eccentricity}</td></tr>
            `;

            // fuel
            let pct = (data.delta_v_used / data.delta_v_budget)*100;
            document.getElementById("fuelBar").style.width = pct + "%";

            // steps
            document.getElementById("steps").innerText = data.step_index + " / " + data.max_steps;

            // score
            document.getElementById("score").innerText = data.mission_analysis?.score_estimate || "-";

            // intel
            let intel = document.getElementById("intel");
            intel.innerHTML = "";
            (data.recommendations || []).forEach(r=>{
                let li = document.createElement("li");
                li.innerText = "• " + r;
                intel.appendChild(li);
            });

            // actions
            let actionsDiv = document.getElementById("actions");
            actionsDiv.innerHTML = "";
            (data.available_maneuvers || []).forEach(m=>{
                let btn = document.createElement("button");
                btn.innerText = m;
                btn.className = "bg-purple-700 px-3 py-1 rounded";
                btn.onclick = ()=>executeManeuver(m);
                actionsDiv.appendChild(btn);
            });

            if(msg.done){
                alert("Mission Complete! Reward: " + msg.reward);
            }
        }
    };
}

function resetMission(){
    let task = document.getElementById("taskSelect").value;
    ws.send(JSON.stringify({type:"reset", task_id:task}));
}

function executeManeuver(m){
    let alt = document.getElementById("altitudeInput").value;
    let inc = document.getElementById("inclinationInput").value;

    let action = {
        type:"execute_maneuver",
        maneuver:m
    };

    if(alt) action.target_altitude_km = Number(alt);
    if(inc) action.target_inclination_deg = Number(inc);

    ws.send(JSON.stringify({type:"step", action}));
}

function submitMission(){
    ws.send(JSON.stringify({
        type:"step",
        action:{type:"submit_mission"}
    }));
}
</script>

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