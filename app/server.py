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
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ORBIT — AI Space Mission Architect</title>

<style>
* { margin:0; padding:0; box-sizing:border-box; }

body {
    font-family: 'Segoe UI', sans-serif;
    color:#e0e0e0;
    background: radial-gradient(circle at 20% 20%, #1a1f4d, #0a0e1a 40%),
                radial-gradient(circle at 80% 30%, #2a0845, transparent 40%),
                radial-gradient(circle at 50% 80%, #003973, transparent 40%);
}

/* HEADER */
.header {
    text-align:center;
    padding:80px 20px 40px;
}
.header h1 {
    font-size:3.5em;
    letter-spacing:6px;
    background: linear-gradient(90deg,#00d4ff,#7b2ff7,#ff6bcb);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

/* SECTION */
.section {
    max-width:1100px;
    margin:auto;
    padding:50px 20px;
}

/* CARD */
.card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(14px);
    border-radius:16px;
    padding:20px;
    margin-bottom:20px;
    transition:0.3s;
}
.card:hover {
    transform:translateY(-5px);
}

/* GRID */
.grid {
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:20px;
}

/* BUTTON */
.btn {
    padding:10px 18px;
    border:none;
    border-radius:8px;
    background: linear-gradient(135deg,#7b2ff7,#00d4ff);
    color:white;
    cursor:pointer;
    margin-top:10px;
}

/* CANVAS */
canvas {
    width:100%;
    height:320px;
    border-radius:12px;
    background:black;
}

/* CONSOLE */
.console {
    height:320px;
    overflow:auto;
    font-family:monospace;
    font-size:0.8em;
    background:#000;
    padding:10px;
    border-radius:10px;
}

/* ACCORDION */
.accordion {
    cursor:pointer;
    padding:15px;
    border-radius:10px;
    background: rgba(123,47,247,0.2);
    margin-bottom:10px;
    font-weight:600;
}

.panel {
    max-height:0;
    overflow:hidden;
    transition:max-height 0.3s ease;
    padding:0 10px;
    color:#9aa4c0;
}

.panel.open {
    max-height:200px;
    padding:10px;
}
</style>
</head>

<body>

<div class="header">
    <h1>ORBIT</h1>
    <p>AI Space Mission Architect</p>
</div>

<!-- SIMULATOR -->
<div class="section">
<div class="grid">

<div class="card">
<h3>🛰️ Orbit Simulation</h3>
<canvas id="orbitCanvas"></canvas>
</div>

<div class="card">
<h3>🔌 Live Console</h3>
<div class="console" id="console"></div>

<button class="btn" onclick="connectWS()">Connect</button>
<button class="btn" onclick="resetMission()">Reset</button>
<button class="btn" onclick="stepMission()">Step</button>

<input id="taskInput" placeholder="task_id">
</div>

</div>
</div>

<!-- MISSIONS -->
<div class="section">
<h2>🚀 Missions</h2>

<div class="accordion">LEO Satellite Deployment</div>
<div class="panel">
Launch satellite to 400km orbit at ISS inclination with high fuel margin.
</div>

<div class="accordion">Lunar Orbit Insertion</div>
<div class="panel">
Two-step Earth → Moon transfer using TLI and LOI burns.
</div>

<div class="accordion">Asteroid Bennu Rendezvous</div>
<div class="panel">
Deep-space mission using gravity assists and multi-step planning.
</div>

</div>

<script>
/* ACCORDION */
document.querySelectorAll(".accordion").forEach((btn,i)=>{
    btn.addEventListener("click",()=>{
        document.querySelectorAll(".panel").forEach(p=>p.classList.remove("open"));
        document.querySelectorAll(".panel")[i].classList.toggle("open");
    });
});

/* ORBIT SIMULATION */
const canvas = document.getElementById("orbitCanvas");
const ctx = canvas.getContext("2d");
canvas.width = canvas.offsetWidth;
canvas.height = canvas.offsetHeight;

let angle=0;

function draw(){
    ctx.fillStyle="rgba(0,0,0,0.3)";
    ctx.fillRect(0,0,canvas.width,canvas.height);

    let cx=canvas.width/2;
    let cy=canvas.height/2;

    // stars
    for(let i=0;i<50;i++){
        ctx.fillStyle="white";
        ctx.fillRect(Math.random()*canvas.width, Math.random()*canvas.height,1,1);
    }

    // planet glow
    let grd = ctx.createRadialGradient(cx,cy,10,cx,cy,40);
    grd.addColorStop(0,"#00d4ff");
    grd.addColorStop(1,"transparent");
    ctx.fillStyle=grd;
    ctx.beginPath();
    ctx.arc(cx,cy,40,0,Math.PI*2);
    ctx.fill();

    // orbit rings
    ctx.strokeStyle="#444";
    ctx.beginPath();
    ctx.arc(cx,cy,100,0,Math.PI*2);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(cx,cy,150,0,Math.PI*2);
    ctx.stroke();

    // satellites
    let x1 = cx + 100*Math.cos(angle);
    let y1 = cy + 100*Math.sin(angle);

    let x2 = cx + 150*Math.cos(-angle*0.7);
    let y2 = cy + 150*Math.sin(-angle*0.7);

    ctx.fillStyle="#fff";
    ctx.beginPath();
    ctx.arc(x1,y1,4,0,Math.PI*2);
    ctx.fill();

    ctx.beginPath();
    ctx.arc(x2,y2,4,0,Math.PI*2);
    ctx.fill();

    angle += 0.01;
    requestAnimationFrame(draw);
}
draw();

/* WEBSOCKET */
let ws;
function log(msg){
    let c=document.getElementById("console");
    c.innerHTML+=msg+"<br>";
    c.scrollTop=c.scrollHeight;
}

function connectWS(){
    ws = new WebSocket("wss://" + location.host + "/ws");
    ws.onopen=()=>log("Connected");
    ws.onmessage=(e)=>log(e.data);
}

function resetMission(){
    let t=document.getElementById("taskInput").value||"leo_satellite";
    ws.send(JSON.stringify({type:"reset",task_id:t}));
}

function stepMission(){
    ws.send(JSON.stringify({
        type:"step",
        action:{
            type:"execute_maneuver",
            maneuver:"hohmann_transfer",
            target_altitude_km:400
        }
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