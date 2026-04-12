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
<script src="https://cdn.tailwindcss.com"></script>
<style>
html, body { height:100%; }
body {
    background:
        radial-gradient(circle at top left, rgba(0,212,255,0.14), transparent 28%),
        radial-gradient(circle at top right, rgba(123,47,247,0.12), transparent 30%),
        linear-gradient(180deg, #060914 0%, #0a1020 45%, #060a14 100%);
}
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
.panel {
    background: rgba(10,16,32,0.72);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(148,163,184,0.12);
    box-shadow: 0 10px 30px rgba(0,0,0,0.22);
}
.panel-soft {
    background: rgba(15,23,42,0.55);
    border: 1px solid rgba(148,163,184,0.10);
}
.metric-glow {
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 0 0 1px rgba(255,255,255,0.02);
}
.scrollbar-thin::-webkit-scrollbar { width: 8px; height: 8px; }
.scrollbar-thin::-webkit-scrollbar-thumb { background: rgba(100,116,139,0.45); border-radius: 999px; }
.scrollbar-thin::-webkit-scrollbar-track { background: transparent; }
.badge-dot::before {
    content:"";
    width:8px;
    height:8px;
    border-radius:999px;
    display:inline-block;
    margin-right:8px;
}
.status-disconnected::before { background:#ef4444; box-shadow:0 0 10px rgba(239,68,68,.7); }
.status-connecting::before { background:#f59e0b; box-shadow:0 0 10px rgba(245,158,11,.7); }
.status-connected::before { background:#22c55e; box-shadow:0 0 10px rgba(34,197,94,.7); }

.tab-btn.active {
    background: linear-gradient(135deg, rgba(14,165,233,0.18), rgba(168,85,247,0.16));
    border-color: rgba(96,165,250,0.35);
    color: #e2e8f0;
}
.mission-card.active {
    border-color: rgba(34,211,238,0.55);
    box-shadow: 0 0 0 1px rgba(34,211,238,0.2), 0 0 25px rgba(34,211,238,0.08);
}
.maneuver-card.feasible:hover {
    border-color: rgba(34,211,238,0.45);
    transform: translateY(-1px);
}
.maneuver-card.selected {
    border-color: rgba(168,85,247,0.55);
    box-shadow: 0 0 0 1px rgba(168,85,247,0.18), 0 0 24px rgba(168,85,247,0.08);
}
.grid-app {
    display:grid;
    grid-template-columns: 320px minmax(0,1fr) 360px;
    grid-template-rows: auto minmax(0,1fr) 260px;
    gap:16px;
    height:100vh;
    padding:16px;
}
.topbar { grid-column: 1 / -1; }
.sidebar { grid-column: 1; min-height:0; }
.center { grid-column: 2; min-height:0; }
.rightbar { grid-column: 3; min-height:0; }
.bottom { grid-column: 1 / -1; min-height:0; }

@media (max-width: 1280px) {
    .grid-app {
        grid-template-columns: 280px minmax(0,1fr) 320px;
        grid-template-rows: auto minmax(0,1fr) 280px;
    }
}
@media (max-width: 1100px) {
    .grid-app {
        grid-template-columns: 1fr;
        grid-template-rows: auto auto auto auto 300px;
        height:auto;
        min-height:100vh;
    }
    .topbar,.sidebar,.center,.rightbar,.bottom { grid-column: 1; }
}
</style>
</head>
<body class="text-slate-100 antialiased overflow-hidden">
<div class="grid-app">

    <!-- TOPBAR -->
    <header class="topbar panel rounded-2xl px-5 py-4 flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div class="flex items-start gap-4">
            <div class="h-12 w-12 rounded-2xl bg-gradient-to-br from-cyan-400 via-blue-500 to-violet-500 flex items-center justify-center text-2xl shadow-lg shadow-cyan-500/20">🚀</div>
            <div>
                <div class="flex items-center gap-3 flex-wrap">
                    <h1 class="text-2xl font-semibold tracking-wide">ORBIT</h1>
                    <span class="px-2.5 py-1 rounded-full text-xs border border-cyan-400/25 bg-cyan-400/10 text-cyan-300">OpenEnv RL Environment</span>
                    <span class="px-2.5 py-1 rounded-full text-xs border border-violet-400/25 bg-violet-400/10 text-violet-300">Deterministic Physics</span>
                </div>
                <p class="text-sm text-slate-400 mt-1">AI Space Mission Architect · Observe state, choose valid actions, inspect reward transitions.</p>
            </div>
        </div>

        <div class="flex flex-wrap items-center gap-3">
            <div id="connectionBadge" class="badge-dot status-disconnected px-3 py-2 rounded-xl border border-slate-700 bg-slate-900/70 text-sm text-slate-300">
                Disconnected
            </div>
            <div id="episodeBadge" class="px-3 py-2 rounded-xl border border-slate-700 bg-slate-900/70 text-sm text-slate-300">
                Episode: Idle
            </div>
            <button id="connectBtn" onclick="connectWS()" class="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-sm transition">
                Connect
            </button>
            <button onclick="requestState()" class="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-sm transition">
                Refresh State
            </button>
            <button onclick="resetMission()" class="px-4 py-2 rounded-xl bg-gradient-to-r from-cyan-500 to-violet-500 hover:opacity-95 text-white text-sm font-medium shadow-lg shadow-cyan-500/15 transition">
                Reset Episode
            </button>
        </div>
    </header>

    <!-- SIDEBAR -->
    <aside class="sidebar min-h-0 flex flex-col gap-4">
        <section class="panel rounded-2xl p-4 min-h-0">
            <div class="flex items-center justify-between mb-3">
                <h2 class="text-sm font-semibold text-slate-200 tracking-wide">Task Selection</h2>
                <span class="text-xs text-slate-500 mono">3 missions</span>
            </div>

            <div class="space-y-3">
                <button class="mission-card active w-full text-left panel-soft rounded-2xl p-4 transition border border-cyan-400/30" data-task="leo_satellite" onclick="selectMission('leo_satellite', this)">
                    <div class="flex items-center justify-between">
                        <div class="font-medium">LEO Satellite Deployment</div>
                        <span class="text-xs px-2 py-1 rounded-lg bg-emerald-500/15 text-emerald-300 border border-emerald-500/20">Easy</span>
                    </div>
                    <div class="mt-3 grid grid-cols-3 gap-2 text-xs text-slate-400">
                        <div><div class="text-slate-500">Optimal</div><div class="mono text-slate-200">9200</div></div>
                        <div><div class="text-slate-500">Budget</div><div class="mono text-slate-200">12000</div></div>
                        <div><div class="text-slate-500">Steps</div><div class="mono text-slate-200">10</div></div>
                    </div>
                </button>

                <button class="mission-card w-full text-left panel-soft rounded-2xl p-4 transition border border-slate-700" data-task="lunar_orbit" onclick="selectMission('lunar_orbit', this)">
                    <div class="flex items-center justify-between">
                        <div class="font-medium">Lunar Orbit Insertion</div>
                        <span class="text-xs px-2 py-1 rounded-lg bg-amber-500/15 text-amber-300 border border-amber-500/20">Medium</span>
                    </div>
                    <div class="mt-3 grid grid-cols-3 gap-2 text-xs text-slate-400">
                        <div><div class="text-slate-500">Optimal</div><div class="mono text-slate-200">3900</div></div>
                        <div><div class="text-slate-500">Budget</div><div class="mono text-slate-200">5000</div></div>
                        <div><div class="text-slate-500">Steps</div><div class="mono text-slate-200">15</div></div>
                    </div>
                </button>

                <button class="mission-card w-full text-left panel-soft rounded-2xl p-4 transition border border-slate-700" data-task="asteroid_rendezvous" onclick="selectMission('asteroid_rendezvous', this)">
                    <div class="flex items-center justify-between">
                        <div class="font-medium">Asteroid Bennu Rendezvous</div>
                        <span class="text-xs px-2 py-1 rounded-lg bg-rose-500/15 text-rose-300 border border-rose-500/20">Hard</span>
                    </div>
                    <div class="mt-3 grid grid-cols-3 gap-2 text-xs text-slate-400">
                        <div><div class="text-slate-500">Optimal</div><div class="mono text-slate-200">5800</div></div>
                        <div><div class="text-slate-500">Budget</div><div class="mono text-slate-200">8000</div></div>
                        <div><div class="text-slate-500">Steps</div><div class="mono text-slate-200">25</div></div>
                    </div>
                </button>
            </div>
        </section>

        <section class="panel rounded-2xl p-4">
            <div class="flex items-center justify-between mb-3">
                <h2 class="text-sm font-semibold tracking-wide">Mission Briefing</h2>
                <span id="selectedTaskPill" class="text-xs px-2 py-1 rounded-lg bg-cyan-400/10 text-cyan-300 border border-cyan-400/20">leo_satellite</span>
            </div>

            <div id="briefingTitle" class="text-lg font-semibold">LEO Satellite Deployment</div>
            <p id="briefingSummary" class="text-sm text-slate-400 mt-2">
                Launch to a 400 km circular orbit matching ISS inclination. Strategic objective: choose the correct transfer and submit efficiently.
            </p>

            <div class="grid grid-cols-2 gap-3 mt-4 text-sm">
                <div class="panel-soft rounded-xl p-3">
                    <div class="text-slate-500 text-xs">Real Reference</div>
                    <div id="briefingReference" class="mt-1">Falcon 9 → ISS</div>
                </div>
                <div class="panel-soft rounded-xl p-3">
                    <div class="text-slate-500 text-xs">Suggested Plan</div>
                    <div id="briefingPlan" class="mt-1 text-slate-200">Hohmann transfer → submit</div>
                </div>
                <div class="panel-soft rounded-xl p-3">
                    <div class="text-slate-500 text-xs">Optimal Δ-v</div>
                    <div id="briefingOptimal" class="mt-1 mono">9200 m/s</div>
                </div>
                <div class="panel-soft rounded-xl p-3">
                    <div class="text-slate-500 text-xs">Budget</div>
                    <div id="briefingBudget" class="mt-1 mono">12000 m/s</div>
                </div>
            </div>
        </section>

        <section class="panel rounded-2xl p-4">
            <div class="flex items-center justify-between mb-3">
                <h2 class="text-sm font-semibold tracking-wide">Episode Summary</h2>
                <span class="text-xs text-slate-500 mono" id="summaryDone">done: false</span>
            </div>
            <div class="grid grid-cols-2 gap-3 text-sm">
                <div class="panel-soft rounded-xl p-3 metric-glow">
                    <div class="text-slate-500 text-xs">Step</div>
                    <div id="summaryStep" class="mono mt-1 text-lg">0 / 0</div>
                </div>
                <div class="panel-soft rounded-xl p-3 metric-glow">
                    <div class="text-slate-500 text-xs">Reward</div>
                    <div id="summaryReward" class="mono mt-1 text-lg">0.00</div>
                </div>
                <div class="panel-soft rounded-xl p-3 metric-glow">
                    <div class="text-slate-500 text-xs">Δ-v Used</div>
                    <div id="summaryDvUsed" class="mono mt-1 text-lg">0</div>
                </div>
                <div class="panel-soft rounded-xl p-3 metric-glow">
                    <div class="text-slate-500 text-xs">Remaining</div>
                    <div id="summaryDvRemaining" class="mono mt-1 text-lg">—</div>
                </div>
            </div>
        </section>
    </aside>

    <!-- CENTER -->
    <main class="center min-h-0 flex flex-col gap-4">
        <section class="panel rounded-2xl p-4 flex flex-col min-h-0">
            <div class="flex items-center justify-between mb-3">
                <div>
                    <h2 class="text-sm font-semibold tracking-wide">Observation Workspace</h2>
                    <p class="text-xs text-slate-500 mt-1">State visualization and orbit comparison driven by environment observations.</p>
                </div>
                <div class="text-xs text-slate-500 mono" id="lastActionLabel">last_action_result: awaiting reset</div>
            </div>

            <div class="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_330px] gap-4 min-h-0 flex-1">
                <div class="panel-soft rounded-2xl p-3 min-h-[320px] flex flex-col">
                    <div class="flex items-center justify-between mb-2">
                        <span class="text-xs uppercase tracking-[0.18em] text-slate-500">Orbit Visualizer</span>
                        <span id="vizMode" class="text-xs px-2 py-1 rounded-lg border border-slate-700 bg-slate-900/60 text-slate-300">Earth Orbit</span>
                    </div>
                    <div class="flex-1 rounded-2xl overflow-hidden border border-slate-800 bg-black/25">
                        <canvas id="orbitCanvas" class="w-full h-full"></canvas>
                    </div>
                </div>

                <div class="flex flex-col gap-4 min-h-0">
                    <div class="panel-soft rounded-2xl p-4">
                        <div class="flex items-center justify-between mb-3">
                            <h3 class="text-sm font-medium">Fuel & Step Constraints</h3>
                            <span id="fuelPctLabel" class="mono text-xs text-cyan-300">0%</span>
                        </div>
                        <div class="mb-2 flex items-center justify-between text-xs text-slate-500">
                            <span>Δ-v usage</span>
                            <span id="fuelText" class="mono text-slate-300">0 / 0 m/s</span>
                        </div>
                        <div class="w-full h-3 rounded-full bg-slate-800 overflow-hidden border border-slate-700">
                            <div id="fuelBar" class="h-full bg-gradient-to-r from-cyan-400 to-violet-500 transition-all duration-500" style="width:0%"></div>
                        </div>

                        <div class="mt-4">
                            <div class="mb-2 flex items-center justify-between text-xs text-slate-500">
                                <span>Step budget</span>
                                <span id="stepText" class="mono text-slate-300">0 / 0</span>
                            </div>
                            <div class="w-full h-3 rounded-full bg-slate-800 overflow-hidden border border-slate-700">
                                <div id="stepBar" class="h-full bg-gradient-to-r from-emerald-400 to-cyan-400 transition-all duration-500" style="width:0%"></div>
                            </div>
                        </div>
                    </div>

                    <div class="panel-soft rounded-2xl p-4">
                        <div class="flex items-center justify-between mb-3">
                            <h3 class="text-sm font-medium">Score Estimate</h3>
                            <span class="text-xs text-slate-500">mission_analysis</span>
                        </div>
                        <div class="text-3xl mono font-semibold text-slate-100" id="scoreEstimate">0.00</div>
                        <div class="grid grid-cols-2 gap-3 mt-4 text-sm">
                            <div>
                                <div class="text-xs text-slate-500">Altitude Error</div>
                                <div id="altError" class="mono mt-1">—</div>
                            </div>
                            <div>
                                <div class="text-xs text-slate-500">Inclination Error</div>
                                <div id="incError" class="mono mt-1">—</div>
                            </div>
                            <div>
                                <div class="text-xs text-slate-500">Eccentricity Error</div>
                                <div id="eccError" class="mono mt-1">—</div>
                            </div>
                            <div>
                                <div class="text-xs text-slate-500">Fuel Margin</div>
                                <div id="fuelMargin" class="mono mt-1">—</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <section class="grid grid-cols-1 xl:grid-cols-3 gap-4">
            <div class="panel rounded-2xl p-4">
                <div class="flex items-center justify-between mb-3">
                    <h3 class="text-sm font-semibold">Current Orbit</h3>
                    <span class="text-xs text-slate-500">observation.current_orbit</span>
                </div>
                <div class="grid grid-cols-2 gap-3 text-sm">
                    <div class="panel-soft rounded-xl p-3"><div class="text-xs text-slate-500">Altitude</div><div id="curAltitude" class="mono mt-1">—</div></div>
                    <div class="panel-soft rounded-xl p-3"><div class="text-xs text-slate-500">Inclination</div><div id="curInclination" class="mono mt-1">—</div></div>
                    <div class="panel-soft rounded-xl p-3"><div class="text-xs text-slate-500">Eccentricity</div><div id="curEccentricity" class="mono mt-1">—</div></div>
                    <div class="panel-soft rounded-xl p-3"><div class="text-xs text-slate-500">Velocity</div><div id="curVelocity" class="mono mt-1">—</div></div>
                </div>
            </div>

            <div class="panel rounded-2xl p-4">
                <div class="flex items-center justify-between mb-3">
                    <h3 class="text-sm font-semibold">Target Orbit</h3>
                    <span class="text-xs text-slate-500">observation.target_orbit</span>
                </div>
                <div class="grid grid-cols-2 gap-3 text-sm">
                    <div class="panel-soft rounded-xl p-3"><div class="text-xs text-slate-500">Altitude</div><div id="tgtAltitude" class="mono mt-1">—</div></div>
                    <div class="panel-soft rounded-xl p-3"><div class="text-xs text-slate-500">Inclination</div><div id="tgtInclination" class="mono mt-1">—</div></div>
                    <div class="panel-soft rounded-xl p-3"><div class="text-xs text-slate-500">Eccentricity</div><div id="tgtEccentricity" class="mono mt-1">—</div></div>
                    <div class="panel-soft rounded-xl p-3"><div class="text-xs text-slate-500">Estimated Δ-v Needed</div><div id="dvNeeded" class="mono mt-1">—</div></div>
                </div>
            </div>

            <div class="panel rounded-2xl p-4 min-h-0">
                <div class="flex items-center justify-between mb-3">
                    <h3 class="text-sm font-semibold">Recommendations</h3>
                    <span class="text-xs text-slate-500">observation.recommendations</span>
                </div>
                <div id="recommendationsList" class="space-y-2 max-h-[220px] overflow-auto scrollbar-thin">
                    <div class="panel-soft rounded-xl p-3 text-sm text-slate-400">Reset an episode to receive context-aware recommendations.</div>
                </div>
            </div>
        </section>
    </main>

    <!-- RIGHTBAR -->
    <aside class="rightbar min-h-0 flex flex-col gap-4">
        <section class="panel rounded-2xl p-4 flex flex-col min-h-0">
            <div class="flex items-center justify-between mb-3">
                <h2 class="text-sm font-semibold tracking-wide">Action Space</h2>
                <span class="text-xs text-slate-500">Strategic + Utility + Advanced</span>
            </div>

            <div class="mb-4 flex items-center gap-2">
                <button onclick="sendRunSimulation()" class="flex-1 px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-sm transition">
                    Preview Score
                </button>
                <button onclick="submitMission()" class="flex-1 px-3 py-2 rounded-xl bg-gradient-to-r from-emerald-500 to-cyan-500 hover:opacity-95 text-white text-sm font-medium transition">
                    Submit Mission
                </button>
            </div>

            <div class="mb-3">
                <div class="text-xs uppercase tracking-[0.18em] text-slate-500 mb-2">Available Maneuvers</div>
                <div id="maneuversList" class="space-y-2 max-h-[290px] overflow-auto scrollbar-thin pr-1">
                    <div class="panel-soft rounded-xl p-3 text-sm text-slate-400">No maneuver data yet. Connect and reset a task.</div>
                </div>
            </div>

            <div class="panel-soft rounded-2xl p-4 mt-auto">
                <div class="flex items-center justify-between mb-3">
                    <h3 class="text-sm font-medium">Selected Action</h3>
                    <span id="selectedActionName" class="text-xs text-violet-300 mono">none</span>
                </div>

                <div class="space-y-3">
                    <div>
                        <label class="block text-xs text-slate-500 mb-1">Maneuver</label>
                        <input id="maneuverNameInput" class="w-full bg-slate-950/80 border border-slate-700 rounded-xl px-3 py-2 text-sm text-slate-200 outline-none" placeholder="e.g. hohmann_transfer">
                    </div>

                    <div class="grid grid-cols-2 gap-3">
                        <div>
                            <label class="block text-xs text-slate-500 mb-1">Target Altitude (km)</label>
                            <input id="targetAltitudeInput" type="number" class="w-full bg-slate-950/80 border border-slate-700 rounded-xl px-3 py-2 text-sm text-slate-200 outline-none" placeholder="400">
                        </div>
                        <div>
                            <label class="block text-xs text-slate-500 mb-1">Target Inclination</label>
                            <input id="targetInclinationInput" type="number" step="0.1" class="w-full bg-slate-950/80 border border-slate-700 rounded-xl px-3 py-2 text-sm text-slate-200 outline-none" placeholder="51.6">
                        </div>
                    </div>

                    <div class="grid grid-cols-2 gap-3">
                        <div>
                            <label class="block text-xs text-slate-500 mb-1">Gravity Assist Body</label>
                            <input id="bodyInput" class="w-full bg-slate-950/80 border border-slate-700 rounded-xl px-3 py-2 text-sm text-slate-200 outline-none" placeholder="venus">
                        </div>
                        <div>
                            <label class="block text-xs text-slate-500 mb-1">Correction Δ-v (m/s)</label>
                            <input id="deltaVInput" type="number" class="w-full bg-slate-950/80 border border-slate-700 rounded-xl px-3 py-2 text-sm text-slate-200 outline-none" placeholder="50">
                        </div>
                    </div>

                    <button onclick="executeSelectedManeuver()" class="w-full px-4 py-3 rounded-xl bg-gradient-to-r from-violet-500 to-fuchsia-500 hover:opacity-95 text-white text-sm font-semibold transition">
                        Execute Maneuver
                    </button>
                </div>
            </div>
        </section>

        <section class="panel rounded-2xl p-4">
            <div class="flex items-center justify-between mb-3">
                <h2 class="text-sm font-semibold tracking-wide">Advanced Action</h2>
                <span class="text-xs text-slate-500">legacy add_burn</span>
            </div>
            <div class="grid grid-cols-2 gap-3 text-sm">
                <input id="burnDvInput" type="number" class="bg-slate-950/80 border border-slate-700 rounded-xl px-3 py-2 outline-none" placeholder="delta_v_ms">
                <input id="burnProgradeInput" type="number" step="0.1" class="bg-slate-950/80 border border-slate-700 rounded-xl px-3 py-2 outline-none" placeholder="prograde">
                <input id="burnRadialInput" type="number" step="0.1" class="bg-slate-950/80 border border-slate-700 rounded-xl px-3 py-2 outline-none" placeholder="radial">
                <input id="burnNormalInput" type="number" step="0.1" class="bg-slate-950/80 border border-slate-700 rounded-xl px-3 py-2 outline-none" placeholder="normal">
            </div>
            <button onclick="sendLegacyBurn()" class="w-full mt-3 px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-sm transition">
                Execute Low-Level Burn
            </button>
        </section>
    </aside>

    <!-- BOTTOM -->
    <section class="bottom panel rounded-2xl p-4 min-h-0 flex flex-col">
        <div class="flex items-center justify-between mb-3">
            <div>
                <h2 class="text-sm font-semibold tracking-wide">Transition & Logs</h2>
                <p class="text-xs text-slate-500 mt-1">Inspect reward, done state, transition info, and raw protocol traffic.</p>
            </div>
            <button onclick="clearLogs()" class="px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs transition">
                Clear Logs
            </button>
        </div>

        <div class="flex gap-2 mb-3 flex-wrap">
            <button class="tab-btn active px-3 py-2 rounded-xl text-sm border border-slate-700 bg-slate-900/60 text-slate-400" data-tab="activity" onclick="switchTab('activity', this)">Activity</button>
            <button class="tab-btn px-3 py-2 rounded-xl text-sm border border-slate-700 bg-slate-900/60 text-slate-400" data-tab="transition" onclick="switchTab('transition', this)">Last Transition</button>
            <button class="tab-btn px-3 py-2 rounded-xl text-sm border border-slate-700 bg-slate-900/60 text-slate-400" data-tab="raw" onclick="switchTab('raw', this)">Raw JSON</button>
            <button class="tab-btn px-3 py-2 rounded-xl text-sm border border-slate-700 bg-slate-900/60 text-slate-400" data-tab="history" onclick="switchTab('history', this)">Action History</button>
        </div>

        <div class="flex-1 min-h-0">
            <div id="tab-activity" class="tab-panel h-full">
                <div id="activityLog" class="h-full overflow-auto scrollbar-thin rounded-2xl panel-soft p-4 mono text-sm text-slate-300 space-y-2"></div>
            </div>

            <div id="tab-transition" class="tab-panel h-full hidden">
                <div class="h-full overflow-auto scrollbar-thin rounded-2xl panel-soft p-4">
                    <div class="grid grid-cols-1 md:grid-cols-4 gap-3 text-sm">
                        <div class="rounded-xl bg-slate-950/60 border border-slate-800 p-3">
                            <div class="text-xs text-slate-500">Reward</div>
                            <div id="transitionReward" class="mono text-lg mt-1">0.00</div>
                        </div>
                        <div class="rounded-xl bg-slate-950/60 border border-slate-800 p-3">
                            <div class="text-xs text-slate-500">Done</div>
                            <div id="transitionDone" class="mono text-lg mt-1">false</div>
                        </div>
                        <div class="rounded-xl bg-slate-950/60 border border-slate-800 p-3">
                            <div class="text-xs text-slate-500">Info</div>
                            <div id="transitionInfo" class="mono text-sm mt-1 text-slate-300 break-words">—</div>
                        </div>
                        <div class="rounded-xl bg-slate-950/60 border border-slate-800 p-3">
                            <div class="text-xs text-slate-500">Last Action Result</div>
                            <div id="transitionLastAction" class="mono text-sm mt-1 text-slate-300 break-words">—</div>
                        </div>
                    </div>
                </div>
            </div>

            <div id="tab-raw" class="tab-panel h-full hidden">
                <pre id="rawLog" class="h-full overflow-auto scrollbar-thin rounded-2xl panel-soft p-4 mono text-xs text-slate-300 whitespace-pre-wrap break-words"></pre>
            </div>

            <div id="tab-history" class="tab-panel h-full hidden">
                <div id="historyLog" class="h-full overflow-auto scrollbar-thin rounded-2xl panel-soft p-4 text-sm text-slate-300 space-y-2"></div>
            </div>
        </div>
    </section>
</div>

<script>
const missionMeta = {
    leo_satellite: {
        title: "LEO Satellite Deployment",
        summary: "Launch to a 400 km circular orbit matching ISS inclination. Strategic objective: choose the correct transfer and submit efficiently.",
        reference: "Falcon 9 → ISS",
        plan: "Hohmann transfer → submit",
        optimal: "9200 m/s",
        budget: "12000 m/s",
        viz: "Earth Orbit"
    },
    lunar_orbit: {
        title: "Lunar Orbit Insertion",
        summary: "Transfer from a 200 km Earth parking orbit to lunar orbit using correct multi-burn sequencing and careful fuel management.",
        reference: "Apollo / Artemis",
        plan: "TLI → LOI → optional circularize",
        optimal: "3900 m/s",
        budget: "5000 m/s",
        viz: "Earth–Moon Transfer"
    },
    asteroid_rendezvous: {
        title: "Asteroid Bennu Rendezvous",
        summary: "Reach Bennu using gravity assists, combined transfer planning, and correction burns while staying within deep-space fuel and step limits.",
        reference: "OSIRIS-REx",
        plan: "Gravity assist → gravity assist → combined transfer → corrections",
        optimal: "5800 m/s",
        budget: "8000 m/s",
        viz: "Deep Space Transfer"
    }
};

let selectedTask = "leo_satellite";
let ws = null;
let lastObservation = null;
let latestReward = 0;
let latestDone = false;
let historyItems = [];
let rawItems = [];
let selectedManeuver = null;

let vizState = {
    task: "leo_satellite",
    current_altitude_km: 0,
    current_inclination_deg: 0,
    current_eccentricity: 0,
    target_altitude_km: 400,
    target_inclination_deg: 51.6,
    target_eccentricity: 0,
    step_index: 0,
    max_steps: 10
};

function byId(id){ return document.getElementById(id); }

function selectMission(taskId, el) {
    selectedTask = taskId;
    document.querySelectorAll(".mission-card").forEach(card => {
        card.classList.remove("active", "border-cyan-400/30");
        card.classList.add("border-slate-700");
    });
    if (el) {
        el.classList.add("active");
        el.classList.remove("border-slate-700");
    }
    const meta = missionMeta[taskId];
    byId("selectedTaskPill").textContent = taskId;
    byId("briefingTitle").textContent = meta.title;
    byId("briefingSummary").textContent = meta.summary;
    byId("briefingReference").textContent = meta.reference;
    byId("briefingPlan").textContent = meta.plan;
    byId("briefingOptimal").textContent = meta.optimal;
    byId("briefingBudget").textContent = meta.budget;
    byId("vizMode").textContent = meta.viz;
}

function setConnectionState(state) {
    const badge = byId("connectionBadge");
    badge.className = "badge-dot px-3 py-2 rounded-xl border text-sm";
    if (state === "connecting") {
        badge.classList.add("status-connecting", "border-amber-500/20", "bg-amber-500/10", "text-amber-300");
        badge.textContent = "Connecting";
    } else if (state === "connected") {
        badge.classList.add("status-connected", "border-emerald-500/20", "bg-emerald-500/10", "text-emerald-300");
        badge.textContent = "Connected";
    } else {
        badge.classList.add("status-disconnected", "border-rose-500/20", "bg-rose-500/10", "text-rose-300");
        badge.textContent = "Disconnected";
    }
}

function setEpisodeState(text) {
    byId("episodeBadge").textContent = "Episode: " + text;
}

function logActivity(msg, tone="normal") {
    const colors = {
        normal: "text-slate-300",
        success: "text-emerald-300",
        warn: "text-amber-300",
        error: "text-rose-300",
        info: "text-cyan-300"
    };
    const time = new Date().toLocaleTimeString();
    const line = document.createElement("div");
    line.className = colors[tone] || colors.normal;
    line.innerHTML = `<span class="text-slate-500">[${time}]</span> ${escapeHtml(String(msg))}`;
    byId("activityLog").appendChild(line);
    byId("activityLog").scrollTop = byId("activityLog").scrollHeight;
}

function addRaw(item) {
    rawItems.push(item);
    if (rawItems.length > 100) rawItems.shift();
    byId("rawLog").textContent = rawItems.map(x => JSON.stringify(x, null, 2)).join("\n\n");
}

function addHistory(item) {
    historyItems.push(item);
    if (historyItems.length > 50) historyItems.shift();
    renderHistory();
}

function renderHistory() {
    const box = byId("historyLog");
    if (!historyItems.length) {
        box.innerHTML = '<div class="text-slate-500">No actions executed yet.</div>';
        return;
    }
    box.innerHTML = historyItems.map((h, i) => `
        <div class="rounded-xl bg-slate-950/55 border border-slate-800 p-3">
            <div class="flex items-center justify-between gap-3">
                <div class="font-medium text-slate-200">Step ${escapeHtml(String(h.step))}: ${escapeHtml(h.label)}</div>
                <div class="mono text-xs text-slate-400">reward ${escapeHtml(String(h.reward))}</div>
            </div>
            <div class="mt-1 text-sm text-slate-400">${escapeHtml(h.result || "—")}</div>
        </div>
    `).join("");
}

function switchTab(tab, btn) {
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.add("hidden"));
    byId("tab-" + tab).classList.remove("hidden");
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    if (btn) btn.classList.add("active");
}

function clearLogs() {
    byId("activityLog").innerHTML = "";
    byId("rawLog").textContent = "";
    byId("historyLog").innerHTML = "";
    rawItems = [];
    historyItems = [];
    logActivity("Logs cleared.", "info");
}

function connectWS() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        logActivity("WebSocket already connected.", "info");
        return;
    }

    setConnectionState("connecting");
    logActivity("Opening WebSocket connection...", "info");

    const proto = window.location.protocol === "https:" ? "wss://" : "ws://";
    ws = new WebSocket(proto + window.location.host + "/ws");

    ws.onopen = () => {
        setConnectionState("connected");
        logActivity("Connected to Orbit environment.", "success");
    };

    ws.onclose = () => {
        setConnectionState("disconnected");
        setEpisodeState("Idle");
        logActivity("WebSocket connection closed.", "warn");
    };

    ws.onerror = () => {
        setConnectionState("disconnected");
        logActivity("WebSocket error occurred.", "error");
    };

    ws.onmessage = (e) => {
        try {
            const msg = JSON.parse(e.data);
            addRaw(msg);

            if (msg.type === "welcome") {
                logActivity(msg.message || "Received welcome message.", "info");
                setEpisodeState("Ready");
                return;
            }

            if (msg.type === "task_list") {
                logActivity("Received task list.", "info");
                return;
            }

            if (msg.type === "observation") {
                latestReward = 0;
                latestDone = false;
                setEpisodeState("Active");
                handleObservation(msg.data, "reset");
                logActivity("Episode reset completed.", "success");
                return;
            }

            if (msg.type === "state") {
                logActivity("State snapshot received.", "info");
                if (msg.data) {
                    handleObservation(msg.data, "state");
                }
                return;
            }

            if (msg.type === "step_result") {
                latestReward = Number(msg.reward || 0);
                latestDone = Boolean(msg.done);
                handleObservation(msg.observation, "step", msg);
                byId("transitionReward").textContent = formatNumber(latestReward, 4);
                byId("transitionDone").textContent = String(latestDone);
                byId("transitionInfo").textContent = typeof msg.info === "object" ? JSON.stringify(msg.info) : String(msg.info || "—");
                byId("transitionLastAction").textContent = (msg.observation && msg.observation.last_action_result) ? msg.observation.last_action_result : "—";

                setEpisodeState(latestDone ? "Done" : "Active");
                logActivity(`Step transition received. Reward=${formatNumber(latestReward, 4)} done=${latestDone}`, latestDone ? "warn" : "success");

                addHistory({
                    step: (msg.observation && msg.observation.step_index != null) ? msg.observation.step_index : historyItems.length + 1,
                    label: selectedManeuver || "action",
                    reward: formatNumber(latestReward, 4),
                    result: (msg.observation && msg.observation.last_action_result) ? msg.observation.last_action_result : "Transition received"
                });

                return;
            }

            if (msg.type === "error") {
                logActivity(msg.message || "Server error", "error");
                return;
            }

            logActivity("Unhandled message type: " + msg.type, "warn");
        } catch (err) {
            logActivity("Failed to parse server message.", "error");
        }
    };
}

function ensureConnected() {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        logActivity("WebSocket is not connected. Click Connect first.", "error");
        return false;
    }
    return true;
}

function resetMission() {
    if (!ensureConnected()) return;
    ws.send(JSON.stringify({ type: "reset", task_id: selectedTask }));
    logActivity("Sending reset for task: " + selectedTask, "info");
    setEpisodeState("Resetting");
}

function requestState() {
    if (!ensureConnected()) return;
    ws.send(JSON.stringify({ type: "state" }));
    logActivity("Requested current state snapshot.", "info");
}

function sendRunSimulation() {
    if (!ensureConnected()) return;
    ws.send(JSON.stringify({ type: "step", action: { type: "run_simulation" } }));
    selectedManeuver = "run_simulation";
    logActivity("Requested simulation preview.", "info");
}

function submitMission() {
    if (!ensureConnected()) return;
    ws.send(JSON.stringify({ type: "step", action: { type: "submit_mission" } }));
    selectedManeuver = "submit_mission";
    logActivity("Submitting mission.", "warn");
}

function executeSelectedManeuver() {
    if (!ensureConnected()) return;

    const maneuver = (byId("maneuverNameInput").value || "").trim();
    if (!maneuver) {
        logActivity("Select or enter a maneuver first.", "error");
        return;
    }

    const action = {
        type: "execute_maneuver",
        maneuver: maneuver
    };

    const alt = byId("targetAltitudeInput").value;
    const inc = byId("targetInclinationInput").value;
    const body = (byId("bodyInput").value || "").trim();
    const dv = byId("deltaVInput").value;

    if (alt !== "") action.target_altitude_km = Number(alt);
    if (inc !== "") action.target_inclination_deg = Number(inc);
    if (body) action.body = body;
    if (dv !== "") action.delta_v_ms = Number(dv);

    ws.send(JSON.stringify({ type: "step", action }));
    selectedManeuver = maneuver;
    logActivity("Executing maneuver: " + maneuver, "info");
}

function sendLegacyBurn() {
    if (!ensureConnected()) return;

    const action = {
        type: "add_burn",
        delta_v_ms: Number(byId("burnDvInput").value || 0),
        prograde: Number(byId("burnProgradeInput").value || 0),
        radial: Number(byId("burnRadialInput").value || 0),
        normal: Number(byId("burnNormalInput").value || 0)
    };

    ws.send(JSON.stringify({ type: "step", action }));
    selectedManeuver = "add_burn";
    logActivity("Executing legacy low-level burn.", "info");
}

function handleObservation(obs, source, fullMessage=null) {
    if (!obs) return;
    lastObservation = obs;

    const co = obs.current_orbit || {};
    const to = obs.target_orbit || {};
    const ma = obs.mission_analysis || {};
    const recs = Array.isArray(obs.recommendations) ? obs.recommendations : [];
    const maneuvers = Array.isArray(obs.available_maneuvers) ? obs.available_maneuvers : [];

    vizState.task = selectedTask;
    vizState.current_altitude_km = Number(co.altitude_km || 0);
    vizState.current_inclination_deg = Number(co.inclination_deg || 0);
    vizState.current_eccentricity = Number(co.eccentricity || 0);
    vizState.target_altitude_km = Number(to.altitude_km || 0);
    vizState.target_inclination_deg = Number(to.inclination_deg || 0);
    vizState.target_eccentricity = Number(to.eccentricity || 0);
    vizState.step_index = Number(obs.step_index || 0);
    vizState.max_steps = Number(obs.max_steps || 0);

    byId("lastActionLabel").textContent = "last_action_result: " + (obs.last_action_result || "—");

    // Current orbit
    byId("curAltitude").textContent = formatMetric(co.altitude_km, " km");
    byId("curInclination").textContent = formatMetric(co.inclination_deg, "°");
    byId("curEccentricity").textContent = formatMetric(co.eccentricity, "");
    byId("curVelocity").textContent = formatMetric(co.velocity_km_s ?? co.velocity_kms ?? co.velocity_ms, co.velocity_ms != null ? " m/s" : " km/s");

    // Target orbit
    byId("tgtAltitude").textContent = formatMetric(to.altitude_km, " km");
    byId("tgtInclination").textContent = formatMetric(to.inclination_deg, "°");
    byId("tgtEccentricity").textContent = formatMetric(to.eccentricity, "");
    byId("dvNeeded").textContent = formatMetric(ma.estimated_delta_v_needed, " m/s");

    // Mission analysis
    byId("scoreEstimate").textContent = formatNumber(ma.current_score_estimate ?? 0, 2);
    byId("altError").textContent = formatMetric(ma.altitude_error_km, " km");
    byId("incError").textContent = formatMetric(ma.inclination_error_deg, "°");
    byId("eccError").textContent = formatMetric(ma.eccentricity_error, "");
    byId("fuelMargin").textContent = formatMetric(ma.fuel_margin_percent, "%");

    // Fuel + step
    const used = Number(obs.delta_v_used || 0);
    const budget = Number(obs.delta_v_budget || 0);
    const remaining = Math.max(0, budget - used);
    const fuelPct = budget > 0 ? Math.min(100, (used / budget) * 100) : 0;

    byId("fuelText").textContent = `${formatNumber(used,0)} / ${formatNumber(budget,0)} m/s`;
    byId("fuelPctLabel").textContent = `${formatNumber(fuelPct,1)}% used`;
    byId("fuelBar").style.width = fuelPct + "%";

    if (fuelPct < 55) {
        byId("fuelBar").className = "h-full bg-gradient-to-r from-cyan-400 to-violet-500 transition-all duration-500";
    } else if (fuelPct < 85) {
        byId("fuelBar").className = "h-full bg-gradient-to-r from-amber-400 to-orange-500 transition-all duration-500";
    } else {
        byId("fuelBar").className = "h-full bg-gradient-to-r from-rose-500 to-red-500 transition-all duration-500";
    }

    const stepIndex = Number(obs.step_index || 0);
    const maxSteps = Number(obs.max_steps || 0);
    const stepPct = maxSteps > 0 ? Math.min(100, (stepIndex / maxSteps) * 100) : 0;

    byId("stepText").textContent = `${stepIndex} / ${maxSteps}`;
    byId("stepBar").style.width = stepPct + "%";

    // Summary
    byId("summaryStep").textContent = `${stepIndex} / ${maxSteps}`;
    byId("summaryReward").textContent = formatNumber(latestReward, 4);
    byId("summaryDvUsed").textContent = `${formatNumber(used,0)} m/s`;
    byId("summaryDvRemaining").textContent = `${formatNumber(remaining,0)} m/s`;
    byId("summaryDone").textContent = `done: ${latestDone}`;
    byId("transitionLastAction").textContent = obs.last_action_result || "—";

    // Recommendations
    renderRecommendations(recs);

    // Maneuvers
    renderManeuvers(maneuvers);

    drawOrbit();
}

function renderRecommendations(recs) {
    const container = byId("recommendationsList");
    if (!recs.length) {
        container.innerHTML = `<div class="panel-soft rounded-xl p-3 text-sm text-slate-400">No recommendations available for the current observation.</div>`;
        return;
    }

    container.innerHTML = recs.map((r, idx) => `
        <div class="panel-soft rounded-xl p-3">
            <div class="flex items-start gap-3">
                <div class="mt-0.5 h-6 w-6 rounded-lg bg-cyan-400/10 border border-cyan-400/20 text-cyan-300 flex items-center justify-center text-xs">${idx+1}</div>
                <div class="text-sm text-slate-300">${escapeHtml(String(r))}</div>
            </div>
        </div>
    `).join("");
}

function renderManeuvers(items) {
    const container = byId("maneuversList");
    if (!items.length) {
        container.innerHTML = `<div class="panel-soft rounded-xl p-3 text-sm text-slate-400">No maneuver metadata returned yet for this observation.</div>`;
        return;
    }

    container.innerHTML = items.map(item => {
        const feasible = item.feasible !== false;
        const reason = item.reason ? `<div class="text-xs text-rose-300 mt-2">${escapeHtml(String(item.reason))}</div>` : "";
        return `
            <button
                class="maneuver-card ${feasible ? 'feasible' : ''} w-full text-left rounded-xl border ${feasible ? 'border-slate-700 bg-slate-950/45 hover:bg-slate-900/70' : 'border-rose-500/20 bg-rose-500/5 opacity-80'} p-3 transition"
                ${feasible ? `onclick="selectManeuverCard(${escapeAttr(JSON.stringify(item))}, this)"` : ''}
            >
                <div class="flex items-center justify-between gap-3">
                    <div class="font-medium text-slate-200">${escapeHtml(String(item.name || 'maneuver'))}</div>
                    <span class="text-[11px] px-2 py-1 rounded-lg border ${feasible ? 'border-emerald-500/20 bg-emerald-500/10 text-emerald-300' : 'border-rose-500/20 bg-rose-500/10 text-rose-300'}">
                        ${feasible ? 'feasible' : 'blocked'}
                    </span>
                </div>
                <div class="text-xs text-slate-400 mt-1">${escapeHtml(String(item.description || ''))}</div>
                <div class="grid grid-cols-2 gap-2 mt-3 text-xs">
                    <div><span class="text-slate-500">est Δ-v</span><div class="mono text-slate-200 mt-1">${item.estimated_delta_v != null ? escapeHtml(formatNumber(item.estimated_delta_v,1) + ' m/s') : '—'}</div></div>
                    <div><span class="text-slate-500">fuel %</span><div class="mono text-slate-200 mt-1">${item.fuel_percentage != null ? escapeHtml(formatNumber(item.fuel_percentage,1) + '%') : '—'}</div></div>
                </div>
                ${reason}
            </button>
        `;
    }).join("");
}

function selectManeuverCard(item, el) {
    document.querySelectorAll(".maneuver-card").forEach(card => card.classList.remove("selected"));
    if (el) el.classList.add("selected");

    byId("selectedActionName").textContent = item.name || "none";
    byId("maneuverNameInput").value = item.name || "";
    selectedManeuver = item.name || null;

    // helpful autofill defaults by maneuver type
    if (item.name === "hohmann_transfer" && lastObservation && lastObservation.target_orbit && lastObservation.target_orbit.altitude_km != null) {
        byId("targetAltitudeInput").value = lastObservation.target_orbit.altitude_km;
    }
    if (item.name === "plane_change" && lastObservation && lastObservation.target_orbit && lastObservation.target_orbit.inclination_deg != null) {
        byId("targetInclinationInput").value = lastObservation.target_orbit.inclination_deg;
    }
    if (item.name === "lunar_orbit_insertion") {
        if (!byId("targetAltitudeInput").value) byId("targetAltitudeInput").value = 100;
    }
    if (item.name === "gravity_assist") {
        if (!byId("bodyInput").value) byId("bodyInput").value = "venus";
    }
    if (item.name === "correction_burn") {
        if (!byId("deltaVInput").value) byId("deltaVInput").value = 50;
    }

    logActivity("Selected maneuver: " + (item.name || "unknown"), "info");
}

function formatMetric(v, suffix="") {
    if (v === null || v === undefined || v === "") return "—";
    if (typeof v === "number") {
        const decimals = Math.abs(v) >= 100 ? 0 : Math.abs(v) >= 10 ? 2 : 3;
        return formatNumber(v, decimals) + suffix;
    }
    return String(v) + suffix;
}

function formatNumber(v, decimals=2) {
    const n = Number(v);
    if (!isFinite(n)) return "—";
    return n.toFixed(decimals);
}

function escapeHtml(str) {
    return str
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function escapeAttr(str) {
    return String(str)
        .replaceAll("&", "&amp;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
}

// CANVAS VISUALIZER
const canvas = document.getElementById("orbitCanvas");
const ctx = canvas.getContext("2d");
let orbitAnim = 0;
let starField = [];

function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * Math.min(window.devicePixelRatio || 1, 2);
    canvas.height = rect.height * Math.min(window.devicePixelRatio || 1, 2);
    ctx.setTransform(1,0,0,1,0,0);
    ctx.scale(Math.min(window.devicePixelRatio || 1, 2), Math.min(window.devicePixelRatio || 1, 2));
    initStars();
}
window.addEventListener("resize", resizeCanvas);

function initStars() {
    const rect = canvas.getBoundingClientRect();
    starField = Array.from({length: 80}, () => ({
        x: Math.random() * rect.width,
        y: Math.random() * rect.height,
        r: Math.random() * 1.6 + 0.2,
        a: Math.random() * 0.6 + 0.2
    }));
}

function drawOrbit() {
    const rect = canvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    if (!w || !h) return;

    ctx.clearRect(0,0,w,h);

    // background
    const bg = ctx.createLinearGradient(0,0,0,h);
    bg.addColorStop(0, "#030712");
    bg.addColorStop(1, "#020617");
    ctx.fillStyle = bg;
    ctx.fillRect(0,0,w,h);

    // stars
    for (const s of starField) {
        ctx.fillStyle = `rgba(255,255,255,${s.a})`;
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI*2);
        ctx.fill();
    }

    const cx = w / 2;
    const cy = h / 2;

    if (vizState.task === "lunar_orbit") {
        drawEarthMoonView(cx, cy, w, h);
    } else if (vizState.task === "asteroid_rendezvous") {
        drawDeepSpaceView(cx, cy, w, h);
    } else {
        drawEarthOrbitView(cx, cy, w, h);
    }

    orbitAnim += 0.008;
    requestAnimationFrame(drawOrbit);
}

function drawEarth(cx, cy, radius=28) {
    const g = ctx.createRadialGradient(cx-8, cy-8, 6, cx, cy, radius);
    g.addColorStop(0, "#38bdf8");
    g.addColorStop(1, "#1d4ed8");
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI*2);
    ctx.fill();

    ctx.strokeStyle = "rgba(56,189,248,0.45)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, radius+3, 0, Math.PI*2);
    ctx.stroke();
}

function drawEarthOrbitView(cx, cy, w, h) {
    drawEarth(cx, cy, 30);

    const currentR = scaleAltitude(vizState.current_altitude_km, 70, 180);
    const targetR = scaleAltitude(vizState.target_altitude_km, 70, 180);

    // target
    ctx.setLineDash([8,8]);
    ctx.strokeStyle = "rgba(168,85,247,0.7)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, targetR, 0, Math.PI*2);
    ctx.stroke();
    ctx.setLineDash([]);

    // current
    drawOrbitEllipse(cx, cy, currentR, Math.max(0.15, 1 - Math.min(vizState.current_eccentricity, 0.85)), "rgba(34,211,238,0.9)");

    // spacecraft
    const satX = cx + currentR * Math.cos(orbitAnim);
    const satY = cy + currentR * Math.sin(orbitAnim) * Math.max(0.15, 1 - Math.min(vizState.current_eccentricity, 0.85));
    ctx.fillStyle = "#ffffff";
    ctx.shadowBlur = 18;
    ctx.shadowColor = "rgba(34,211,238,0.8)";
    ctx.beginPath();
    ctx.arc(satX, satY, 4, 0, Math.PI*2);
    ctx.fill();
    ctx.shadowBlur = 0;

    labelOrbit(cx + targetR + 10, cy - 6, "target");
    labelOrbit(cx + currentR + 10, cy + 14, "current");
}

function drawEarthMoonView(cx, cy, w, h) {
    const earthX = cx - 90;
    const earthY = cy;
    const moonX = cx + 150;
    const moonY = cy - 40;

    drawEarth(earthX, earthY, 28);

    // moon
    const moonGrad = ctx.createRadialGradient(moonX-4, moonY-4, 4, moonX, moonY, 16);
    moonGrad.addColorStop(0, "#f8fafc");
    moonGrad.addColorStop(1, "#94a3b8");
    ctx.fillStyle = moonGrad;
    ctx.beginPath();
    ctx.arc(moonX, moonY, 16, 0, Math.PI*2);
    ctx.fill();

    const currentR = scaleAltitude(vizState.current_altitude_km, 45, 95);
    ctx.strokeStyle = "rgba(34,211,238,0.9)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(earthX, earthY, currentR, 0, Math.PI*2);
    ctx.stroke();

    ctx.setLineDash([7,7]);
    ctx.strokeStyle = "rgba(168,85,247,0.8)";
    ctx.beginPath();
    ctx.arc(moonX, moonY, 45, 0, Math.PI*2);
    ctx.stroke();
    ctx.setLineDash([]);

    // transfer arc
    ctx.strokeStyle = "rgba(250,204,21,0.7)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(earthX + currentR, earthY);
    ctx.quadraticCurveTo(cx + 20, cy - 140, moonX - 45, moonY);
    ctx.stroke();

    const t = (Math.sin(orbitAnim) + 1) / 2;
    const px = quadraticPoint(earthX + currentR, cx + 20, moonX - 45, t);
    const py = quadraticPoint(earthY, cy - 140, moonY, t);

    ctx.fillStyle = "#fff";
    ctx.shadowBlur = 18;
    ctx.shadowColor = "rgba(250,204,21,0.75)";
    ctx.beginPath();
    ctx.arc(px, py, 4, 0, Math.PI*2);
    ctx.fill();
    ctx.shadowBlur = 0;

    labelOrbit(moonX + 24, moonY - 2, "target lunar orbit");
    labelOrbit(earthX + currentR + 10, earthY + 12, "current orbit");
}

function drawDeepSpaceView(cx, cy, w, h) {
    // sun/earth-ish origin node
    const originX = cx - 180;
    const originY = cy + 40;
    const venusX = cx - 40;
    const venusY = cy - 70;
    const earthAssistX = cx + 70;
    const earthAssistY = cy + 10;
    const bennuX = cx + 200;
    const bennuY = cy - 80;

    // route lines
    drawNode(originX, originY, 18, "#38bdf8", "LEO");
    drawNode(venusX, venusY, 10, "#f59e0b", "Venus");
    drawNode(earthAssistX, earthAssistY, 12, "#22d3ee", "Earth");
    drawNode(bennuX, bennuY, 9, "#e2e8f0", "Bennu");

    ctx.strokeStyle = "rgba(168,85,247,0.55)";
    ctx.lineWidth = 2;
    ctx.setLineDash([8,8]);
    ctx.beginPath();
    ctx.moveTo(originX, originY);
    ctx.quadraticCurveTo(cx - 90, cy - 160, venusX, venusY);
    ctx.quadraticCurveTo(cx + 20, cy + 100, earthAssistX, earthAssistY);
    ctx.quadraticCurveTo(cx + 130, cy - 150, bennuX, bennuY);
    ctx.stroke();
    ctx.setLineDash([]);

    const t = (Math.sin(orbitAnim * 0.7) + 1) / 2;
    const segments = [
        {x1: originX, y1: originY, cx: cx - 90, cy: cy - 160, x2: venusX, y2: venusY},
        {x1: venusX, y1: venusY, cx: cx + 20, cy: cy + 100, x2: earthAssistX, y2: earthAssistY},
        {x1: earthAssistX, y1: earthAssistY, cx: cx + 130, cy: cy - 150, x2: bennuX, y2: bennuY}
    ];
    const seg = segments[Math.min(2, Math.floor(t * 3))];
    const localT = (t * 3) % 1;
    const px = quadraticPoint(seg.x1, seg.cx, seg.x2, localT);
    const py = quadraticPoint(seg.y1, seg.cy, seg.y2, localT);

    ctx.fillStyle = "#fff";
    ctx.shadowBlur = 18;
    ctx.shadowColor = "rgba(34,211,238,0.75)";
    ctx.beginPath();
    ctx.arc(px, py, 4, 0, Math.PI*2);
    ctx.fill();
    ctx.shadowBlur = 0;
}

function drawNode(x, y, r, color, label) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI*2);
    ctx.fill();
    ctx.fillStyle = "rgba(226,232,240,0.8)";
    ctx.font = "12px ui-monospace, monospace";
    ctx.fillText(label, x + r + 8, y + 4);
}

function drawOrbitEllipse(cx, cy, rx, ratio, stroke) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, rx * ratio, 0, 0, Math.PI*2);
    ctx.stroke();
}

function quadraticPoint(a, b, c, t) {
    return (1 - t) * (1 - t) * a + 2 * (1 - t) * t * b + t * t * c;
}

function scaleAltitude(alt, minR, maxR) {
    const a = Math.max(0, Number(alt || 0));
    const norm = Math.log10(a + 10) / Math.log10(400000 + 10);
    const clamped = Math.max(0.02, Math.min(1, norm));
    return minR + (maxR - minR) * clamped;
}

function labelOrbit(x, y, label) {
    ctx.fillStyle = "rgba(226,232,240,0.78)";
    ctx.font = "12px ui-monospace, monospace";
    ctx.fillText(label, x, y);
}

// init
selectMission("leo_satellite", document.querySelector('[data-task="leo_satellite"]'));
resizeCanvas();
initStars();
drawOrbit();
logActivity("Orbit environment console initialized. Connect to begin.", "info");
renderHistory();
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