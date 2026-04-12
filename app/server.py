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
html, body { margin:0; padding:0; min-height:100%; }
body {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color:#e5edf7;
    overflow-x:hidden;
    background:
        radial-gradient(circle at 15% 20%, rgba(0,212,255,0.12), transparent 28%),
        radial-gradient(circle at 85% 15%, rgba(123,47,247,0.15), transparent 30%),
        radial-gradient(circle at 50% 85%, rgba(255,255,255,0.04), transparent 18%),
        linear-gradient(180deg, #050814 0%, #07101f 50%, #050914 100%);
}

/* stars */
body::before{
    content:"";
    position:fixed;
    inset:0;
    pointer-events:none;
    background-image: radial-gradient(rgba(255,255,255,0.20) 1px, transparent 1px);
    background-size: 54px 54px;
    opacity:.16;
}

/* subtle grid */
body::after{
    content:"";
    position:fixed;
    inset:0;
    pointer-events:none;
    background:
        linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
    background-size: 42px 42px;
    mask-image: radial-gradient(circle at center, black 42%, transparent 94%);
    opacity:.16;
}

.mono {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
}

.panel {
    background: rgba(8, 14, 30, 0.72);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(148,163,184,0.12);
    box-shadow:
        0 10px 25px rgba(0,0,0,0.24),
        inset 0 1px 0 rgba(255,255,255,0.03);
}

.panel-soft {
    background: rgba(15,23,42,0.54);
    border: 1px solid rgba(148,163,184,0.10);
}

.glow-cyan {
    box-shadow: 0 0 0 1px rgba(34,211,238,0.10), 0 0 28px rgba(34,211,238,0.08);
}

.glow-violet {
    box-shadow: 0 0 0 1px rgba(168,85,247,0.10), 0 0 28px rgba(168,85,247,0.08);
}

.status-dot {
    width:10px;
    height:10px;
    border-radius:999px;
    display:inline-block;
}
.dot-red { background:#ef4444; box-shadow:0 0 12px rgba(239,68,68,.75); }
.dot-amber { background:#f59e0b; box-shadow:0 0 12px rgba(245,158,11,.75); }
.dot-green { background:#22c55e; box-shadow:0 0 12px rgba(34,197,94,.75); }

.console-line {
    animation: fadeInLine .22s ease;
}
@keyframes fadeInLine {
    from { opacity:0; transform:translateY(6px); }
    to { opacity:1; transform:translateY(0); }
}

.boot-screen {
    position:fixed;
    inset:0;
    z-index:50;
    background:
        radial-gradient(circle at center, rgba(0,212,255,0.06), transparent 35%),
        linear-gradient(180deg, #04070f 0%, #060b16 100%);
    display:flex;
    align-items:center;
    justify-content:center;
    transition:opacity .7s ease, visibility .7s ease;
}
.boot-screen.hidden {
    opacity:0;
    visibility:hidden;
}
.boot-inner {
    width:min(700px, 90vw);
    text-align:center;
}
.boot-logo {
    font-size: clamp(2.8rem, 8vw, 5rem);
    font-weight: 800;
    letter-spacing: .28em;
    color: transparent;
    background: linear-gradient(90deg, #22d3ee, #60a5fa, #a855f7);
    -webkit-background-clip:text;
    background-clip:text;
}
.boot-sub {
    margin-top: 14px;
    color:#94a3b8;
    letter-spacing:.18em;
    font-size:.92rem;
}
.boot-line {
    margin-top: 26px;
    color:#cbd5e1;
    font-size:.95rem;
}
.boot-progress-wrap {
    margin-top: 24px;
    height: 10px;
    border-radius: 999px;
    background: rgba(15,23,42,.85);
    border:1px solid rgba(148,163,184,.14);
    overflow:hidden;
}
.boot-progress {
    height:100%;
    width:0%;
    background: linear-gradient(90deg, #22d3ee, #3b82f6, #a855f7);
    box-shadow: 0 0 20px rgba(59,130,246,.35);
    transition: width .25s ease;
}
.boot-button {
    margin-top: 22px;
    display:inline-flex;
    align-items:center;
    gap:10px;
    padding: 12px 18px;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(34,211,238,.18), rgba(168,85,247,.18));
    border:1px solid rgba(96,165,250,.25);
    color:#f8fafc;
    font-weight:600;
    cursor:pointer;
    transition: transform .2s ease, box-shadow .2s ease;
}
.boot-button:hover {
    transform: translateY(-1px);
    box-shadow: 0 0 22px rgba(34,211,238,.14);
}

.main-shell {
    opacity:0;
    transform: translateY(10px);
    transition: opacity .8s ease, transform .8s ease;
}
.main-shell.ready {
    opacity:1;
    transform: translateY(0);
}

.reveal {
    opacity:0;
    transform: translateY(18px);
}
.reveal.show {
    opacity:1;
    transform: translateY(0);
    transition: opacity .7s ease, transform .7s ease;
}

.btn-primary {
    background: linear-gradient(135deg, #22d3ee, #3b82f6 50%, #8b5cf6);
    color:white;
    border:none;
}
.btn-primary:hover { filter:brightness(1.05); box-shadow:0 0 24px rgba(59,130,246,.22); }

.btn-dark {
    background: rgba(15,23,42,.75);
    border:1px solid rgba(148,163,184,.15);
}
.btn-dark:hover {
    background: rgba(30,41,59,.85);
    border-color: rgba(148,163,184,.22);
}

.btn-amber {
    background: linear-gradient(135deg, rgba(245,158,11,.22), rgba(249,115,22,.22));
    border:1px solid rgba(251,191,36,.22);
}
.btn-amber:hover {
    box-shadow:0 0 20px rgba(245,158,11,.12);
}

.scrollbar-thin::-webkit-scrollbar { width:8px; height:8px; }
.scrollbar-thin::-webkit-scrollbar-thumb {
    background: rgba(100,116,139,.42);
    border-radius:999px;
}
.scrollbar-thin::-webkit-scrollbar-track { background:transparent; }

.mission-tile {
    transition: all .22s ease;
}
.mission-tile:hover {
    transform: translateY(-2px);
    border-color: rgba(96,165,250,.28);
}
.mission-tile.active {
    border-color: rgba(34,211,238,.5);
    box-shadow: 0 0 0 1px rgba(34,211,238,.18), 0 0 26px rgba(34,211,238,.08);
}

.telemetry-card {
    transition: transform .25s ease, border-color .25s ease;
}
.telemetry-card:hover {
    transform: translateY(-2px);
    border-color: rgba(148,163,184,.22);
}
</style>
</head>
<body>

<!-- BOOT / INTRO -->
<div id="bootScreen" class="boot-screen">
    <div class="boot-inner">
        <div class="boot-logo">ORBIT</div>
        <div class="boot-sub mono">AI SPACE MISSION ARCHITECT</div>
        <div id="bootLine" class="boot-line mono">Initializing mission control interface...</div>
        <div class="boot-progress-wrap">
            <div id="bootProgress" class="boot-progress"></div>
        </div>
        <button id="enterBtn" class="boot-button hidden">
            <span>Enter Mission Control</span>
            <span>→</span>
        </button>
    </div>
</div>

<div id="mainShell" class="main-shell max-w-[1800px] mx-auto px-4 py-4 pb-8">
    <!-- TOPBAR -->
    <header class="panel rounded-2xl px-5 py-4 reveal mb-4">
        <div class="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div class="flex items-start gap-4">
                <div class="h-12 w-12 rounded-2xl bg-gradient-to-br from-cyan-400 via-blue-500 to-violet-500 flex items-center justify-center text-2xl shadow-lg shadow-cyan-500/20">🚀</div>
                <div>
                    <div class="flex items-center gap-3 flex-wrap">
                        <h1 class="text-2xl font-semibold tracking-[0.18em]">ORBIT</h1>
                        <span class="px-2.5 py-1 rounded-full text-xs border border-cyan-400/20 bg-cyan-400/10 text-cyan-300">OpenEnv RL Environment</span>
                        <span class="px-2.5 py-1 rounded-full text-xs border border-violet-400/20 bg-violet-400/10 text-violet-300">Deterministic Physics</span>
                    </div>
                    <p class="text-sm text-slate-400 mt-1">Observe state · Execute actions · Inspect transitions · Plan fuel-efficient missions</p>
                </div>
            </div>

            <div class="flex flex-wrap items-center gap-3">
                <div class="flex items-center gap-2 px-3 py-2 rounded-xl panel-soft">
                    <span id="connectionDot" class="status-dot dot-red"></span>
                    <span id="connectionText" class="text-sm text-slate-300">Disconnected</span>
                </div>
                <div class="px-3 py-2 rounded-xl panel-soft text-sm text-slate-300">
                    <span class="text-slate-500">Episode:</span>
                    <span id="episodeText" class="ml-1">Idle</span>
                </div>
                <div class="px-3 py-2 rounded-xl panel-soft text-sm text-slate-300">
                    <span class="text-slate-500">Task:</span>
                    <span id="activeTaskText" class="ml-1 mono">leo_satellite</span>
                </div>
            </div>
        </div>
    </header>

    <!-- MAIN CONTENT -->
    <div class="grid grid-cols-1 xl:grid-cols-[320px_minmax(0,1fr)_360px] gap-4 items-start">

        <!-- LEFT COLUMN -->
        <div class="flex flex-col gap-4">
            <section class="panel rounded-2xl p-4 reveal">
                <div class="flex items-center justify-between mb-3">
                    <h2 class="text-sm font-semibold tracking-[0.16em] text-slate-200">MISSION SET</h2>
                    <span class="text-xs text-slate-500 mono">3 tasks</span>
                </div>

                <div class="space-y-3">
                    <button class="mission-tile active w-full text-left rounded-2xl panel-soft border border-cyan-400/30 p-4" data-task="leo_satellite" onclick="selectTask('leo_satellite', this)">
                        <div class="flex items-center justify-between">
                            <div class="font-medium">LEO Satellite Deployment</div>
                            <span class="text-xs px-2 py-1 rounded-lg bg-emerald-500/15 text-emerald-300 border border-emerald-500/20">Easy</span>
                        </div>
                        <div class="grid grid-cols-3 gap-2 mt-3 text-xs text-slate-400">
                            <div><div class="text-slate-500">Optimal</div><div class="mono text-slate-200">9200</div></div>
                            <div><div class="text-slate-500">Budget</div><div class="mono text-slate-200">12000</div></div>
                            <div><div class="text-slate-500">Steps</div><div class="mono text-slate-200">10</div></div>
                        </div>
                    </button>

                    <button class="mission-tile w-full text-left rounded-2xl panel-soft border border-slate-700 p-4" data-task="lunar_orbit" onclick="selectTask('lunar_orbit', this)">
                        <div class="flex items-center justify-between">
                            <div class="font-medium">Lunar Orbit Insertion</div>
                            <span class="text-xs px-2 py-1 rounded-lg bg-amber-500/15 text-amber-300 border border-amber-500/20">Medium</span>
                        </div>
                        <div class="grid grid-cols-3 gap-2 mt-3 text-xs text-slate-400">
                            <div><div class="text-slate-500">Optimal</div><div class="mono text-slate-200">3900</div></div>
                            <div><div class="text-slate-500">Budget</div><div class="mono text-slate-200">5000</div></div>
                            <div><div class="text-slate-500">Steps</div><div class="mono text-slate-200">15</div></div>
                        </div>
                    </button>

                    <button class="mission-tile w-full text-left rounded-2xl panel-soft border border-slate-700 p-4" data-task="asteroid_rendezvous" onclick="selectTask('asteroid_rendezvous', this)">
                        <div class="flex items-center justify-between">
                            <div class="font-medium">Asteroid Bennu Rendezvous</div>
                            <span class="text-xs px-2 py-1 rounded-lg bg-rose-500/15 text-rose-300 border border-rose-500/20">Hard</span>
                        </div>
                        <div class="grid grid-cols-3 gap-2 mt-3 text-xs text-slate-400">
                            <div><div class="text-slate-500">Optimal</div><div class="mono text-slate-200">5800</div></div>
                            <div><div class="text-slate-500">Budget</div><div class="mono text-slate-200">8000</div></div>
                            <div><div class="text-slate-500">Steps</div><div class="mono text-slate-200">25</div></div>
                        </div>
                    </button>
                </div>
            </section>

            <!-- MOVED CONTROLS HIGHER -->
            <section class="panel rounded-2xl p-4 reveal glow-cyan xl:sticky xl:top-4">
                <div class="flex items-center justify-between mb-3">
                    <h2 class="text-sm font-semibold tracking-[0.16em] text-slate-200">CONTROL ACTIONS</h2>
                    <span class="text-xs text-slate-500 mono">safe mode</span>
                </div>

                <div class="grid grid-cols-1 gap-3">
                    <button class="h-12 rounded-xl btn-dark text-sm transition" onclick="connectWS()">Connect</button>
                    <button class="h-12 rounded-xl btn-primary text-sm font-medium transition" onclick="resetMission()">Reset Episode</button>
                    <button class="h-12 rounded-xl btn-amber text-sm transition" onclick="stepMission()">Execute Step</button>
                </div>

                <div class="mt-4 panel-soft rounded-xl p-3">
                    <div class="text-xs text-slate-500 mb-2">Selected Task ID</div>
                    <input id="taskInput" class="w-full bg-slate-950/80 border border-slate-700 rounded-xl px-3 py-2 text-sm text-slate-200 outline-none mono" value="leo_satellite" placeholder="task_id">
                </div>

                <div class="mt-4 grid grid-cols-3 gap-2 text-center text-xs">
                    <div class="panel-soft rounded-xl py-3 px-2">
                        <div class="text-slate-500">Connect</div>
                        <div class="mt-1 text-slate-200">WS</div>
                    </div>
                    <div class="panel-soft rounded-xl py-3 px-2">
                        <div class="text-slate-500">Reset</div>
                        <div class="mt-1 text-slate-200">Episode</div>
                    </div>
                    <div class="panel-soft rounded-xl py-3 px-2">
                        <div class="text-slate-500">Step</div>
                        <div class="mt-1 text-slate-200">Action</div>
                    </div>
                </div>
            </section>

            <section class="panel rounded-2xl p-4 reveal">
                <div class="flex items-center justify-between mb-3">
                    <h2 class="text-sm font-semibold tracking-[0.16em] text-slate-200">MISSION BRIEFING</h2>
                    <span id="taskBadge" class="text-xs px-2 py-1 rounded-lg bg-cyan-400/10 text-cyan-300 border border-cyan-400/20 mono">leo_satellite</span>
                </div>

                <div id="briefTitle" class="text-lg font-semibold">LEO Satellite Deployment</div>
                <p id="briefSummary" class="text-sm text-slate-400 mt-2">
                    Launch to a 400 km circular orbit matching ISS inclination. Choose the correct strategic transfer and submit efficiently.
                </p>

                <div class="grid grid-cols-2 gap-3 mt-4 text-sm">
                    <div class="panel-soft rounded-xl p-3">
                        <div class="text-xs text-slate-500">Reference</div>
                        <div id="briefReference" class="mt-1">Falcon 9 → ISS</div>
                    </div>
                    <div class="panel-soft rounded-xl p-3">
                        <div class="text-xs text-slate-500">Suggested Plan</div>
                        <div id="briefPlan" class="mt-1">Hohmann transfer → submit</div>
                    </div>
                </div>
            </section>
        </div>

        <!-- CENTER COLUMN -->
        <div class="flex flex-col gap-4">
            <section class="panel rounded-2xl p-4 reveal glow-cyan">
                <div class="flex items-center justify-between mb-3">
                    <div>
                        <h2 class="text-sm font-semibold tracking-[0.16em] text-slate-200">ORBIT VISUALIZATION</h2>
                        <p class="text-xs text-slate-500 mt-1">Live visual layer for the current environment session</p>
                    </div>
                    <div class="text-xs text-slate-500 mono">mission-control view</div>
                </div>
                <div class="rounded-2xl overflow-hidden border border-slate-800 bg-black/35 h-[340px] md:h-[420px] xl:h-[460px]">
                    <canvas id="orbitCanvas" class="w-full h-full"></canvas>
                </div>
            </section>

            <section class="grid grid-cols-1 sm:grid-cols-2 2xl:grid-cols-4 gap-4 reveal">
                <div class="telemetry-card panel rounded-2xl p-4">
                    <div class="text-xs tracking-[0.16em] text-slate-500">CURRENT TASK</div>
                    <div id="metricTask" class="mt-2 text-lg font-semibold">leo_satellite</div>
                    <div class="text-xs text-slate-500 mt-1">Active selected mission</div>
                </div>
                <div class="telemetry-card panel rounded-2xl p-4">
                    <div class="text-xs tracking-[0.16em] text-slate-500">SESSION STATUS</div>
                    <div id="metricStatus" class="mt-2 text-lg font-semibold">Awaiting Connection</div>
                    <div class="text-xs text-slate-500 mt-1">WebSocket-driven interaction state</div>
                </div>
                <div class="telemetry-card panel rounded-2xl p-4">
                    <div class="text-xs tracking-[0.16em] text-slate-500">LAST ACTION</div>
                    <div id="metricAction" class="mt-2 text-lg font-semibold">None</div>
                    <div class="text-xs text-slate-500 mt-1">Latest control operation sent</div>
                </div>
                <div class="telemetry-card panel rounded-2xl p-4">
                    <div class="text-xs tracking-[0.16em] text-slate-500">MESSAGE FLOW</div>
                    <div id="metricMessages" class="mt-2 text-lg font-semibold mono">0</div>
                    <div class="text-xs text-slate-500 mt-1">Incoming WebSocket frames</div>
                </div>
            </section>
        </div>

        <!-- RIGHT COLUMN -->
        <div class="flex flex-col gap-4">
            <section class="panel rounded-2xl p-4 reveal glow-violet">
                <div class="flex items-center justify-between mb-3">
                    <h2 class="text-sm font-semibold tracking-[0.16em] text-slate-200">ENVIRONMENT LOOP</h2>
                    <span class="text-xs text-slate-500 mono">observe → act → transition</span>
                </div>

                <div class="space-y-3 text-sm">
                    <div class="panel-soft rounded-xl p-3">
                        <div class="text-xs text-slate-500">1. Connect</div>
                        <div class="mt-1 text-slate-300">Open a WebSocket session with the environment.</div>
                    </div>
                    <div class="panel-soft rounded-xl p-3">
                        <div class="text-xs text-slate-500">2. Reset</div>
                        <div class="mt-1 text-slate-300">Initialize the selected mission as a fresh episode.</div>
                    </div>
                    <div class="panel-soft rounded-xl p-3">
                        <div class="text-xs text-slate-500">3. Step</div>
                        <div class="mt-1 text-slate-300">Execute the current test maneuver and inspect returned transition data.</div>
                    </div>
                </div>
            </section>

            <section class="panel rounded-2xl p-4 reveal">
                <div class="flex items-center justify-between mb-3">
                    <h2 class="text-sm font-semibold tracking-[0.16em] text-slate-200">LIVE METRICS</h2>
                    <span class="text-xs text-slate-500 mono">safe placeholders</span>
                </div>

                <div class="space-y-3">
                    <div class="panel-soft rounded-xl p-3">
                        <div class="flex items-center justify-between">
                            <span class="text-xs text-slate-500">Socket State</span>
                            <span id="socketStatePill" class="text-xs mono text-rose-300">disconnected</span>
                        </div>
                    </div>
                    <div class="panel-soft rounded-xl p-3">
                        <div class="flex items-center justify-between">
                            <span class="text-xs text-slate-500">Episode State</span>
                            <span id="episodeStatePill" class="text-xs mono text-slate-300">idle</span>
                        </div>
                    </div>
                    <div class="panel-soft rounded-xl p-3">
                        <div class="flex items-center justify-between">
                            <span class="text-xs text-slate-500">Last Reward</span>
                            <span id="rewardPill" class="text-xs mono text-slate-300">—</span>
                        </div>
                    </div>
                    <div class="panel-soft rounded-xl p-3">
                        <div class="flex items-center justify-between">
                            <span class="text-xs text-slate-500">Done Flag</span>
                            <span id="donePill" class="text-xs mono text-slate-300">false</span>
                        </div>
                    </div>
                </div>
            </section>

            <section class="panel rounded-2xl p-4 reveal">
                <div class="flex items-center justify-between mb-3">
                    <h2 class="text-sm font-semibold tracking-[0.16em] text-slate-200">MISSION NOTES</h2>
                    <span class="text-xs text-slate-500 mono">brief</span>
                </div>
                <div id="notesBox" class="text-sm text-slate-400 leading-6">
                    LEO mission is the recommended first episode. Connect, reset the mission, then execute a step to inspect the transition response in the console feed below.
                </div>
            </section>
        </div>
    </div>

    <!-- CONSOLE -->
    <section class="panel rounded-2xl p-4 reveal mt-4">
        <div class="flex items-center justify-between mb-3">
            <div>
                <h2 class="text-sm font-semibold tracking-[0.16em] text-slate-200">TELEMETRY CONSOLE</h2>
                <p class="text-xs text-slate-500 mt-1">Raw WebSocket event stream and protocol traffic</p>
            </div>
            <button onclick="clearConsole()" class="px-3 py-2 rounded-xl btn-dark text-xs transition">Clear Console</button>
        </div>

        <div id="console" class="h-[220px] overflow-auto scrollbar-thin rounded-2xl bg-black/40 border border-slate-800 p-4 mono text-[13px] leading-6 text-slate-300"></div>
    </section>
</div>

<script>
/* -------------------------------------------------------------------------- */
/* BOOT / INTRO */
/* -------------------------------------------------------------------------- */
const bootMessages = [
    "Initializing mission control interface...",
    "Calibrating visual telemetry layers...",
    "Preparing orbital console...",
    "Routing WebSocket control channel...",
    "Mission control ready."
];

const bootScreen = document.getElementById("bootScreen");
const bootProgress = document.getElementById("bootProgress");
const bootLine = document.getElementById("bootLine");
const enterBtn = document.getElementById("enterBtn");
const mainShell = document.getElementById("mainShell");

let bootIndex = 0;
let bootPercent = 0;

function runBootSequence() {
    const timer = setInterval(() => {
        bootPercent += 20;
        bootProgress.style.width = bootPercent + "%";
        bootLine.textContent = bootMessages[Math.min(bootIndex, bootMessages.length - 1)];
        bootIndex++;

        if (bootPercent >= 100) {
            clearInterval(timer);
            enterBtn.classList.remove("hidden");
        }
    }, 420);
}

function enterMissionControl() {
    bootScreen.classList.add("hidden");
    mainShell.classList.add("ready");

    const reveals = document.querySelectorAll(".reveal");
    reveals.forEach((el, i) => {
        setTimeout(() => el.classList.add("show"), 120 + i * 80);
    });
}

enterBtn.addEventListener("click", enterMissionControl);
runBootSequence();

/* -------------------------------------------------------------------------- */
/* MISSION META */
/* -------------------------------------------------------------------------- */
const missionMeta = {
    leo_satellite: {
        title: "LEO Satellite Deployment",
        summary: "Launch to a 400 km circular orbit matching ISS inclination. Choose the correct strategic transfer and submit efficiently.",
        reference: "Falcon 9 → ISS",
        plan: "Hohmann transfer → submit",
        notes: "Recommended starting task. Reset the episode, then execute a step to inspect the environment transition."
    },
    lunar_orbit: {
        title: "Lunar Orbit Insertion",
        summary: "Transfer from Earth parking orbit to lunar orbit using correct sequencing and careful fuel management.",
        reference: "Apollo / Artemis",
        plan: "TLI → LOI → optional circularize",
        notes: "Multi-burn mission. Use this after verifying the environment loop on the LEO task."
    },
    asteroid_rendezvous: {
        title: "Asteroid Bennu Rendezvous",
        summary: "Deep-space rendezvous scenario requiring gravity-assist aware planning and more strategic sequencing.",
        reference: "OSIRIS-REx",
        plan: "Gravity assist → gravity assist → combined transfer → corrections",
        notes: "Most complex mission. Best for testing longer strategic trajectories and richer multi-step interaction."
    }
};

let selectedTask = "leo_satellite";
let ws = null;
let angle = 0;
let messageCount = 0;

/* -------------------------------------------------------------------------- */
/* HELPERS */
/* -------------------------------------------------------------------------- */
function setConnection(state) {
    const dot = document.getElementById("connectionDot");
    const text = document.getElementById("connectionText");
    const pill = document.getElementById("socketStatePill");

    dot.className = "status-dot";
    if (state === "connecting") {
        dot.classList.add("dot-amber");
        text.textContent = "Connecting";
        pill.textContent = "connecting";
        pill.className = "text-xs mono text-amber-300";
        document.getElementById("metricStatus").textContent = "Connecting";
    } else if (state === "connected") {
        dot.classList.add("dot-green");
        text.textContent = "Connected";
        pill.textContent = "connected";
        pill.className = "text-xs mono text-emerald-300";
        document.getElementById("metricStatus").textContent = "Connected";
    } else {
        dot.classList.add("dot-red");
        text.textContent = "Disconnected";
        pill.textContent = "disconnected";
        pill.className = "text-xs mono text-rose-300";
        document.getElementById("metricStatus").textContent = "Awaiting Connection";
    }
}

function setEpisodeState(state) {
    document.getElementById("episodeText").textContent = state;
    document.getElementById("episodeStatePill").textContent = state.toLowerCase();
}

function log(msg, type="default") {
    const consoleEl = document.getElementById("console");
    const time = new Date().toLocaleTimeString();
    let color = "text-slate-300";
    if (type === "success") color = "text-emerald-300";
    if (type === "error") color = "text-rose-300";
    if (type === "warn") color = "text-amber-300";
    if (type === "info") color = "text-cyan-300";

    const line = document.createElement("div");
    line.className = "console-line " + color;
    line.innerHTML = `<span class="text-slate-500">[${time}]</span> ${escapeHtml(msg)}`;
    consoleEl.appendChild(line);
    consoleEl.scrollTop = consoleEl.scrollHeight;
}

function clearConsole() {
    document.getElementById("console").innerHTML = "";
    log("Console cleared.", "info");
}

function escapeHtml(str) {
    return String(str)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function ensureSocket() {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        log("WebSocket not connected. Click Connect first.", "error");
        return false;
    }
    return true;
}

/* -------------------------------------------------------------------------- */
/* TASK SELECTION */
/* -------------------------------------------------------------------------- */
function selectTask(taskId, el) {
    selectedTask = taskId;
    document.getElementById("taskInput").value = taskId;
    document.getElementById("activeTaskText").textContent = taskId;
    document.getElementById("metricTask").textContent = taskId;
    document.getElementById("taskBadge").textContent = taskId;

    document.querySelectorAll(".mission-tile").forEach(card => {
        card.classList.remove("active");
        card.classList.remove("border-cyan-400/30");
        card.classList.add("border-slate-700");
    });

    if (el) {
        el.classList.add("active");
        el.classList.remove("border-slate-700");
    }

    const meta = missionMeta[taskId];
    if (meta) {
        document.getElementById("briefTitle").textContent = meta.title;
        document.getElementById("briefSummary").textContent = meta.summary;
        document.getElementById("briefReference").textContent = meta.reference;
        document.getElementById("briefPlan").textContent = meta.plan;
        document.getElementById("notesBox").textContent = meta.notes;
    }

    log("Task selected: " + taskId, "info");
}

/* -------------------------------------------------------------------------- */
/* WEBSOCKET ACTIONS */
/* -------------------------------------------------------------------------- */
function connectWS() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        log("Already connected to environment.", "info");
        return;
    }

    setConnection("connecting");
    log("Opening WebSocket connection...", "info");

    const protocol = location.protocol === "https:" ? "wss://" : "ws://";
    ws = new WebSocket(protocol + location.host + "/ws");

    ws.onopen = () => {
        setConnection("connected");
        setEpisodeState("Ready");
        log("Connected to Orbit WebSocket endpoint.", "success");
    };

    ws.onclose = () => {
        setConnection("disconnected");
        setEpisodeState("Idle");
        log("WebSocket disconnected.", "warn");
    };

    ws.onerror = () => {
        setConnection("disconnected");
        setEpisodeState("Error");
        log("WebSocket error occurred.", "error");
    };

    ws.onmessage = (e) => {
        messageCount += 1;
        document.getElementById("metricMessages").textContent = String(messageCount);

        log("Received: " + e.data, "info");

        try {
            const msg = JSON.parse(e.data);

            if (msg.type === "welcome") {
                setEpisodeState("Ready");
                document.getElementById("metricAction").textContent = "welcome";
                return;
            }

            if (msg.type === "observation") {
                setEpisodeState("Active");
                document.getElementById("metricAction").textContent = "reset";
                document.getElementById("rewardPill").textContent = "—";
                document.getElementById("donePill").textContent = "false";
                return;
            }

            if (msg.type === "step_result") {
                setEpisodeState(msg.done ? "Done" : "Active");
                document.getElementById("metricAction").textContent = "step";
                document.getElementById("rewardPill").textContent =
                    msg.reward !== undefined ? String(msg.reward) : "—";
                document.getElementById("donePill").textContent =
                    msg.done !== undefined ? String(msg.done) : "false";
                return;
            }

            if (msg.type === "state") {
                document.getElementById("metricAction").textContent = "state";
                return;
            }

            if (msg.type === "task_list") {
                document.getElementById("metricAction").textContent = "list_tasks";
                return;
            }

            if (msg.type === "error") {
                document.getElementById("metricAction").textContent = "error";
                log(msg.message || "Server returned error.", "error");
                return;
            }
        } catch (err) {
            log("Message parsed as raw text.", "warn");
        }
    };
}

function resetMission() {
    if (!ensureSocket()) return;

    const task = document.getElementById("taskInput").value || selectedTask || "leo_satellite";
    selectedTask = task;
    document.getElementById("activeTaskText").textContent = task;
    document.getElementById("metricTask").textContent = task;

    ws.send(JSON.stringify({ type: "reset", task_id: task }));
    document.getElementById("metricAction").textContent = "reset";
    setEpisodeState("Resetting");
    log("Sent reset for task_id=" + task, "success");
}

function stepMission() {
    if (!ensureSocket()) return;

    ws.send(JSON.stringify({
        type: "step",
        action: {
            type: "execute_maneuver",
            maneuver: "hohmann_transfer",
            target_altitude_km: 400
        }
    }));

    document.getElementById("metricAction").textContent = "execute_maneuver";
    log("Sent step action: execute_maneuver / hohmann_transfer / target_altitude_km=400", "warn");
}

/* -------------------------------------------------------------------------- */
/* CANVAS VISUALIZER */
/* -------------------------------------------------------------------------- */
const canvas = document.getElementById("orbitCanvas");
const ctx = canvas.getContext("2d");
let stars = [];

function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);
    initStars();
}
window.addEventListener("resize", resizeCanvas);

function initStars() {
    const rect = canvas.getBoundingClientRect();
    stars = Array.from({ length: 90 }, () => ({
        x: Math.random() * rect.width,
        y: Math.random() * rect.height,
        r: Math.random() * 1.5 + 0.3,
        a: Math.random() * 0.6 + 0.15
    }));
}

function drawOrbit() {
    const rect = canvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    if (!w || !h) {
        requestAnimationFrame(drawOrbit);
        return;
    }

    ctx.clearRect(0, 0, w, h);

    const bg = ctx.createLinearGradient(0, 0, 0, h);
    bg.addColorStop(0, "#020617");
    bg.addColorStop(1, "#030712");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);

    for (const s of stars) {
        ctx.fillStyle = `rgba(255,255,255,${s.a})`;
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fill();
    }

    const cx = w / 2;
    const cy = h / 2;

    ctx.strokeStyle = "rgba(148,163,184,0.10)";
    ctx.lineWidth = 1;
    [70, 120, 170, 220].forEach(r => {
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.stroke();
    });

    ctx.strokeStyle = "rgba(148,163,184,0.08)";
    ctx.beginPath();
    ctx.moveTo(cx - 260, cy);
    ctx.lineTo(cx + 260, cy);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(cx, cy - 180);
    ctx.lineTo(cx, cy + 180);
    ctx.stroke();

    const earthGlow = ctx.createRadialGradient(cx, cy, 12, cx, cy, 48);
    earthGlow.addColorStop(0, "rgba(34,211,238,0.85)");
    earthGlow.addColorStop(1, "rgba(34,211,238,0)");
    ctx.fillStyle = earthGlow;
    ctx.beginPath();
    ctx.arc(cx, cy, 48, 0, Math.PI * 2);
    ctx.fill();

    const earthCore = ctx.createRadialGradient(cx - 8, cy - 8, 6, cx, cy, 26);
    earthCore.addColorStop(0, "#67e8f9");
    earthCore.addColorStop(1, "#2563eb");
    ctx.fillStyle = earthCore;
    ctx.beginPath();
    ctx.arc(cx, cy, 24, 0, Math.PI * 2);
    ctx.fill();

    ctx.setLineDash([8, 8]);
    ctx.strokeStyle = "rgba(168,85,247,0.75)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, 150, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.strokeStyle = "rgba(34,211,238,0.85)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, 105, 0, Math.PI * 2);
    ctx.stroke();

    const sx = cx + Math.cos(angle) * 105;
    const sy = cy + Math.sin(angle) * 105;

    ctx.shadowBlur = 18;
    ctx.shadowColor = "rgba(255,255,255,.8)";
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(sx, sy, 4.2, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;

    ctx.fillStyle = "rgba(226,232,240,0.75)";
    ctx.font = "12px ui-monospace, monospace";
    ctx.fillText("current orbit", cx + 112, cy + 10);
    ctx.fillText("target orbit", cx + 158, cy - 8);
    ctx.fillText("Earth", cx - 14, cy + 42);

    angle += 0.008;
    requestAnimationFrame(drawOrbit);
}

resizeCanvas();
initStars();
drawOrbit();

/* -------------------------------------------------------------------------- */
/* INIT */
/* -------------------------------------------------------------------------- */
setConnection("disconnected");
setEpisodeState("Idle");
log("Interface initialized. Complete boot sequence, then connect to the environment.", "info");
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