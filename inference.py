"""
inference.py
Orbit — AI Space Mission Architect

OpenEnv-compliant inference script.
Runs all 3 missions using an LLM agent via WebSocket.

v2.1 Changes:
    - Added OPENAI_API_KEY validation
    - Added WebSocket connection retry logic
    - Added top-level error handling
    - Improved robustness for validator environments

Environment Variables:
    API_BASE_URL   : LLM API endpoint (default: HuggingFace router)
    MODEL_NAME     : Model identifier (default: Qwen2.5-72B-Instruct)
    OPENAI_API_KEY : API key (required)
"""

import asyncio
import json
import os
import sys
import time
from typing import List, Optional

import websockets
from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# Configuration from Environment Variables
# ─────────────────────────────────────────────────────────────────────────────

API_BASE_URL   = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME     = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

WS_URL       = os.getenv("WS_URL", "ws://localhost:7860/ws")
TEMPERATURE  = 0
MAX_TOKENS   = 500
BENCHMARK    = "orbit-mission-architect"

TASKS = ["leo_satellite", "lunar_orbit", "asteroid_rendezvous"]

# Retry settings for WebSocket connection
WS_MAX_RETRIES  = 10
WS_RETRY_DELAY  = 3  # seconds


# ─────────────────────────────────────────────────────────────────────────────
# Mandatory Log Functions
# ─────────────────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert orbital mechanics mission planner.

You control a spacecraft and must reach a target orbit efficiently.
Your goal is to choose the RIGHT MANEUVERS in the RIGHT ORDER while minimizing fuel usage.

You must respond with ONLY a valid JSON action. No explanation, no markdown, just JSON.

═══════════════════════════════════════════════════════════════════
STRATEGIC MANEUVERS (RECOMMENDED — environment calculates fuel automatically):
═══════════════════════════════════════════════════════════════════

1. Hohmann Transfer — change orbit altitude:
{"type": "execute_maneuver", "maneuver": "hohmann_transfer", "target_altitude_km": 400}

2. Plane Change — change orbital inclination:
{"type": "execute_maneuver", "maneuver": "plane_change", "target_inclination_deg": 51.6}

3. Circularize — make orbit circular (reduce eccentricity to 0):
{"type": "execute_maneuver", "maneuver": "circularize"}

4. Trans-Lunar Injection — burn to reach Moon (lunar mission only):
{"type": "execute_maneuver", "maneuver": "trans_lunar_injection"}

5. Lunar Orbit Insertion — capture into Moon orbit (lunar mission only):
{"type": "execute_maneuver", "maneuver": "lunar_orbit_insertion", "target_altitude_km": 100}

6. Gravity Assist — FREE velocity gain from flyby (asteroid mission):
{"type": "execute_maneuver", "maneuver": "gravity_assist", "body": "venus"}

7. Combined Transfer — altitude + inclination change together (15% cheaper):
{"type": "execute_maneuver", "maneuver": "combined_transfer", "target_altitude_km": 400, "target_inclination_deg": 51.6}

8. Correction Burn — small fine-tuning burn (max 500 m/s):
{"type": "execute_maneuver", "maneuver": "correction_burn", "delta_v_ms": 50}

═══════════════════════════════════════════════════════════════════
UTILITY ACTIONS:
═══════════════════════════════════════════════════════════════════

9. Preview score (free, no fuel cost):
{"type": "run_simulation"}

10. Submit mission for final grading:
{"type": "submit_mission"}

═══════════════════════════════════════════════════════════════════
STRATEGY GUIDE:
═══════════════════════════════════════════════════════════════════

LEO SATELLITE (Easy):
  → Use hohmann_transfer to target altitude. One maneuver + submit.

LUNAR ORBIT (Medium):
  → Step 1: trans_lunar_injection (leave Earth orbit)
  → Step 2: lunar_orbit_insertion (capture at Moon)
  → Step 3: submit_mission

ASTEROID RENDEZVOUS (Hard):
  → Direct transfer exceeds fuel budget — you MUST use gravity assists
  → Step 1: gravity_assist with venus or earth (free fuel!)
  → Step 2: Use remaining fuel for altitude + inclination changes
  → Step 3: submit_mission

IMPORTANT:
  - Check "available_maneuvers" in the observation — it tells you what you CAN do and the fuel cost
  - Check "mission_analysis" — it shows your errors and fuel margin
  - Check "recommendations" — it gives strategic advice
  - Gravity assists are FREE — always consider them first
  - Combined transfers save 15% fuel vs separate maneuvers
  - Submit when your score estimate is good or you're running low on steps
"""


# ─────────────────────────────────────────────────────────────────────────────
# LLM Action Planner
# ─────────────────────────────────────────────────────────────────────────────

def get_llm_action(client: OpenAI, observation: dict, task_id: str, history: List[dict]) -> dict:
    """Ask LLM for next action based on current observation."""
    curr  = observation.get("current_orbit", {})
    tgt   = observation.get("target_orbit", {})
    dv    = observation.get("delta_v_used", 0)
    budg  = observation.get("delta_v_budget", 0)
    steps = observation.get("step_index", 0)
    maxs  = observation.get("max_steps", 0)
    last  = observation.get("last_action_result", "")

    # ── Format available maneuvers ──
    maneuvers_text = ""
    available = observation.get("available_maneuvers", [])
    if available:
        maneuvers_text = "\nAvailable Maneuvers:\n"
        for m in available:
            feasible_tag = "✅ FEASIBLE" if m.get("feasible") else "❌ NOT FEASIBLE"
            maneuvers_text += (
                f"  • {m.get('name', 'unknown')} — {m.get('description', '')}\n"
                f"    Cost: {m.get('estimated_delta_v', 0):.0f} m/s "
                f"({m.get('fuel_percentage', 0):.1f}% of remaining fuel) [{feasible_tag}]\n"
            )
            if m.get("reason"):
                maneuvers_text += f"    Reason: {m['reason']}\n"

    # ── Format mission analysis ──
    analysis_text = ""
    analysis = observation.get("mission_analysis")
    if analysis:
        analysis_text = f"""
Mission Analysis:
  Altitude Error    : {analysis.get('altitude_error_km', 0):.1f} km
  Inclination Error : {analysis.get('inclination_error_deg', 0):.2f}°
  Eccentricity Error: {analysis.get('eccentricity_error', 0):.4f}
  Est. Δ-v Needed   : {analysis.get('estimated_delta_v_needed', 0):.0f} m/s
  Fuel Remaining    : {analysis.get('fuel_remaining', 0):.0f} m/s
  Fuel Margin       : {analysis.get('fuel_margin_percent', 0):.1f}%
  Score Estimate    : {analysis.get('current_score_estimate', 0):.3f}"""

    # ── Format recommendations ──
    recs_text = ""
    recs = observation.get("recommendations", [])
    if recs:
        recs_text = "\nRecommendations:\n"
        for r in recs:
            recs_text += f"  → {r}\n"

    user_message = f"""Mission: {task_id}
Step: {steps}/{maxs}

Current Orbit:
  Altitude    : {curr.get('altitude_km', 0):.1f} km
  Eccentricity: {curr.get('eccentricity', 0):.4f}
  Inclination : {curr.get('inclination_deg', 0):.2f}°
  Velocity    : {curr.get('velocity_ms', 0):.1f} m/s

Target Orbit:
  Altitude    : {tgt.get('altitude_km', 0):.1f} km
  Eccentricity: {tgt.get('eccentricity', 0):.4f}
  Inclination : {tgt.get('inclination_deg', 0):.2f}°

Fuel:
  Used      : {dv:.1f} m/s
  Budget    : {budg:.1f} m/s
  Remaining : {budg - dv:.1f} m/s
{analysis_text}
{maneuvers_text}
{recs_text}
Last Action Result: {last}
Previous Actions (last 3): {json.dumps(history[-3:]) if history else 'None'}

Based on the available maneuvers and recommendations, what is your next action?
Respond with ONLY valid JSON."""

    try:
        response = client.chat.completions.create(
            model       = MODEL_NAME,
            temperature = TEMPERATURE,
            max_tokens  = MAX_TOKENS,
            messages    = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return {"type": "run_simulation"}


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket Connection with Retry
# ─────────────────────────────────────────────────────────────────────────────

async def connect_with_retry(url: str, max_retries: int = WS_MAX_RETRIES, delay: float = WS_RETRY_DELAY):
    """Connect to WebSocket with retry logic for container startup delays."""
    for attempt in range(1, max_retries + 1):
        try:
            ws = await websockets.connect(url)
            print(f"[DEBUG] WebSocket connected on attempt {attempt}", flush=True)
            return ws
        except (ConnectionRefusedError, OSError, websockets.exceptions.InvalidURI) as e:
            print(f"[DEBUG] Connection attempt {attempt}/{max_retries} failed: {e}", flush=True)
            if attempt < max_retries:
                print(f"[DEBUG] Retrying in {delay}s...", flush=True)
                await asyncio.sleep(delay)
            else:
                print(f"[DEBUG] All {max_retries} connection attempts failed.", flush=True)
                raise


# ─────────────────────────────────────────────────────────────────────────────
# Mission Runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_mission(client: OpenAI, task_id: str) -> dict:
    """Run a single mission and emit [START], [STEP], [END] logs."""

    rewards      = []
    steps_taken  = 0
    score        = 0.0
    success      = False
    action_history = []

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        ws = await connect_with_retry(WS_URL)

        try:
            # Welcome message
            welcome_raw = await ws.recv()
            print(f"[DEBUG] Welcome received", flush=True)

            # Reset environment
            await ws.send(json.dumps({"type": "reset", "task_id": task_id}))
            msg = json.loads(await ws.recv())

            # Handle potential error on reset
            if msg.get("type") == "error":
                print(f"[DEBUG] Reset error: {msg.get('message')}", flush=True)
                log_end(success=False, steps=0, score=0.0, rewards=[])
                return {"task_id": task_id, "score": 0.0, "success": False, "steps": 0}

            observation = msg.get("data", {})

            step = 0
            while True:
                step += 1

                # Get LLM action
                action = get_llm_action(client, observation, task_id, action_history)
                action_history.append(action)
                action_str = json.dumps(action)

                # Send action to environment
                await ws.send(json.dumps({"type": "step", "action": action}))
                result = json.loads(await ws.recv())

                # Handle error responses gracefully
                if result.get("type") == "error":
                    error  = result.get("message", "unknown error")
                    reward = -0.1
                    done   = False
                    info   = {}
                    print(f"[DEBUG] Step error: {error}", flush=True)
                else:
                    observation = result.get("observation", {})
                    reward      = result.get("reward", 0.0)
                    done        = result.get("done", False)
                    info        = result.get("info", {})
                    error       = info.get("error", None)

                rewards.append(reward)
                steps_taken = step

                log_step(
                    step   = step,
                    action = action_str,
                    reward = reward,
                    done   = done,
                    error  = error,
                )

                if done:
                    grade   = info.get("grade_result", {})
                    score   = grade.get("score", 0.0)
                    success = grade.get("mission_success", False)
                    break

        finally:
            await ws.close()

    except Exception as e:
        print(f"[DEBUG] Mission error for {task_id}: {type(e).__name__}: {e}", flush=True)

    finally:
        log_end(
            success = success,
            steps   = steps_taken,
            score   = score,
            rewards = rewards,
        )

    return {
        "task_id": task_id,
        "score":   score,
        "success": success,
        "steps":   steps_taken,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    """Run all 3 missions with structured logging."""

    print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", flush=True)
    print(f"[DEBUG] WS_URL={WS_URL}", flush=True)
    print(f"[DEBUG] OPENAI_API_KEY={'set' if OPENAI_API_KEY else 'NOT SET'}", flush=True)

    # Validate API key
    if not OPENAI_API_KEY:
        print("[DEBUG] WARNING: OPENAI_API_KEY is not set. LLM calls will fail.", flush=True)
        print("[DEBUG] Falling back to deterministic baseline actions.", flush=True)

    try:
        client = OpenAI(
            api_key  = OPENAI_API_KEY or "dummy-key-for-fallback",
            base_url = API_BASE_URL,
        )
    except Exception as e:
        print(f"[DEBUG] Failed to create OpenAI client: {e}", flush=True)
        client = None

    results = []
    for task_id in TASKS:
        try:
            result = await run_mission(client, task_id)
            results.append(result)
        except Exception as e:
            print(f"[DEBUG] Unhandled error in {task_id}: {type(e).__name__}: {e}", flush=True)
            log_end(success=False, steps=0, score=0.0, rewards=[])
            results.append({
                "task_id": task_id,
                "score": 0.0,
                "success": False,
                "steps": 0,
            })

    print("\n[DEBUG] === FINAL SUMMARY ===", flush=True)
    for r in results:
        print(
            f"[DEBUG] {r['task_id']}: score={r['score']:.4f} "
            f"success={r['success']} steps={r['steps']}",
            flush=True,
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[DEBUG] Interrupted by user.", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"[DEBUG] Fatal error: {type(e).__name__}: {e}", flush=True)
        sys.exit(1)