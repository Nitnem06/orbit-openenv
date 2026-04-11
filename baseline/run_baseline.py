"""
baseline/run_baseline.py
Orbit — AI Space Mission Architect

Baseline LLM agent that solves all 3 missions via WebSocket.
Uses strategic execute_maneuver actions with enriched observations.

Usage:
    # Set your API key first:
    export OPENAI_API_KEY=your_key_here       # macOS/Linux
    set OPENAI_API_KEY=your_key_here          # Windows

    # Optional: configure model and endpoint
    export API_BASE_URL=https://router.huggingface.co/v1
    export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

    # Make sure the server is running:
    uvicorn app.server:app --host 0.0.0.0 --port 7860

    # Then run this script:
    python baseline/run_baseline.py

Output:
    Prints [START]/[STEP]/[END] logs (OpenEnv-compliant) plus
    human-readable step details. Fully reproducible at temperature=0.

Environment Variables:
    OPENAI_API_KEY  : API key (required)
    API_BASE_URL    : LLM API endpoint (default: HuggingFace router)
    MODEL_NAME      : Model identifier (default: Qwen2.5-72B-Instruct)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional

import websockets
from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

WS_URL       = "ws://localhost:7860/ws"
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

TEMPERATURE = 0       # Deterministic — same inputs always produce same outputs
MAX_TOKENS  = 500
BENCHMARK   = "orbit-mission-architect"

TASKS = ["leo_satellite", "lunar_orbit", "asteroid_rendezvous"]


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI Client Setup
# ─────────────────────────────────────────────────────────────────────────────

def get_client() -> OpenAI:
    """
    Create OpenAI-compatible client using OPENAI_API_KEY.
    Exits with helpful error if key is not set.
    """
    if not OPENAI_API_KEY:
        print("❌ ERROR: OPENAI_API_KEY environment variable not set.")
        print("   Set it with:")
        print("     Windows : set OPENAI_API_KEY=your_key_here")
        print("     macOS   : export OPENAI_API_KEY=your_key_here")
        sys.exit(1)

    return OpenAI(
        api_key  = OPENAI_API_KEY,
        base_url = API_BASE_URL,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Mandatory Log Functions (OpenEnv spec)
# ─────────────────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt — Strategic Maneuvers (v2)
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
  → Step 3: Consider circularize if eccentricity is high
  → Step 4: submit_mission

ASTEROID RENDEZVOUS (Hard):
  → Direct transfer exceeds fuel budget — you MUST use gravity assists
  → Step 1: gravity_assist with venus (free fuel!)
  → Step 2: gravity_assist with earth (more free fuel!)
  → Step 3: combined_transfer for altitude + inclination change
  → Step 4: Consider correction_burn or circularize for fine-tuning
  → Step 5: submit_mission

IMPORTANT:
  - Check "available_maneuvers" in the observation — it tells you what you CAN do and the fuel cost
  - Check "mission_analysis" — it shows your errors and fuel margin
  - Check "recommendations" — it gives strategic advice
  - Gravity assists are FREE — always consider them first on hard missions
  - Combined transfers save 15% fuel vs separate maneuvers
  - Submit when your score estimate is good or you're running low on steps
"""


# ─────────────────────────────────────────────────────────────────────────────
# LLM Action Planner
# ─────────────────────────────────────────────────────────────────────────────

def get_llm_action(
    client: OpenAI,
    observation: Dict,
    task_id: str,
    history: List[Dict],
) -> Dict:
    """
    Ask LLM for next action based on current observation.
    Uses enriched v2 observations: available_maneuvers, mission_analysis,
    recommendations.
    """
    curr  = observation["current_orbit"]
    tgt   = observation["target_orbit"]
    dv    = observation["delta_v_used"]
    budg  = observation["delta_v_budget"]
    steps = observation["step_index"]
    maxs  = observation["max_steps"]
    last  = observation.get("last_action_result", "")

    # ── Format available maneuvers ──
    maneuvers_text = ""
    available = observation.get("available_maneuvers", [])
    if available:
        maneuvers_text = "\nAvailable Maneuvers:\n"
        for m in available:
            feasible_tag = "✅ FEASIBLE" if m["feasible"] else "❌ NOT FEASIBLE"
            maneuvers_text += (
                f"  • {m['name']} — {m['description']}\n"
                f"    Cost: {m['estimated_delta_v']:.0f} m/s "
                f"({m['fuel_percentage']:.1f}% of remaining fuel) "
                f"[{feasible_tag}]\n"
            )
            if m.get("reason"):
                maneuvers_text += f"    Reason: {m['reason']}\n"

    # ── Format mission analysis ──
    analysis_text = ""
    analysis = observation.get("mission_analysis")
    if analysis:
        analysis_text = f"""
Mission Analysis:
  Altitude Error    : {analysis['altitude_error_km']:.1f} km
  Inclination Error : {analysis['inclination_error_deg']:.2f}°
  Eccentricity Error: {analysis['eccentricity_error']:.4f}
  Est. Δ-v Needed   : {analysis['estimated_delta_v_needed']:.0f} m/s
  Fuel Remaining    : {analysis['fuel_remaining']:.0f} m/s
  Fuel Margin       : {analysis['fuel_margin_percent']:.1f}%
  Score Estimate    : {analysis['current_score_estimate']:.3f}"""

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
  Altitude    : {curr['altitude_km']:.1f} km
  Eccentricity: {curr['eccentricity']:.4f}
  Inclination : {curr['inclination_deg']:.2f}°
  Velocity    : {curr['velocity_ms']:.1f} m/s

Target Orbit:
  Altitude    : {tgt['altitude_km']:.1f} km
  Eccentricity: {tgt['eccentricity']:.4f}
  Inclination : {tgt['inclination_deg']:.2f}°

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

        # Clean up accidental markdown
        raw = raw.replace("```json", "").replace("```", "").strip()

        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"   ⚠️  LLM returned invalid JSON: {raw}", flush=True)
        print("   ⚠️  Falling back to run_simulation", flush=True)
        return {"type": "run_simulation"}
    except Exception as e:
        print(f"   ⚠️  LLM error: {e}", flush=True)
        return {"type": "run_simulation"}


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket Mission Runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_mission(client: OpenAI, task_id: str) -> Dict[str, Any]:
    """
    Run a single mission using LLM agent.
    Emits [START]/[STEP]/[END] logs (OpenEnv-compliant) plus human-readable detail.
    """
    rewards        = []
    steps_taken    = 0
    score          = 0.0
    success        = False
    action_history = []

    # ── OpenEnv-compliant start log ──
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    # ── Human-readable header ──
    print(f"\n{'='*60}", flush=True)
    print(f"🚀 Starting Mission: {task_id.upper()}", flush=True)
    print(f"{'='*60}", flush=True)

    try:
        async with websockets.connect(WS_URL) as ws:
            # Welcome message
            welcome = json.loads(await ws.recv())
            print(f"✅ Connected to server", flush=True)

            # Reset environment
            await ws.send(json.dumps({"type": "reset", "task_id": task_id}))
            msg         = json.loads(await ws.recv())
            observation = msg["data"]

            print(f"📋 Task initialized", flush=True)
            print(f"   Target : {observation['target_orbit']['altitude_km']:.0f} km altitude", flush=True)
            print(f"   Budget : {observation['delta_v_budget']:.0f} m/s", flush=True)
            print(f"   Steps  : {observation['max_steps']}", flush=True)

            step = 0
            while True:
                step += 1

                # ── Human-readable step detail ──
                print(f"\n--- Step {step} ---", flush=True)
                print(f"   Altitude : {observation['current_orbit']['altitude_km']:.1f} km", flush=True)
                print(f"   Ecc      : {observation['current_orbit']['eccentricity']:.4f}", flush=True)
                print(f"   Inc      : {observation['current_orbit']['inclination_deg']:.2f}°", flush=True)
                print(f"   Δv Used  : {observation['delta_v_used']:.1f} m/s", flush=True)

                # Get LLM action
                action = get_llm_action(client, observation, task_id, action_history)
                action_history.append(action)
                action_str = json.dumps(action)

                print(f"   Action   : {action_str}", flush=True)

                # Send action to environment
                await ws.send(json.dumps({"type": "step", "action": action}))
                result = json.loads(await ws.recv())

                # Handle response
                if result.get("type") == "error":
                    error  = result.get("message", "unknown error")
                    reward = -0.1
                    done   = False
                    info   = {}
                    print(f"   ⚠️ Error: {error}", flush=True)
                else:
                    observation = result["observation"]
                    reward      = result["reward"]
                    done        = result["done"]
                    info        = result.get("info", {})
                    error       = info.get("error", None)

                rewards.append(reward)
                steps_taken = step

                print(f"   Reward   : {reward:.4f}", flush=True)
                msg_text = info.get("message", "")
                if msg_text:
                    print(f"   Message  : {msg_text[:100]}", flush=True)

                # ── OpenEnv-compliant step log ──
                log_step(
                    step   = step,
                    action = action_str,
                    reward = reward,
                    done   = done,
                    error  = error,
                )

                if done:
                    grade = info.get("grade_result", {})
                    score   = grade.get("score", 0.0)
                    success = grade.get("mission_success", False)

                    # ── Human-readable result ──
                    status = "✅ SUCCESS" if success else "❌ INCOMPLETE"
                    print(f"\n{status}", flush=True)
                    print(f"   Final Score : {score:.4f}", flush=True)
                    print(f"   Steps Used  : {grade.get('steps_used', step)}", flush=True)
                    print(f"   Δv Used     : {grade.get('delta_v_used', 0):.1f} m/s", flush=True)
                    if grade.get("component_scores"):
                        print(f"   Components  :", flush=True)
                        for k, v in grade["component_scores"].items():
                            print(f"     {k:15s}: {v:.4f}", flush=True)
                    break

    except Exception as e:
        print(f"   ❌ Mission error: {e}", flush=True)

    finally:
        # ── OpenEnv-compliant end log ──
        log_end(
            success = success,
            steps   = steps_taken,
            score   = score,
            rewards = rewards,
        )

    return {
        "task_id":         task_id,
        "score":           score,
        "success":         success,
        "steps":           steps_taken,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    """Run all 3 missions and print a final summary."""

    print("🌌 Orbit — AI Space Mission Architect", flush=True)
    print("   Baseline LLM Agent", flush=True)
    print(f"   Model      : {MODEL_NAME}", flush=True)
    print(f"   API Base   : {API_BASE_URL}", flush=True)
    print(f"   Server     : {WS_URL}", flush=True)
    print(f"   Temperature: {TEMPERATURE}", flush=True)

    # Setup client
    client = get_client()
    print("✅ LLM client ready\n", flush=True)

    # Run all missions
    results = []
    for task_id in TASKS:
        result = await run_mission(client, task_id)
        results.append(result)

    # ── Final Summary ──
    print(f"\n{'='*60}", flush=True)
    print("📊 BASELINE RESULTS SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(
        f"{'Mission':<30} {'Score':>8} {'Steps':>7} {'Success':>8}",
        flush=True,
    )
    print(f"{'-'*60}", flush=True)

    total_score = 0.0
    for r in results:
        status = "✅" if r["success"] else "❌"
        print(
            f"{r['task_id']:<30} "
            f"{r['score']:>8.4f} "
            f"{r['steps']:>7} "
            f"{status:>8}",
            flush=True,
        )
        total_score += r["score"]

    avg_score = total_score / len(results)
    print(f"{'-'*60}", flush=True)
    print(f"{'Average Score':<30} {avg_score:>8.4f}", flush=True)
    print(f"{'='*60}", flush=True)
    print(
        "\n✅ Baseline complete. Results are reproducible at temperature=0.",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())