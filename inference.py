"""
inference.py
Orbit — AI Space Mission Architect

OpenEnv-compliant inference script.
Runs all 3 missions using an LLM agent via WebSocket.

Environment Variables:
    API_BASE_URL  : LLM API endpoint (default: HuggingFace router)
    MODEL_NAME    : Model identifier (default: Qwen2.5-72B-Instruct)
    HF_TOKEN      : HuggingFace / API key
"""

import asyncio
import json
import os
import sys
from typing import List, Optional

import websockets
from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# Configuration from Environment Variables
# ─────────────────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN")     or os.getenv("API_KEY", "dummy")

WS_URL       = "ws://localhost:7860/ws"
TEMPERATURE  = 0
MAX_TOKENS   = 500
BENCHMARK    = "orbit-mission-architect"

TASKS = ["leo_satellite", "lunar_orbit", "asteroid_rendezvous"]

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

SYSTEM_PROMPT = """You are an expert orbital mechanics engineer planning space missions.

You control a spacecraft and must reach a target orbit efficiently.
Your goal is to minimize fuel (Δ-v) usage while reaching the target orbit.

You must respond with ONLY a valid JSON action. No explanation, no markdown, just JSON.

Available actions:

1. Set orbit plan (no fuel cost):
{"type": "set_orbit", "altitude_km": 400.0, "eccentricity": 0.0, "inclination_deg": 51.6}

2. Execute a burn (costs fuel):
{"type": "add_burn", "delta_v_ms": 9400.0, "prograde": 1.0, "radial": 0.0, "normal": 0.0}
- prograde +1.0 = forward (raises orbit), -1.0 = retrograde (lowers orbit)
- radial   +1.0 = away from Earth (changes eccentricity)
- normal   +1.0 = north (changes inclination, very expensive!)

3. Plan gravity assist (Task 3 only):
{"type": "set_flyby", "body": "venus", "periapsis_km": 500.0}

4. Preview current score (free):
{"type": "run_simulation"}

5. Submit mission for final grading:
{"type": "submit_mission"}

Strategy tips:
- LEO mission   : One large prograde burn ~9400 m/s from ground
- Lunar mission : TLI burn ~3100 m/s prograde, then LOI burn ~850 m/s retrograde
- Asteroid mission: Use gravity assists to save fuel, plan flybys first
- Always run_simulation before submitting to check your score
- Submit when you are close to target or running low on steps/budget
"""


# ─────────────────────────────────────────────────────────────────────────────
# LLM Action Planner
# ─────────────────────────────────────────────────────────────────────────────

def get_llm_action(client: OpenAI, observation: dict, task_id: str, history: List[dict]) -> dict:
    """Ask LLM for next action based on current observation."""
    curr  = observation["current_orbit"]
    tgt   = observation["target_orbit"]
    dv    = observation["delta_v_used"]
    budg  = observation["delta_v_budget"]
    steps = observation["step_index"]
    maxs  = observation["max_steps"]
    last  = observation.get("last_action_result", "")

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

Last Action Result: {last}
Previous Actions: {json.dumps(history[-3:]) if history else 'None'}

What is your next action? Respond with ONLY valid JSON."""

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
        async with websockets.connect(WS_URL) as ws:
            # Welcome message
            await ws.recv()

            # Reset environment
            await ws.send(json.dumps({"type": "reset", "task_id": task_id}))
            msg         = json.loads(await ws.recv())
            observation = msg["data"]

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

                observation = result["observation"]
                reward      = result["reward"]
                done        = result["done"]
                info        = result.get("info", {})
                error       = info.get("error", None)

                rewards.append(reward)
                steps_taken = step

                # Emit [STEP] log
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

    except Exception as e:
        print(f"[DEBUG] Mission error: {e}", flush=True)

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

    # Setup OpenAI client pointing to configured LLM endpoint
    client = OpenAI(
        api_key  = API_KEY,
        base_url = API_BASE_URL,
    )

    results = []
    for task_id in TASKS:
        result = await run_mission(client, task_id)
        results.append(result)

    # Final summary to stderr (not stdout — keeps stdout clean for parser)
    print("\n[DEBUG] === FINAL SUMMARY ===", flush=True)
    for r in results:
        print(
            f"[DEBUG] {r['task_id']}: score={r['score']:.4f} "
            f"success={r['success']} steps={r['steps']}",
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())