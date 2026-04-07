"""
baseline/run_baseline.py
Orbit — AI Space Mission Architect

GPT-4 baseline agent that solves all 3 missions via WebSocket.

Usage:
    # Set your OpenAI API key first:
    export OPENAI_API_KEY=your_key_here   # macOS/Linux
    set OPENAI_API_KEY=your_key_here      # Windows

    # Make sure the server is running:
    uvicorn app.server:app --host 0.0.0.0 --port 7860

    # Then run this script:
    python baseline/run_baseline.py

Output:
    Prints step-by-step actions and final scores for all 3 missions.
    Results are fully reproducible (temperature=0).
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

WS_URL      = "ws://localhost:7860/ws"
MODEL       = "llama-3.3-70b-versatile"
TEMPERATURE = 0  # Deterministic — same inputs always produce same outputs
MAX_TOKENS  = 500

TASKS = ["leo_satellite", "lunar_orbit", "asteroid_rendezvous"]

# ─────────────────────────────────────────────────────────────────────────────
# OpenAI Client Setup
# ─────────────────────────────────────────────────────────────────────────────

def get_openai_client() -> OpenAI:
    """
    Create Groq client using GROQ_API_KEY from environment variables.
    Exits with helpful error if key is not set.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("❌ ERROR: GROQ_API_KEY environment variable not set.")
        print("   Set it with:")
        print("   Windows : set GROQ_API_KEY=your_key_here")
        print("   macOS   : export GROQ_API_KEY=your_key_here")
        sys.exit(1)
    return OpenAI(
        api_key  = api_key,
        base_url = "https://api.groq.com/openai/v1"
    )


# ─────────────────────────────────────────────────────────────────────────────
# GPT-4 Action Planner
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


def get_gpt4_action(
    client: OpenAI,
    observation: Dict,
    step: int,
    task_id: str,
    history: List[Dict],
) -> Dict:
    """
    Ask GPT-4 to plan the next action based on current observation.

    Args:
        client:      OpenAI client
        observation: Current environment observation
        step:        Current step number
        task_id:     Current mission ID
        history:     List of previous actions taken

    Returns:
        Action dict ready to send to the environment
    """
    # Format observation clearly for GPT-4
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
  Altitude   : {curr['altitude_km']:.1f} km
  Eccentricity: {curr['eccentricity']:.4f}
  Inclination: {curr['inclination_deg']:.2f}°
  Velocity   : {curr['velocity_ms']:.1f} m/s

Target Orbit:
  Altitude   : {tgt['altitude_km']:.1f} km
  Eccentricity: {tgt['eccentricity']:.4f}
  Inclination: {tgt['inclination_deg']:.2f}°

Fuel:
  Used      : {dv:.1f} m/s
  Budget    : {budg:.1f} m/s
  Remaining : {budg - dv:.1f} m/s

Last Action Result: {last}

Previous Actions: {json.dumps(history[-3:]) if history else 'None'}

What is your next action? Respond with ONLY valid JSON."""

    response = client.chat.completions.create(
        model       = MODEL,
        temperature = TEMPERATURE,
        max_tokens  = MAX_TOKENS,
        messages    = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Clean up any accidental markdown
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"   ⚠️  GPT-4 returned invalid JSON: {raw}")
        print("   ⚠️  Falling back to run_simulation")
        return {"type": "run_simulation"}


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket Mission Runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_mission(
    client: OpenAI,
    task_id: str,
) -> Dict[str, Any]:
    """
    Run a single mission using GPT-4 as the agent.

    Args:
        client:  OpenAI client
        task_id: Mission to run

    Returns:
        Dict with task_id, final_score, steps_used, delta_v_used, success
    """
    print(f"\n{'='*60}")
    print(f"🚀 Starting Mission: {task_id.upper()}")
    print(f"{'='*60}")

    async with websockets.connect(WS_URL) as ws:
        # Receive welcome message
        welcome = json.loads(await ws.recv())
        print(f"✅ Connected to server")

        # Reset environment for this task
        await ws.send(json.dumps({"type": "reset", "task_id": task_id}))
        msg         = json.loads(await ws.recv())
        observation = msg["data"]

        print(f"📋 Task initialized")
        print(f"   Target: {observation['target_orbit']['altitude_km']:.0f} km altitude")
        print(f"   Budget: {observation['delta_v_budget']:.0f} m/s")
        print(f"   Max Steps: {observation['max_steps']}")

        action_history = []
        final_score    = 0.0
        steps_used     = 0
        delta_v_used   = 0.0
        mission_success = False

        # Main agent loop
        while True:
            step = observation["step_index"]

            # Ask GPT-4 for next action
            print(f"\n--- Step {step + 1} ---")
            print(f"   Altitude : {observation['current_orbit']['altitude_km']:.1f} km")
            print(f"   Δv Used  : {observation['delta_v_used']:.1f} m/s")

            action = get_gpt4_action(
                client      = client,
                observation = observation,
                step        = step,
                task_id     = task_id,
                history     = action_history,
            )

            print(f"   Action   : {json.dumps(action)}")
            action_history.append(action)

            # Send action to environment
            await ws.send(json.dumps({"type": "step", "action": action}))
            result = json.loads(await ws.recv())

            # Handle step result
            observation  = result["observation"]
            reward       = result["reward"]
            done         = result["done"]
            info         = result["info"]

            print(f"   Reward   : {reward:.4f}")
            print(f"   Message  : {info.get('message', '')[:80]}")

            # Check if mission ended
            if done:
                grade = info.get("grade_result", {})
                final_score     = grade.get("score", 0.0)
                steps_used      = grade.get("steps_used", step + 1)
                delta_v_used    = grade.get("delta_v_used", observation["delta_v_used"])
                mission_success = grade.get("mission_success", False)

                status = "✅ SUCCESS" if mission_success else "❌ INCOMPLETE"
                print(f"\n{status}")
                print(f"   Final Score  : {final_score:.4f}")
                print(f"   Steps Used   : {steps_used}")
                print(f"   Δv Used      : {delta_v_used:.1f} m/s")
                if grade:
                    print(f"   Components   :")
                    for k, v in grade.get("component_scores", {}).items():
                        print(f"     {k:15s}: {v:.4f}")
                break

    return {
        "task_id":        task_id,
        "final_score":    final_score,
        "steps_used":     steps_used,
        "delta_v_used":   delta_v_used,
        "mission_success": mission_success,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    """Run all 3 missions and print a final summary."""

    print("🌌 Orbit — AI Space Mission Architect")
    print("   GPT-4 Baseline Agent")
    print(f"   Model: {MODEL} | Temperature: {TEMPERATURE}")
    print(f"   Server: {WS_URL}")

    # Setup OpenAI client
    client = get_openai_client()
    print("✅ OpenAI client ready\n")

    # Run all missions
    results = []
    for task_id in TASKS:
        result = await run_mission(client, task_id)
        results.append(result)

    # Final Summary
    print(f"\n{'='*60}")
    print("📊 BASELINE RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Mission':<30} {'Score':>8} {'Δv Used':>10} {'Steps':>7} {'Success':>8}")
    print(f"{'-'*60}")

    total_score = 0.0
    for r in results:
        status = "✅" if r["mission_success"] else "❌"
        print(
            f"{r['task_id']:<30} "
            f"{r['final_score']:>8.4f} "
            f"{r['delta_v_used']:>10.1f} "
            f"{r['steps_used']:>7} "
            f"{status:>8}"
        )
        total_score += r["final_score"]

    avg_score = total_score / len(results)
    print(f"{'-'*60}")
    print(f"{'Average Score':<30} {avg_score:>8.4f}")
    print(f"{'='*60}")
    print("\n✅ Baseline run complete! Results are reproducible at temperature=0.")


if __name__ == "__main__":
    asyncio.run(main())