"""
mock_agent.py
Orbit — AI Space Mission Architect

Rule-based mock agent that tests all 3 missions without needing an API key.
Uses hardcoded physics-based actions for each mission.
"""

import asyncio
import json
import websockets

WS_URL = "ws://localhost:7860/ws"

# ─────────────────────────────────────────────────────────────────────────────
# Hardcoded Mission Plans (physics-based)
# ─────────────────────────────────────────────────────────────────────────────

MISSION_PLANS = {

    # LEO: One large prograde burn from ground + inclination correction
    "leo_satellite": [
        {"type": "run_simulation"},
        {"type": "add_burn", "delta_v_ms": 9400.0, "prograde": 1.0, "radial": 0.0, "normal": 0.0},
        {"type": "add_burn", "delta_v_ms": 500.0,  "prograde": 0.0, "radial": 0.0, "normal": 1.0},
        {"type": "run_simulation"},
        {"type": "submit_mission"},
    ],

    # Lunar: TLI burn + LOI braking burn
    "lunar_orbit": [
        {"type": "run_simulation"},
        {"type": "add_burn", "delta_v_ms": 3133.0, "prograde": 1.0,  "radial": 0.0, "normal": 0.0},
        {"type": "run_simulation"},
        {"type": "add_burn", "delta_v_ms": 834.0,  "prograde": -1.0, "radial": 0.0, "normal": 0.0},
        {"type": "run_simulation"},
        {"type": "submit_mission"},
    ],

    # Asteroid: Gravity assists + burns
    "asteroid_rendezvous": [
        {"type": "run_simulation"},
        {"type": "set_flyby", "body": "earth", "periapsis_km": 300.0},
        {"type": "set_flyby", "body": "venus", "periapsis_km": 400.0},
        {"type": "add_burn", "delta_v_ms": 3000.0, "prograde": 1.0, "radial": 0.0, "normal": 0.0},
        {"type": "run_simulation"},
        {"type": "add_burn", "delta_v_ms": 2000.0, "prograde": 1.0, "radial": 0.0, "normal": 0.0},
        {"type": "run_simulation"},
        {"type": "add_burn", "delta_v_ms": 500.0,  "prograde": 0.0, "radial": 0.0, "normal": 1.0},
        {"type": "run_simulation"},
        {"type": "submit_mission"},
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Mission Runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_mission(task_id: str) -> dict:
    """Run a single mission using hardcoded plan."""

    print(f"\n{'='*60}")
    print(f"🚀 Starting Mission: {task_id.upper()}")
    print(f"{'='*60}")

    async with websockets.connect(WS_URL) as ws:
        # Welcome message
        await ws.recv()
        print(f"✅ Connected to server")

        # Reset
        await ws.send(json.dumps({"type": "reset", "task_id": task_id}))
        msg         = json.loads(await ws.recv())
        observation = msg["data"]

        print(f"📋 Task initialized")
        print(f"   Target Altitude : {observation['target_orbit']['altitude_km']:.0f} km")
        print(f"   Budget          : {observation['delta_v_budget']:.0f} m/s")
        print(f"   Max Steps       : {observation['max_steps']}")

        plan = MISSION_PLANS[task_id]

        final_score     = 0.0
        steps_used      = 0
        delta_v_used    = 0.0
        mission_success = False

        for action in plan:
            print(f"\n--- Step {observation['step_index'] + 1} ---")
            print(f"   Altitude : {observation['current_orbit']['altitude_km']:.1f} km")
            print(f"   Δv Used  : {observation['delta_v_used']:.1f} m/s")
            print(f"   Action   : {json.dumps(action)}")

            await ws.send(json.dumps({"type": "step", "action": action}))
            result = json.loads(await ws.recv())

            observation = result["observation"]
            reward      = result["reward"]
            done        = result["done"]
            info        = result["info"]

            print(f"   Reward   : {reward:.4f}")
            print(f"   Message  : {info.get('message', '')[:80]}")

            if done:
                grade           = info.get("grade_result", {})
                final_score     = grade.get("score", 0.0)
                steps_used      = grade.get("steps_used", 0)
                delta_v_used    = grade.get("delta_v_used", 0.0)
                mission_success = grade.get("mission_success", False)

                status = "✅ SUCCESS" if mission_success else "❌ INCOMPLETE"
                print(f"\n{status}")
                print(f"   Final Score : {final_score:.4f}")
                print(f"   Steps Used  : {steps_used}")
                print(f"   Δv Used     : {delta_v_used:.1f} m/s")
                if grade:
                    print(f"   Components  :")
                    for k, v in grade.get("component_scores", {}).items():
                        print(f"     {k:15s}: {v:.4f}")
                break

    return {
        "task_id":         task_id,
        "final_score":     final_score,
        "steps_used":      steps_used,
        "delta_v_used":    delta_v_used,
        "mission_success": mission_success,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    """Run all 3 missions and print summary."""

    print("🌌 Orbit — AI Space Mission Architect")
    print("   Mock Rule-Based Agent (No API Key Required)")
    print(f"   Server: {WS_URL}")

    results = []
    for task_id in ["leo_satellite", "lunar_orbit", "asteroid_rendezvous"]:
        result = await run_mission(task_id)
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("📊 MOCK AGENT RESULTS SUMMARY")
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

    avg = total_score / len(results)
    print(f"{'-'*60}")
    print(f"{'Average Score':<30} {avg:>8.4f}")
    print(f"{'='*60}")
    print("\n✅ Mock agent run complete!")


if __name__ == "__main__":
    asyncio.run(main())