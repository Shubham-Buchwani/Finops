"""
LLM-based baseline agent using ReAct-style single-turn prompting.
Calls the OpenEnv /step API in a loop until done.

Usage:
  python run_baseline.py --task sandbox_cleanup --model meta-llama/Llama-3-8B-Instruct
  python run_baseline.py --task cross_account_rightsizing --model gpt-4o-mini
"""

from __future__ import annotations
import argparse
import json
import os
# pyre-ignore[21]
import httpx

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

SYSTEM_PROMPT = """You are an expert FinOps analyst. You are given a cloud infrastructure snapshot.
Your goal: reduce monthly spend by the specified budget target percentage without breaking production services.

STRATEGY:
1. First, use ANALYZE on resources to understand utilization details.
2. Use CHECK_DEPENDENCIES before modifying production or critical resources.
3. Start with obvious waste: sandbox, idle, or development resources with 0 utilization.
4. Then rightsize oversized resources (low p95 CPU and memory).
5. Consider SCHEDULE for resources with predictable off-hours.
6. Consider RESERVE for stable long-running production resources.
7. Use FLAG_FOR_REVIEW for complex architectural changes.
8. When done, call FINALIZE_PLAN.

CRITICAL RULES:
- Never terminate a production resource without first calling CHECK_DEPENDENCIES.
- Never rightsize a resource if p95 CPU > 70% (it may have bursty workloads).
- Average CPU alone is NOT sufficient — always check p95 after ANALYZE.

Always output ONLY valid JSON in this exact format:
{"action_type": "<type>", "target_resource": "<resource_id or null>", "parameters": {}}

Valid action types: analyze, check_deps, rightsize, terminate, schedule, reserve, migrate, flag, finalize
"""

def call_llm(messages: list, client: httpx.Client) -> str:
    headers = {}
    auth_token = HF_TOKEN or os.environ.get('OPENAI_API_KEY', '')
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    response = client.post(
        f"{API_BASE_URL}/chat/completions",
        headers=headers,
        json={
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 200,
        },
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def run_episode(task_id: str) -> dict:
    env_client = httpx.Client(base_url=ENV_URL)
    llm_client = httpx.Client()

    # Reset
    reset_resp = env_client.post("/reset", json={"task_id": task_id})
    reset_resp.raise_for_status()
    obs = reset_resp.json()["observation"]

    total_reward = 0.0
    steps = []

    while not obs.get("done", False):
        # Build prompt
        obs_summary = {
            "step": obs["step_number"],
            "max_steps": obs["max_steps"],
            "savings_achieved": obs["savings_achieved_so_far"],
            "budget_target": obs["budget_target_dollars"],
            "incidents": obs["incidents_caused"],
            "resources": [
                {
                    "id": r["resource_id"],
                    "type": r["resource_type"],
                    "env": r["environment"],
                    "config": r["instance_config"],
                    "cost": r["monthly_cost"],
                    "avg_cpu": r["avg_cpu_utilization"],
                    "analyzed": r["analyzed"],
                    "detail": r["analysis_detail"] if r["analyzed"] else None,
                }
                for r in obs["resources"]
            ],
            "last_actions": obs["action_history"][-5:],
            "system_message": obs.get("system_message"),
        }

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Current state:\n{json.dumps(obs_summary, indent=2)}\n\n"
                    f"What is your next action? Output only valid JSON."
                ),
            },
        ]

        try:
            llm_output = call_llm(messages, llm_client)
            action = json.loads(llm_output)
        except Exception as e:
            print(f"LLM error or parse error: {e}. Using finalize.")
            action = {"action_type": "finalize", "target_resource": None, "parameters": {}}

        # Step
        step_resp = env_client.post("/step", json=action)
        step_resp.raise_for_status()
        step_data = step_resp.json()
        obs = step_data["observation"]
        reward = step_data["reward"]
        total_reward += reward
        steps.append({"action": action, "reward": reward, "info": step_data["info"]})

        print(f"Step {obs['step_number']}: {action['action_type']} | reward={reward:.3f} | savings=${obs['savings_achieved_so_far']:.0f}")

    # Grade
    grade_resp = env_client.get("/grade")
    final_grade = grade_resp.json()

    return {
        "task_id": task_id,
        "total_reward": float(f"{total_reward:.4f}"),
        "final_grade": final_grade,
        "steps_taken": len(steps),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="sandbox_cleanup",
                        choices=["sandbox_cleanup", "cross_account_rightsizing", "enterprise_transformation"])
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    if args.model:
        MODEL_NAME = args.model

    result = run_episode(args.task)
    print("\n=== EPISODE RESULT ===")
    print(json.dumps(result, indent=2))
