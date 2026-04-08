"""
Meta Hackathon: Mandatory Compliant Inference Script
Implements exactly the logging format and structure from the sample inference.py.
"""

import os
import json
import asyncio
import textwrap
import argparse
from typing import List, Optional, Dict, Any
import httpx
from openai import OpenAI

# Required environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
TASK_NAME = os.getenv("FINOPS_TASK", "sandbox_cleanup")
BENCHMARK = "finops-cloud-optimizer"

TEMPERATURE = 0.0
MAX_TOKENS = 512
SUCCESS_SCORE_THRESHOLD = 0.1

class OpenEnvAPIWrapper:
    """Wrapper to make the FastAPI environment look like a standard OpenEnv class."""
    def __init__(self, base_url: str):
        self.client = httpx.AsyncClient(base_url=base_url, timeout=60.0)

    async def reset(self, task_id: str):
        resp = await self.client.post("/reset", json={"task_id": task_id})
        resp.raise_for_status()
        return resp.json()

    async def step(self, action: Dict[str, Any]):
        resp = await self.client.post("/step", json=action)
        resp.raise_for_status()
        return resp.json()

    async def grade(self):
        resp = await self.client.get("/grade")
        resp.raise_for_status()
        return resp.json()

    async def close(self):
        await self.client.aclose()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def get_model_action(client: OpenAI, obs: Any, history: List[str]) -> Dict[str, Any]:
    system_prompt = textwrap.dedent(
        """
        You are a FinOps AI agent. Analyze cloud infrastructure and optimize for cost savings.
        Maintain production safety — breaking resources with active dependencies is heavily penalized.
        
        Output ONLY valid JSON in this exact format:
        {"action_type": "<type>", "target_resource": "<resource_id or null>", "parameters": {}}
        
        Valid action types: analyze, check_deps, rightsize, terminate, schedule, reserve, migrate, flag, finalize
        """
    ).strip()

    user_prompt = f"Observation:\n{json.dumps(obs, indent=2)}\n\nAction History:\n" + "\n".join(history[-5:])
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        
        # Robust parsing for JSON block
        import re
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        
        return json.loads(text)
    except Exception as exc:
        print(f"[DEBUG] Model request failed or invalid JSON: {exc}", flush=True)
        return {"action_type": "finalize", "target_resource": None, "parameters": {}}

async def main(task_id: str) -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "EMPTY")
    env = OpenEnvAPIWrapper(ENV_URL)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_data = await env.reset(task_id)
        obs = reset_data["observation"]
        done = False

        while not done:
            steps_taken += 1
            action = get_model_action(client, obs, history)
            
            result = await env.step(action)
            obs = result["observation"]
            reward = result["reward"]
            done = result["done"]
            error = result["info"].get("error")

            rewards.append(reward)
            log_step(step=steps_taken, action=action["action_type"], reward=reward, done=done, error=error)
            
            history.append(f"Step {steps_taken}: {action['action_type']} -> reward {reward:+.2f}")

            if done or steps_taken >= 60:
                break

        grade_data = await env.grade()
        score = grade_data.get("total_score", 0.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=TASK_NAME)
    args = parser.parse_args()
    asyncio.run(main(args.task))
