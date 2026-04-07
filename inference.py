"""
OpenEnv Compliance Inference Script
Emits structured logs: [START], [STEP], [END]
Uses the OpenAI Python client as mandated.
"""

import os
import json
import argparse
from openai import OpenAI

# Required environment variables
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Llama-2-7b-chat-hf")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

def run_inference(task_id: str):
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "EMPTY")
    import httpx
    env_client = httpx.Client(base_url=ENV_URL)

    # 1. Start the episode
    try:
        reset_resp = env_client.post("/reset", json={"task_id": task_id})
        reset_resp.raise_for_status()
        obs = reset_resp.json()["observation"]
    except Exception as e:
        print(f"Failed to reset environment: {e}")
        return

    # [START] Logging
    print(f"[START] task_id={task_id} model={MODEL_NAME}")

    system_prompt = (
        "You are a FinOps AI agent. Analyze the cloud infrastructure and optimize for cost savings "
        "without breaking production. Output ONLY valid JSON: "
        '{"action_type": "<type>", "target_resource": "<resource_id or null>", "parameters": {}}'
    )

    total_reward = 0.0
    step_count = 0
    done = False

    while not done:
        # 2. Get LLM response
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Observations:\n{json.dumps(obs, indent=2)}"}
                ],
                temperature=0.0,
                max_tokens=256
            )
            llm_text = response.choices[0].message.content.strip()
            # Handle possible markdown wrapping
            if "```json" in llm_text:
                llm_text = llm_text.split("```json")[-1].split("```")[0].strip()
            action = json.loads(llm_text)
        except Exception as e:
            # Fallback to finalize if LLM fails or task is done
            action = {"action_type": "finalize", "target_resource": None, "parameters": {}}

        # 3. Take step in environment
        try:
            step_resp = env_client.post("/step", json=action)
            step_resp.raise_for_status()
            step_data = step_resp.json()
            obs = step_data["observation"]
            reward = step_data["reward"]
            done = step_data["done"]
            total_reward += reward
            step_count += 1

            # [STEP] Logging
            print(f"[STEP] step_number={step_count} action={action['action_type']} reward={reward:.4f} done={done}")

        except Exception as e:
            print(f"Error taking step: {e}")
            break

    # 4. Final grade
    try:
        grade_resp = env_client.get("/grade")
        grade_data = grade_resp.json()
        final_score = grade_data.get("total_score", 0.0)
    except Exception:
        final_score = 0.0

    # [END] Logging
    print(f"[END] task_id={task_id} total_reward={total_reward:.4f} final_grade={final_score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="sandbox_cleanup")
    args = parser.parse_args()
    run_inference(args.task)
