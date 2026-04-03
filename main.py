# pyre-ignore-all-errors
"""
FastAPI server — exposes OpenEnv-compliant REST API:
  POST /reset       → Observation + info
  POST /step        → Observation + reward + done + info
  GET  /state       → InternalState
  GET  /grade       → GradeResult
  POST /run_baseline → runs full baseline agent episode
  GET  /health      → {"status": "ok"}
"""

from __future__ import annotations
import os
from typing import Any, Dict, Optional

# pyre-ignore[21]
from fastapi import FastAPI, HTTPException
# pyre-ignore[21]
from fastapi.responses import RedirectResponse
# pyre-ignore[21]
from fastapi.middleware.cors import CORSMiddleware
# pyre-ignore[21]
from pydantic import BaseModel

from env import FinOpsEnv  # type: ignore
from schemas.models import Action, ActionType  # type: ignore
from graders.graders import grade  # type: ignore

app = FastAPI(
    title="FinOps Cloud Optimizer — OpenEnv",
    description="Production-grade RL environment for cloud infrastructure cost optimization.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global env instance (single-session for demo; production should use sessions)
_env: Optional[FinOpsEnv] = None


# ---------------------------------------------------------------------------
# Request / Response schemas for API
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "sandbox_cleanup"


class StepRequest(BaseModel):
    action_type: str
    target_resource: Optional[str] = None
    parameters: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def read_root():
    """Redirect root to API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/reset")
def reset(req: ResetRequest):
    global _env
    try:
        _env = FinOpsEnv(task_id=req.task_id)
        obs, info = _env.reset()
        return {"observation": obs.model_dump(), "info": info}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(req: StepRequest):
    global _env
    env = _env
    if env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    try:
        action = Action(
            action_type=ActionType(req.action_type),
            target_resource=req.target_resource,
            parameters=req.parameters,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid action: {e}")
    try:
        obs, reward, done, info = env.step(action)  # type: ignore
        return {
            "observation": obs.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def get_state():
    global _env
    env = _env
    if env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    return env.state().model_dump()


@app.get("/grade")
def get_grade():
    global _env
    env = _env
    if env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    state = env.state()  # type: ignore
    result = grade(state.task_id, state)
    return {
        "total_score": result.total_score,
        "breakdown": result.breakdown,
        "bonuses": result.bonuses,
        "notes": result.notes,
    }


@app.post("/run_baseline")
def run_baseline(req: ResetRequest):
    """Run a simple heuristic baseline agent and return the episode result."""
    env = FinOpsEnv(task_id=req.task_id)
    global _env
    _env = env
    obs, info_reset = env.reset()
    max_steps: int = int(info_reset.get("max_steps", 30))

    history = []
    step_num = 0
    done = False

    # Phase 1: Analyze all resources
    for r in obs.resources:  # type: ignore
        if step_num >= max_steps - 2 or done:
            break
        action = Action(action_type=ActionType.ANALYZE, target_resource=r.resource_id)
        obs, reward, done, info = env.step(action)  # type: ignore
        history.append({"action": "analyze", "target": r.resource_id, "reward": reward})
        step_num += 1  # type: ignore

    # Phase 2: Terminate idle resources with no dependents
    for r in obs.resources:  # type: ignore
        if step_num >= max_steps - 2 or done:
            break
        env_state = env.state()  # type: ignore
        no_deps = not env_state.dependency_graph.get(r.resource_id)
        if r.avg_cpu_utilization < 0.05 and r.monthly_cost > 0 and no_deps:
            action = Action(action_type=ActionType.TERMINATE, target_resource=r.resource_id)
            obs, reward, done, info = env.step(action)  # type: ignore
            history.append({"action": "terminate", "target": r.resource_id,
                            "reward": reward, "savings": info["savings_this_step"]})
            step_num += 1  # type: ignore

    # Phase 3: Rightsize low-cpu compute resources
    from env import RIGHTSIZE_COST_MAP  # type: ignore
    for r in obs.resources:  # type: ignore
        if step_num >= max_steps - 2 or done:
            break
        if r.resource_type == "ec2" and r.avg_cpu_utilization < 0.15 and r.monthly_cost > 80:
            cheaper: Dict[str, float] = {
                k: v for k, v in RIGHTSIZE_COST_MAP.items()  # type: ignore
                if v is not None and v < r.monthly_cost * 0.7 and v > 0 and "db." not in k
            }
            if cheaper:
                new_config = min(cheaper.keys(), key=lambda k: cheaper[k])
                action = Action(
                    action_type=ActionType.RIGHTSIZE,
                    target_resource=r.resource_id,
                    parameters={"new_config": new_config},
                )
                obs, reward, done, info = env.step(action)  # type: ignore
                history.append({"action": "rightsize", "target": r.resource_id,
                                "new_config": new_config, "reward": reward})
                step_num += 1  # type: ignore

    # Finalize
    if not done:
        action = Action(action_type=ActionType.FINALIZE_PLAN)
        obs, reward, done, info = env.step(action)  # type: ignore
        history.append({"action": "finalize", "reward": reward})

    state = env.state()  # type: ignore
    result = grade(state.task_id, state)

    return {
        "task_id": req.task_id,
        "steps_taken": step_num + 1,  # type: ignore
        "savings_achieved": state.savings_achieved,
        "incidents_caused": state.incidents_caused,
        "grade": {
            "total_score": result.total_score,
            "breakdown": result.breakdown,
            "notes": result.notes,
        },
        "episode_history": history,
    }
