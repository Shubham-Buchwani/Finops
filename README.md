---
title: FinOps Cloud Optimizer
emoji: 💰
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# FinOps Cloud Optimizer — OpenEnv Environment


> **A production-grade RL environment for cloud infrastructure cost optimization.**
> Evaluates AI agents on multi-step reasoning, numerical analysis, constraint-aware planning, and safety-first decision-making.

---

## Problem

Cloud waste costs enterprises **$17.6B/year** (Flexera 2024). Traditional FinOps tools are rule-based and miss context-dependent decisions. This environment challenges AI agents to perform real-world FinOps:

1. Analyze resource utilization (average vs. p95 traps)
2. Identify waste across multi-account, multi-region infrastructure
3. Propose safe optimizations (rightsize, terminate, schedule, reserve, migrate)
4. Maintain production SLAs — **breaking prod is heavily penalized**

No existing RL/AI benchmark covers this domain.

---

## Why It Matters

- **Direct dollar-value outcomes** — not proxy metrics
- **Tests investigation-first reasoning** — agents must ANALYZE before acting
- **Safety-aware optimization** — production incidents reduce score by 40% each
- **Multi-step planning** — 15–60 step episodes with dependency chains and traps
- **Real-world deployable** — companies could use this to evaluate their FinOps AI tools today

---

## Environment API

The environment runs as a FastAPI server:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/reset` | POST | Start a new episode. Body: `{"task_id": "..."}` |
| `/step` | POST | Take an action. Body: `{"action_type": "...", "target_resource": "...", "parameters": {}}` |
| `/state` | GET | Get full internal state (for graders/debugging) |
| `/grade` | GET | Get deterministic grade for current episode |
| `/run_baseline` | POST | Run the heuristic baseline agent |

---

## Action Space

| Action Type | Description | Required Parameters |
|---|---|---|
| `analyze` | Reveal full utilization metrics for a resource | `target_resource` |
| `check_deps` | Query dependency graph for a resource | `target_resource` |
| `rightsize` | Change instance size | `target_resource`, `parameters.new_config` |
| `terminate` | Shut down resource | `target_resource` |
| `schedule` | Set active hours for a resource | `target_resource`, `parameters.active_hours` |
| `reserve` | Purchase reserved capacity | `target_resource`, `parameters.term`, `parameters.payment` |
| `migrate` | Change storage class | `target_resource`, `parameters.new_class` |
| `flag` | Flag for human architectural review | `target_resource` |
| `finalize` | Submit plan and end episode | — |

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `step_number` | int | Current step |
| `max_steps` | int | Episode step budget |
| `total_monthly_spend` | float | Total current monthly cost |
| `budget_target_pct` | float | Target reduction percentage |
| `savings_achieved_so_far` | float | Savings accumulated this episode |
| `incidents_caused` | int | Production incidents triggered |
| `resources` | list | Resource inventory (partial visibility) |
| `action_history` | list | Last 10 actions + results |
| `system_message` | string | Feedback from last action |

> **Partial observability**: Detailed utilization metrics (p95, memory, IOPS) for each resource are hidden until the agent calls `ANALYZE` on that resource.

---

## Tasks

### Task 1 — Sandbox Cleanup (Easy)
- **Scope**: 1 AWS account, 1 region, 8 resources
- **Spend**: $4,200/month | **Target**: -15% ($630)
- **Optimal savings**: $890
- **Max steps**: 15
- **Trap**: Staging EC2 has low *average* CPU but high p95 from CI/CD — don't terminate it, rightsize it.

### Task 2 — Cross-Account Right-Sizing (Medium)
- **Scope**: 3 AWS accounts, 2 regions, 20 resources
- **Spend**: $28,500/month | **Target**: -20% ($5,700)
- **Optimal savings**: $7,200
- **Max steps**: 30
- **Traps**: (1) Production RDS looks idle on average but p95 is 78% during nightly batch. (2) Two reserved instances expire in 30 days — proactively identify and renew.

### Task 3 — Enterprise Transformation (Hard)
- **Scope**: 5 AWS accounts, 3 regions, 45 resources
- **Spend**: $142,000/month | **Target**: -25% ($35,500)
- **Optimal savings**: $48,000
- **Max steps**: 60
- **Traps**: (1) Mislabeled "development" account serves production tools. (2) GPU instances need scheduling, not termination. (3) NAT Gateway can be replaced with VPC endpoints. (4) Circular dependency between cross-account resources.

---

## Reward Design

Dense per-step reward:
```
R = R_savings + R_investigation + R_safety + R_efficiency
```

| Signal | Value |
|---|---|
| Savings produced | `+(savings/max) × 5.0` |
| Cost increased | `-2.0 × (increase/max)` |
| ANALYZE on wasteful resource | `+0.3` |
| CHECK_DEPS before production change | `+0.2` |
| Production incident | `-3.0` |
| 3+ incidents (critical) | `-5.0` (terminal) |
| Blind modification | `-0.5` |
| No-op / duplicate | `-0.1` |
| Per-step cost | `-0.02` |

---

## Grading

Each episode is scored 0.0–1.0:

```
Score = 0.40×Coverage + 0.30×Safety + 0.15×Efficiency + 0.15×Diligence
```

| Component | Formula |
|---|---|
| Coverage | `min(1.0, savings / target)` |
| Safety | `max(0.0, 1.0 - incidents × 0.4)` |
| Efficiency | `1.0 - (wasted_actions / total_actions)` |
| Diligence | `resources_analyzed_before_modify / resources_modified` |

Task-specific bonus modifiers reward proactive optimization (RI renewal, architectural flags, trap avoidance).

---

## Setup

### Prerequisites
- Python 3.11+
- Docker (for containerized deployment)

### Local Setup

```bash
git clone <repo>
cd finops-env
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t finops-env .
docker run -p 7860:7860 finops-env
```

---

## Running the Baseline Agent

```bash
export API_BASE_URL=https://api.openai.com/v1
export OPENAI_API_KEY=sk-...
export MODEL_NAME=gpt-4o-mini
export ENV_URL=http://localhost:7860

python run_baseline.py --task sandbox_cleanup
python run_baseline.py --task cross_account_rightsizing
python run_baseline.py --task enterprise_transformation
```

---

## Baseline Results

| Task | Expected Score | Key Failure Modes |
|---|---|---|
| Easy | ~0.72 | Staging EC2 trap; duplicate analyses |
| Medium | ~0.47 | p95 CPU trap; misses RI renewal |
| Hard | ~0.22 | Mislabeled account; circular deps; scale |

---

## Hugging Face Space Deployment

Set environment variables in Space settings:
- `API_BASE_URL` — LLM inference endpoint
- `MODEL_NAME` — Model identifier
- `HF_TOKEN` — HuggingFace token
- `OPENENV_TASK` — Task to run

Space type: **Docker** | Port: **7860**

---

## Validation

```bash
# Validate openenv.yaml
openenv validate openenv.yaml

# Quick smoke test
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id":"sandbox_cleanup"}'
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"action_type":"analyze","target_resource":"ec2-sandbox-dev-01","parameters":{}}'
curl http://localhost:7860/grade
```

- All 3 tasks run in **< 3.5 minutes** total
- Environment memory: **< 160MB RAM** (pure Python, no ML models)
- Runs comfortably on **2 vCPU / 8GB RAM**
