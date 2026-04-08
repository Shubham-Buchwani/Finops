"""
Core FinOps environment logic — step(), reset(), state() implementation.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# pyre-ignore[21]
from schemas.models import (
    Action, ActionRecord, ActionType, InternalState, Observation,
    OptimalAction, Resource, ResourceSummary, UtilizationDetail,
)


SCENARIO_DIR = Path(__file__).resolve().parent / "scenarios"

TASK_CONFIG: Dict[str, Dict[str, Any]] = {
    "sandbox_cleanup": {
        "scenario_file": "easy_001.json",
        "max_steps": 15,
        "difficulty": "easy",
    },
    "cross_account_rightsizing": {
        "scenario_file": "medium_001.json",
        "max_steps": 30,
        "difficulty": "medium",
    },
    "enterprise_transformation": {
        "scenario_file": "hard_001.json",
        "max_steps": 60,
        "difficulty": "hard",
    },
}

# Monthly cost per config string. None = not applicable for compute resize.
RIGHTSIZE_COST_MAP: Dict[str, Optional[float]] = {
    # EC2
    "m5.large": 70.0,
    "m5.xlarge": 140.0,
    "m5.2xlarge": 280.0,
    "m5.4xlarge": 560.0,
    "m5.8xlarge": 1120.0,
    "t3.medium": 30.0,
    "t3.large": 60.0,
    "t3.xlarge": 120.0,
    "c5.xlarge": 120.0,
    "c5.2xlarge": 240.0,
    "c5.4xlarge": 480.0,
    "r5.large": 110.0,
    "r5.xlarge": 220.0,
    "r5.2xlarge": 440.0,
    "r5.4xlarge": 880.0,
    # RDS
    "db.t3.medium": 50.0,
    "db.r5.large": 175.0,
    "db.r5.xlarge": 350.0,
    "db.r5.2xlarge": 700.0,
    "db.r5.4xlarge": 1400.0,
    # Lambda memory tiers
    "1024MB": None,
    "2048MB": None,
    "3008MB": None,
    # Storage classes (not a resize target in same sense)
    "S3_STANDARD": None,
    "S3_IA": None,
    "Glacier": None,
    "gp3": None,
}

RESERVE_DISCOUNT: Dict[str, float] = {
    "1yr_all_upfront": 0.40,
    "1yr_partial": 0.30,
    "1yr_none": 0.20,
    "3yr_all_upfront": 0.60,
    "3yr_partial": 0.50,
    "3yr_none": 0.40,
}

# Normalized reward weights [0.0, 1.0]
R_SAVINGS_MAX = 0.8
R_INVESTIGATE_WASTE = 0.05
R_INVESTIGATE_ANY = 0.01
R_CHECK_DEP = 0.04
R_INCIDENT = 0.0
R_BLIND_MODIFY = 0.0
R_BLIND_PROD_CHANGE = 0.0
R_TERMINAL_FAIL = 0.0
R_NOOP = 0.0
R_URGENCY = 0.0
R_STEP_COST = 0.0


class FinOpsEnv:
    """FinOps Cloud Optimizer environment."""

    def __init__(self, task_id: str = "sandbox_cleanup") -> None:
        if task_id not in TASK_CONFIG:
            raise ValueError(
                f"Unknown task_id: {task_id}. Choose from {list(TASK_CONFIG.keys())}"
            )
        self.task_id = task_id
        self._state: Optional[InternalState] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> Tuple[Observation, Dict[str, Any]]:
        config = TASK_CONFIG[self.task_id]
        scenario_path = SCENARIO_DIR / str(config["scenario_file"])
        raw: Dict[str, Any] = json.loads(scenario_path.read_text(encoding="utf-8"))

        resources = [Resource(**r) for r in raw["resources"]]
        dep_graph: Dict[str, List[str]] = {
            r.resource_id: r.dependents for r in resources
        }
        gt_plan = [OptimalAction(**p) for p in raw["ground_truth_plan"]]
        total_spend: float = float(raw["total_monthly_spend"])
        bpct: float = float(raw["budget_target_pct"])

        state = InternalState(
            scenario_id=str(raw["scenario_id"]),
            task_id=self.task_id,
            resources=resources,
            dependency_graph=dep_graph,
            ground_truth_plan=gt_plan,
            ground_truth_max_savings=float(raw["ground_truth_max_savings"]),
            budget_target_pct=bpct,
            total_monthly_spend=total_spend,
            step_count=0,
            max_steps=int(config["max_steps"]),
            actions_taken=[],
            savings_achieved=0.0,
            incidents_caused=0,
            resources_analyzed=[],
            resources_modified=[],
            done=False,
        )
        self._state = state
        obs = self._build_observation()
        info: Dict[str, Any] = {
            "task_id": self.task_id,
            "max_steps": state.max_steps,
            "total_monthly_spend": total_spend,
            "budget_target_pct": bpct,
            "num_resources": len(resources),
        }
        return obs, info

    def step(
        self, action: Action
    ) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        s = self._state
        assert s is not None, "Call reset() first."
        if s.done:
            raise RuntimeError("Episode already done. Call reset().")

        result_msg, savings_delta, reward, valid = self._execute_action(action, s)
        record = ActionRecord(
            step=s.step_count,
            action_type=action.action_type.value,
            target_resource=action.target_resource,
            parameters=action.parameters,
            result=(
                "success"
                if valid and savings_delta >= 0
                else ("error" if not valid else "warning")
            ),
            savings_delta=savings_delta,
            reward=float(f"{reward:.4f}"),
            message=result_msg,
        )
        s.actions_taken.append(record)
        s.savings_achieved += savings_delta
        s.step_count += 1

        # Check terminal conditions
        if action.action_type == ActionType.FINALIZE_PLAN:
            s.done = True
        elif s.step_count >= s.max_steps:
            s.done = True
            result_msg += " | Max steps reached."
        elif s.incidents_caused >= 3:
            reward += R_TERMINAL_FAIL
            s.done = True
            result_msg += " | CRITICAL: 3+ incidents. Episode terminated."

        obs = self._build_observation(system_message=result_msg)
        info: Dict[str, Any] = {
            "savings_this_step": savings_delta,
            "total_savings": s.savings_achieved,
            "incidents": s.incidents_caused,
            "step": s.step_count,
            "valid_action": valid,
        }
        # Clamp reward to range specified in openenv.yaml [0.0, 1.0]
        clamped_reward = max(0.0, min(1.0, reward))
        return obs, float(f"{clamped_reward:.4f}"), s.done, info

    def state(self) -> InternalState:
        s = self._state
        assert s is not None, "Call reset() first."
        return s

    # ------------------------------------------------------------------
    # Action Execution
    # ------------------------------------------------------------------

    def _execute_action(
        self, action: Action, s: InternalState
    ) -> Tuple[str, float, float, bool]:
        """Returns (message, savings_delta, reward, is_valid)."""
        reward: float = R_STEP_COST
        atype = action.action_type
        rid = action.target_resource

        # Urgency penalty
        progress_pct = s.step_count / max(s.max_steps, 1)
        budget_target = s.total_monthly_spend * s.budget_target_pct
        if progress_pct > 0.8 and s.savings_achieved < 0.3 * budget_target:
            reward += R_URGENCY

        # --- FINALIZE ---
        if atype == ActionType.FINALIZE_PLAN:
            return "Plan finalized.", 0.0, reward, True

        # --- Validate resource target ---
        resource = self._find_resource(rid, s)
        if rid and resource is None:
            return f"Resource {rid} not found.", 0.0, reward + R_NOOP, False

        # --- ANALYZE ---
        if atype == ActionType.ANALYZE:
            if not rid:
                return "analyze requires target_resource.", 0.0, reward + R_NOOP, False
            if rid in s.resources_analyzed:
                return f"Already analyzed {rid}.", 0.0, reward + R_NOOP, False
            s.resources_analyzed.append(rid)
            is_wasteful = (
                resource is not None
                and (
                    resource.avg_cpu_utilization < 0.10
                    or (
                        resource.utilization.last_accessed_days_ago is not None
                        and resource.utilization.last_accessed_days_ago > 20
                    )
                )
            )
            reward += R_INVESTIGATE_WASTE if is_wasteful else R_INVESTIGATE_ANY
            name = resource.name if resource else rid
            return f"Analyzed {name}. Full utilization data now available.", 0.0, reward, True

        # --- CHECK_DEPENDENCIES ---
        if atype == ActionType.CHECK_DEPENDENCIES:
            if not rid:
                return "check_deps requires target_resource.", 0.0, reward + R_NOOP, False
            dependents = resource.dependents if resource else []
            reward += R_CHECK_DEP
            dep_str = ", ".join(dependents) if dependents else "none"
            return f"Dependencies for {rid}: dependents=[{dep_str}]", 0.0, reward, True

        # From here, resource must be present for modification actions
        if resource is None:
            return f"Resource {rid} not found.", 0.0, reward + R_NOOP, False

        # --- RIGHTSIZE ---
        if atype == ActionType.RIGHTSIZE:
            return self._do_rightsize(action, resource, rid or "", reward, s)

        # --- TERMINATE ---
        if atype == ActionType.TERMINATE:
            return self._do_terminate(resource, rid or "", reward, s)

        # --- SCHEDULE ---
        if atype == ActionType.SCHEDULE:
            return self._do_schedule(action, resource, rid or "", reward, s)

        # --- RESERVE ---
        if atype == ActionType.RESERVE:
            return self._do_reserve(action, resource, rid or "", reward, s)

        # --- MIGRATE_STORAGE_CLASS ---
        if atype == ActionType.MIGRATE_STORAGE_CLASS:
            return self._do_migrate(action, resource, rid or "", reward, s)

        # --- FLAG_FOR_REVIEW ---
        if atype == ActionType.FLAG_FOR_REVIEW:
            reward += 0.1
            return f"Flagged {rid or 'item'} for human review.", 0.0, reward, True

        return "Unknown action.", 0.0, reward + R_NOOP, False

    # ------------------------------------------------------------------
    # Sub-action helpers
    # ------------------------------------------------------------------

    def _do_rightsize(
        self,
        action: Action,
        resource: Resource,
        rid: str,
        reward: float,
        s: InternalState,
    ) -> Tuple[str, float, float, bool]:
        new_config = action.parameters.get("new_config", "")
        if not new_config or new_config not in RIGHTSIZE_COST_MAP:
            return f"Invalid new_config: {new_config}", 0.0, reward + R_NOOP, False
        if rid in s.resources_modified:
            return f"{rid} already modified.", 0.0, reward + R_NOOP, False

        if rid not in s.resources_analyzed:
            reward += R_BLIND_MODIFY
        if resource.environment == "production":
            reward += R_BLIND_PROD_CHANGE

        old_cost = resource.monthly_cost
        new_cost_raw = RIGHTSIZE_COST_MAP.get(new_config)
        # For Lambda/storage, approximate: use 1/3 of old cost as new cost
        new_cost: float = new_cost_raw if new_cost_raw is not None else old_cost * 0.34
        savings = old_cost - new_cost

        s.resources_modified.append(rid)
        resource.instance_config = new_config
        resource.monthly_cost = new_cost

        if savings < 0:
            reward += 2.0 * (abs(savings) / max(s.ground_truth_max_savings, 1))
            return (
                f"Rightsized {rid} to {new_config} — INCREASED cost by ${abs(savings):.0f}!",
                0.0,
                reward,
                True,
            )

        savings_reward = (savings / max(s.ground_truth_max_savings, 1)) * R_SAVINGS_MAX
        reward += savings_reward

        # Incident check: production resource with high p95
        if (
            resource.environment == "production"
            and resource.utilization.cpu_p95 > 0.70
            and new_cost < old_cost
        ):
            s.incidents_caused += 1
            reward += R_INCIDENT
            return (
                f"INCIDENT: Rightsized production {rid} but p95 CPU was "
                f"{resource.utilization.cpu_p95:.0%}! SLA degradation.",
                savings,
                reward,
                True,
            )

        return (
            f"Rightsized {rid} to {new_config}. Saving ${savings:.0f}/month.",
            savings,
            reward,
            True,
        )

    def _do_terminate(
        self,
        resource: Resource,
        rid: str,
        reward: float,
        s: InternalState,
    ) -> Tuple[str, float, float, bool]:
        if rid in s.resources_modified:
            return f"{rid} already modified.", 0.0, reward + R_NOOP, False
        if rid not in s.resources_analyzed:
            reward += R_BLIND_MODIFY
        if resource.environment == "production":
            reward += R_BLIND_PROD_CHANGE

        savings = resource.monthly_cost if resource.is_safe_to_terminate else 0.0
        s.resources_modified.append(rid)

        if not resource.is_safe_to_terminate:
            s.incidents_caused += 1
            reward += R_INCIDENT
            resource.monthly_cost = 0.0
            return (
                f"INCIDENT: Terminated {rid} ({resource.environment}) — NOT safe!",
                savings,
                reward,
                True,
            )
        else:
            savings_reward = (savings / max(s.ground_truth_max_savings, 1)) * R_SAVINGS_MAX
            reward += savings_reward
            resource.monthly_cost = 0.0
            return f"Terminated {rid}. Saving ${savings:.0f}/month.", savings, reward, True

    def _do_schedule(
        self,
        action: Action,
        resource: Resource,
        rid: str,
        reward: float,
        s: InternalState,
    ) -> Tuple[str, float, float, bool]:
        if rid in s.resources_modified:
            return f"{rid} already modified.", 0.0, reward + R_NOOP, False
        if rid not in s.resources_analyzed:
            reward += R_BLIND_MODIFY
        active_hours_str = str(action.parameters.get("active_hours", "08:00-20:00"))
        savings_pct = 0.35 if "-" in active_hours_str else 0.50
        savings = resource.monthly_cost * savings_pct
        savings_reward = (savings / max(s.ground_truth_max_savings, 1)) * R_SAVINGS_MAX
        reward += savings_reward
        s.resources_modified.append(rid)
        resource.monthly_cost -= savings
        return (
            f"Scheduled {rid} to {active_hours_str}. Saving ${savings:.0f}/month.",
            savings,
            reward,
            True,
        )

    def _do_reserve(
        self,
        action: Action,
        resource: Resource,
        rid: str,
        reward: float,
        s: InternalState,
    ) -> Tuple[str, float, float, bool]:
        if rid in s.resources_modified:
            return f"{rid} already reserved/modified.", 0.0, reward + R_NOOP, False
        term = str(action.parameters.get("term", "1yr"))
        payment = str(action.parameters.get("payment", "all_upfront")).replace("-", "_")
        key = f"{term}_{payment}"
        discount = RESERVE_DISCOUNT.get(key, 0.30)
        savings = resource.monthly_cost * discount
        savings_reward = (savings / max(s.ground_truth_max_savings, 1)) * R_SAVINGS_MAX
        reward += savings_reward
        s.resources_modified.append(rid)
        resource.monthly_cost -= savings
        return (
            f"Reserved {rid} ({term}, {payment}). Saving ${savings:.0f}/month.",
            savings,
            reward,
            True,
        )

    def _do_migrate(
        self,
        action: Action,
        resource: Resource,
        rid: str,
        reward: float,
        s: InternalState,
    ) -> Tuple[str, float, float, bool]:
        if rid in s.resources_modified:
            return f"{rid} already modified.", 0.0, reward + R_NOOP, False
        new_class = str(action.parameters.get("new_class", "S3_IA"))
        class_savings: Dict[str, float] = {
            "S3_IA": 0.40,
            "Glacier": 0.72,
            "gp3": 0.20,
        }
        savings_pct = class_savings.get(new_class, 0.20)
        savings = resource.monthly_cost * savings_pct
        savings_reward = (savings / max(s.ground_truth_max_savings, 1)) * R_SAVINGS_MAX
        reward += savings_reward
        s.resources_modified.append(rid)
        resource.monthly_cost -= savings
        resource.instance_config = new_class
        return (
            f"Migrated {rid} to {new_class}. Saving ${savings:.0f}/month.",
            savings,
            reward,
            True,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_resource(
        self, rid: Optional[str], s: InternalState
    ) -> Optional[Resource]:
        if rid is None:
            return None
        for r in s.resources:
            if r.resource_id == rid:
                return r
        return None

    def _build_observation(
        self, system_message: Optional[str] = None
    ) -> Observation:
        s = self._state
        assert s is not None, "No state — call reset() first."
        summaries: List[ResourceSummary] = []
        for r in s.resources:
            detail: Optional[UtilizationDetail] = None
            if r.resource_id in s.resources_analyzed:
                detail = r.utilization
            summaries.append(
                ResourceSummary(
                    resource_id=r.resource_id,
                    resource_type=r.resource_type,
                    name=r.name,
                    region=r.region,
                    account=r.account,
                    environment=r.environment,
                    instance_config=r.instance_config,
                    monthly_cost=r.monthly_cost,
                    avg_cpu_utilization=r.avg_cpu_utilization,
                    tags=r.tags,
                    analyzed=r.resource_id in s.resources_analyzed,
                    analysis_detail=detail,
                )
            )
        budget_target = s.total_monthly_spend * s.budget_target_pct
        return Observation(
            step_number=s.step_count,
            max_steps=s.max_steps,
            total_monthly_spend=s.total_monthly_spend,
            budget_target_pct=s.budget_target_pct,
            budget_target_dollars=float(f"{budget_target:.2f}"),
            savings_achieved_so_far=float(f"{s.savings_achieved:.2f}"),
            incidents_caused=s.incidents_caused,
            resources=summaries,
            action_history=s.actions_taken[-10:],
            system_message=system_message,
            done=s.done,
        )
