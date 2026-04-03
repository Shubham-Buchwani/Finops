# pyre-ignore-all-errors
"""
Deterministic graders for all 3 tasks.
Input: InternalState
Output: score in [0.0, 1.0] with breakdown
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

# pyre-ignore[21]
from schemas.models import InternalState


@dataclass
class GradeResult:
    total_score: float
    coverage: float
    safety: float
    efficiency: float
    diligence: float
    bonuses: float
    breakdown: Dict[str, float]
    notes: List[str] = field(default_factory=list)


def _base_grade(state: InternalState) -> GradeResult:
    """Shared 4-component grader logic."""
    budget_target = state.total_monthly_spend * state.budget_target_pct
    actions = state.actions_taken
    total_actions = len(actions)

    # ---- Coverage ----
    coverage: float = (
        min(1.0, state.savings_achieved / budget_target) if budget_target > 0 else 0.0
    )

    # ---- Safety ----
    safety: float = max(0.0, 1.0 - (state.incidents_caused * 0.4))

    # ---- Efficiency ----
    wasted_count: int = 0
    for a in actions:
        if a.result in ("error", "no_effect"):
            wasted_count = wasted_count + 1  # pyre-ignore[58]
    # Count duplicate analyses as waste
    analyzed_ids: List[str] = []
    for a in actions:
        if a.action_type == "analyze" and a.target_resource:
            if a.target_resource in analyzed_ids:
                wasted_count = wasted_count + 1  # pyre-ignore[58]
            else:
                analyzed_ids.append(a.target_resource)
                
    safe_total = int(total_actions) if total_actions > 0 else 1
    effic_ratio = float(wasted_count) / float(safe_total)
    efficiency: float = (
        float(max(0.0, min(1.0, 1.0 - effic_ratio))) if total_actions > 0 else 0.0
    )

    # ---- Diligence ----
    modified = set(state.resources_modified)
    analyzed_before_modify = int(sum(
        1 for rid in modified if rid in state.resources_analyzed
    ))
    safe_mod = int(len(modified)) if modified else 1
    diligence: float = (
        float(analyzed_before_modify) / float(safe_mod) if modified else 1.0
    )
    diligence = float(max(0.0, min(1.0, diligence)))

    # Weights
    w_cov, w_safe, w_eff, w_dil = 0.40, 0.30, 0.15, 0.15
    base: float = (
        w_cov * coverage + w_safe * safety + w_eff * efficiency + w_dil * diligence
    )
    base = max(0.0, min(1.0, base))

    return GradeResult(
        total_score=base,
        coverage=coverage,
        safety=safety,
        efficiency=efficiency,
        diligence=diligence,
        bonuses=0.0,
        breakdown={
            "coverage": float(f"{coverage:.4f}"),
            "safety": float(f"{safety:.4f}"),
            "efficiency": float(f"{efficiency:.4f}"),
            "diligence": float(f"{diligence:.4f}"),
        },
        notes=[],
    )


def grade_easy(state: InternalState) -> GradeResult:
    """Task 1: Sandbox Cleanup grader."""
    if len(state.actions_taken) == 0:
        return GradeResult(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, ["No actions taken."]
        )
    result = _base_grade(state)
    notes = list(result.notes)

    # Trap: incorrectly terminated staging EC2 (bursty CI/CD)
    staging_terminate = any(
        a for a in state.actions_taken
        if a.target_resource == "ec2-staging-01" and a.action_type == "terminate"
    )
    if staging_terminate:
        notes.append("Trap triggered: incorrectly terminated bursty staging EC2.")

    result.notes = notes
    result.total_score = float(f"{max(0.0, min(1.0, result.total_score)):.4f}")
    return result


def grade_medium(state: InternalState) -> GradeResult:
    """Task 2: Cross-Account Right-Sizing grader."""
    if len(state.actions_taken) == 0:
        return GradeResult(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, ["No actions taken."]
        )
    result = _base_grade(state)
    bonuses: float = 0.0
    notes = list(result.notes)

    # Bonus: renewed both expiring reserved instances
    ri_renewals = [
        a for a in state.actions_taken if a.action_type == "reserve"
    ]
    if len(ri_renewals) >= 2:
        bonuses += 0.05
        notes.append("+0.05 bonus: Renewed expiring Reserved Instances.")

    # Trap: rightsized production RDS (high p95 disguised as idle)
    prod_rds_rightsize = any(
        a for a in state.actions_taken
        if a.target_resource == "rds-prod-master-01" and a.action_type == "rightsize"
    )
    if prod_rds_rightsize:
        notes.append("Trap triggered: prod RDS has p95 CPU=78% — rightsizing caused incident.")

    result.bonuses = bonuses
    result.total_score = float(f"{max(0.0, min(1.0, result.total_score + bonuses)):.4f}")
    result.notes = notes
    return result


def grade_hard(state: InternalState) -> GradeResult:
    """Task 3: Enterprise Transformation grader."""
    if len(state.actions_taken) == 0:
        return GradeResult(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, ["No actions taken."]
        )
    result = _base_grade(state)
    bonuses: float = 0.0
    notes = list(result.notes)

    # Bonus: used >= 5 different action types (strategy diversity)
    used_types = set(a.action_type for a in state.actions_taken)
    if len(used_types) >= 5:
        bonuses += 0.05
        notes.append(f"+0.05 bonus: Used {len(used_types)} different action types.")

    # Bonus: avoided mislabeled dev account resources
    mislabeled_terminated = any(
        a for a in state.actions_taken
        if a.target_resource in ("ec2-internal-tools-01", "rds-internal-apps-01")
        and a.action_type == "terminate"
    )
    if not mislabeled_terminated:
        bonuses += 0.03
        notes.append("+0.03 bonus: Correctly avoided mislabeled 'dev' account resources.")

    # Bonus: flagged NAT gateway for architectural review
    nat_flagged = any(
        a for a in state.actions_taken
        if a.action_type == "flag"
        and a.target_resource in ("nat-gateway-prod-01", "nat-gateway-shared-01")
    )
    if nat_flagged:
        bonuses += 0.02
        notes.append("+0.02 bonus: Flagged NAT Gateway for VPC endpoint migration.")

    result.bonuses = bonuses
    result.total_score = float(f"{max(0.0, min(1.0, result.total_score + bonuses)):.4f}")
    result.notes = notes
    return result


TASK_GRADERS = {
    "sandbox_cleanup": grade_easy,
    "cross_account_rightsizing": grade_medium,
    "enterprise_transformation": grade_hard,
}


def grade(task_id: str, state: InternalState) -> GradeResult:
    grader = TASK_GRADERS.get(task_id)
    if grader is None:
        raise ValueError(f"No grader for task: {task_id}")
    return grader(state)
