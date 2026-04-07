"""
Typed Pydantic models for the FinOps Cloud Optimizer OpenEnv environment.
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
# pyre-ignore[21]
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Resource & Utilization Models
# ---------------------------------------------------------------------------

class UtilizationDetail(BaseModel):
    """Detailed metrics — only revealed after ANALYZE action."""
    cpu_p50: float = Field(..., ge=0.0, le=1.0)
    cpu_p95: float = Field(..., ge=0.0, le=1.0)
    cpu_max: float = Field(..., ge=0.0, le=1.0)
    memory_avg: float = Field(..., ge=0.0, le=1.0)
    memory_p95: float = Field(..., ge=0.0, le=1.0)
    network_in_gb: float
    network_out_gb: float
    iops_avg: float
    storage_used_pct: float = Field(default=0.0, ge=0.0, le=1.0)
    last_accessed_days_ago: Optional[int] = None
    connection_count_avg: Optional[int] = None
    invocation_count_30d: Optional[int] = None
    dependencies_known: List[str] = Field(default_factory=list)


class ResourceSummary(BaseModel):
    """What the agent sees about each resource (possibly partial)."""
    resource_id: str
    resource_type: str  # ec2, rds, s3, lambda, elb, eks, elasticache, sagemaker, nat
    name: str
    region: str
    account: str
    environment: str  # production, staging, development, sandbox
    instance_config: str  # e.g. "m5.2xlarge" or "db.r5.large"
    monthly_cost: float
    avg_cpu_utilization: float = Field(..., ge=0.0, le=1.0)
    tags: Dict[str, str] = Field(default_factory=dict)
    analyzed: bool = False
    analysis_detail: Optional[UtilizationDetail] = None


# ---------------------------------------------------------------------------
# Action Models
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    # Investigation
    ANALYZE = "analyze"
    CHECK_DEPENDENCIES = "check_deps"
    # Optimization
    RIGHTSIZE = "rightsize"
    TERMINATE = "terminate"
    SCHEDULE = "schedule"
    RESERVE = "reserve"
    MIGRATE_STORAGE_CLASS = "migrate"
    # Plan
    FLAG_FOR_REVIEW = "flag"
    FINALIZE_PLAN = "finalize"


class Action(BaseModel):
    action_type: ActionType
    target_resource: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Action History Record
# ---------------------------------------------------------------------------

class ActionRecord(BaseModel):
    step: int
    action_type: str
    target_resource: Optional[str]
    parameters: Dict[str, Any]
    result: str  # "success", "warning", "error", "no_effect"
    savings_delta: float = 0.0
    reward: float = 0.0
    message: str = ""


# ---------------------------------------------------------------------------
# Observation Model
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    step_number: int
    max_steps: int
    total_monthly_spend: float
    budget_target_pct: float
    budget_target_dollars: float
    savings_achieved_so_far: float
    incidents_caused: int
    resources: List[ResourceSummary]
    action_history: List[ActionRecord]
    system_message: Optional[str] = None
    done: bool = False


# ---------------------------------------------------------------------------
# Internal State (grader / state() endpoint)
# ---------------------------------------------------------------------------

class OptimalAction(BaseModel):
    action_type: str
    target_resource: str
    parameters: Dict[str, Any]
    expected_savings: float
    rationale: str


class Resource(BaseModel):
    """Full internal resource — superset of ResourceSummary."""
    resource_id: str
    resource_type: str
    name: str
    region: str
    account: str
    environment: str
    instance_config: str
    monthly_cost: float
    avg_cpu_utilization: float
    tags: Dict[str, str] = Field(default_factory=dict)
    utilization: UtilizationDetail
    dependencies: List[str] = Field(default_factory=list)  # IDs this resource depends on
    dependents: List[str] = Field(default_factory=list)    # IDs that depend on this
    is_safe_to_terminate: bool = False
    is_safe_to_rightsize: bool = False
    optimal_action: Optional[OptimalAction] = None


class InternalState(BaseModel):
    scenario_id: str
    task_id: str
    resources: List[Resource]
    dependency_graph: Dict[str, List[str]]  # resource_id → [dependents]
    ground_truth_plan: List[OptimalAction]
    ground_truth_max_savings: float
    budget_target_pct: float
    total_monthly_spend: float
    step_count: int
    max_steps: int
    actions_taken: List[ActionRecord]
    savings_achieved: float
    incidents_caused: int
    resources_analyzed: List[str]
    resources_modified: List[str]
    terminated: bool = False
    done: bool = False
