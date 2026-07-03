"""localize — grader for task-agent eval."""
from .contract import Contract, ToolArgSpec, ToolSpec
from .dataset_models import DatasetRow
from .grading import (
    GradeReport,
    Finding,
    ObservedStep,
    ObservedTrajectory,
    ResponseContext,
    aggregate,
    grade,
    heuristic_response_judge,
    per_layer_stats,
    render_text_report,
)
from .generators import build_dataset, generate_scenario_grid, validate
from .judges import LLMResponseJudge, default_judge

from typing import Callable
ResponseJudge = Callable[[ResponseContext], list[Finding]]

__all__ = [
    "Contract", "ToolArgSpec", "ToolSpec",
    "DatasetRow",
    "GradeReport", "Finding",
    "ObservedStep", "ObservedTrajectory",
    "ResponseContext", "ResponseJudge",
    "aggregate", "grade", "heuristic_response_judge",
    "per_layer_stats", "render_text_report",
    "build_dataset", "generate_scenario_grid", "validate",
    "LLMResponseJudge", "default_judge",
]
