"""
Domain-agnostic contract schema for the grader.

A Contract centralizes every piece of domain knowledge so the engine
(grading.py, generators.py, judges.py) never hardcodes domain vocabulary.
Swap the Contract → grade a different agent with zero engine changes.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, computed_field


# --------------------------------------------------------------------------- #
# Enums (moved here from dataset_models.py)                                    #
# --------------------------------------------------------------------------- #

class ArgSource(str, Enum):
    """Where a (gold) arg value came from. This is the arg-level localizer."""
    from_user = "from_user"               # extracted from a user message
    from_tool_result = "from_tool_result" # carried from a prior observation
    from_context = "from_context"         # pulled from session/customer context
    computed = "computed"                 # derived (see compute_type_hint)
    missing = "missing"                   # required but unavailable -> must clarify
    fabricated = "fabricated"             # FAILURE-ONLY: value with no valid source


class Behavior(str, Enum):
    act = "act"          # calls a tool
    clarify = "clarify"  # asks the user for a missing arg
    respond = "respond"  # final natural-language reply, no tool


# --------------------------------------------------------------------------- #
# Tool specifications                                                          #
# --------------------------------------------------------------------------- #

class ToolArgSpec(BaseModel):
    """Specification for a single argument of a tool."""
    model_config = ConfigDict(extra="forbid")

    required: bool = True
    provenance_hint: Optional[str] = None    # "from_user" | "from_context" | "computed" | ...
    compute_type_hint: Optional[str] = None  # domain compute-type name, e.g. "relative_time"
    fail_bucket: Optional[str] = None        # arg_fail bucket name, e.g. "relative_time_resolution"
                                             # Drives measured_knobs without drift


class ToolSpec(BaseModel):
    """Specification for a single tool in the contract."""
    model_config = ConfigDict(extra="forbid")

    args: dict[str, ToolArgSpec]
    applicable_statuses: list[str] = ["ok"]  # result statuses the generator tries for this tool
    prerequisite_tools: list[str] = []       # tools whose result is checked before calling this one


# --------------------------------------------------------------------------- #
# Contract — the domain-global specification                                   #
# --------------------------------------------------------------------------- #

class Contract(BaseModel):
    """All domain knowledge the grader needs, in one swappable object."""
    model_config = ConfigDict(extra="forbid")

    role_description: str                    # fills the judge's system prompt
    tools: dict[str, ToolSpec]
    terminal_tools: list[str]                # tools that produce the "final effect"
    success_lexicon: list[str]               # marker strings for success claims
    success_fields: list[str]                # tool-result keys carrying a STRING success token
    language: str                            # default response language code
    outcome_predicates: list[str]            # allowed must_not_happen names

    @computed_field                          # Pydantic v2; never hand-listed -> no drift
    @property
    def resolution_buckets(self) -> list[str]:
        # Catch 2b: filter on compute_type_hint, not fail_bucket — the arg_resolution
        # injection knob only corrupts computed args, so measured_knobs must sum
        # exactly the buckets of computed args. Filtering on fail_bucket would pull
        # in from_user/from_context buckets the knob never injects — silent desync.
        return list(dict.fromkeys(
            spec.fail_bucket
            for tool in self.tools.values()
            for spec in tool.args.values()
            if spec.compute_type_hint is not None and spec.fail_bucket is not None
        ))
