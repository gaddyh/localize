"""
Pydantic v2 models for the ReAct appointment-booking eval dataset.

Three independent localization layers are encoded here:
  1. behavior   -> act / clarify / respond            (Step discriminated union)
  2. args       -> value + provenance + fail_bucket    (ArgSpec)
  3. response   -> speech_act + grounding + screens_for (ResponseCheck)

Load a row with:  DatasetRow.model_validate(json_obj)
Validators reject rows that violate the schema's localization invariants,
so bad data fails at parse time instead of skewing your aggregate metrics.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contract import ArgSource, Behavior


# --------------------------------------------------------------------------- #
# Arg-level provenance                                                         #
# --------------------------------------------------------------------------- #

class ArgSpec(BaseModel):
    """A single expected argument, tagged with its provenance so a wrong value
    can be filed into a fail_bucket instead of a generic 'arg mismatch'."""
    model_config = ConfigDict(extra="forbid")

    value: Optional[Any] = None
    source: ArgSource
    source_detail: Optional[str] = None
    compute_type: Optional[str] = None
    raw_span: Optional[str] = None          # the literal user text, e.g. "השבוע"
    fail_bucket: Optional[str] = None        # counter key, e.g. "relative_time_resolution"

    @model_validator(mode="after")
    def _check_provenance(self) -> "ArgSpec":
        if self.source == ArgSource.computed and self.compute_type is None:
            raise ValueError(
                "source='computed' requires a compute_type "
                "so the failure is localizable."
            )
        if self.source != ArgSource.computed and self.compute_type is not None:
            raise ValueError("compute_type is only valid when source='computed'.")
        if self.source == ArgSource.missing and self.value not in (None, ""):
            raise ValueError("source='missing' must not carry a concrete value.")
        return self


# --------------------------------------------------------------------------- #
# Forbidden block (eager-acting / wrong-tool / wrong-value traps)              #
# --------------------------------------------------------------------------- #

class Forbidden(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tools: list[str] = Field(default_factory=list)        # forbidden tool calls
    args: list[str] = Field(default_factory=list)         # forbidden arg KEYS (fabrication)
    behaviors: list[Behavior] = Field(default_factory=list)
    args_values: dict[str, list[Any]] = Field(default_factory=dict)  # forbidden VALUES per arg
    reason: str


# --------------------------------------------------------------------------- #
# Response-level checks (the communication layer)                             #
# --------------------------------------------------------------------------- #

class ResponseCheck(BaseModel):
    """Grades the text the user actually sees, independent of the tool call."""
    model_config = ConfigDict(extra="forbid")

    speech_act: str                          # e.g. report_availability | confirm_booking
    language: str = "he"
    expected_status: Optional[str] = None    # status of the observation this reply reacts to:
                                             # None/"ok" = normal; "empty" = no availability;
                                             # "error" = tool failed. Non-ok -> handling is graded.
    must_reflect: list[str] = Field(default_factory=list)
    must_not_contain: list[str] = Field(default_factory=list)
    grounding: Optional[str] = None          # every fact must trace to an observation
    faithfulness: Optional[str] = None       # claimed action == what actually happened
    screens_for: list[str] = Field(default_factory=list)  # response-side counter buckets


# --------------------------------------------------------------------------- #
# Steps: discriminated union on `behavior`                                     #
# --------------------------------------------------------------------------- #

class _StepBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int
    user_message: Optional[str] = None   # present when this step is triggered by the user
    gloss: Optional[str] = None
    reacts_to: Optional[str] = None       # e.g. "observation_from_step_1"
    forbidden: Optional[Forbidden] = None


class ActStep(_StepBase):
    behavior: Literal[Behavior.act] = Behavior.act
    tool: str
    args: dict[str, ArgSpec]


class ClarifyStep(_StepBase):
    behavior: Literal[Behavior.clarify] = Behavior.clarify
    tool: None = None
    clarify_target: str                      # which missing arg we're asking for
    response_check: Optional[ResponseCheck] = None


class RespondStep(_StepBase):
    behavior: Literal[Behavior.respond] = Behavior.respond
    tool: None = None
    response_check: ResponseCheck            # required: a reply must be graded


Step = Annotated[
    Union[ActStep, ClarifyStep, RespondStep],
    Field(discriminator="behavior"),
]


# --------------------------------------------------------------------------- #
# Environment (the scripted world the agent reacts to)                         #
# --------------------------------------------------------------------------- #

class CustomerContext(BaseModel):
    model_config = ConfigDict(extra="allow")  # context shape varies by deployment


class ToolOutcome(BaseModel):
    """What a tool returns when called. `returns` is intentionally loose because
    each tool's payload differs (slots vs confirmation_id vs error)."""
    model_config = ConfigDict(extra="forbid")
    returns: dict[str, Any]


class Env(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference_time: Optional[str] = None      # anchor for relative_time args (required when used)
    customer_context: CustomerContext = Field(default_factory=CustomerContext)
    tool_outcomes: dict[str, ToolOutcome] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Outcome check (loose / end-state grader)                                     #
# --------------------------------------------------------------------------- #

class OutcomeCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    final_state: str
    expected_effect: Optional[dict[str, Any]] = None
    must_not_happen: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Root dataset row                                                             #
# --------------------------------------------------------------------------- #

class DatasetRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    intent: str
    turn_pattern: str
    env: Env
    expected_trajectory: list[Step]
    outcome_check: OutcomeCheck

    @model_validator(mode="after")
    def _cross_checks(self) -> "DatasetRow":
        # steps must be 1..N, contiguous, in order
        nums = [s.step for s in self.expected_trajectory]
        if nums != list(range(1, len(nums) + 1)):
            raise ValueError(f"step numbers must be 1..N contiguous, got {nums}")

        for s in self.expected_trajectory:
            # gold trajectories describe correct behavior -> never 'fabricated'
            if isinstance(s, ActStep):
                for name, spec in s.args.items():
                    if spec.source == ArgSource.fabricated:
                        raise ValueError(
                            f"step {s.step} arg '{name}': 'fabricated' is a "
                            "failure-only tag and must not appear in a gold trajectory."
                        )
                    # any relative_time arg needs the env anchor to resolve
                    if spec.compute_type == "relative_time" and not self.env.reference_time:
                        raise ValueError(
                            f"step {s.step} arg '{name}' is relative_time but "
                            "env.reference_time is empty."
                        )
        return self


