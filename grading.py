"""
Observed trajectory + grader for the ReAct booking-agent eval dataset.

dataset_models.py  -> the GOLD (what should happen)
ObservedTrajectory -> what the agent ACTUALLY did (fabricated is allowed here)
grade()            -> walks the two step-by-step and files every discrepancy
                      into a bucket, across all three localization layers:
                        behavior_buckets   (act/clarify/respond errors)
                        arg_fail_buckets    (provenance-aware arg errors)
                        response_buckets    (communication errors ~ screens_for)
                      plus trajectory-level (premature stop / loop) and a loose
                      outcome check.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, field_validator

try:  # works when run as a script from the same directory
    from contract import Contract
    from dataset_models import (
        ActStep,
        ArgSource,
        Behavior,
        ClarifyStep,
        DatasetRow,
        RespondStep,
    )
except ImportError:  # works when imported as part of a package
    from .contract import Contract  # type: ignore
    from .dataset_models import (  # type: ignore
        ActStep,
        ArgSource,
        Behavior,
        ClarifyStep,
        DatasetRow,
        RespondStep,
    )


# --------------------------------------------------------------------------- #
# Observed side: what the agent actually emitted                              #
# --------------------------------------------------------------------------- #

class ObservedArg(BaseModel):
    """An arg the agent actually produced. `source` is OPTIONAL: most harnesses
    only know the value, but if yours can report provenance, the grader will
    catch 'right value, wrong source' (e.g. re-extracted instead of carried)."""
    model_config = ConfigDict(extra="forbid")
    value: Any = None
    source: Optional[ArgSource] = None


class ObservedStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step: int
    behavior: Behavior
    tool: Optional[str] = None
    args: dict[str, ObservedArg] = {}
    response_text: Optional[str] = None

    @field_validator("args", mode="before")
    @classmethod
    def _wrap_scalars(cls, v: Any) -> Any:
        # accept {"date": "2026-06-29"} as shorthand for {"date": {"value": ...}}
        if isinstance(v, dict):
            return {
                k: (val if isinstance(val, dict) else {"value": val})
                for k, val in v.items()
            }
        return v


class ObservedTrajectory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    steps: list[ObservedStep]


# --------------------------------------------------------------------------- #
# Grade report models                                                          #
# --------------------------------------------------------------------------- #

class Failure(BaseModel):
    layer: str          # behavior | tool | arg | response | trajectory | outcome
    code: str           # the bucket this failure is counted under
    detail: str
    step: Optional[int] = None
    arg: Optional[str] = None


class StepGrade(BaseModel):
    step: int
    behavior_expected: Optional[str] = None
    behavior_observed: Optional[str] = None
    behavior_ok: bool = True
    tool_ok: Optional[bool] = None
    arg_checked: bool = False        # gold step was an act -> args graded
    response_checked: bool = False   # gold step had a response_check
    tool_expected: Optional[str] = None   # gold tool for this step (None if not an act)
    tool_observed: Optional[str] = None   # tool the agent actually called (None if no call)
    failures: list[Failure] = []


class GradeReport(BaseModel):
    id: str
    strict_pass: bool          # every step matches gold, no failures anywhere
    outcome_pass: bool         # loose: end state ok, nothing in must_not_happen fired
    step_grades: list[StepGrade]
    trajectory_failures: list[Failure]
    outcome_failures: list[Failure]
    behavior_buckets: dict[str, int]
    arg_fail_buckets: dict[str, int]
    response_buckets: dict[str, int]
    # observation-handling records: one per non-ok respond decision point
    status_handling: list[dict[str, str]] = []

    def all_failures(self) -> list[Failure]:
        out: list[Failure] = list(self.trajectory_failures) + list(self.outcome_failures)
        for sg in self.step_grades:
            out.extend(sg.failures)
        return out


# --------------------------------------------------------------------------- #
# Text helpers (response layer)                                                #
# --------------------------------------------------------------------------- #

def _norm(s: Any) -> str:
    return " ".join(str(s).split()).lower()


def _contains(haystack: Optional[str], needle: Any) -> bool:
    if haystack is None:
        return False
    return _norm(needle) in _norm(haystack)


def _has_script_chars(s: str, language: str) -> bool:
    """Check whether *s* contains characters in the script for *language*.
    Generic replacement for the old _has_hebrew helper."""
    script_ranges = {
        "he": ("\u0590", "\u05FF"),
        "ar": ("\u0600", "\u06FF"),
        "ru": ("\u0400", "\u04FF"),
    }
    lo, hi = script_ranges.get(language, ("\u0000", "\uFFFF"))
    return any(lo <= c <= hi for c in s)


# --------------------------------------------------------------------------- #
# Pluggable response layer: judge the user-facing reply                        #
# --------------------------------------------------------------------------- #
# A "response judge" is any callable ResponseContext -> list[Finding].
# The heuristic judge below is the default (deterministic, free, validated).
# judges.py provides an LLMResponseJudge with the SAME signature, so it drops
# in without touching grade() or anything downstream.

class Finding(BaseModel):
    """One response-layer failure, in the shared bucket vocabulary."""
    model_config = ConfigDict(extra="forbid")
    bucket: str
    detail: str = ""


class ResponseContext(BaseModel):
    """Everything a response judge needs to grade ONE reply. Self-contained so a
    judge (heuristic or LLM) needs no access to the rest of the grader."""
    model_config = ConfigDict(extra="forbid")
    step: int
    text: Optional[str]                 # the reply the agent actually showed the user
    speech_act: str                     # the intended move (report_availability, confirm_booking, ...)
    language: str = "he"
    expected_status: Optional[str] = None   # ok | empty | error
    must_reflect: list[str] = []
    must_not_contain: list[str] = []
    screens_for: list[str] = []
    tool: Optional[str] = None          # the tool whose result this reply reacts to
    observation: dict[str, Any] = {}    # that tool's returned payload (for grounding)
    # shared lexicons so heuristic and LLM judges agree on what counts as a claim
    confirmation_ids: list[str] = []
    forbidden_values: list[str] = []
    booking_markers: list[str] = []


# response buckets that mean "the agent asserted success that didn't happen"
STATUS_MISHANDLE_BUCKETS = {"false_success_claim", "fabricated_on_empty"}


def heuristic_response_judge(ctx: ResponseContext) -> list[Finding]:
    """Deterministic substring/lexicon rules. Default judge; also the regression
    baseline the LLM judge is checked against."""
    text = ctx.text
    if text is None or not text.strip():
        return [Finding(bucket="no_response_text",
                        detail="gold expects a reply but agent produced no text.")]

    out: list[Finding] = []
    conf = {_norm(c) for c in ctx.confirmation_ids}
    forb = {_norm(v) for v in ctx.forbidden_values}
    markers = {_norm(m) for m in ctx.booking_markers}

    for fact in ctx.must_reflect:
        if not _contains(text, fact):
            if _norm(fact) in conf:
                code = "omitted_confirmation_id"
            else:
                code = "omitted_slot" if "omitted_slot" in ctx.screens_for else "omitted_fact"
            out.append(Finding(bucket=code, detail=f"reply omits required fact '{fact}'."))

    for phrase in ctx.must_not_contain:
        if _contains(text, phrase):
            if _norm(phrase) in forb:
                code = "wrong_slot_confirmed"
            elif _norm(phrase) in markers and ctx.speech_act != "confirm_booking":
                code = "unfaithful_action_claim"
            else:
                code = "forbidden_phrase"
            out.append(Finding(bucket=code, detail=f"reply contains forbidden phrase '{phrase}'."))

    if ctx.language and not _has_script_chars(text, ctx.language):
        out.append(Finding(bucket="language_mismatch",
                           detail="reply expected in Hebrew but no Hebrew characters found."))

    if ctx.expected_status in ("empty", "error"):
        leaked = [m for m in markers if _contains(text, m)]
        leaked += [c for c in ctx.confirmation_ids if _contains(text, c)]
        if leaked:
            code = ("fabricated_on_empty" if ctx.expected_status == "empty"
                    else "false_success_claim")
            out.append(Finding(
                bucket=code,
                detail=f"observation was '{ctx.expected_status}' but reply asserts success "
                       f"({', '.join(leaked)})."))
    return out


# --------------------------------------------------------------------------- #
# The grader                                                                   #
# --------------------------------------------------------------------------- #

def grade(expected: DatasetRow, observed: ObservedTrajectory,
          contract: Contract,
          response_judge=heuristic_response_judge) -> GradeReport:
    exp_steps = {s.step: s for s in expected.expected_trajectory}
    obs_steps = {s.step: s for s in observed.steps}

    behavior_ctr: Counter[str] = Counter()
    arg_ctr: Counter[str] = Counter()
    resp_ctr: Counter[str] = Counter()

    # facts that, if forbidden-and-present, mean the agent confirmed the wrong slot
    forbidden_values: set[str] = set()
    for s in expected.expected_trajectory:
        fb = getattr(s, "forbidden", None)
        if fb:
            for vals in fb.args_values.values():
                forbidden_values.update(_norm(v) for v in vals)

    # confirmation ids the env will return (for omitted_confirmation_id detection)
    # Scan tool_outcomes for keys in contract.success_fields where the value is a str
    # (guards against dict-valued fields like "cancelled").
    confirmation_ids: set[str] = set()
    for outcome in expected.env.tool_outcomes.values():
        for k, v in outcome.returns.items():
            if k in contract.success_fields and isinstance(v, str):
                confirmation_ids.add(_norm(v))

    step_grades: list[StepGrade] = []
    traj_failures: list[Failure] = []
    status_handling: list[dict[str, str]] = []
    last_act_tool: Optional[str] = None   # most recent gold act tool (for status attribution)

    # --- trajectory-level: stopped early / kept going ---------------------- #
    exp_max = max(exp_steps) if exp_steps else 0
    obs_max = max(obs_steps) if obs_steps else 0
    if obs_max < exp_max:
        traj_failures.append(Failure(
            layer="trajectory", code="premature_stop",
            detail=f"agent produced {obs_max} steps; gold has {exp_max}.",
        ))
    if obs_max > exp_max:
        traj_failures.append(Failure(
            layer="trajectory", code="doesnt_stop",
            detail=f"agent produced {obs_max} steps; gold has {exp_max}.",
        ))

    # repeated identical act call (classic ReAct loop)
    ordered = sorted(observed.steps, key=lambda s: s.step)
    for a, b in zip(ordered, ordered[1:]):
        if (a.behavior == Behavior.act and b.behavior == Behavior.act
                and a.tool == b.tool
                and {k: v.value for k, v in a.args.items()}
                == {k: v.value for k, v in b.args.items()}):
            traj_failures.append(Failure(
                layer="trajectory", code="repeated_call", step=b.step,
                detail=f"step {b.step} repeats the step {a.step} call to '{a.tool}'.",
            ))

    # --- per-step ---------------------------------------------------------- #
    for n in sorted(set(exp_steps) | set(obs_steps)):
        exp = exp_steps.get(n)
        obs = obs_steps.get(n)

        if exp is not None and obs is None:
            sg = StepGrade(step=n, behavior_expected=exp.behavior.value,
                           behavior_ok=False,
                           tool_expected=exp.tool if isinstance(exp, ActStep) else None)
            sg.failures.append(Failure(
                layer="behavior", code="step_missing", step=n,
                detail=f"gold step {n} ({exp.behavior.value}) was never produced.",
            ))
            behavior_ctr["step_missing"] += 1
            step_grades.append(sg)
            continue

        if exp is None and obs is not None:
            sg = StepGrade(step=n, behavior_observed=obs.behavior.value,
                           behavior_ok=False, tool_observed=obs.tool)
            sg.failures.append(Failure(
                layer="behavior", code="unexpected_step", step=n,
                detail=f"agent produced an extra step {n} ({obs.behavior.value}).",
            ))
            behavior_ctr["unexpected_step"] += 1
            step_grades.append(sg)
            continue

        # both present
        sg = StepGrade(
            step=n,
            behavior_expected=exp.behavior.value,
            behavior_observed=obs.behavior.value,
            behavior_ok=(exp.behavior == obs.behavior),
            tool_expected=exp.tool if isinstance(exp, ActStep) else None,
            tool_observed=obs.tool,
        )

        # ---- behavior layer ---- #
        if exp.behavior != obs.behavior:
            if exp.behavior == Behavior.act and obs.behavior == Behavior.clarify:
                code = "over_clarify"
            elif exp.behavior in (Behavior.clarify, Behavior.respond) and obs.behavior == Behavior.act:
                code = "eager_act"
            else:
                code = "behavior_mismatch"
            behavior_ctr[code] += 1
            sg.failures.append(Failure(
                layer="behavior", code=code, step=n,
                detail=f"expected {exp.behavior.value}, observed {obs.behavior.value}.",
            ))

        # ---- tool + arg layer (only when gold expects an act) ---- #
        fb = getattr(exp, "forbidden", None)

        if isinstance(exp, ActStep):
            sg.arg_checked = True
            last_act_tool = exp.tool
            sg.tool_ok = (obs.tool == exp.tool)
            if obs.tool != exp.tool:
                arg_ctr["wrong_tool"] += 1
                sg.failures.append(Failure(
                    layer="tool", code="wrong_tool", step=n,
                    detail=f"expected tool '{exp.tool}', observed '{obs.tool}'.",
                ))

            # expected args, provenance-aware
            for name, espec in exp.args.items():
                if espec.source == ArgSource.missing:
                    if name in obs.args:
                        arg_ctr["fabricated"] += 1
                        sg.failures.append(Failure(
                            layer="arg", code="fabricated", step=n, arg=name,
                            detail=f"arg '{name}' should be MISSING (clarify) but agent supplied "
                                   f"'{obs.args[name].value}'.",
                        ))
                    continue

                bucket = espec.fail_bucket or "arg_value_mismatch"
                if name not in obs.args:
                    arg_ctr[bucket] += 1
                    sg.failures.append(Failure(
                        layer="arg", code=bucket, step=n, arg=name,
                        detail=f"arg '{name}' missing (expected '{espec.value}', "
                               f"source={espec.source.value}).",
                    ))
                    continue

                oarg = obs.args[name]
                if _norm(oarg.value) != _norm(espec.value):
                    arg_ctr[bucket] += 1
                    sg.failures.append(Failure(
                        layer="arg", code=bucket, step=n, arg=name,
                        detail=f"arg '{name}'='{oarg.value}', expected '{espec.value}' "
                               f"(source={espec.source.value}).",
                    ))
                elif oarg.source is not None and oarg.source != espec.source:
                    # right value, wrong provenance: 'got lucky re-extracting'
                    arg_ctr["provenance_mismatch"] += 1
                    sg.failures.append(Failure(
                        layer="arg", code="provenance_mismatch", step=n, arg=name,
                        detail=f"arg '{name}' correct but source={oarg.source.value}, "
                               f"expected {espec.source.value}.",
                    ))

            # forbidden arg keys (fabrication) and forbidden values (wrong slot)
            if fb:
                for k in fb.args:
                    if k in obs.args and k not in exp.args:
                        arg_ctr["fabricated"] += 1
                        sg.failures.append(Failure(
                            layer="arg", code="fabricated", step=n, arg=k,
                            detail=f"forbidden arg '{k}'='{obs.args[k].value}' was supplied.",
                        ))
                for k, badvals in fb.args_values.items():
                    if k in obs.args and _norm(obs.args[k].value) in {_norm(b) for b in badvals}:
                        arg_ctr["forbidden_arg_value"] += 1
                        sg.failures.append(Failure(
                            layer="arg", code="forbidden_arg_value", step=n, arg=k,
                            detail=f"arg '{k}'='{obs.args[k].value}' is a forbidden value "
                                   f"(wrong slot / wrong-arg act).",
                        ))

        else:
            # gold is clarify/respond -> any tool call is eager acting.
            # NOTE: the eager_act COUNT is recorded once in the behavior layer above
            # (matches the confusion matrix). Here we only add a tool-layer DETAIL so
            # we know which tool was wrongly called -- without re-counting eager_act.
            if obs.tool is not None:
                sg.failures.append(Failure(
                    layer="tool", code="eager_act_tool_call", step=n,
                    detail=f"gold step is {exp.behavior.value} (no tool); agent called '{obs.tool}'.",
                ))
                for k, oarg in obs.args.items():
                    arg_ctr["fabricated"] += 1
                    sg.failures.append(Failure(
                        layer="arg", code="fabricated", step=n, arg=k,
                        detail=f"arg '{k}'='{oarg.value}' fabricated while acting eagerly.",
                    ))

        # forbidden tool (applies to any gold step that declares one)
        if fb and obs.tool is not None and obs.tool in fb.tools:
            behavior_ctr["forbidden_tool"] += 1
            sg.failures.append(Failure(
                layer="tool", code="forbidden_tool", step=n,
                detail=f"called forbidden tool '{obs.tool}'.",
            ))

        # ---- response layer (pluggable judge) ---- #
        rc = getattr(exp, "response_check", None)
        if rc is not None:
            sg.response_checked = True
            tool_for_obs = last_act_tool
            obs_payload: dict[str, Any] = {}
            if tool_for_obs and tool_for_obs in expected.env.tool_outcomes:
                obs_payload = expected.env.tool_outcomes[tool_for_obs].returns

            ctx = ResponseContext(
                step=n,
                text=obs.response_text,
                speech_act=rc.speech_act,
                language=rc.language,
                expected_status=rc.expected_status,
                must_reflect=rc.must_reflect,
                must_not_contain=rc.must_not_contain,
                screens_for=rc.screens_for,
                tool=tool_for_obs,
                observation=obs_payload,
                confirmation_ids=sorted(confirmation_ids),
                forbidden_values=sorted(forbidden_values),
                booking_markers=sorted(contract.success_lexicon),
            )

            findings = response_judge(ctx)
            for f in findings:
                resp_ctr[f.bucket] += 1
                sg.failures.append(Failure(
                    layer="response", code=f.bucket, step=n, detail=f.detail))

            if rc.expected_status in ("empty", "error"):
                mishandled = any(f.bucket in STATUS_MISHANDLE_BUCKETS for f in findings)
                status_handling.append({
                    "status": rc.expected_status,
                    "tool": last_act_tool or "(unknown)",
                    "result": "mishandled" if mishandled else "handled",
                })

        step_grades.append(sg)

    # --- loose outcome check ---------------------------------------------- #
    outcome_failures = _check_outcome(expected, observed, resp_ctr, contract)

    strict_pass = (
        not traj_failures
        and not outcome_failures
        and all(not sg.failures for sg in step_grades)
    )
    outcome_pass = not outcome_failures

    return GradeReport(
        id=expected.id,
        strict_pass=strict_pass,
        outcome_pass=outcome_pass,
        step_grades=step_grades,
        trajectory_failures=traj_failures,
        outcome_failures=outcome_failures,
        behavior_buckets=dict(behavior_ctr),
        arg_fail_buckets=dict(arg_ctr),
        response_buckets=dict(resp_ctr),
        status_handling=status_handling,
    )


def _check_outcome(expected: DatasetRow, observed: ObservedTrajectory,
                   resp_ctr: Counter, contract: Contract) -> list[Failure]:
    """Mechanically checkable predicates for outcome_check.must_not_happen.
    Unknown predicates are skipped (reported nowhere) rather than failing."""
    out: list[Failure] = []
    terminal_calls = [s for s in observed.steps
                      if s.behavior == Behavior.act and s.tool in contract.terminal_tools]
    any_act = any(s.behavior == Behavior.act for s in observed.steps)
    eb = expected.outcome_check.expected_effect or {}

    def _slot_of(step: ObservedStep) -> dict[str, Any]:
        return {k: v.value for k, v in step.args.items()}

    # Check prerequisite tools for terminal_action_on_unavailable_result.
    # .get() guard: if the prerequisite tool was never called (no entry in
    # tool_outcomes), the predicate does NOT fire.
    prereq_unavailable = False
    for tc in terminal_calls:
        tool_spec = contract.tools.get(tc.tool)
        if not tool_spec:
            continue
        for prereq in tool_spec.prerequisite_tools:
            prereq_outcome = expected.env.tool_outcomes.get(prereq)
            if prereq_outcome and prereq_outcome.returns.get("status") in ("empty", "error"):
                prereq_unavailable = True

    predicates = {
        "duplicate_terminal_action": len(terminal_calls) > 1,
        "answered_without_calling_tool":
            (not any_act) and any(s.response_text for s in observed.steps),
        "final_reply_contradicts_tool_result":
            resp_ctr.get("unfaithful_action_claim", 0) > 0
            or resp_ctr.get("wrong_slot_confirmed", 0) > 0
            or resp_ctr.get("false_success_claim", 0) > 0
            or resp_ctr.get("fabricated_on_empty", 0) > 0,
        "terminal_action_on_unavailable_result": prereq_unavailable,
        "effect_mismatch": any(
            eb and any(
                str(slot.get(k)) != str(eb.get(k))
                for k in eb if k in slot
            )
            for slot in map(_slot_of, terminal_calls)
        ),
        # Legacy predicate names (for backward compat with hand-authored rows):
        "double_booked": len(terminal_calls) > 1,
        "booked_despite_no_availability": prereq_unavailable,
        "booked_wrong_slot": any(
            eb and any(
                str(slot.get(k)) != str(eb.get(k))
                for k in ("date", "time") if k in eb
            )
            for slot in map(_slot_of, terminal_calls)
        ),
    }

    for cond in expected.outcome_check.must_not_happen:
        if predicates.get(cond):
            out.append(Failure(
                layer="outcome", code=cond,
                detail=f"must_not_happen '{cond}' was observed.",
            ))

    # expected_effect match
    if eb:
        if not terminal_calls:
            out.append(Failure(layer="outcome", code="effect_missing",
                               detail="expected an effect; none occurred."))
        else:
            slot = _slot_of(terminal_calls[-1])
            diff = {k: (slot.get(k), eb[k]) for k in eb if str(slot.get(k)) != str(eb[k])}
            if diff:
                out.append(Failure(layer="outcome", code="effect_mismatch",
                                   detail=f"final effect differs from expected: {diff}."))
    return out


def behavior_pairs(reports: list[GradeReport]) -> tuple[list[str], list[str]]:
    """(y_true, y_pred) of behaviors over every step. '(none)' = missing/extra."""
    yt, yp = [], []
    for r in reports:
        for sg in r.step_grades:
            yt.append(sg.behavior_expected or "(none)")
            yp.append(sg.behavior_observed or "(none)")
    return yt, yp


def tool_choice_pairs(reports: list[GradeReport]) -> tuple[list[str], list[str]]:
    """(y_true, y_pred) of tool choice. Scored on steps where the gold expects a
    tool OR the agent called one; pure no-call/no-call steps are skipped so the
    matrix isn't drowned by respond-step true-negatives. '(no_call)' = clarified
    or replied instead of calling a tool."""
    yt, yp = [], []
    for r in reports:
        for sg in r.step_grades:
            gt = sg.tool_expected or "(no_call)"
            pt = sg.tool_observed or "(no_call)"
            if gt == "(no_call)" and pt == "(no_call)":
                continue
            yt.append(gt)
            yp.append(pt)
    return yt, yp


def aggregate(reports: list[GradeReport]) -> dict[str, Any]:
    """Dataset-level profile: where do failures cluster across the whole set?"""
    behavior, arg, resp = Counter(), Counter(), Counter()
    for r in reports:
        behavior.update(r.behavior_buckets)
        arg.update(r.arg_fail_buckets)
        resp.update(r.response_buckets)
    n = len(reports)
    return {
        "rows": n,
        "strict_pass_rate": round(sum(r.strict_pass for r in reports) / n, 3) if n else 0,
        "outcome_pass_rate": round(sum(r.outcome_pass for r in reports) / n, 3) if n else 0,
        "behavior_buckets": dict(behavior.most_common()),
        "arg_fail_buckets": dict(arg.most_common()),
        "response_buckets": dict(resp.most_common()),
    }


# --------------------------------------------------------------------------- #
# Readable reporting: behavior confusion matrix + precision/recall/F1          #
# --------------------------------------------------------------------------- #

_BEHAVIOR_LABELS = ["act", "clarify", "respond", "(none)"]   # (none) = missing/extra step
_REPORT_LABELS = ["act", "clarify", "respond"]               # scored classes


def _safe_div(num: float, den: float) -> Optional[float]:
    return num / den if den else None


def behavior_confusion(reports: list[GradeReport]) -> dict[str, dict[str, int]]:
    """cm[gold][pred] over every graded step. gold/pred = behavior or '(none)'."""
    cm = {g: {p: 0 for p in _BEHAVIOR_LABELS} for g in _BEHAVIOR_LABELS}
    for r in reports:
        for sg in r.step_grades:
            g = sg.behavior_expected or "(none)"
            p = sg.behavior_observed or "(none)"
            cm[g][p] += 1
    return cm


def classification_report(cm: dict[str, dict[str, int]]) -> dict[str, Any]:
    """sklearn-style per-class precision / recall / f1 / support + averages."""
    total = sum(cm[g][p] for g in _BEHAVIOR_LABELS for p in _BEHAVIOR_LABELS)
    per_class: dict[str, dict[str, Any]] = {}
    for c in _REPORT_LABELS:
        tp = cm[c][c]
        fp = sum(cm[g][c] for g in _BEHAVIOR_LABELS) - tp
        fn = sum(cm[c][p] for p in _BEHAVIOR_LABELS) - tp
        support = sum(cm[c][p] for p in _BEHAVIOR_LABELS)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, support)
        f1 = (_safe_div(2 * precision * recall, precision + recall)
              if precision and recall else (0.0 if support else None))
        per_class[c] = {"precision": precision, "recall": recall,
                        "f1": f1, "support": support}

    correct = sum(cm[c][c] for c in _REPORT_LABELS)
    accuracy = _safe_div(correct, total)

    def _avg(metric: str, weighted: bool) -> Optional[float]:
        num = den = 0.0
        for c in _REPORT_LABELS:
            v = per_class[c][metric]
            w = per_class[c]["support"] if weighted else 1
            num += (v or 0.0) * w
            den += w
        return _safe_div(num, den)

    return {
        "per_class": per_class,
        "accuracy": accuracy,
        "total": total,
        "macro_avg": {m: _avg(m, False) for m in ("precision", "recall", "f1")},
        "weighted_avg": {m: _avg(m, True) for m in ("precision", "recall", "f1")},
        # the two Act-vs-Clarify failure modes you named, read off the matrix:
        "eager_act": cm["clarify"]["act"] + cm["respond"]["act"],
        "over_clarify": cm["act"]["clarify"],
        "premature_stop": sum(cm[g]["(none)"] for g in _REPORT_LABELS),
        "runaway_steps": sum(cm["(none)"][p] for p in _REPORT_LABELS),
    }


def per_layer_stats(reports: list[GradeReport]) -> dict[str, dict[str, Any]]:
    """Clean-rate per localization layer: of the steps where a layer applies,
    how many had zero failures in that layer."""
    rows = {"behavior": [0, 0], "arg": [0, 0], "response": [0, 0]}  # [applicable, clean]
    for r in reports:
        for sg in r.step_grades:
            matched = sg.behavior_expected is not None and sg.behavior_observed is not None
            checks = {
                "behavior": matched,
                "arg": sg.arg_checked,
                "response": sg.response_checked,
            }
            for layer, applies in checks.items():
                if not applies:
                    continue
                rows[layer][0] += 1
                if not any(f.layer == layer for f in sg.failures):
                    rows[layer][1] += 1
    return {
        layer: {
            "applicable": appl,
            "clean": clean,
            "clean_rate": round(clean / appl, 3) if appl else None,
        }
        for layer, (appl, clean) in rows.items()
    }


def _fmt(v: Optional[float]) -> str:
    return f"{v:.3f}" if isinstance(v, float) else "  -  "


def render_text_report(reports: list[GradeReport], title: str = "EVAL REPORT") -> str:
    cm = behavior_confusion(reports)
    rep = classification_report(cm)
    layers = per_layer_stats(reports)
    agg = aggregate(reports)
    out: list[str] = []

    out.append(f"\n{'=' * 62}\n {title}\n{'=' * 62}")
    out.append(f" rows: {agg['rows']}   "
               f"strict_pass_rate: {agg['strict_pass_rate']}   "
               f"outcome_pass_rate: {agg['outcome_pass_rate']}")

    # --- behavior confusion matrix ---
    out.append("\n— Behavior confusion (rows = gold, cols = agent) —")
    header = "  gold \\ pred │" + "".join(f"{p:>10}" for p in _BEHAVIOR_LABELS) + f"{'support':>10}"
    out.append(header)
    out.append("  " + "─" * (len(header) - 2))
    for g in _BEHAVIOR_LABELS:
        support = sum(cm[g][p] for p in _BEHAVIOR_LABELS)
        if support == 0 and sum(cm[gg][g] for gg in _BEHAVIOR_LABELS) == 0:
            continue  # skip labels that never appear as gold or pred
        cells = "".join(f"{cm[g][p]:>10}" for p in _BEHAVIOR_LABELS)
        out.append(f"  {g:>11} │{cells}{support:>10}")

    # --- classification report ---
    out.append("\n— Behavior metrics (Act / Clarify / Respond as classes) —")
    out.append(f"  {'class':>10} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>9}")
    for c in _REPORT_LABELS:
        m = rep["per_class"][c]
        out.append(f"  {c:>10} {_fmt(m['precision']):>10} {_fmt(m['recall']):>10} "
                   f"{_fmt(m['f1']):>10} {m['support']:>9}")
    out.append(f"  {'accuracy':>10} {'':>10} {'':>10} {_fmt(rep['accuracy']):>10} {rep['total']:>9}")
    for avg in ("macro_avg", "weighted_avg"):
        a = rep[avg]
        out.append(f"  {avg:>10} {_fmt(a['precision']):>10} {_fmt(a['recall']):>10} "
                   f"{_fmt(a['f1']):>10} {rep['total']:>9}")

    out.append(f"\n  eager_act (acted when gold=clarify/respond): {rep['eager_act']}")
    out.append(f"  over_clarify (clarified when gold=act):      {rep['over_clarify']}")
    out.append(f"  premature_stop (gold step never produced):   {rep['premature_stop']}")
    out.append(f"  runaway_steps  (extra step over gold):       {rep['runaway_steps']}")

    # --- per-layer clean rates ---
    out.append("\n— Clean-rate by localization layer —")
    out.append(f"  {'layer':>10} {'applicable':>11} {'clean':>7} {'clean_rate':>11}")
    for layer, s in layers.items():
        out.append(f"  {layer:>10} {s['applicable']:>11} {s['clean']:>7} "
                   f"{_fmt(s['clean_rate']):>11}")

    # --- failure-bucket frequency ---
    def _bucket_block(name: str, d: dict[str, int]) -> None:
        out.append(f"\n— {name} —")
        if not d:
            out.append("  (none)")
        for k, v in d.items():
            out.append(f"  {v:>4}  {k}")

    _bucket_block("arg_fail_buckets", agg["arg_fail_buckets"])
    _bucket_block("response_buckets", agg["response_buckets"])
    _bucket_block("behavior_buckets", agg["behavior_buckets"])

    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Demo: clean run, content-bug run, behavior-bug run                           #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import json
    from dataset_models import DatasetRow, EXAMPLE
    from examples.salon.contract import SALON_CONTRACT

    gold = DatasetRow.model_validate(EXAMPLE)

    # ---- RUN A: agent does everything right ---- #
    good = ObservedTrajectory(
        id=gold.id,
        steps=[
            {"step": 1, "behavior": "act", "tool": "check_availability",
             "args": {"service": "gel_polish",
                      "date_range": "2026-06-29..2026-07-04"}},
            {"step": 2, "behavior": "respond",
             "response_text": "יש לי שני תורים: ראשון 2026-06-29 14:00 או 2026-07-01 10:30."},
            {"step": 3, "behavior": "act", "tool": "book_appointment",
             "args": {"service": "gel_polish", "date": "2026-06-29",
                      "time": "14:00", "customer_name": "מאיה לוי"}},
            {"step": 4, "behavior": "respond",
             "response_text": "מעולה, נקבע לך תור ל-2026-06-29 בשעה 14:00. אישור: BK-5512."},
        ],
    )

    # ---- RUN B: several distinct bugs, one per layer ---- #
    bad = ObservedTrajectory(
        id=gold.id,
        steps=[
            # arg bug: wrong relative-time resolution
            {"step": 1, "behavior": "act", "tool": "check_availability",
             "args": {"service": "gel_polish",
                      "date_range": "2026-07-06..2026-07-11"}},  # wrong week
            # response bug: claims a booking that never happened
            {"step": 2, "behavior": "respond",
             "response_text": "מעולה, כבר נקבע לך תור!"},        # 'נקבע' = booking claim
            # arg bug: books the OTHER slot (forbidden value)
            {"step": 3, "behavior": "act", "tool": "book_appointment",
             "args": {"service": "gel_polish", "date": "2026-07-01",
                      "time": "10:30", "customer_name": "מאיה לוי"}},
            # response bug: confirms the wrong slot
            {"step": 4, "behavior": "respond",
             "response_text": "נקבע ל-2026-07-01 בשעה 10:30. אישור BK-5512."},
        ],
    )

    # ---- RUN C: behavior-level bugs (over-clarify, eager-act, premature stop) ---- #
    behavior_bug = ObservedTrajectory(
        id=gold.id,
        steps=[
            # over-clarify: gold says act, agent asks instead of looking up
            {"step": 1, "behavior": "clarify",
             "response_text": "באיזה יום בדיוק?"},
            # eager-act: gold says respond, agent books before the user chose
            {"step": 2, "behavior": "act", "tool": "book_appointment",
             "args": {"service": "gel_polish", "date": "2026-06-29", "time": "14:00",
                      "customer_name": "מאיה לוי"}},
            # correct act
            {"step": 3, "behavior": "act", "tool": "book_appointment",
             "args": {"service": "gel_polish", "date": "2026-06-29", "time": "14:00",
                      "customer_name": "מאיה לוי"}},
            # step 4 (gold respond) never produced -> premature_stop
        ],
    )

    runs = [("RUN A (clean)", good), ("RUN B (content bugs)", bad),
            ("RUN C (behavior bugs)", behavior_bug)]

    reports = []
    for label, obs in runs:
        rep = grade(gold, obs, contract=SALON_CONTRACT)
        reports.append(rep)
        print(f"\n----- {label}: strict_pass={rep.strict_pass} "
              f"outcome_pass={rep.outcome_pass} -----")
        for sg in rep.step_grades:
            for f in sg.failures:
                print(f"  step {f.step} [{f.layer}/{f.code}] {f.detail}")

    print(render_text_report(reports, title="DATASET REPORT (A + B + C)"))
