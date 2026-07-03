"""
Volume generator for the booking-agent eval (contract-driven).

Two pieces:
  make_gold(scenario, idx, rng, contract, fixtures)  -> a gold DatasetRow
  simulate_agent(gold, profile, rng, contract)       -> an ObservedTrajectory

build_dataset(n_per_cell, profile, seed, contract, fixtures) crosses every
(scenario, status) cell n_per_cell times, so you get controlled volume.

The scenario grid is derived from the Contract's tools and their
applicable_statuses — no hardcoded scenario names.
"""

from __future__ import annotations

import math
import random
from typing import Optional

from contract import Contract, ArgSource
from dataset_models import DatasetRow
from grading import ObservedTrajectory


def _arg(value, source, fail_bucket=None, **kw):
    d = {"value": value, "source": source}
    if fail_bucket:
        d["fail_bucket"] = fail_bucket
    d.update(kw)
    return d


# --------------------------------------------------------------------------- #
# Scenario grid: derived from contract + fixtures                              #
# --------------------------------------------------------------------------- #

def generate_scenario_grid(contract: Contract, fixtures: dict) -> list[dict]:
    """Derive all (tool, status) cells from the contract.

    For each tool, each applicable_status produces a scenario cell.
    Additionally, for each tool, each required arg with provenance_hint
    != "from_context" produces a clarify-then-act cell (the arg is missing).

    Excludes from_context args from clarify rungs (Catch 2a: those are
    pulled from session context, not from the user, so the agent should
    never clarify them).

    Returns a list of scenario dicts with keys:
        tool, status, missing_arg (optional)
    """
    cells: list[dict] = []

    for tool_name, tool_spec in contract.tools.items():
        for status in tool_spec.applicable_statuses:
            # direct call with all args present
            cells.append({"tool": tool_name, "status": status})

            # clarify-then-act for each required arg that is NOT from_context
            for arg_name, arg_spec in tool_spec.args.items():
                if not arg_spec.required:
                    continue
                if arg_spec.provenance_hint == "from_context":
                    continue
                cells.append({
                    "tool": tool_name,
                    "status": status,
                    "missing_arg": arg_name,
                })

    return cells


# --------------------------------------------------------------------------- #
# Gold construction                                                            #
# --------------------------------------------------------------------------- #

def make_gold(scenario: dict, idx: int, rng: random.Random,
              contract: Contract, fixtures: dict) -> DatasetRow:
    """Build a gold DatasetRow from a scenario dict produced by
    generate_scenario_grid().  No hardcoded scenario names — fully driven
    by contract + fixtures."""
    tool_name = scenario["tool"]
    status = scenario["status"]
    missing_arg = scenario.get("missing_arg")
    tool_spec = contract.tools[tool_name]

    # Pick fixture values
    services = fixtures.get("services", {})
    svc_key = rng.choice(list(services)) if services else None
    svc_he = services.get(svc_key, "")
    names = fixtures.get("customer_names", [""])
    name = rng.choice(names)
    days = fixtures.get("days", {})
    day_key = rng.choice(list(days)) if days else None
    day_he, day_iso = days.get(day_key, ("", ""))
    times = fixtures.get("times", [""])
    t = rng.choice(times)
    week_range = fixtures.get("week_range", "")
    ref = fixtures.get("reference_time", "")
    ctx_source = fixtures.get("context_source", "whatsapp")
    env_cust = {"source": ctx_source, "customer_name": name}
    base_id = f"{tool_name}__{status}"
    if missing_arg:
        base_id += f"__missing_{missing_arg}"
    base_id += f"__{idx:03d}"

    # Build tool outcome
    if tool_name == "check_availability":
        if status == "ok":
            returns = {"status": "ok", "slots": [{"date": day_iso, "time": t}]}
        elif status == "empty":
            returns = {"status": "empty", "slots": []}
        else:
            returns = {"status": status}
    elif tool_name == "book_appointment":
        if status == "ok":
            cid = f"BK-{7000 + idx}"
            returns = {"status": "ok", "confirmation_id": cid}
        elif status == "error":
            returns = {"status": "error", "reason": "slot_taken"}
        else:
            returns = {"status": status}
    elif tool_name == "cancel_appointment":
        if status == "ok":
            returns = {"status": "ok", "cancelled": {"date": day_iso, "time": t}}
        else:
            returns = {"status": status}
    else:
        returns = {"status": status}

    env_dict = {
        "reference_time": ref,
        "customer_context": env_cust,
        "tool_outcomes": {tool_name: {"returns": returns}},
    }

    # Build args dict from contract spec
    def build_args(exclude: Optional[str] = None) -> dict:
        args = {}
        for arg_name, arg_spec in tool_spec.args.items():
            if arg_name == exclude:
                continue
            if arg_name == "service":
                args["service"] = _arg(svc_key, "from_user", arg_spec.fail_bucket,
                                       source_detail=f"'{svc_he}'")
            elif arg_name == "date_range":
                args["date_range"] = _arg(week_range, "computed", arg_spec.fail_bucket,
                                          compute_type=arg_spec.compute_type_hint,
                                          raw_span="השבוע")
            elif arg_name == "date":
                args["date"] = _arg(day_iso, "computed", arg_spec.fail_bucket,
                                    compute_type=arg_spec.compute_type_hint,
                                    raw_span=day_he)
            elif arg_name == "time":
                args["time"] = _arg(t, "from_user")
            elif arg_name == "customer_name":
                args["customer_name"] = _arg(name, "from_context", arg_spec.fail_bucket,
                                             source_detail="env.customer_context")
            else:
                args[arg_name] = _arg(None, arg_spec.provenance_hint or "from_user",
                                      arg_spec.fail_bucket)
        return args

    # Build forbidden tools: all tools except the current one
    other_tools = [t for t in contract.tools if t != tool_name]

    # Determine speech_act and response_check based on tool + status
    if tool_name == "check_availability":
        if status == "empty":
            speech_act = "report_no_availability"
            expected_status = "empty"
            screens_for = ["fabricated_on_empty"]
        else:
            speech_act = "report_availability"
            expected_status = None
            screens_for = ["omitted_slot"]
        must_reflect = [f"{day_iso} {t}"] if status == "ok" else []
        resp_check = {
            "speech_act": speech_act, "language": contract.language,
            "screens_for": screens_for,
        }
        if expected_status:
            resp_check["expected_status"] = expected_status
        if must_reflect:
            resp_check["must_reflect"] = must_reflect
        final_state = "slots_reported_to_user" if status == "ok" else "no_availability_reported"
        must_not_happen = ["appointment_booked"]
        if status == "empty":
            must_not_happen += ["booked_despite_no_availability",
                                "final_reply_contradicts_tool_result"]
        expected_effect = None
    elif tool_name == "book_appointment":
        if status == "error":
            speech_act = "report_booking_failed"
            expected_status = "error"
            screens_for = ["false_success_claim"]
            must_reflect = []
        else:
            speech_act = "confirm_booking"
            expected_status = None
            screens_for = ["omitted_confirmation_id"]
            must_reflect = [returns.get("confirmation_id", ""), day_iso, t]
        resp_check = {
            "speech_act": speech_act, "language": contract.language,
            "screens_for": screens_for,
        }
        if expected_status:
            resp_check["expected_status"] = expected_status
        if must_reflect:
            resp_check["must_reflect"] = must_reflect
        final_state = "appointment_booked" if status == "ok" else "booking_failed_reported"
        must_not_happen = ["double_booked"] if status == "ok" else ["final_reply_contradicts_tool_result"]
        expected_effect = {"service": svc_key, "date": day_iso, "time": t} if status == "ok" else None
    elif tool_name == "cancel_appointment":
        speech_act = "confirm_cancellation"
        resp_check = {
            "speech_act": speech_act, "language": contract.language,
            "must_reflect": [day_iso, t],
            "must_not_contain": ["BK-"],
            "screens_for": ["wrong_slot_confirmed"],
        }
        final_state = "appointment_cancelled"
        must_not_happen = ["double_booked"]
        expected_effect = None
    else:
        speech_act = "report_result"
        resp_check = {"speech_act": speech_act, "language": contract.language}
        final_state = "completed"
        must_not_happen = []
        expected_effect = None

    # Build trajectory
    if missing_arg is None:
        trajectory = [
            {"step": 1, "behavior": "act",
             "user_message": f"יש {svc_he} השבוע?" if tool_name == "check_availability"
                             else f"תקבעי לי {svc_he} ל{day_he} ב-{t}" if tool_name == "book_appointment"
                             else f"תבטלי את התור של {day_he} ב-{t}",
             "tool": tool_name,
             "args": build_args(),
             "forbidden": {"tools": other_tools, "reason": "wrong tool"}},
            {"step": 2, "behavior": "respond", "reacts_to": "observation_from_step_1",
             "response_check": resp_check},
        ]
    else:
        clarify_speech_act = f"ask_{missing_arg}"
        trajectory = [
            {"step": 1, "behavior": "clarify",
             "user_message": f"יש לכם מקום השבוע?" if missing_arg == "service"
                             else f"תקבעי לי {svc_he} ל{day_he}" if missing_arg == "time"
                             else f"יש {svc_he} השבוע?",
             "clarify_target": missing_arg,
             "response_check": {"speech_act": clarify_speech_act,
                                "language": contract.language,
                                "must_not_contain": ["BK-", "נקבע"]},
             "forbidden": {"tools": [tool_name] + other_tools,
                           "args": [missing_arg],
                           "reason": f"{missing_arg} unknown"}},
            {"step": 2, "behavior": "act",
             "user_message": svc_he if missing_arg == "service"
                             else f"ב-{t}" if missing_arg == "time"
                             else "השבוע",
             "tool": tool_name,
             "args": build_args(),
             "forbidden": {"tools": other_tools, "reason": "now act"}},
            {"step": 3, "behavior": "respond", "reacts_to": "observation_from_step_2",
             "response_check": resp_check},
        ]

    outcome_check = {
        "final_state": final_state,
        "must_not_happen": must_not_happen,
    }
    if expected_effect:
        outcome_check["expected_effect"] = expected_effect

    row = DatasetRow.model_validate({
        "id": base_id,
        "intent": "availability_check" if tool_name == "check_availability"
                  else "book" if tool_name == "book_appointment"
                  else "cancel",
        "turn_pattern": "clarify_then_act" if missing_arg else "single_turn",
        "env": env_dict,
        "expected_trajectory": trajectory,
        "outcome_check": outcome_check,
    })

    # Catch 1: assert clarify trajectory shape for generated rows
    if missing_arg is not None:
        assert len(row.expected_trajectory) == 3, (
            f"clarify row {row.id} must have 3 steps, got {len(row.expected_trajectory)}"
        )
        assert row.expected_trajectory[0].behavior.value == "clarify"
        assert row.expected_trajectory[1].behavior.value == "act"
        assert row.expected_trajectory[2].behavior.value == "respond"

    return row


# --------------------------------------------------------------------------- #
# Agent simulator                                                             #
# --------------------------------------------------------------------------- #

DEFAULT_PROFILE = {
    "eager_act": 0.25,
    "wrong_tool": 0.15,
    "arg_resolution": 0.20,
    "omit_fact": 0.15,
    "false_success": 0.40,
    "premature_stop": 0.08,
}


def _gold_args(act_step) -> dict:
    return {k: v.value for k, v in act_step.args.items()}


def _wrong_tool_for(tool: str, contract: Contract, rng: random.Random) -> str:
    alts = [t for t in contract.tools if t != tool]
    return rng.choice(alts) if alts else tool


def _corrupt_value(value: str, rng: random.Random) -> str:
    if ".." in value:
        return "2026-07-06..2026-07-11"
    if len(value) == 10 and value[4] == "-":
        return "2026-07-06"
    return value + "x"


def simulate_agent(gold: DatasetRow, profile: dict, rng: random.Random,
                   contract: Contract) -> ObservedTrajectory:
    steps = []
    for st in gold.expected_trajectory:
        if st.behavior.value == "respond" and rng.random() < profile.get("premature_stop", 0):
            break

        if st.behavior.value == "clarify":
            if rng.random() < profile.get("eager_act", 0):
                intent = gold.intent
                if "book" in intent:
                    tool = "book_appointment"
                elif "cancel" in intent:
                    tool = "cancel_appointment"
                else:
                    tool = "check_availability"

                tool_spec = contract.tools.get(tool)
                args = {}
                if tool_spec:
                    for arg_name, arg_spec in tool_spec.args.items():
                        if arg_spec.provenance_hint == "from_user":
                            args[arg_name] = "manicure" if arg_name == "service" else "09:00"
                        elif arg_spec.provenance_hint == "from_context":
                            args[arg_name] = "דנה כהן"
                        elif arg_spec.compute_type_hint:
                            args[arg_name] = "2026-06-30"
                        else:
                            args[arg_name] = "unknown"

                steps.append({"step": st.step, "behavior": "act", "tool": tool,
                              "args": args, "response_text": None})
            else:
                steps.append({"step": st.step, "behavior": "clarify",
                              "response_text": "אפשר פרט נוסף?"})
            continue

        if st.behavior.value == "act":
            tool = st.tool
            args = _gold_args(st)
            if rng.random() < profile.get("wrong_tool", 0):
                tool = _wrong_tool_for(tool, contract, rng)
            elif rng.random() < profile.get("arg_resolution", 0):
                # Catch 2a: only corrupt computed args (compute_type is set)
                for key, spec in st.args.items():
                    if spec.compute_type is not None and key in args:
                        args[key] = _corrupt_value(str(args[key]), rng)
                        break
            steps.append({"step": st.step, "behavior": "act", "tool": tool,
                          "args": args, "response_text": None})
            continue

        # respond
        rc = st.response_check
        if rc.expected_status in ("empty", "error"):
            if rng.random() < profile.get("false_success", 0):
                text = "מצוין! קבעתי לך תור. אישור BK-9999."
            else:
                text = ("מצטערת, אין מקום פנוי השבוע." if rc.expected_status == "empty"
                        else "מצטערת, ההזמנה נכשלה — התור נתפס.")
        else:
            facts = list(rc.must_reflect)
            if facts and rng.random() < profile.get("omit_fact", 0):
                facts.pop(rng.randrange(len(facts)))
            lead = ("נקבע, " if rc.speech_act == "confirm_booking"
                    else "בוטל, " if rc.speech_act == "confirm_cancellation"
                    else "יש תורים: ")
            text = lead + " ".join(facts) if facts else lead + "בוצע."
        steps.append({"step": st.step, "behavior": "respond", "response_text": text})

    return ObservedTrajectory(id=gold.id, steps=steps)


def build_dataset(n_per_cell: int = 6, profile: Optional[dict] = None,
                  seed: int = 7, contract: Optional[Contract] = None,
                  fixtures: Optional[dict] = None):
    """Returns (cases, profile). cases = list of (label, gold, observed)."""
    if contract is None:
        from examples.salon.contract import SALON_CONTRACT, FIXTURES
        contract = SALON_CONTRACT
        fixtures = fixtures or FIXTURES
    if fixtures is None:
        from examples.salon.contract import FIXTURES
        fixtures = FIXTURES

    profile = {**DEFAULT_PROFILE, **(profile or {})}
    rng = random.Random(seed)
    grid = generate_scenario_grid(contract, fixtures)
    cases = []
    for scenario in grid:
        for i in range(n_per_cell):
            gold = make_gold(scenario, i, rng, contract, fixtures)
            obs = simulate_agent(gold, profile, rng, contract)
            label = f"{scenario['tool']}__{scenario['status']}"
            if "missing_arg" in scenario:
                label += f"__{scenario['missing_arg']}"
            label += f"#{i}"
            cases.append((label, gold, obs))
    return cases, profile


# --------------------------------------------------------------------------- #
# Validation: injected vs measured knobs                                       #
# --------------------------------------------------------------------------- #

def _denominators(cases) -> dict:
    n_rows = len(cases)
    n_act = n_clarify = n_respond = n_nonok = 0
    for _, gold, _ in cases:
        for st in gold.expected_trajectory:
            b = st.behavior.value
            if b == "act":
                n_act += 1
            elif b == "clarify":
                n_clarify += 1
            elif b == "respond":
                n_respond += 1
                if getattr(st.response_check, "expected_status", None) in ("empty", "error"):
                    n_nonok += 1
    return {"rows": n_rows, "act": n_act, "clarify": n_clarify,
            "respond": n_respond, "nonok": n_nonok}


def expected_knobs(cases, profile: dict, contract: Contract) -> list[dict]:
    """For each injected knob: effective probability, denominator, expected count.
    Accounts for two simulator couplings:
      - arg_resolution is an `elif` after wrong_tool  -> p *= (1 - wrong_tool)
      - false_success fires only if premature_stop didn't pre-empt the respond step

    Symmetric injected-side invariant: the structural couplings in
    simulate_agent's control flow are encoded here so expected_knobs and
    measured_knobs stay in sync.
    """
    d = _denominators(cases)
    wt = profile["wrong_tool"]
    ps = profile["premature_stop"]
    return [
        {"knob": "premature_stop", "p": ps, "denom": d["respond"]},
        {"knob": "wrong_tool", "p": wt, "denom": d["act"]},
        {"knob": "eager_act", "p": profile["eager_act"], "denom": d["clarify"]},
        {"knob": "arg_resolution", "p": (1 - wt) * profile["arg_resolution"], "denom": d["act"]},
        {"knob": "false_success", "p": profile["false_success"],
         "denom": d["nonok"] * (1 - ps)},
    ]


def measured_knobs(reports, contract: Contract) -> dict:
    """Pull the measured count for each knob out of the grader aggregates.
    Uses contract.resolution_buckets to sum arg-resolution failures without drift."""
    from grading import aggregate, behavior_pairs
    agg = aggregate(reports)
    arg = agg["arg_fail_buckets"]
    resp = agg["response_buckets"]
    beh = agg["behavior_buckets"]
    bt, bp = behavior_pairs(reports)
    eager = sum(1 for t, p in zip(bt, bp) if t == "clarify" and p == "act")
    return {
        "premature_stop": beh.get("step_missing", 0),
        "wrong_tool": arg.get("wrong_tool", 0),
        "eager_act": eager,
        "arg_resolution": sum(arg.get(b, 0) for b in contract.resolution_buckets),
        "false_success": (resp.get("fabricated_on_empty", 0)
                          + resp.get("false_success_claim", 0)),
    }


def validate(cases, reports, profile: dict, contract: Contract,
             z: float = 2.576) -> list[dict]:
    """Compare injected vs measured per knob. Pass if measured falls inside the
    z-sigma binomial interval around the expected count (default z=2.576 ~ 99%)."""
    measured = measured_knobs(reports, contract)
    rows = []
    for k in expected_knobs(cases, profile, contract):
        p, denom = k["p"], k["denom"]
        exp = p * denom
        sd = math.sqrt(denom * p * (1 - p)) if denom > 0 else 0.0
        lo, hi = max(0.0, exp - z * sd), exp + z * sd
        m = measured[k["knob"]]
        rows.append({
            "knob": k["knob"], "rate": round(p, 3), "denom": round(denom, 1),
            "expected": round(exp, 1), "lo": round(lo, 1), "hi": round(hi, 1),
            "measured": m, "pass": lo <= m <= hi,
        })
    return rows


if __name__ == "__main__":
    cases, profile = build_dataset(n_per_cell=6, seed=7)
    print(f"generated {len(cases)} runs across {len(set(l.split('#')[0] for l, _, _ in cases))} scenarios")
    print("profile:", profile)
