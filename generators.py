"""
Volume generator for the booking-agent eval.

Two pieces:
  make_gold(scenario, status, params)  -> a gold DatasetRow
  simulate_agent(gold, profile, rng)   -> an ObservedTrajectory whose errors are
                                          injected at tunable per-failure-mode rates

build_dataset(n_per_cell, profile, seed) crosses every (scenario, status) cell
n_per_cell times, so you get controlled volume. Because the agent's mistakes are
sampled from `profile`, the measured failure-bucket rates should track the
injected rates -- which is how you sanity-check that the GRADER itself is faithful.

Scenarios:  check_ok, check_empty, book_ok, book_error,
            cancel_ok, check_service_missing, book_time_missing
"""

from __future__ import annotations

import math
import random
from typing import Optional

from dataset_models import DatasetRow
from grading import ObservedTrajectory

REF = "2026-06-27T09:00:00+03:00"
CUST_NAMES = ["דנה כהן", "מאיה לוי", "נועה ברק", "רוני אבני", "תמר שגב"]
SERVICES = {"gel_polish": "לק ג׳ל", "manicure": "מניקור", "pedicure": "פדיקור"}
DAYS = {  # day-name -> (hebrew, iso)  (synthetic calendar, kept internally consistent)
    "sunday": ("יום ראשון", "2026-06-29"),
    "monday": ("יום שני", "2026-06-30"),
    "tuesday": ("יום שלישי", "2026-07-01"),
    "wednesday": ("יום רביעי", "2026-07-02"),
}
TIMES = ["09:30", "11:00", "14:00", "16:30"]
WEEK_RANGE = "2026-06-29..2026-07-04"


def _arg(value, source, fail_bucket=None, **kw):
    d = {"value": value, "source": source}
    if fail_bucket:
        d["fail_bucket"] = fail_bucket
    d.update(kw)
    return d


# --------------------------------------------------------------------------- #
# Gold construction                                                            #
# --------------------------------------------------------------------------- #

def make_gold(scenario: str, idx: int, rng: random.Random) -> DatasetRow:
    svc_key = rng.choice(list(SERVICES))
    svc_he = SERVICES[svc_key]
    name = rng.choice(CUST_NAMES)
    day_key = rng.choice(list(DAYS))
    day_he, day_iso = DAYS[day_key]
    t = rng.choice(TIMES)
    env_cust = {"source": "whatsapp", "customer_name": name}
    base_id = f"{scenario}__{idx:03d}"

    def env(tool, returns):
        return {"reference_time": REF, "customer_context": env_cust,
                "tool_outcomes": {tool: {"returns": returns}}}

    if scenario == "check_ok":
        return DatasetRow.model_validate({
            "id": base_id, "intent": "availability_check", "turn_pattern": "lookup",
            "env": env("check_availability",
                       {"status": "ok", "slots": [{"date": day_iso, "time": t}]}),
            "expected_trajectory": [
                {"step": 1, "behavior": "act",
                 "user_message": f"יש {svc_he} השבוע?", "tool": "check_availability",
                 "args": {"service": _arg(svc_key, "from_user", "service_extraction"),
                          "date_range": _arg(WEEK_RANGE, "computed", "relative_time_resolution",
                                             compute_type="relative_time", raw_span="השבוע")},
                 "forbidden": {"tools": ["book_appointment"], "args": ["date", "time"],
                               "reason": "availability only"}},
                {"step": 2, "behavior": "respond", "reacts_to": "observation_from_step_1",
                 "response_check": {"speech_act": "report_availability", "language": "he",
                                    "must_reflect": [f"{day_iso} {t}"],
                                    "screens_for": ["omitted_slot"]}},
            ],
            "outcome_check": {"final_state": "slots_reported_to_user",
                              "must_not_happen": ["appointment_booked"]},
        })

    if scenario == "check_empty":
        return DatasetRow.model_validate({
            "id": base_id, "intent": "availability_check", "turn_pattern": "lookup_empty",
            "env": env("check_availability", {"status": "empty", "slots": []}),
            "expected_trajectory": [
                {"step": 1, "behavior": "act",
                 "user_message": f"יש {svc_he} השבוע?", "tool": "check_availability",
                 "args": {"service": _arg(svc_key, "from_user", "service_extraction"),
                          "date_range": _arg(WEEK_RANGE, "computed", "relative_time_resolution",
                                             compute_type="relative_time", raw_span="השבוע")},
                 "forbidden": {"tools": ["book_appointment"],
                               "reason": "availability only"}},
                {"step": 2, "behavior": "respond", "reacts_to": "observation_from_step_1",
                 "response_check": {"speech_act": "report_no_availability", "language": "he",
                                    "expected_status": "empty",
                                    "screens_for": ["fabricated_on_empty"]}},
            ],
            "outcome_check": {"final_state": "no_availability_reported",
                              "must_not_happen": ["appointment_booked",
                                                  "booked_despite_no_availability",
                                                  "final_reply_contradicts_tool_result"]},
        })

    if scenario == "book_ok":
        cid = f"BK-{7000 + idx}"
        return DatasetRow.model_validate({
            "id": base_id, "intent": "book", "turn_pattern": "single_turn_book",
            "env": env("book_appointment", {"status": "ok", "confirmation_id": cid}),
            "expected_trajectory": [
                {"step": 1, "behavior": "act",
                 "user_message": f"תקבעי לי {svc_he} ל{day_he} ב-{t}", "tool": "book_appointment",
                 "args": {"service": _arg(svc_key, "from_user", "service_extraction"),
                          "date": _arg(day_iso, "computed", "day_name_resolution",
                                       compute_type="resolved", raw_span=day_he),
                          "time": _arg(t, "from_user"),
                          "customer_name": _arg(name, "from_context", "context_lookup")},
                 "forbidden": {"tools": ["check_availability"], "behaviors": ["clarify"],
                               "reason": "all args present"}},
                {"step": 2, "behavior": "respond", "reacts_to": "observation_from_step_1",
                 "response_check": {"speech_act": "confirm_booking", "language": "he",
                                    "must_reflect": [cid, day_iso, t],
                                    "screens_for": ["omitted_confirmation_id"]}},
            ],
            "outcome_check": {"final_state": "appointment_booked",
                              "expected_booking": {"service": svc_key, "date": day_iso, "time": t},
                              "must_not_happen": ["double_booked"]},
        })

    if scenario == "book_error":
        return DatasetRow.model_validate({
            "id": base_id, "intent": "book", "turn_pattern": "book_fails",
            "env": env("book_appointment", {"status": "error", "reason": "slot_taken"}),
            "expected_trajectory": [
                {"step": 1, "behavior": "act",
                 "user_message": f"תקבעי לי {svc_he} ל{day_he} ב-{t}", "tool": "book_appointment",
                 "args": {"service": _arg(svc_key, "from_user", "service_extraction"),
                          "date": _arg(day_iso, "computed", "day_name_resolution",
                                       compute_type="resolved", raw_span=day_he),
                          "time": _arg(t, "from_user"),
                          "customer_name": _arg(name, "from_context", "context_lookup")},
                 "forbidden": {"tools": ["check_availability"],
                               "reason": "all args present"}},
                {"step": 2, "behavior": "respond", "reacts_to": "observation_from_step_1",
                 "response_check": {"speech_act": "report_booking_failed", "language": "he",
                                    "expected_status": "error",
                                    "screens_for": ["false_success_claim"]}},
            ],
            "outcome_check": {"final_state": "booking_failed_reported",
                              "must_not_happen": ["final_reply_contradicts_tool_result"]},
        })

    if scenario == "cancel_ok":
        return DatasetRow.model_validate({
            "id": base_id, "intent": "cancel", "turn_pattern": "single_turn_cancel",
            "env": env("cancel_appointment",
                       {"status": "ok", "cancelled": {"date": day_iso, "time": t}}),
            "expected_trajectory": [
                {"step": 1, "behavior": "act",
                 "user_message": f"תבטלי את התור של {day_he} ב-{t}", "tool": "cancel_appointment",
                 "args": {"date": _arg(day_iso, "computed", "day_name_resolution",
                                       compute_type="resolved", raw_span=day_he),
                          "time": _arg(t, "from_user")},
                 "forbidden": {"tools": ["book_appointment", "check_availability"],
                               "args": ["service"], "reason": "cancellation, not booking"}},
                {"step": 2, "behavior": "respond", "reacts_to": "observation_from_step_1",
                 "response_check": {"speech_act": "confirm_cancellation", "language": "he",
                                    "must_reflect": [day_iso, t],
                                    "must_not_contain": ["BK-"],
                                    "screens_for": ["wrong_slot_confirmed"]}},
            ],
            "outcome_check": {"final_state": "appointment_cancelled",
                              "must_not_happen": ["double_booked"]},
        })

    if scenario == "check_service_missing":
        return DatasetRow.model_validate({
            "id": base_id, "intent": "availability_check", "turn_pattern": "clarify_then_lookup",
            "env": env("check_availability",
                       {"status": "ok", "slots": [{"date": day_iso, "time": t}]}),
            "expected_trajectory": [
                {"step": 1, "behavior": "clarify",
                 "user_message": "יש לכם מקום השבוע?", "clarify_target": "service",
                 "response_check": {"speech_act": "ask_service", "language": "he",
                                    "must_not_contain": ["BK-", "נקבע"]},
                 "forbidden": {"tools": ["check_availability", "book_appointment"],
                               "args": ["service"], "reason": "service unknown"}},
                {"step": 2, "behavior": "act",
                 "user_message": svc_he, "tool": "check_availability",
                 "args": {"service": _arg(svc_key, "from_user", "service_extraction"),
                          "date_range": _arg(WEEK_RANGE, "computed", "relative_time_resolution",
                                             compute_type="relative_time", raw_span="השבוע")},
                 "forbidden": {"tools": ["book_appointment"], "reason": "look up"}},
                {"step": 3, "behavior": "respond", "reacts_to": "observation_from_step_2",
                 "response_check": {"speech_act": "report_availability", "language": "he",
                                    "must_reflect": [f"{day_iso} {t}"],
                                    "screens_for": ["omitted_slot"]}},
            ],
            "outcome_check": {"final_state": "slots_reported_to_user",
                              "must_not_happen": ["appointment_booked"]},
        })

    if scenario == "book_time_missing":
        cid = f"BK-{7500 + idx}"
        return DatasetRow.model_validate({
            "id": base_id, "intent": "book", "turn_pattern": "clarify_then_book",
            "env": env("book_appointment", {"status": "ok", "confirmation_id": cid}),
            "expected_trajectory": [
                {"step": 1, "behavior": "clarify",
                 "user_message": f"תקבעי לי {svc_he} ל{day_he}", "clarify_target": "time",
                 "response_check": {"speech_act": "ask_time", "language": "he",
                                    "must_not_contain": ["BK-", "נקבע"]},
                 "forbidden": {"tools": ["book_appointment"], "args": ["time"],
                               "reason": "time unknown"}},
                {"step": 2, "behavior": "act",
                 "user_message": f"ב-{t}", "tool": "book_appointment",
                 "args": {"service": _arg(svc_key, "from_user", "service_extraction"),
                          "date": _arg(day_iso, "computed", "day_name_resolution",
                                       compute_type="resolved", raw_span=day_he),
                          "time": _arg(t, "from_user"),
                          "customer_name": _arg(name, "from_context", "context_lookup")},
                 "forbidden": {"tools": ["check_availability"], "reason": "now book"}},
                {"step": 3, "behavior": "respond", "reacts_to": "observation_from_step_2",
                 "response_check": {"speech_act": "confirm_booking", "language": "he",
                                    "must_reflect": [cid, day_iso, t],
                                    "screens_for": ["omitted_confirmation_id"]}},
            ],
            "outcome_check": {"final_state": "appointment_booked",
                              "expected_booking": {"service": svc_key, "date": day_iso, "time": t},
                              "must_not_happen": ["double_booked",
                                                  "clarified_name_already_in_context"]},
        })

    raise ValueError(f"unknown scenario {scenario}")


SCENARIOS = ["check_ok", "check_empty", "book_ok", "book_error",
             "cancel_ok", "check_service_missing", "book_time_missing"]

DEFAULT_PROFILE = {
    "eager_act": 0.25,        # clarify step -> acts instead
    "wrong_tool": 0.15,       # act step -> calls the wrong tool
    "arg_resolution": 0.20,   # corrupt a computed (date/range) arg
    "omit_fact": 0.15,        # respond -> drop a required fact
    "false_success": 0.40,    # non-ok respond -> claims success anyway
    "premature_stop": 0.08,   # truncate before the final reply
}


# --------------------------------------------------------------------------- #
# Agent simulator                                                             #
# --------------------------------------------------------------------------- #

def _gold_args(act_step) -> dict:
    return {k: v.value for k, v in act_step.args.items()}


def _wrong_tool_for(tool: str, rng: random.Random) -> str:
    alts = [t for t in ("check_availability", "book_appointment", "cancel_appointment")
            if t != tool]
    return rng.choice(alts)


def _corrupt_date(value: str, rng: random.Random) -> str:
    # shift a date or a range by a week -> a resolution-style error
    if ".." in value:
        return "2026-07-06..2026-07-11"
    return rng.choice([iso for _, iso in DAYS.values() if iso != value] or [value])


def simulate_agent(gold: DatasetRow, profile: dict, rng: random.Random) -> ObservedTrajectory:
    steps = []
    derailed_book = False  # agent booked eagerly -> may carry into later steps
    for st in gold.expected_trajectory:
        # premature stop before a final respond
        if st.behavior.value == "respond" and rng.random() < profile.get("premature_stop", 0):
            break

        if st.behavior.value == "clarify":
            if rng.random() < profile.get("eager_act", 0):
                # eager-act: skip the question and act (guessing/ fabricating the missing arg)
                # pick the tool implied by the row's intent
                tool = "book_appointment" if "book" in gold.intent else "check_availability"
                args = {"service": "manicure", "date_range": WEEK_RANGE} \
                    if tool == "check_availability" else \
                    {"service": "manicure", "date": "2026-06-30", "time": "09:00",
                     "customer_name": "דנה כהן"}
                steps.append({"step": st.step, "behavior": "act", "tool": tool, "args": args,
                              "response_text": None})
            else:
                steps.append({"step": st.step, "behavior": "clarify",
                              "response_text": "אפשר פרט נוסף?"})
            continue

        if st.behavior.value == "act":
            tool = st.tool
            args = _gold_args(st)
            if rng.random() < profile.get("wrong_tool", 0):
                tool = _wrong_tool_for(tool, rng)
            elif rng.random() < profile.get("arg_resolution", 0):
                for key in ("date", "date_range"):
                    if key in args:
                        args[key] = _corrupt_date(args[key], rng)
                        break
            if tool == "book_appointment":
                derailed_book = True
            steps.append({"step": st.step, "behavior": "act", "tool": tool, "args": args,
                          "response_text": None})
            continue

        # respond
        rc = st.response_check
        if rc.expected_status in ("empty", "error"):
            if rng.random() < profile.get("false_success", 0):
                text = "מצוין! קבעתי לך תור. אישור BK-9999."   # claims success after failure
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
                  seed: int = 7):
    """Returns (cases, profile). cases = list of (label, gold, observed)."""
    profile = {**DEFAULT_PROFILE, **(profile or {})}
    rng = random.Random(seed)
    cases = []
    for scenario in SCENARIOS:
        for i in range(n_per_cell):
            gold = make_gold(scenario, i, rng)
            obs = simulate_agent(gold, profile, rng)
            cases.append((f"{scenario}#{i}", gold, obs))
    return cases, profile


def _denominators(cases) -> dict:
    """Count how many opportunities each injected error mode had, from the gold."""
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


def expected_knobs(cases, profile: dict) -> list[dict]:
    """For each injected knob: effective probability, denominator, expected count.
    Accounts for two simulator couplings:
      - arg_resolution is an `elif` after wrong_tool  -> p *= (1 - wrong_tool)
      - false_success fires only if premature_stop didn't pre-empt the respond step
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


def measured_knobs(reports) -> dict:
    """Pull the measured count for each knob out of the grader aggregates."""
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
        "arg_resolution": (arg.get("relative_time_resolution", 0)
                           + arg.get("day_name_resolution", 0)),
        "false_success": (resp.get("fabricated_on_empty", 0)
                          + resp.get("false_success_claim", 0)),
    }


def validate(cases, reports, profile: dict, z: float = 2.576) -> list[dict]:
    """Compare injected vs measured per knob. Pass if measured falls inside the
    z-sigma binomial interval around the expected count (default z=2.576 ~ 99%)."""
    measured = measured_knobs(reports)
    rows = []
    for k in expected_knobs(cases, profile):
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
    print(f"generated {len(cases)} runs across {len(SCENARIOS)} scenarios")
    print("profile:", profile)
