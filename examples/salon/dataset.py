"""
Curated salon dataset (R1-R5) re-bound to SALON_CONTRACT.

Five gold rows spanning all three tools and the Act/Clarify ladder:
  R1  check_availability -> book   (all args, multi-turn)
  R2  check_availability           (service MISSING -> clarify)
  R3  book_appointment             (all args, single turn)
  R4  book_appointment             (time MISSING -> clarify)
  R5  cancel_appointment           (all args)

For each row we include a CLEAN agent run and a BUGGY one.
"""

from __future__ import annotations

from localize.dataset_models import DatasetRow
from localize.grading import ObservedTrajectory

REF = "2026-06-27T09:00:00+03:00"
CUST = {"source": "whatsapp", "customer_name": "דנה כהן"}


def arg(value, source, fail_bucket=None, **kw):
    d = {"value": value, "source": source}
    if fail_bucket:
        d["fail_bucket"] = fail_bucket
    d.update(kw)
    return d


# --------------------------------------------------------------------------- #
# EXAMPLE — the canonical gel-polish gold row (moved from dataset_models.py)   #
# --------------------------------------------------------------------------- #

EXAMPLE = {
        "id": "gel_polish__lookup_then_book__traj__01",
        "intent": "availability_check_then_book",
        "turn_pattern": "lookup_then_book",
        "env": {
            "reference_time": "2026-06-27T09:00:00+03:00",
            "customer_context": {"source": "whatsapp", "customer_name": "מאיה לוי"},
            "tool_outcomes": {
                "check_availability": {
                    "returns": {
                        "status": "ok",
                        "slots": [
                            {"date": "2026-06-29", "time": "14:00"},
                            {"date": "2026-07-01", "time": "10:30"},
                        ],
                    }
                },
                "book_appointment": {
                    "returns": {"status": "ok", "confirmation_id": "BK-5512"}
                },
            },
        },
        "expected_trajectory": [
            {
                "step": 1,
                "behavior": "act",
                "user_message": "יש לק ג׳ל השבוע?",
                "gloss": "Is there gel polish this week?",
                "tool": "check_availability",
                "args": {
                    "service": {
                        "value": "gel_polish",
                        "source": "from_user",
                        "source_detail": "step_1: 'לק ג׳ל'",
                    },
                    "date_range": {
                        "value": "2026-06-29..2026-07-04",
                        "source": "computed",
                        "compute_type": "relative_time",
                        "raw_span": "השבוע",
                        "source_detail": "resolve('this week', ref=env.reference_time)",
                        "fail_bucket": "relative_time_resolution",
                    },
                },
                "forbidden": {
                    "tools": ["book_appointment"],
                    "args": ["date", "time", "customer_name"],
                    "reason": "Availability question -> look up only.",
                },
            },
            {
                "step": 2,
                "behavior": "respond",
                "reacts_to": "observation_from_step_1",
                "response_check": {
                    "speech_act": "report_availability",
                    "language": "he",
                    "must_reflect": ["2026-06-29 14:00", "2026-07-01 10:30"],
                    "must_not_contain": ["BK-", "booked", "נקבע"],
                    "grounding": "every slot stated must exist in step_1 observation",
                    "faithfulness": "must_NOT_claim_a_booking",
                    "screens_for": ["hallucinated_slot", "unfaithful_action_claim"],
                },
                "forbidden": {
                    "tools": ["book_appointment"],
                    "behaviors": ["clarify"],
                    "reason": "Slots in hand -> report them.",
                },
            },
            {
                "step": 3,
                "behavior": "act",
                "user_message": "מעולה, תקבעי לי ליום ראשון ב-14:00",
                "gloss": "Great, book me for Sunday at 14:00",
                "tool": "book_appointment",
                "args": {
                    "service": {
                        "value": "gel_polish",
                        "source": "from_tool_result",
                        "source_detail": "carried from step_1, NOT re-asked",
                        "fail_bucket": "carryover",
                    },
                    "date": {
                        "value": "2026-06-29",
                        "source": "computed",
                        "compute_type": "resolved",
                        "raw_span": "יום ראשון",
                        "source_detail": "resolve('Sunday') against step_1 slots + ref",
                        "fail_bucket": "day_name_resolution",
                    },
                    "time": {
                        "value": "14:00",
                        "source": "from_user",
                        "source_detail": "step_3: 'ב-14:00'",
                    },
                    "customer_name": {
                        "value": "מאיה לוי",
                        "source": "from_context",
                        "source_detail": "env.customer_context.customer_name",
                        "fail_bucket": "context_lookup",
                    },
                },
                "forbidden": {
                    "tools": ["check_availability"],
                    "behaviors": ["clarify"],
                    "args_values": {"time": ["10:30"], "date": ["2026-07-01"]},
                    "reason": "All args resolvable -> act.",
                },
            },
            {
                "step": 4,
                "behavior": "respond",
                "reacts_to": "observation_from_step_3",
                "response_check": {
                    "speech_act": "confirm_booking",
                    "language": "he",
                    "must_reflect": ["BK-5512", "2026-06-29", "14:00"],
                    "must_not_contain": ["10:30", "2026-07-01"],
                    "grounding": "confirmation_id + slot trace to step_3 observation",
                    "faithfulness": "claim_of_booking_is_TRUE",
                    "screens_for": ["wrong_slot_confirmed", "omitted_confirmation_id"],
                },
                "forbidden": {
                    "tools": ["book_appointment"],
                    "reason": "Booking confirmed -> state it.",
                },
            },
        ],
        "outcome_check": {
            "final_state": "appointment_booked",
            "expected_effect": {
                "service": "gel_polish",
                "date": "2026-06-29",
                "time": "14:00",
            },
            "must_not_happen": [
                "booked_wrong_slot",
                "double_booked",
                "answered_without_calling_tool",
                "clarified_name_already_in_context",
                "final_reply_contradicts_tool_result",
            ],
        },
    }


# --------------------------------------------------------------------------- #
# Gold rows                                                                     #
# --------------------------------------------------------------------------- #

R1 = DatasetRow.model_validate(EXAMPLE)   # gel_polish lookup -> book

R2 = DatasetRow.model_validate({
    "id": "check_availability__service_missing__01",
    "intent": "availability_check",
    "turn_pattern": "clarify_then_lookup",
    "env": {
        "reference_time": REF, "customer_context": CUST,
        "tool_outcomes": {"check_availability": {"returns": {
            "status": "ok", "slots": [{"date": "2026-06-29", "time": "09:30"}]}}},
    },
    "expected_trajectory": [
        {"step": 1, "behavior": "clarify",
         "user_message": "יש לכם מקום השבוע?", "gloss": "Do you have availability this week?",
         "clarify_target": "service",
         "response_check": {"speech_act": "ask_service", "language": "he",
                            "must_not_contain": ["BK-", "נקבע"],
                            "screens_for": ["language_mismatch"]},
         "forbidden": {"tools": ["check_availability", "book_appointment"],
                       "args": ["service", "date", "time"],
                       "reason": "Service unknown -> cannot pick a calendar."}},
        {"step": 2, "behavior": "act",
         "user_message": "מניקור", "gloss": "Manicure", "tool": "check_availability",
         "args": {
             "service": arg("manicure", "from_user", "service_extraction",
                            source_detail="step_2: 'מניקור'"),
             "date_range": arg("2026-06-29..2026-07-04", "computed",
                               "relative_time_resolution", compute_type="relative_time",
                               raw_span="השבוע", source_detail="carried from step_1 'השבוע'"),
         },
         "forbidden": {"tools": ["book_appointment"], "args": ["date", "time"],
                       "reason": "Service now known -> look up."}},
        {"step": 3, "behavior": "respond", "reacts_to": "observation_from_step_2",
         "response_check": {"speech_act": "report_availability", "language": "he",
                            "must_reflect": ["2026-06-29 09:30"],
                            "must_not_contain": ["BK-", "נקבע"],
                            "grounding": "slots from step_2 observation",
                            "faithfulness": "no booking claimed",
                            "screens_for": ["omitted_slot", "unfaithful_action_claim"]}},
    ],
    "outcome_check": {"final_state": "slots_reported_to_user",
                      "must_not_happen": ["appointment_booked",
                                          "answered_without_calling_tool"]},
})

R3 = DatasetRow.model_validate({
    "id": "book__all_args__single_turn__01",
    "intent": "book",
    "turn_pattern": "single_turn_book",
    "env": {
        "reference_time": REF, "customer_context": CUST,
        "tool_outcomes": {"book_appointment": {"returns": {
            "status": "ok", "confirmation_id": "BK-7781"}}},
    },
    "expected_trajectory": [
        {"step": 1, "behavior": "act",
         "user_message": "תקבעי לי לק ג׳ל ליום ראשון ב-14:00",
         "gloss": "Book me gel polish Sunday at 14:00", "tool": "book_appointment",
         "args": {
             "service": arg("gel_polish", "from_user", "service_extraction",
                            source_detail="'לק ג׳ל'"),
             "date": arg("2026-06-29", "computed", "day_name_resolution",
                         compute_type="resolved", raw_span="יום ראשון",
                         source_detail="resolve('Sunday', ref)"),
             "time": arg("14:00", "from_user", source_detail="'14:00'"),
             "customer_name": arg("דנה כהן", "from_context", "context_lookup",
                                  source_detail="env.customer_context"),
         },
         "forbidden": {"tools": ["check_availability"], "behaviors": ["clarify"],
                       "reason": "All args resolvable from message + context -> act."}},
        {"step": 2, "behavior": "respond", "reacts_to": "observation_from_step_1",
         "response_check": {"speech_act": "confirm_booking", "language": "he",
                            "must_reflect": ["BK-7781", "2026-06-29", "14:00"],
                            "grounding": "confirmation_id from observation",
                            "faithfulness": "booking claim is TRUE",
                            "screens_for": ["omitted_confirmation_id", "language_mismatch"]}},
    ],
    "outcome_check": {"final_state": "appointment_booked",
                      "expected_effect": {"service": "gel_polish",
                                           "date": "2026-06-29", "time": "14:00"},
                      "must_not_happen": ["double_booked",
                                          "answered_without_calling_tool"]},
})

R4 = DatasetRow.model_validate({
    "id": "book__time_missing__01",
    "intent": "book",
    "turn_pattern": "clarify_then_book",
    "env": {
        "reference_time": REF, "customer_context": CUST,
        "tool_outcomes": {"book_appointment": {"returns": {
            "status": "ok", "confirmation_id": "BK-7782"}}},
    },
    "expected_trajectory": [
        {"step": 1, "behavior": "clarify",
         "user_message": "תקבעי לי פדיקור ליום שלישי", "gloss": "Book me pedicure on Tuesday",
         "clarify_target": "time",
         "response_check": {"speech_act": "ask_time", "language": "he",
                            "must_not_contain": ["BK-", "נקבע"],
                            "screens_for": ["language_mismatch"]},
         "forbidden": {"tools": ["book_appointment"], "args": ["time"],
                       "reason": "No time given and none in context -> must ask."}},
        {"step": 2, "behavior": "act",
         "user_message": "ב-11:00", "gloss": "At 11:00", "tool": "book_appointment",
         "args": {
             "service": arg("pedicure", "from_user", "service_extraction",
                            source_detail="step_1: 'פדיקור'"),
             "date": arg("2026-06-30", "computed", "day_name_resolution",
                         compute_type="resolved", raw_span="יום שלישי",
                         source_detail="resolve('Tuesday', ref)"),
             "time": arg("11:00", "from_user", source_detail="step_2: 'ב-11:00'"),
             "customer_name": arg("דנה כהן", "from_context", "context_lookup",
                                  source_detail="env.customer_context"),
         },
         "forbidden": {"tools": ["check_availability"],
                       "reason": "All args now present -> book."}},
        {"step": 3, "behavior": "respond", "reacts_to": "observation_from_step_2",
         "response_check": {"speech_act": "confirm_booking", "language": "he",
                            "must_reflect": ["BK-7782", "2026-06-30", "11:00"],
                            "grounding": "confirmation_id",
                            "faithfulness": "booking is TRUE",
                            "screens_for": ["omitted_confirmation_id"]}},
    ],
    "outcome_check": {"final_state": "appointment_booked",
                      "expected_effect": {"service": "pedicure",
                                           "date": "2026-06-30", "time": "11:00"},
                      "must_not_happen": ["double_booked",
                                          "clarified_name_already_in_context"]},
})

R5 = DatasetRow.model_validate({
    "id": "cancel__all_args__01",
    "intent": "cancel",
    "turn_pattern": "single_turn_cancel",
    "env": {
        "reference_time": REF, "customer_context": CUST,
        "tool_outcomes": {"cancel_appointment": {"returns": {
            "status": "ok", "cancelled": {"date": "2026-06-28", "time": "16:00"}}}},
    },
    "expected_trajectory": [
        {"step": 1, "behavior": "act",
         "user_message": "תבטלי לי את התור למחר ב-16:00",
         "gloss": "Cancel my appointment tomorrow at 16:00", "tool": "cancel_appointment",
         "args": {
             "date": arg("2026-06-28", "computed", "relative_time_resolution",
                         compute_type="relative_time", raw_span="מחר",
                         source_detail="resolve('tomorrow', ref)"),
             "time": arg("16:00", "from_user", source_detail="'16:00'"),
         },
         "forbidden": {"tools": ["book_appointment", "check_availability"],
                       "args": ["service"],
                       "reason": "Cancellation, not a new booking."}},
        {"step": 2, "behavior": "respond", "reacts_to": "observation_from_step_1",
         "response_check": {"speech_act": "confirm_cancellation", "language": "he",
                            "must_reflect": ["2026-06-28", "16:00"],
                            "must_not_contain": ["BK-", "נקבע תור חדש"],
                            "grounding": "cancelled slot from observation",
                            "faithfulness": "claims a CANCELLATION, not a booking",
                            "screens_for": ["unfaithful_action_claim",
                                            "wrong_slot_confirmed", "language_mismatch"]}},
    ],
    "outcome_check": {"final_state": "appointment_cancelled",
                      "must_not_happen": ["double_booked",
                                          "answered_without_calling_tool",
                                          "final_reply_contradicts_tool_result"]},
})


# --------------------------------------------------------------------------- #
# Observed agent runs                                                          #
# --------------------------------------------------------------------------- #

def obs(gold_id, steps):
    return ObservedTrajectory(id=gold_id, steps=steps)


# R1 -------------------------------------------------------------------------
R1_clean = obs(R1.id, [
    {"step": 1, "behavior": "act", "tool": "check_availability",
     "args": {"service": "gel_polish", "date_range": "2026-06-29..2026-07-04"}},
    {"step": 2, "behavior": "respond",
     "response_text": "יש תורים: ראשון 2026-06-29 14:00 או 2026-07-01 10:30."},
    {"step": 3, "behavior": "act", "tool": "book_appointment",
     "args": {"service": "gel_polish", "date": "2026-06-29", "time": "14:00",
              "customer_name": "מאיה לוי"}},
    {"step": 4, "behavior": "respond",
     "response_text": "נקבע ל-2026-06-29 בשעה 14:00. אישור: BK-5512."},
])
R1_bug = obs(R1.id, [
    {"step": 1, "behavior": "act", "tool": "book_appointment",
     "args": {"service": "gel_polish", "date": "2026-06-29", "time": "14:00",
              "customer_name": "מאיה לוי"}},
])

# R2 -------------------------------------------------------------------------
R2_clean = obs(R2.id, [
    {"step": 1, "behavior": "clarify", "response_text": "איזה טיפול תרצי? מניקור, פדיקור או לק ג׳ל?"},
    {"step": 2, "behavior": "act", "tool": "check_availability",
     "args": {"service": "manicure", "date_range": "2026-06-29..2026-07-04"}},
    {"step": 3, "behavior": "respond", "response_text": "יש מקום ביום שני 2026-06-29 09:30."},
])
R2_bug = obs(R2.id, [
    {"step": 1, "behavior": "act", "tool": "check_availability",
     "args": {"service": "manicure", "date_range": "2026-06-29..2026-07-04"}},
    {"step": 2, "behavior": "respond", "response_text": "יש מקום ביום שני."},
])

# R3 -------------------------------------------------------------------------
R3_clean = obs(R3.id, [
    {"step": 1, "behavior": "act", "tool": "book_appointment",
     "args": {"service": "gel_polish", "date": "2026-06-29", "time": "14:00",
              "customer_name": "דנה כהן"}},
    {"step": 2, "behavior": "respond", "response_text": "מצוין דנה, נקבע ל-2026-06-29 14:00. BK-7781."},
])
R3_bug = obs(R3.id, [
    {"step": 1, "behavior": "act", "tool": "check_availability",
     "args": {"service": "gel_polish", "date_range": "2026-06-29..2026-06-29"}},
])

# R4 -------------------------------------------------------------------------
R4_clean = obs(R4.id, [
    {"step": 1, "behavior": "clarify", "response_text": "באיזו שעה ביום שלישי?"},
    {"step": 2, "behavior": "act", "tool": "book_appointment",
     "args": {"service": "pedicure", "date": "2026-06-30", "time": "11:00",
              "customer_name": "דנה כהן"}},
    {"step": 3, "behavior": "respond", "response_text": "נקבע פדיקור ל-2026-06-30 11:00. BK-7782."},
])
R4_bug = obs(R4.id, [
    {"step": 1, "behavior": "act", "tool": "book_appointment",
     "args": {"service": "pedicure", "date": "2026-06-30", "time": "09:00",
              "customer_name": "דנה כהן"}},
    {"step": 2, "behavior": "respond", "response_text": "קבעתי לך פדיקור ב-09:00. BK-7782."},
])

# R5 -------------------------------------------------------------------------
R5_clean = obs(R5.id, [
    {"step": 1, "behavior": "act", "tool": "cancel_appointment",
     "args": {"date": "2026-06-28", "time": "16:00"}},
    {"step": 2, "behavior": "respond", "response_text": "ביטלתי את התור ל-2026-06-28 16:00."},
])
R5_bug = obs(R5.id, [
    {"step": 1, "behavior": "act", "tool": "book_appointment",
     "args": {"service": "manicure", "date": "2026-06-28", "time": "16:00",
              "customer_name": "דנה כהן"}},
    {"step": 2, "behavior": "respond", "response_text": "קבעתי לך תור ל-2026-06-28 16:00."},
])


CASES = [
    ("R1 check->book  CLEAN", R1, R1_clean),
    ("R1 check->book  BUG (books first)", R1, R1_bug),
    ("R2 clarify svc  CLEAN", R2, R2_clean),
    ("R2 clarify svc  BUG (eager check)", R2, R2_bug),
    ("R3 book single  CLEAN", R3, R3_clean),
    ("R3 book single  BUG (redundant check)", R3, R3_bug),
    ("R4 clarify time CLEAN", R4, R4_clean),
    ("R4 clarify time BUG (fabricated time)", R4, R4_bug),
    ("R5 cancel       CLEAN", R5, R5_clean),
    ("R5 cancel       BUG (books instead!)", R5, R5_bug),
]

GOLD_ROWS = [R1, R2, R3, R4, R5]


if __name__ == "__main__":
    print(f"validated {len(GOLD_ROWS)} gold rows, {len(CASES)} observed runs")
    for label, _, o in CASES:
        print(f"  {label:42s} steps={len(o.steps)}")
