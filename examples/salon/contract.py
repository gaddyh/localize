"""
Salon booking-agent contract + fixture pack.

This module centralizes all domain knowledge for the WhatsApp nail-salon
booking agent.  Swap this Contract for another to grade a different agent
without touching any engine code (grading.py, generators.py, judges.py).
"""

from __future__ import annotations

from contract import Contract, ToolArgSpec, ToolSpec

SALON_CONTRACT = Contract(
    role_description=(
        "You are a strict evaluator for a WhatsApp salon-booking agent. "
        "You grade ONE agent reply against a rubric and the tool result the "
        "agent saw. You judge only the reply text; you do not re-do the "
        "agent's job. Report only failures that are clearly supported. "
        "Reply with JSON only, no prose."
    ),
    tools={
        "check_availability": ToolSpec(
            args={
                "service": ToolArgSpec(
                    provenance_hint="from_user",
                    fail_bucket="service_extraction",
                ),
                "date_range": ToolArgSpec(
                    provenance_hint="computed",
                    compute_type_hint="relative_time",
                    fail_bucket="relative_time_resolution",
                ),
            },
            applicable_statuses=["ok", "empty"],
        ),
        "book_appointment": ToolSpec(
            args={
                "service": ToolArgSpec(
                    provenance_hint="from_user",
                    fail_bucket="service_extraction",
                ),
                "date": ToolArgSpec(
                    provenance_hint="computed",
                    compute_type_hint="resolved",
                    fail_bucket="day_name_resolution",
                ),
                "time": ToolArgSpec(
                    provenance_hint="from_user",
                ),
                "customer_name": ToolArgSpec(
                    provenance_hint="from_context",
                    fail_bucket="context_lookup",
                ),
            },
            applicable_statuses=["ok", "error"],
            prerequisite_tools=["check_availability"],
        ),
        "cancel_appointment": ToolSpec(
            args={
                "date": ToolArgSpec(
                    provenance_hint="computed",
                    compute_type_hint="resolved",
                    fail_bucket="day_name_resolution",
                ),
                "time": ToolArgSpec(
                    provenance_hint="from_user",
                ),
            },
            applicable_statuses=["ok"],
        ),
    },
    terminal_tools=["book_appointment"],
    success_lexicon=["bk-", "booked", "נקבע", "אישור", "confirmed"],
    success_fields=["confirmation_id"],
    language="he",
    outcome_predicates=[
        "duplicate_terminal_action",
        "answered_without_calling_tool",
        "final_reply_contradicts_tool_result",
        "terminal_action_on_unavailable_result",
        "effect_mismatch",
        # legacy names kept for backward compat with hand-authored rows
        "double_booked",
        "booked_despite_no_availability",
        "booked_wrong_slot",
    ],
)

# resolution_buckets is computed: ["relative_time_resolution", "day_name_resolution"]
# (2 buckets, filtered on compute_type_hint — not fail_bucket, which would
#  also pull in service_extraction and context_lookup that the arg_resolution
#  knob never injects.)


# --------------------------------------------------------------------------- #
# Fixture pack: domain values the generator needs to produce concrete rows.    #
# --------------------------------------------------------------------------- #

FIXTURES = {
    "reference_time": "2026-06-27T09:00:00+03:00",
    "customer_names": ["דנה כהן", "מאיה לוי", "נועה ברק", "רוני אבני", "תמר שגב"],
    "services": {"gel_polish": "לק ג׳ל", "manicure": "מניקור", "pedicure": "פדיקור"},
    "days": {
        "sunday": ("יום ראשון", "2026-06-29"),
        "monday": ("יום שני", "2026-06-30"),
        "tuesday": ("יום שלישי", "2026-07-01"),
        "wednesday": ("יום רביעי", "2026-07-02"),
    },
    "times": ["09:30", "11:00", "14:00", "16:30"],
    "week_range": "2026-06-29..2026-07-04",
    "context_source": "whatsapp",
}
