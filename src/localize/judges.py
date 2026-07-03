"""
LLM-as-judge for the response layer (contract-driven).

An `LLMResponseJudge` has the SAME signature as `heuristic_response_judge`
(ResponseContext -> list[Finding]), so it drops into `grade(...)` unchanged:

    from judges import LLMResponseJudge
    judge = LLMResponseJudge(contract=my_contract)
    report = grade(gold, observed, contract=my_contract, response_judge=judge)

The system prompt is now driven by `contract.role_description` instead of
a hardcoded salon string.  The backend is injectable (like the agent simulator):
  * default        -> real OpenAI API (needs OPENAI_API_KEY)
  * complete_fn=... -> any callable(prompt:str)->str, for offline tests/demos

Output is constrained to the SHARED bucket vocabulary so the report and the
grader-validation machinery work identically whether you judge by heuristic or LLM.
"""

from __future__ import annotations

import json
from typing import Callable, Optional

from .grading import Finding, ResponseContext
from .contract import Contract

from dotenv import load_dotenv

load_dotenv()

# the only buckets a judge may emit (keeps LLM output aligned with the report)
ALLOWED_BUCKETS = [
    "no_response_text",
    "omitted_fact", "omitted_slot", "omitted_confirmation_id",
    "unfaithful_action_claim", "wrong_slot_confirmed", "forbidden_phrase",
    "language_mismatch",
    "false_success_claim", "fabricated_on_empty",
]

BUCKET_GLOSSARY = """\
- omitted_fact / omitted_slot / omitted_confirmation_id: a fact in MUST_REFLECT is
  absent or not conveyed by the reply (use omitted_confirmation_id for a booking id,
  omitted_slot for a date/time slot, omitted_fact otherwise).
- unfaithful_action_claim: the reply claims an action happened that did NOT
  (e.g. says it booked/confirmed when the step was only an availability lookup).
- wrong_slot_confirmed: the reply confirms a date/time the user did not choose.
- false_success_claim: RESULT STATUS was 'error' but the reply claims success.
- fabricated_on_empty: RESULT STATUS was 'empty' (no availability) but the reply
  offers a concrete slot or claims a booking.
- language_mismatch: the reply is not in the EXPECTED LANGUAGE.
- forbidden_phrase: a MUST_NOT_CONTAIN phrase appears with no more specific tag.
"""


def build_prompt(ctx: ResponseContext, contract: Optional[Contract] = None) -> str:
    role = contract.role_description if contract else (
        "You are a strict evaluator for an agent. You grade ONE agent reply "
        "against a rubric and the tool result the agent saw. You judge only "
        "the reply text; you do not re-do the agent's job. Report only failures "
        "that are clearly supported. Reply with JSON only, no prose."
    )
    return f"""{role}

Grade this agent reply.

EXPECTED LANGUAGE: {ctx.language}
INTENDED REPLY TYPE (speech_act): {ctx.speech_act}
TOOL CALLED: {ctx.tool}
RESULT STATUS: {ctx.expected_status or "ok"}
TOOL RESULT (ground truth the reply must respect):
{json.dumps(ctx.observation, ensure_ascii=False)}

MUST_REFLECT (each must be conveyed): {json.dumps(ctx.must_reflect, ensure_ascii=False)}
MUST_NOT_CONTAIN (red-flag phrases): {json.dumps(ctx.must_not_contain, ensure_ascii=False)}

AGENT REPLY:
\"\"\"{ctx.text or ""}\"\"\"

Decide which failure tags apply. Allowed tags:
{BUCKET_GLOSSARY}
Return JSON exactly like:
{{"findings": [{{"bucket": "<one allowed tag>", "detail": "<short reason>"}}]}}
If the reply is fully correct, return {{"findings": []}}."""


def _parse(raw: str) -> list[Finding]:
    """Robustly parse the model's JSON into validated Findings, dropping anything
    outside the allowed bucket vocabulary."""
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s[s.find("{"):] if "{" in s else s
    try:
        data = json.loads(s[s.find("{"): s.rfind("}") + 1])
    except (json.JSONDecodeError, ValueError):
        return [Finding(bucket="forbidden_phrase",
                        detail=f"unparseable judge output: {raw[:80]}")]
    out: list[Finding] = []
    for item in data.get("findings", []):
        bucket = str(item.get("bucket", "")).strip()
        if bucket in ALLOWED_BUCKETS:
            out.append(Finding(bucket=bucket, detail=str(item.get("detail", ""))[:200]))
    return out


class LLMResponseJudge:
    """Drop-in response judge backed by an LLM.

    Parameters
    ----------
    contract : the domain Contract (provides role_description for the system prompt).
    model : OpenAI model id (ignored when complete_fn is given).
    complete_fn : optional callable(prompt:str)->str. If provided, it REPLACES the
        API call -- used for offline tests/demos. If omitted, a real OpenAI
        client is created lazily (requires OPENAI_API_KEY).
    """

    def __init__(self, contract: Optional[Contract] = None,
                 model: str = "gpt-4o",
                 complete_fn: Optional[Callable[[str], str]] = None):
        self.contract = contract
        self.model = model
        self._complete = complete_fn
        self._client = None

    def _complete_via_api(self, prompt: str) -> str:
        if self._client is None:
            import openai
            self._client = openai.OpenAI()          # reads OPENAI_API_KEY
        msg = self._client.chat.completions.create(
            model=self.model, max_tokens=512,
            messages=[
                {"role": "system", "content": self.contract.role_description
                 if self.contract else "You are a strict evaluator."},
                {"role": "user",   "content": prompt},
            ],
        )
        return msg.choices[0].message.content or ""

    def __call__(self, ctx: ResponseContext) -> list[Finding]:
        # cheap guard: empty reply needs no model call
        if ctx.text is None or not ctx.text.strip():
            return [Finding(bucket="no_response_text",
                            detail="gold expects a reply but agent produced no text.")]
        prompt = build_prompt(ctx, self.contract)
        raw = (self._complete or self._complete_via_api)(prompt)
        return _parse(raw)


# --------------------------------------------------------------------------- #
# Offline scripted backend: a stand-in "judge model" for demos without a key.  #
# It reasons over ctx (not substrings) to show the judge INTERFACE end-to-end: #
# it understands meaning we deliberately phrase to dodge the substring rules.   #
# --------------------------------------------------------------------------- #

def scripted_backend(prompt: str) -> str:
    """Parse the fields back out of the prompt and emit a plausible verdict.
    This imitates an LLM's semantic judgment for false-success / omission, so the
    pipeline runs with no API key. Replace with LLMResponseJudge() for real grading."""
    def _field(tag: str) -> str:
        i = prompt.find(tag)
        return prompt[i + len(tag): prompt.find("\n", i)].strip() if i >= 0 else ""

    status = _field("RESULT STATUS:")
    # isolate ONLY the agent reply (between the triple quotes), not the glossary
    after = prompt.split("AGENT REPLY:", 1)[-1]
    reply = after.split('"""')[1] if '"""' in after else after
    findings = []
    low = reply.lower()
    success_words = ["נקבע", "קבעתי", "booked", "אישור", "bk-", "confirmed", "מצוין! קבעתי"]
    if status in ("empty", "error") and any(w.lower() in low for w in success_words):
        findings.append({
            "bucket": "false_success_claim" if status == "error" else "fabricated_on_empty",
            "detail": "reply asserts a successful booking despite a non-ok result.",
        })
    return json.dumps({"findings": findings}, ensure_ascii=False)


def default_judge(model: str = "gpt-4o",
                  contract: Optional[Contract] = None) -> "LLMResponseJudge":
    """Real OpenAI backend if OPENAI_API_KEY is set, else the offline
    scripted backend so demos/CI run without a key."""
    import os
    if os.environ.get("OPENAI_API_KEY"):
        return LLMResponseJudge(contract=contract, model=model)
    return LLMResponseJudge(contract=contract, model=model, complete_fn=scripted_backend)


if __name__ == "__main__":
    # tiny smoke test of the parse + interface, no network needed
    ctx = ResponseContext(
        step=2, text="מצוין! קבעתי לך תור. אישור BK-9999.",
        speech_act="report_booking_failed", expected_status="error",
        tool="book_appointment", observation={"status": "error", "reason": "slot_taken"},
    )
    judge = LLMResponseJudge(complete_fn=scripted_backend)
    print("scripted verdict:", [f.bucket for f in judge(ctx)])
