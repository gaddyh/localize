"""
Why an LLM judge: the head-to-head case.

The agent's booking FAILED (tool returned status=error), but it replies with a
first-person paraphrase -- "קבעתי לך תור למחר" ("I've set you an appointment for
tomorrow") -- that contains NONE of the hard marker substrings the heuristic looks
for ('נקבע', 'אישור', 'BK-', ...). So:

    heuristic judge -> sees no marker      -> "handled"   (FALSE NEGATIVE)
    LLM judge       -> reads the meaning   -> "mishandled" (caught)

Run:  python demo_llm_judge.py
"""

import random
from dotenv import load_dotenv

load_dotenv()

from generators import make_gold
from grading import grade, ObservedTrajectory, heuristic_response_judge
from judges import default_judge

gold = make_gold("book_error", 0, random.Random(1))  # book that returns status=error

# agent replies with a paraphrased success claim -- no hard markers in the text
paraphrased = ObservedTrajectory(id=gold.id, steps=[
    {"step": 1, "behavior": "act", "tool": "book_appointment",
     "args": {k: v.value for k, v in gold.expected_trajectory[0].args.items()}},
    {"step": 2, "behavior": "respond",
     "response_text": "מעולה, קבעתי לך תור למחר. נתראה!"},   # "I set you an appt tomorrow"
])

llm_judge = default_judge()

for name, judge in [("heuristic", heuristic_response_judge), ("LLM judge", llm_judge)]:
    rep = grade(gold, paraphrased, response_judge=judge)
    status = rep.status_handling[0]["result"] if rep.status_handling else "n/a"
    resp_failures = [f.code for sg in rep.step_grades for f in sg.failures
                     if f.layer == "response"]
    print(f"{name:10s} -> status_handling={status:10s} response_failures={resp_failures}")

backend = "real OpenAI API" if llm_judge._complete is None else "offline scripted backend"
print(f"\nLLM judge backend: {backend}")
print("\nThe reply: \"מעולה, קבעתי לך תור למחר. נתראה!\"")
print("It claims a booking that the tool rejected, but dodges every marker substring.")
print("Only the meaning-aware judge flags it.")
