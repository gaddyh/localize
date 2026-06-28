# localize — a failure-localization eval harness for tool-using agents

A small, self-contained evaluation harness for a multi-turn **ReAct booking agent**
(a WhatsApp nail-salon bot). It doesn't just tell you *that* an agent run failed — it
tells you **which decision broke and why**: wrong behavior, wrong tool, a mis-resolved
date, a hallucinated argument, or a reply that lied about what happened.

The headline idea: an agent can call every tool correctly and still fail the user, so
the harness grades **three independent layers** of every step —

1. **Behavior** — did it *Act* when it should have *Clarified* (or vice-versa)?
2. **Arguments** — were the args right, and did each come from the right *source*
   (the user, a prior tool result, context, or a computed value)?
3. **Response** — did the text shown to the user faithfully reflect the tool result,
   or did it invent a slot / claim a booking that never happened? The response layer is
   **pluggable**: a fast deterministic heuristic, or a meaning-aware **LLM-as-judge**.

Plus tool-choice and tool-result-status checks, and a **self-validation mode** that
proves the grader itself is trustworthy before you point it at a real model — the same
mode that caught a faulty LLM judge during development (see below).

---

## Why this exists

Most agent evals report a single pass/fail. That number tells you nothing actionable:
if 30% of runs fail, *where do you spend the next week?* This harness is built around one
principle — **failure localization** — so the output is a profile like:

> behavior is healthy (0.96 clean), but arguments (0.81) and especially
> observation-handling (0.63) are where it falls apart — and 70% of the arg errors
> are relative-time resolution.

That sentence points at an engineering task. A bare pass-rate doesn't.

---

## What's in the box (7 files, ~2.5k LOC, no framework)

| File | Role |
|---|---|
| `dataset_models.py` | Pydantic schema for a **gold** row: the user turns, a scripted `env` (what the tools return), the correct trajectory, and per-step *forbidden* traps. Validators reject malformed data at load time. |
| `examples_dataset.py` | Hand-written **seed dataset**: 5 real gold rows across `check_availability` / `book_appointment` / `cancel_appointment` (incl. clarify cases) + 10 realistic agent runs (clean and buggy). |
| `generators.py` | Two jobs: (a) `make_gold()` stamps out **more gold rows** from scenario templates; (b) `simulate_agent()` is a **fake agent** that makes mistakes at tunable rates (a `profile`). Also the grader **validation** logic. |
| `grading.py` | The engine. `grade(gold, observed)` walks the two trajectories step-by-step and files every discrepancy into a labeled bucket across all layers. The response layer is a **pluggable judge** (`ResponseContext -> list[Finding]`). |
| `judges.py` | `LLMResponseJudge` — a meaning-aware response judge with the same signature as the heuristic, so it drops into `grade()` unchanged. **Injectable backend**: real LLM API when a key is set, offline scripted stand-in otherwise. |
| `report.py` | `rich` + `scikit-learn` rendering: behavior & tool confusion matrices, per-class precision/recall/F1, observation-handling, per-layer clean-rates, and the validation table. |
| `demo_llm_judge.py` | The head-to-head: a paraphrased false-success claim the heuristic misses and the LLM judge catches. |

The three core objects:

- **Gold** — the answer key for one situation (question *and* correct answer).
- **Observed** — what an agent actually did on that situation (`ObservedTrajectory`).
- **Grader** — `grade(gold, observed)` → a localized failure report.

---

## Quick start

```bash
git clone https://github.com/<you>/localize.git
cd localize
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python report.py             # dashboard over the 10 curated runs
python report.py gen 20      # generate 140 runs and report on them
python report.py validate 20 # prove the grader is faithful (injected vs measured)

# meaning-aware LLM judge for the response layer:
python demo_llm_judge.py     # heuristic misses a paraphrase; the LLM judge catches it
python report.py gen 20 llm  # full dashboard, replies graded by the LLM judge
python report.py validate 20 llm  # validate the grader WITH the LLM judge in the loop
```

Requires Python 3.10+. Dependencies: `pydantic`, `rich`, `scikit-learn` (and an LLM SDK
+ `*_API_KEY` only if you use the real LLM-judge backend; without a key it falls back to
an offline stand-in so everything still runs).

---

## How it works, end to end

```
                make_gold()                     simulate_agent(gold, profile)
 scenario  ──────────────────► GOLD ──┐        ┌──────────────────────────────► OBSERVED
 params                       (answer  │        │  a FAKE agent whose error        (what the
 (service/day/status)          key)    │        │  rates are dialed in by profile   agent did)
                                       ▼        ▼
                                     grade(gold, observed)
                                               │
                                               ▼
                         failure buckets → confusion matrices → metrics
```

### 1. A gold row is a scripted world + the correct trajectory

`env` fixes what each tool returns (slots found, no availability, or an error), so the
agent's later steps have something deterministic to react to. The gold trajectory then
says, per step: the right **behavior**, **tool**, **args** (each tagged with its
*source*), and a **forbidden** block (tools/args/values that would mean eager-acting or
hallucinating).

### 2. Arguments carry provenance

A "wrong arg" is really four different bugs. Each gold arg is tagged with where its value
should come from, so the grader can file the failure precisely:

| source | failure it localizes |
|---|---|
| `from_user` | extraction (missed what the user said) |
| `from_tool_result` | carry-over (lost state between turns) |
| `from_context` | context-lookup (ignored the WhatsApp profile) |
| `computed` (`relative_time` / `resolved`) | date math ("this week" / "Sunday" → wrong date) |
| `missing` → supplied anyway | **fabrication** (the eager-acting signature) |

### 3. The grader scores three layers per step

Behavior (act/clarify/respond), arguments (with the provenance buckets above), and the
**response text** the user sees (`must_reflect` facts present? `must_not_contain` traps
tripped? did it claim success after a tool error?). Tool-choice and a loose end-state
`outcome_check` round it out.

### 4. The report turns runs into a profile

Behavior and tool-choice **confusion matrices** (via `sklearn`), per-class
precision/recall/F1, an **observation-handling** table (right tool, wrong belief about
the result), and **per-layer clean-rates** that tell you which layer to fix first.

---

## The part that makes it trustworthy: grader self-validation

You can't measure with a ruler you don't trust. The `simulate_agent` fake makes mistakes
at rates *you choose* (`profile`), so the harness can check that the grader **measures
back the rates that were injected**:

```
   injected knob    rate   denom   expected         99% CI   measured   result
  premature_stop   0.080     140       11.2    [2.9, 19.5]         11     PASS
      wrong_tool   0.150     140       21.0   [10.1, 31.9]         21     PASS
       eager_act   0.250      40       10.0    [2.9, 17.1]         12     PASS
  arg_resolution   0.170     140       23.8   [12.4, 35.2]         26     PASS
   false_success   0.400      37       14.7    [7.1, 22.4]         12     PASS

  5/5 knobs within tolerance -> grader is faithful
```

Each measured count must land inside a 99% binomial confidence interval around the
expected count. This is the cheapest possible regression test for an eval harness, and it
drops straight into CI:

```python
from generators import build_dataset, validate
from grading import grade

def test_grader_is_faithful():
    cases, profile = build_dataset(n_per_cell=40, seed=7)
    reports = [grade(g, o) for _, g, o in cases]
    assert all(r["pass"] for r in validate(cases, reports, profile))
```

### This isn't hypothetical — it caught a real bug

When the LLM judge was first dropped in, validation immediately flagged it: `false_success`
measured **37** against an expected ~15 — the judge was firing on every non-ok reply
instead of the 40% that actually claimed success. The fix was a one-line bug in how the
judge isolated the reply text; after it, re-validation went green.

> I swapped in an LLM judge, the validation harness immediately flagged it as
> over-detecting (37 vs ~15), I found the bug, fixed it, and re-validated to green.
> The faulty grader was caught **before it ever touched a real model** — without the
> validation layer it would have happily reported triple the true false-success rate,
> with no way to know.

That's the whole point: a grader you haven't validated is just an opinion with a number on it.

### It validates the LLM judge too

Because the LLM judge emits the same buckets as the heuristic, the *same* command checks
it — here against a real LLM backend, not the offline stand-in:

```
$ python report.py validate 20 llm
─────────────── Grader validation (140 runs) [LLM judge] ───────────────
   injected knob    rate   denom   expected         99% CI   measured   result
  premature_stop   0.080     140       11.2    [2.9, 19.5]         11     PASS
      wrong_tool   0.150     140       21.0   [10.1, 31.9]         21     PASS
       eager_act   0.250      40       10.0    [2.9, 17.1]         12     PASS
  arg_resolution   0.170     140       23.8   [12.4, 35.2]         26     PASS
   false_success   0.400      37       14.7    [7.1, 22.4]         12     PASS

  5/5 knobs within tolerance -> grader is faithful
```

---

## The LLM judge: why it earns its place

The heuristic response layer matches **substrings**. It can't tell that
*"מעולה, קבעתי לך תור למחר"* ("Great, I've set you an appointment for tomorrow") is a
**false success claim** unless that exact phrase is in `must_not_contain`. An LLM reads
meaning. `python demo_llm_judge.py` shows the gap on a booking that the tool *rejected*:

```
heuristic  -> status_handling=handled    response_failures=[]
LLM judge  -> status_handling=mishandled response_failures=['false_success_claim']
```

The agent claimed a booking that never happened, phrased to dodge every marker substring.
The heuristic waves it through; the judge catches it — the exact failure your production
agent will actually produce.

`LLMResponseJudge` has the **same signature** as the heuristic
(`ResponseContext -> list[Finding]`) and emits the **same bucket vocabulary**, so it drops
into `grade(gold, observed, response_judge=judge)` with nothing downstream changing. Its
backend is **injectable**, exactly like the agent simulator: a real LLM API when a key is
set, an offline scripted stand-in otherwise (so the repo runs and self-tests with no key).
The judge's prompt is built from the rubric you *already wrote* in the gold row —
`must_reflect`, `must_not_contain`, `expected_status`, the tool observation — so your gold
data does double duty as the judge's grading instructions.

---

## Plugging in a real agent

The fake agent is a **calibration weight**, not the thing being weighed. To evaluate a
real ReAct agent, replace `simulate_agent` and nothing else: feed each gold row's user
turns and scripted `env` to your agent, capture its steps, and wrap them as an
`ObservedTrajectory` — the same shape the simulator already emits. `grade()` and the
whole report run unchanged.

```python
from dataset_models import DatasetRow
from grading import ObservedTrajectory, grade

def to_observed(gold: DatasetRow, agent) -> ObservedTrajectory:
    steps = []
    for i, gstep in enumerate(gold.expected_trajectory, start=1):
        action = agent.next_step(...)        # your agent's real decision
        steps.append({"step": i, "behavior": action.behavior,
                      "tool": action.tool, "args": action.args,
                      "response_text": action.text})
    return ObservedTrajectory(id=gold.id, steps=steps)
```

---

## Design notes & honest limitations

- **Hebrew-language dataset.** The salon bot speaks Hebrew; arg *values* are normalized to
  English snake_case while raw Hebrew stays in `user_message`. The response layer uses a
  light Hebrew-character heuristic for language checks.
- **Two response judges, different trade-offs.** The heuristic is deterministic, free, and
  fast but matches substrings, so it misses paraphrased failures. The LLM judge reads
  meaning and catches them, but costs API calls and is non-deterministic — for
  `validate ... llm` against a live model, pin temperature low and use a modest N, since
  the binomial CI assumes independent trials and model variance adds noise on top.
- **The offline judge backend is a crude imitation.** It exists only to exercise the judge
  *interface* without a key; its judgments are not a real model's. Use a real backend
  (set the API key) for actual grading.
- **Validation covers injected modes only.** The self-check verifies the failure modes the
  simulator produces. Perfect scores on some confusion cells (e.g. `act` recall) reflect
  that the simulator never produces those errors — not that the grader is infallible.
  Adding those injection paths closes the gap.
- **The simulator is a stand-in.** It load-tests the harness and tunes thresholds; it is
  not a measurement of any real model.

---

## Roadmap

- [ ] `run_agent.py` bridge for a real ReAct agent (the one wire not yet connected).
- [x] LLM-judge backend for the response layer — *and validated by the same harness.*
- [ ] More simulator injection paths so validation covers the full confusion matrix.
- [ ] Held-out split once a real model is in the loop (don't tune on what you report).

---

## License

MIT — see `LICENSE`.
