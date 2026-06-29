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

> "Independent" means each failure is *attributed to one layer*, not that one bad decision
> can't cascade. A wrong tool call often drags args and response down with it — which is
> exactly why per-layer scoring is useful: it lets you trace the cascade back to its
> root layer instead of seeing five symptoms and guessing.

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

## The mental model (read this first)

Everything in the repo is built from **three objects**. Get these and the rest follows.

- **Gold** — the answer key for one situation: the user's message(s), a *scripted world*
  (`env` — exactly what the tools will return, fixed in advance), and the *correct*
  trajectory step by step. This is the exam question **and** its model answer.
- **Observed** — what *an agent* actually did on that situation: the steps it took, tools
  it called, args it passed, text it replied. A candidate answer to be graded.
- **Grader** — `grade(gold, observed)`: lays the two side by side and files every
  difference into a labelled bucket. The localized failure report.

A "test case" is the **pair** `(gold, observed)`. The two halves come from two different
places, and keeping that straight is the key to understanding the whole repo.

---

## How the schema was derived (it grew from one example)

The gold schema in `dataset_models.py` wasn't designed up front — it was *discovered* by
trying to grade a single real message and asking, each time the grader couldn't localize
a failure, "what field is missing?" The first example was a Hebrew availability question,
*"יש לק ג׳ל השבוע?"* ("is there gel polish this week?"), and each gap forced a field into
existence:

- To say whether *acting* was even right, the row needed an explicit **behavior** label
  (act / clarify / respond) — not just "what tool was called."
- To catch *eager acting* (booking when only asked about availability), it needed a
  **forbidden** block — tools/args/values that should *not* appear.
- To tell "wrong date value" apart from "couldn't resolve 'this week' at all," each arg
  needed a **source / provenance** tag (`from_user`, `computed`, `from_context`, …).
- To grade the agent's *reply* — and especially replies that claim success after a tool
  failed — the row needed a scripted **`env`** (what the tool returns) plus a
  **`response_check`** rubric (`must_reflect`, `must_not_contain`, `expected_status`).

So the schema is the residue of one worked example, generalized. That's why every field
earns its place: each exists because, without it, some real failure couldn't be pinned to
a specific cause.

**Inspiration.** The shape echoes **τ-bench / τ²-bench** (Sierra Research & Princeton),
which structures each task as a *policy + tool set + a database/world state + tasks with a
single correct outcome*, and scores *tool use, policy compliance, and multi-turn
communication*. The mapping onto a gold row is direct: `env` is the world state, the
`forbidden` block is the policy, the expected trajectory is the correct outcome, and the
three scoring layers are the communication/tool/compliance axes. This project is a small,
single-domain, **localization-first** take on that idea — with an added twist τ²-bench
doesn't include: a self-validating grader (below).

---

## Where examples come from: a generator, not a pile of files

You could hand-write every `(gold, observed)` pair (the repo ships 5 + 10 of those as
seeds). But to get *volume* — dozens of instances per scenario, enough for the metrics to
mean something — there's a **generator**. It has two completely separate jobs, and
conflating them is the single most common point of confusion:

```
   make_gold(...)              simulate_agent(gold, profile)
  ┌──────────────┐            ┌────────────────────────────┐
  │ writes the   │            │ a FAKE agent that takes the │
  │ EXAM (gold)  │  ───────►  │ exam and makes mistakes at  │
  │              │   gold     │ RATES YOU CHOOSE (profile)  │
  └──────────────┘            └────────────────────────────┘
        │                                   │
        ▼                                   ▼
      gold                              observed   ──►  grade(gold, observed)
```

- `make_gold` produces **gold** (the answer key). It does **not** use the profile.
- `simulate_agent` produces **observed** (a fake attempt). The **profile** governs only
  this fake agent's error rates.

So the profile is not "how we generate examples." It's the dial-set for a *stand-in
agent*. `make_gold` writes the test; `simulate_agent` is a student who flunks it on
purpose, by a controlled amount.

---

## Where the scenarios came from (the tool registry × a completeness ladder)

The scenario list (`check_ok`, `book_ok`, `book_time_missing`, `cancel_ok`, …) isn't a
guess at "things that might go wrong." Like the schema, it was *derived* — by taking the
agent's **tool registry** and crossing it with an **argument-completeness ladder**.

Start from the tools the agent actually has and their required args:

```
check_availability(service, date_range)
book_appointment(service, date, time, customer_name)
cancel_appointment(date, time)
```

Then, for each tool, walk the ladder of *how much the user has supplied so far* — because
that is precisely what decides **Act vs Clarify**, the core behavior axis:

| rung | what the user gave | correct behavior |
|---|---|---|
| all args present | everything the tool needs | **Act** (call the tool) |
| 1 arg missing | enough to know the intent, not to act | **Clarify** (ask for the one gap) |
| 2+ args missing | barely-specified request | **Clarify** (ask for the most informative gap) |

That cross-product *is* the scenario grid. `book_appointment` with all four args →
`book_ok` (Act); `book_appointment` missing `time` → `book_time_missing` (Clarify);
`check_availability` missing `service` → `check_service_missing` (Clarify). Each cell is a
distinct, nameable test of one decision — and crucially, the **forbidden** block on a
"missing" cell is what catches *eager acting* (calling the tool anyway with a fabricated
value instead of asking).

Two more axes layer on top of completeness, once the basic grid exists:

- **Tool-result status** — for a tool that *can* act, what did it *return*? `ok` (slots
  found / booking confirmed), `empty` (no availability), or `error` (booking rejected).
  This is the `*_ok` vs `check_empty` vs `book_error` split, and it's what surfaces the
  "right tool, wrong belief about the result" failure (claiming success after an error).
- **Input ambiguity** — beyond *missing*, an arg can be *present but unclear* (an ambiguous
  service term, a relative date with two readings). That's a natural next rung on the same
  ladder, sitting between "present" and "missing," and a clean place to extend the grid.

So the scenarios are the systematic enumeration of *(tool) × (how complete the request is)
× (what the tool returns)* — not a wish-list. New tool? Add its rows. New failure surface?
Add an axis. The grid tells you exactly which cells you haven't covered yet.

---

## "Deterministic variety" — how that isn't a contradiction

The thing that trips people up: *how do you generate many different examples and have it
be reproducible?* Those sound opposite. The resolution is one idea, used in both halves of
the generator — a **seeded random generator**.

```python
rng = random.Random(7)   # a PRIVATE stream of "random" numbers with a fixed start
```

Same seed → the *exact same sequence* of draws, every run, on every machine, forever.
It looks random but it's a fixed, replayable stream. (It's also separate from Python's
global `random`, so nothing else in the program can disturb it.) That single object is
what makes both pieces below reproducible.

### The generator varies *fillers*, not *structure*

`make_gold` isn't inventing novel test logic. Each scenario (e.g. `book_ok`) is a **fixed
skeleton** — which behavior, which tool, which args carry which provenance, what's
forbidden. The only things that change are *fillers* drawn from small pools:

```python
SERVICES = {"gel_polish": ..., "manicure": ..., "pedicure": ...}
DAYS     = {"sunday": ..., "monday": ..., ...}
TIMES    = ["09:30", "11:00", "14:00", "16:30"]

def make_gold(scenario, idx, rng):
    svc = rng.choice(list(SERVICES))   # the only randomness: which filler
    day = rng.choice(list(DAYS))
    t   = rng.choice(TIMES)
    ...                                # stamp svc/day/t into the fixed skeleton
```

So "100 more examples" = the same handful of skeletons, filled with different
service/day/time draws. Row #3 and row #7 of `book_ok` test the **same decision**
(all-args-present → act → confirm); only the surface details differ. That's the point —
you want *many instances of the same situation* so a metric like "act precision" is
statistically meaningful, not seven unrelated one-offs.

And it's deterministic because the **one seeded `rng` is threaded through the whole loop**,
like dealing from a shuffled deck: each draw advances the stream, so the cards differ, but
re-running with the same seed deals the identical hand.

```python
for scenario in SCENARIOS:
    for i in range(n_per_cell):
        gold = make_gold(scenario, i, rng)   # same rng, advancing each call
```

### The simulator rolls the same dice against the profile

`simulate_agent` walks the gold trajectory and, at each step, **rolls** against the
relevant knob in the profile. The pattern is the same everywhere:

```python
if rng.random() < profile["wrong_tool"]:   # rng.random() in [0,1); < 0.15 ~ 15% of the time
    tool = a_different_tool                 # inject the error
else:
    tool = gold_step.tool                   # behave correctly
```

Step by step:

| gold step says | the simulator rolls | on a hit it… |
|---|---|---|
| `clarify` | `eager_act` (0.25) | acts instead of asking — fabricates the missing arg |
| `act` | `wrong_tool` (0.15), else `arg_resolution` | swaps the tool / shifts a date by a week |
| `respond` (after a failed tool) | `false_success` (0.40) | writes a reply that claims success anyway |
| `respond` (normal) | `omit_fact` (0.15) | drops a required fact from the reply |
| before any final reply | `premature_stop` (0.08) | quits the trajectory early |

The output is an `ObservedTrajectory` — the same shape your real agent's logs would take —
that is *mostly* correct but wrong in **exactly the proportions you dialed in**.

### Why the determinism is the whole point

This is the keystone. Because the simulator's mistakes are seeded, **you know the ground
truth of how many errors were injected.** Dial in `wrong_tool: 0.15` over 140 act-steps
and you *know* ~21 wrong-tool errors exist. Now run the grader: does it report ~21?

- If yes → the grader is faithful.
- If it reports 37 → you've caught a broken grader, with a *known* answer to check against.

A *truly* random simulator would give a different error count every run, leaving you
nothing fixed to validate against. So the seed does double duty: **reproducibility**
(anyone re-running gets your numbers) **and a known ground truth** (the injected counts the
validator checks). That second job is what makes the whole self-validation idea possible.

> One subtlety the validator accounts for: a few knobs are *coupled* by control flow.
> `arg_resolution` only rolls if the `wrong_tool` roll missed, so its effective rate is
> `0.20 × (1 − 0.15) = 0.17`; and `false_success` can only fire on a reply that
> `premature_stop` didn't already cut off. That's why the validation table below shows
> `0.170` for arg-resolution and a reduced denominator for false-success — otherwise it
> would compare against the wrong target.

---

## What's in the box (9 files, ~4k LOC, no framework)

| File | Role |
|---|---|
| `dataset_models.py` | Pydantic schema for a **gold** row: the user turns, a scripted `env` (what the tools return), the correct trajectory, and per-step *forbidden* traps. Validators reject malformed data at load time. |
| `examples_dataset.py` | Hand-written **seed dataset**: 5 real gold rows across `check_availability` / `book_appointment` / `cancel_appointment` (incl. clarify cases) + 10 realistic agent runs (clean and buggy). |
| `generators.py` | Two jobs: (a) `make_gold()` stamps out **more gold rows** from scenario templates; (b) `simulate_agent()` is a **fake agent** that makes mistakes at tunable rates (a `profile`). Also the grader **validation** logic. |
| `grading.py` | The engine. `grade(gold, observed)` walks the two trajectories step-by-step and files every discrepancy into a labeled bucket across all layers. The response layer is a **pluggable judge** (`ResponseContext -> list[Finding]`). |
| `judges.py` | `LLMResponseJudge` — a meaning-aware response judge with the same signature as the heuristic, so it drops into `grade()` unchanged. **Injectable backend**: real LLM API when a key is set, offline scripted stand-in otherwise. |
| `report.py` | `rich` + `scikit-learn` rendering: behavior & tool confusion matrices, per-class precision/recall/F1, observation-handling, per-layer clean-rates, and the validation table. |
| `demo_llm_judge.py` | The head-to-head: a paraphrased false-success claim the heuristic misses and the LLM judge catches. |

---

## Quick start

```bash
git clone https://github.com/gaddyh/localize.git
cd localize
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python report.py             # the 10 CURATED runs (hand-written, readable, real)
python report.py gen 20      # 140 GENERATED runs (volume, for stable metrics)
python report.py validate 20 # prove the grader is faithful (injected vs measured)

# meaning-aware LLM judge for the response layer:
python demo_llm_judge.py     # heuristic misses a paraphrase; the LLM judge catches it
python report.py gen 20 llm  # full dashboard, replies graded by the LLM judge
python report.py validate 20 llm  # validate the grader WITH the LLM judge in the loop
```

`report.py` (no args) reads the curated seeds in `examples_dataset.py` — a handful of
real, hand-written cases you can read end to end. `gen N` generates `N × 7` runs for
metrics that don't wobble on small samples. Same grader, same report, both ways.

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
tripped? did it claim success after a tool error?). Two more surfaces wrap these: a
**tool-choice** check (which tool, not just act-vs-clarify) and a loose end-state
**`outcome_check`** — so it's really three per-step layers plus two cross-cutting ones,
five scored surfaces in total.

### 4. The report turns runs into a profile

Behavior and tool-choice **confusion matrices** (via `sklearn`), per-class
precision/recall/F1, an **observation-handling** table (right tool, wrong belief about
the result), and **per-layer clean-rates** that tell you which layer to fix first.

### What a run actually looks like

`python report.py gen 20` (140 simulated runs, default profile). Abridged:

```
───────────────────── Generated volume (140 runs) ──────────────────────
rows: 140   strict_pass_rate: 0.414   outcome_pass_rate: 0.793

        Behavior confusion (Act / Clarify / Respond)
  gold \ pred   act   clarify   respond   (none)   support
          act   140         ·         ·        ·       140
      clarify    12        28         ·        ·        40     <- 12 eager-acts
      respond     ·         ·       129       11       140     <- 11 premature stops

         class   precision   recall      f1   support
           act       0.921    1.000   0.959       140
       clarify       1.000    0.700   0.824        40

                 Tool-choice confusion
  gold \ pred   book   cancel   check   (no_call)   support
         book     51        3       6           ·        60
       cancel      ·       20       ·           ·        20
        check      7        5      48           ·        60     <- check mis-fired as book/cancel

    Observation handling by result status (right tool, wrong belief)
                tool   result status   handled   mishandled   handling_acc
    book_appointment           error        13            5          0.722
  check_availability           empty        12            7          0.632

       Clean-rate by localization layer
     layer   applicable   clean   clean_rate
  behavior          309     297        0.961     <- healthy
       arg          140     114        0.814
  response          169     122        0.722     <- weakest layer
```

Read it as a diagnosis: *behavior* is healthy (0.96), the damage is in *arguments* and
*responses*, and within those the worst single thing is **observation handling** — when a
tool returns `empty`/`error`, the agent still claims success ~30% of the time. That
sentence is the deliverable. A bare "41% strict pass" would have hidden all of it.

---

## The part that makes it trustworthy: grader self-validation

You can't measure with a ruler you don't trust. Because the `simulate_agent` fake makes
mistakes at rates *you choose* (and, per the section above, in *countable* amounts), the
harness can check that the grader **measures back the rates that were injected**:

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

**Why these five knobs and not every bucket?** The validator checks the failure modes the
simulator *injects directly and countably* — one clean Bernoulli trial per opportunity.
The grader emits many more buckets (`forbidden_tool`, `wrong_slot_confirmed`,
`omitted_confirmation_id`, …), but those are **downstream consequences** of the injected
modes (an eager-act *causes* a forbidden-tool and a fabricated arg), not independently
dialed rates — so there's no clean injected number to check them against. The five are the
levers with a known ground truth; the rest ride along.

### "Isn't this circular?" — no, and here's why

The fair objection: the simulator and the grader were written by the same person against
the same gold, so couldn't a shared wrong assumption make validation pass while both are
wrong together? The reason it isn't circular is that the two sides reach the verdict by
**independent mechanisms**:

- the **simulator** injects an error by *mutating a trajectory* — swap a tool, shift a
  date, truncate a step — driven by a probability it was told;
- the **grader** detects an error by *comparing the observed trajectory against gold* and
  bucketing the diff — it never sees the profile or which step was mutated.

For validation to pass *spuriously*, a bug would have to corrupt the injection and the
detection in the **same direction by the same amount** — far less likely than either being
simply wrong. This isn't hypothetical hand-waving: the `false_success` bug (37 vs ~15)
passed every *other* knob and failed loudly on exactly the one that broke, because the
mechanisms diverged there. The validator can't prove the gold encodes the *right* policy
(that's a human judgement), but it does prove the grader faithfully measures whatever the
gold says — which is the part that silently rots otherwise.

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

The simulator was always a **stand-in** so the grader could be validated against a known
ground truth. Your real agent's mistakes are unknown — which is exactly why you validate
the ruler *first*, then measure with it.

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
- **LLM-judge cost & runtime.** The judge fires once per *respond* step, not per run, so
  `validate 20` (140 runs, ~140 respond steps) is roughly 140 small calls — seconds to a
  couple of minutes and cents, not dollars, on a small judge model. Scale linearly with N;
  the heuristic path is free and instant if you just want the structure.
- **Validation covers injected modes only.** The self-check verifies the failure modes the
  simulator produces. Perfect scores on some confusion cells (e.g. `act` recall) reflect
  that the simulator never produces those errors — not that the grader is infallible.
  Adding those injection paths closes the gap.
- **Synthetic calendar.** `DAYS` maps day-names to fixed ISO dates that need not match a
  real 2026 calendar; the harness only requires that gold and the resolution logic agree
  internally.

---

## Roadmap

- [ ] `run_agent.py` bridge for a real ReAct agent (the one wire not yet connected).
- [x] LLM-judge backend for the response layer — *and validated by the same harness.*
- [ ] More simulator injection paths so validation covers the full confusion matrix.
- [ ] Held-out split once a real model is in the loop (don't tune on what you report).

---

## License

MIT — see `LICENSE`.