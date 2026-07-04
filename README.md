# localize

**A failure-localization eval harness for tool-using agents.** Most agent evals give you
one number: *pass rate*. This one tells you **which decision broke and why** — wrong
behavior, wrong tool, a mis-resolved date, a hallucinated argument, or a reply that lied
about what the tool actually returned.

Write a **Contract** that describes your agent's tools, arguments, and expected behaviors.
The grader does the rest — five independent layers, zero engine code to touch. A bundled
salon-booking example proves it works, with a **self-validating grader** that proves itself
correct before you trust it on a real model.

---

## Why it matters

If your eval says "41% of runs fail," where do you spend next week? A single pass-rate
can't tell you. `localize` grades **five independent surfaces** of every run and reports a
*profile* instead:

> behavior is healthy (0.96 clean) — the damage is in **arguments** (0.81) and
> **responses** (0.72), and the worst single thing is observation-handling: when a tool
> returns *no availability* or *error*, the agent still claims success ~30% of the time.

That sentence points at an engineering task. And because the grader is **validated against
a known ground truth**, you can trust the number — see [docs/Design.md](docs/Design.md)
for how (and the real bug it caught).

---

## Run it in 60 seconds

```bash
git clone https://github.com/gaddyh/localize.git
cd localize
python -m venv .venv && source .venv/bin/activate
pip install -e .

localize curated --contract examples.salon.contract:SALON_CONTRACT --cases examples.salon.dataset:CASES
localize gen 20 --contract examples.salon.contract:SALON_CONTRACT
localize validate 20 --contract examples.salon.contract:SALON_CONTRACT
```

Or with the bundled salon runner (same thing, less typing):

```bash
python examples/salon/run_report.py              # curated, hand-written cases
python examples/salon/run_report.py gen 20       # 140 generated runs — metrics that don't wobble
python examples/salon/run_report.py validate 20  # prove the grader is faithful (injected vs measured)
```

Optional LLM-as-judge for the response layer (set an API key, else it falls back to an
offline stand-in so everything still runs):

```bash
pip install -e ".[llm]"
python examples/salon/demo_llm_judge.py          # a paraphrase the heuristic misses and the judge catches
localize validate 20 --contract examples.salon.contract:SALON_CONTRACT --llm
```

Python 3.10+. Deps: `pydantic`, `rich`, `scikit-learn`, `python-dotenv` (LLM extra: `openai`, `anthropic`).

---

## The Contract — write one, grade any agent

A `Contract` centralizes every piece of domain knowledge so the engine never hardcodes
your vocabulary. Define your tools, their argument provenance, success markers, and
outcome predicates. Swap the Contract → grade a different agent with zero engine changes.

```python
from localize import Contract, ToolArgSpec, ToolSpec

MY_CONTRACT = Contract(
    role_description="You are a strict evaluator for my agent...",
    tools={
        "search": ToolSpec(
            args={
                "query": ToolArgSpec(provenance_hint="from_user",
                                     fail_bucket="query_extraction"),
                "filters": ToolArgSpec(provenance_hint="computed",
                                       compute_type_hint="relative",
                                       fail_bucket="filter_resolution"),
            },
        ),
    },
    terminal_tools=["search"],
    success_lexicon=["found", "matched"],
    success_fields=["status"],
    language="en",
    outcome_predicates=["returned_wrong_results"],
)
```

The bundled example — `examples/salon/contract.py` — is a full working Contract for a
WhatsApp nail-salon booking agent with three tools, Hebrew-language responses, and
relative-time resolution. Clone the repo and run it to see the whole loop.

---

## Example output

`localize gen 20 --contract examples.salon.contract:SALON_CONTRACT` (140 simulated runs). Abridged:

```
rows: 140   strict_pass_rate: 0.414   outcome_pass_rate: 0.793

        Behavior confusion (Act / Clarify / Respond)
  gold \ pred   act   clarify   respond   (none)
          act   140         ·         ·        ·
      clarify    12        28         ·        ·     <- 12 eager-acts (acted, should've asked)
      respond     ·         ·       129       11     <- 11 premature stops

                 Tool-choice confusion          (catches book-vs-cancel — invisible to behavior)
  gold \ pred   book   cancel   check
        check      7        5      48            <- check mis-fired as book/cancel

       Clean-rate by localization layer
     layer   applicable   clean   clean_rate
  behavior          309     297        0.961    <- healthy
       arg          140     114        0.814
  response          169     122        0.722    <- weakest layer
```

And the self-check that makes the numbers trustworthy:

```
$ localize validate 20 --contract examples.salon.contract:SALON_CONTRACT
   injected knob    rate   denom   expected         99% CI   measured   result
  premature_stop   0.080     140       11.2    [2.9, 19.5]         11     PASS
      wrong_tool   0.150     140       21.0   [10.1, 31.9]         21     PASS
       eager_act   0.250      40       10.0    [2.9, 17.1]         12     PASS
  arg_resolution   0.170     140       23.8   [12.4, 35.2]         26     PASS
   false_success   0.400      37       14.7    [7.1, 22.4]         12     PASS

  5/5 knobs within tolerance -> grader is faithful
```

The fake agent makes mistakes at rates *you choose*, so the grader can be checked against a
known answer. This caught a real over-detecting LLM judge during development (37 vs ~15)
**before it ever touched a real model** — story in [docs/Design.md](docs/Design.md).

---

## What failures it localizes

| layer | catches | example |
|---|---|---|
| **Behavior** | Act-vs-Clarify errors | booked when it should have asked which day |
| **Tool choice** | wrong tool (both read as "act") | *cancelled* request → called `book_appointment` |
| **Arguments** | wrong value *and its cause*, via provenance | "Sunday" → wrong ISO date (`computed`); name not pulled from context (`from_context`) |
| **Response** | reply that misrepresents the tool result | claimed a booking the tool *rejected* |
| **Observation-handling** | right tool, wrong belief about the result | invented a slot after "no availability" |

Each failure is attributed to **one** layer (even though a wrong tool call can cascade
into arg and response failures) — which is exactly what lets per-layer scores point at the
*root* instead of the symptoms.

---

## Plug in a real agent

The bundled simulator is a calibration weight — it exists so the grader can be validated.
To evaluate a real agent, write your Contract (above), then run your agent over each
gold row's user turns + scripted tool results and capture its steps as an
`ObservedTrajectory` (the same shape the simulator emits). The grader and report
run unchanged — no engine code to touch.

```python
from localize import grade, ObservedTrajectory, default_judge

observed = ObservedTrajectory(id=gold.id, steps=[
    {"step": 1, "behavior": "act", "tool": "check_availability",
     "args": {...}, "response_text": None},
    {"step": 2, "behavior": "respond", "response_text": "..."},
])
report = grade(gold, observed, contract=MY_CONTRACT,
               response_judge=default_judge(contract=MY_CONTRACT))
```

Validate the ruler first, then measure with it.

---

## Learn more

- **[docs/Design.md](docs/Design.md)** — the full rationale: how the schema and scenarios
  were *derived* (not decreed), how "deterministic variety" works, why the self-validation
  isn't circular, and the LLM-judge design.
- Inspired by [τ²-bench](https://github.com/sierra-research/tau2-bench) (Sierra Research &
  Princeton) — *policy + tools + world-state + correct-outcome* — recast as a small,
  single-domain, **localization-first** harness with a self-validating grader.

## License

MIT — see [LICENSE](LICENSE).
