"""
Rich + scikit-learn reporting for the booking-agent eval.

Renders two confusion matrices with per-class precision / recall / f1:
  1. BEHAVIOR  -> act / clarify / respond   (the Act-vs-Clarify metric)
  2. TOOL      -> check_availability / book_appointment / cancel_appointment
                  (catches wrong-tool errors the behavior matrix can't see,
                   because a book-instead-of-cancel still reads as 'act')

Plus per-layer clean-rates and failure-bucket frequencies.

Run:  python report.py
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from sklearn.metrics import classification_report, confusion_matrix
from rich import box
from rich.console import Console
from rich.table import Table

from grading import (
    aggregate,
    behavior_pairs,
    grade,
    per_layer_stats,
    tool_choice_pairs,
)
from examples_dataset import CASES
from examples.salon.contract import SALON_CONTRACT


def _ordered_labels(y_true, y_pred, tail=("(none)", "(no_call)")):
    """Stable label order: real classes alphabetical, sentinel labels last."""
    seen = set(y_true) | set(y_pred)
    real = sorted(l for l in seen if l not in tail)
    return real + [l for l in tail if l in seen]


def render_confusion(console: Console, y_true, y_pred, title: str) -> None:
    labels = _ordered_labels(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    table = Table(title=title, box=box.SIMPLE_HEAVY, title_style="bold")
    table.add_column("gold \\ pred", style="bold", justify="right")
    for l in labels:
        table.add_column(l, justify="right")
    table.add_column("support", justify="right", style="dim")

    for i, gl in enumerate(labels):
        row = [gl]
        support = int(cm[i].sum())
        for j in range(len(labels)):
            v = int(cm[i][j])
            cell = str(v)
            if v == 0:
                cell = "[dim]·[/dim]"
            elif i == j:
                cell = f"[green]{v}[/green]"      # correct
            else:
                cell = f"[red]{v}[/red]"          # confusion
            row.append(cell)
        row.append(str(support))
        table.add_row(*row)
    console.print(table)


def render_metrics(console: Console, y_true, y_pred, title: str) -> None:
    labels = [l for l in _ordered_labels(y_true, y_pred)
              if l not in ("(none)", "(no_call)")]  # don't score sentinels as classes
    rep = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0,
    )

    table = Table(title=title, box=box.SIMPLE_HEAVY, title_style="bold")
    for col in ("class", "precision", "recall", "f1", "support"):
        table.add_column(col, justify="right")

    def _row(name, d, style=""):
        table.add_row(
            f"[{style}]{name}[/{style}]" if style else name,
            f"{d['precision']:.3f}", f"{d['recall']:.3f}",
            f"{d['f1-score']:.3f}", f"{int(d['support'])}",
        )

    for c in labels:
        _row(c, rep[c])
    table.add_section()
    acc = rep.get("accuracy")
    if acc is not None:
        table.add_row("[bold]accuracy[/bold]", "", "", f"[bold]{acc:.3f}[/bold]",
                      f"{int(rep['macro avg']['support'])}")
    _row("macro avg", rep["macro avg"], style="cyan")
    _row("weighted avg", rep["weighted avg"], style="cyan")
    console.print(table)


def render_layers(console: Console, reports) -> None:
    stats = per_layer_stats(reports)
    table = Table(title="Clean-rate by localization layer", box=box.SIMPLE_HEAVY,
                  title_style="bold")
    for col in ("layer", "applicable", "clean", "clean_rate"):
        table.add_column(col, justify="right")
    for layer, s in stats.items():
        rate = s["clean_rate"]
        colored = (f"[green]{rate:.3f}[/green]" if rate and rate >= 0.8
                   else f"[red]{rate:.3f}[/red]" if rate is not None else "-")
        table.add_row(layer, str(s["applicable"]), str(s["clean"]), colored)
    console.print(table)


def render_buckets(console: Console, reports) -> None:
    agg = aggregate(reports)
    table = Table(title="Failure buckets", box=box.SIMPLE_HEAVY, title_style="bold")
    table.add_column("layer", style="bold")
    table.add_column("bucket")
    table.add_column("count", justify="right")
    for layer_name, key in (("behavior", "behavior_buckets"),
                            ("arg", "arg_fail_buckets"),
                            ("response", "response_buckets")):
        d = agg[key]
        if not d:
            table.add_row(layer_name, "[dim](clean)[/dim]", "0")
        for i, (bucket, cnt) in enumerate(d.items()):
            table.add_row(layer_name if i == 0 else "", bucket, str(cnt))
        table.add_section()
    console.print(table)


def render_status(console: Console, reports) -> None:
    """Observation-handling: given a non-ok tool result, did the agent believe it?"""
    from collections import Counter
    by_status = {}        # status -> [handled, mishandled]
    by_tool = {}          # (tool,status) -> [handled, mishandled]
    for r in reports:
        for rec in r.status_handling:
            s, tool, res = rec["status"], rec["tool"], rec["result"]
            by_status.setdefault(s, [0, 0])
            by_tool.setdefault((tool, s), [0, 0])
            idx = 0 if res == "handled" else 1
            by_status[s][idx] += 1
            by_tool[(tool, s)][idx] += 1

    if not by_status:
        return
    table = Table(title="Observation handling by result status (right tool, wrong belief)",
                  box=box.SIMPLE_HEAVY, title_style="bold")
    for col in ("tool", "result status", "handled", "mishandled", "handling_acc"):
        table.add_column(col, justify="right")
    for (tool, s), (h, m) in sorted(by_tool.items()):
        acc = h / (h + m) if (h + m) else 0
        colored = f"[green]{acc:.3f}[/green]" if acc >= 0.8 else f"[red]{acc:.3f}[/red]"
        table.add_row(tool, s, str(h), f"[red]{m}[/red]" if m else "0", colored)
    table.add_section()
    for s, (h, m) in sorted(by_status.items()):
        acc = h / (h + m) if (h + m) else 0
        table.add_row("[bold]ALL[/bold]", f"[bold]{s}[/bold]", str(h),
                      f"[red]{m}[/red]" if m else "0", f"[bold]{acc:.3f}[/bold]")
    console.print(table)


def render_validation(console: Console, rows: list[dict], n_runs: int) -> None:
    table = Table(
        title=f"Grader validation: injected vs measured ({n_runs} runs, 99% CI)",
        box=box.SIMPLE_HEAVY, title_style="bold")
    for col in ("injected knob", "rate", "denom", "expected",
                "99% CI", "measured", "result"):
        table.add_column(col, justify="right")
    n_pass = 0
    for r in rows:
        ok = r["pass"]
        n_pass += ok
        result = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
        meas = f"{r['measured']}" if ok else f"[red]{r['measured']}[/red]"
        table.add_row(r["knob"], f"{r['rate']:.3f}", f"{r['denom']:.0f}",
                      f"{r['expected']:.1f}", f"[{r['lo']:.1f}, {r['hi']:.1f}]",
                      meas, result)
    console.print(table)
    verdict = ("[green]grader is faithful[/green]" if n_pass == len(rows)
               else f"[red]{len(rows) - n_pass} knob(s) out of interval[/red]")
    console.print(f"  {n_pass}/{len(rows)} knobs within tolerance -> {verdict}\n")


def run(reports, console: Console, title: str) -> None:
    agg = aggregate(reports)
    console.rule(f"[bold]{title}[/bold]")
    console.print(
        f"rows: [bold]{agg['rows']}[/bold]   "
        f"strict_pass_rate: [bold]{agg['strict_pass_rate']}[/bold]   "
        f"outcome_pass_rate: [bold]{agg['outcome_pass_rate']}[/bold]\n"
    )
    bt, bp = behavior_pairs(reports)
    render_confusion(console, bt, bp, "Behavior confusion (Act / Clarify / Respond)")
    render_metrics(console, bt, bp, "Behavior metrics")
    tt, tp = tool_choice_pairs(reports)
    console.print()
    render_confusion(console, tt, tp, "Tool-choice confusion")
    render_metrics(console, tt, tp, "Tool-choice metrics")
    console.print()
    render_status(console, reports)
    console.print()
    render_layers(console, reports)
    console.print()
    render_buckets(console, reports)


def main() -> None:
    import sys
    console = Console()
    if len(sys.argv) > 1 and sys.argv[1] in ("gen", "validate"):
        from generators import build_dataset, validate
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
        use_llm = "llm" in sys.argv
        judge = None
        if use_llm:
            from judges import default_judge
            judge = default_judge(contract=SALON_CONTRACT)
        cases, profile = build_dataset(n_per_cell=n, seed=7, contract=SALON_CONTRACT)
        reports = [grade(gold, observed, contract=SALON_CONTRACT,
                         **({"response_judge": judge} if judge else {}))
                   for _, gold, observed in cases]
        tag = " [LLM judge]" if use_llm else ""
        if sys.argv[1] == "validate":
            console.rule(f"[bold]Grader validation ({len(cases)} runs){tag}[/bold]")
            rows = validate(cases, reports, profile, contract=SALON_CONTRACT)
            render_validation(console, rows, len(cases))
            console.print(f"[dim]injected agent profile: {profile}[/dim]")
        else:
            run(reports, console, f"Generated volume ({len(cases)} runs){tag}")
            console.print(f"\n[dim]injected agent profile: {profile}[/dim]")
    else:
        reports = [grade(gold, observed, contract=SALON_CONTRACT) for _, gold, observed in CASES]
        run(reports, console, "Curated dataset")


if __name__ == "__main__":
    main()
