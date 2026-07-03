"""Generic CLI for the localize grader.

Usage:
    localize gen 8 --contract examples.salon.contract:SALON_CONTRACT
    localize validate 20 --contract examples.salon.contract:SALON_CONTRACT
    localize curated --contract examples.salon.contract:SALON_CONTRACT --cases examples.salon.dataset:CASES
"""
from __future__ import annotations

import argparse
import importlib


def _resolve(path: str):
    """Resolve 'module:attribute' to the actual object."""
    module_name, _, attr = path.partition(":")
    module = importlib.import_module(module_name)
    if attr:
        return getattr(module, attr)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(prog="localize", description="Grade task-agent eval.")
    parser.add_argument("mode", choices=["curated", "gen", "validate"],
                        help="curated: use pre-built cases; gen: build dataset; validate: build + validate")
    parser.add_argument("n", type=int, nargs="?", default=8,
                        help="runs per cell (gen/validate)")
    parser.add_argument("--contract", required=True,
                        help="module:attr for the Contract (e.g. examples.salon.contract:SALON_CONTRACT)")
    parser.add_argument("--cases", default=None,
                        help="module:attr for curated cases (required for curated mode)")
    parser.add_argument("--fixtures", default=None,
                        help="module:attr for fixtures (gen/validate). If omitted, looks for FIXTURES in the contract module.")
    parser.add_argument("--llm", action="store_true",
                        help="use LLM judge instead of heuristic")
    args = parser.parse_args()

    from rich.console import Console
    from .report import run_eval

    console = Console()
    contract = _resolve(args.contract)

    judge = None
    if args.llm:
        from .judges import default_judge
        judge = default_judge(contract=contract)

    cases = None
    if args.mode == "curated":
        if not args.cases:
            parser.error("--cases is required for curated mode")
        cases = _resolve(args.cases)

    fixtures = None
    if args.mode in ("gen", "validate"):
        if args.fixtures:
            fixtures = _resolve(args.fixtures)
        else:
            contract_module = importlib.import_module(args.contract.partition(":")[0])
            fixtures = getattr(contract_module, "FIXTURES", None)
            if fixtures is None:
                parser.error(
                    "--fixtures module:attr required for gen/validate "
                    "(no FIXTURES found in contract module)")

    run_eval(contract, mode=args.mode, cases=cases, fixtures=fixtures,
             n=args.n, judge=judge, console=console)


if __name__ == "__main__":
    main()
