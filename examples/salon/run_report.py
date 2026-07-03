"""Salon report runner — point the library reporter at the salon agent."""
from __future__ import annotations

import sys

from rich.console import Console

from localize.report import run_eval
from examples.salon.contract import SALON_CONTRACT, FIXTURES
from examples.salon.dataset import CASES


def main():
    console = Console()
    if len(sys.argv) > 1 and sys.argv[1] in ("gen", "validate"):
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
        use_llm = "llm" in sys.argv
        judge = None
        if use_llm:
            from localize.judges import default_judge
            judge = default_judge(contract=SALON_CONTRACT)
        run_eval(SALON_CONTRACT, mode=sys.argv[1], fixtures=FIXTURES, n=n,
                 judge=judge, console=console)
    else:
        run_eval(SALON_CONTRACT, mode="curated", cases=CASES, console=console)


if __name__ == "__main__":
    main()
