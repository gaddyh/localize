"""
Backward-compat shim: re-exports from examples.salon.dataset.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from examples.salon.dataset import (  # noqa: F401
    CASES,
    GOLD_ROWS,
    R1, R1_clean, R1_bug,
    R2, R2_clean, R2_bug,
    R3, R3_clean, R3_bug,
    R4, R4_clean, R4_bug,
    R5, R5_clean, R5_bug,
)

if __name__ == "__main__":
    print(f"validated {len(GOLD_ROWS)} gold rows, {len(CASES)} observed runs")
    for label, _, o in CASES:
        print(f"  {label:42s} steps={len(o.steps)}")
