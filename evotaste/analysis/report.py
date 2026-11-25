"""Output of the analysis commands: CSV tables and a short plain-text report under results/<run>/<command>/."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


def render(headers: list[str], rows: list[list], left: int = 1) -> list[str]:
    """Fixed-width table: the first `left` columns left-aligned, the others right-aligned."""
    rows = [[str(c) for c in r] for r in rows]
    widths = [max(len(str(c)) for c in col) for col in zip(headers, *rows)]

    def line(cells):
        return "  ".join(c.ljust(w) if i < left else c.rjust(w) for i, (c, w) in enumerate(zip(cells, widths))).rstrip()

    return [line([str(h) for h in headers])] + [line(r) for r in rows]


def pct(x, digits: int = 1) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{digits}f}"


def pci(v, lo, hi, digits: int = 1, sign: bool = False) -> str:
    """A percentage with its interval, e.g. 41.2 [38.0, 44.1]."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    f = f"{{:{'+' if sign else ''}.{digits}f}}"
    return f"{f.format(100 * v)} [{f.format(100 * lo)}, {f.format(100 * hi)}]"


def num(v, lo=None, hi=None, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    s = f"{v:.{digits}f}"
    return s if lo is None else f"{s} [{lo:.{digits}f}, {hi:.{digits}f}]"


def write(out: Path, tables: dict[str, pd.DataFrame], lines: list[str]) -> None:
    """Write every table as <name>.csv and the report as report.txt, then print the report."""
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
    text = "\n".join(lines) + "\n"
    (out / "report.txt").write_text(text)
    print(text + f"tables and report: {out}/")
