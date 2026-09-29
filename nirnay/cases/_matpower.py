"""Reader for MATPOWER case files (.m), the format of the IEEE PES PGLib-OPF library.

Only the numeric matrices are read: mpc.baseMVA, mpc.bus, mpc.gen, mpc.gencost, mpc.branch.
Column meanings follow MATPOWER's CASEFORMAT (https://matpower.org/docs/ref/matpower7.1/lib/caseformat.html).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def read_matpower(path) -> dict:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    out: dict = {}
    m = re.search(r"mpc\.baseMVA\s*=\s*([0-9.eE+-]+)", text)
    out["baseMVA"] = float(m.group(1)) if m else 100.0
    for key in ("bus", "gen", "gencost", "branch"):
        m = re.search(rf"mpc\.{key}\s*=\s*\[(.*?)\];", text, re.S)
        if not m:
            continue
        rows = []
        for line in m.group(1).splitlines():
            line = line.split("%", 1)[0].strip().rstrip(";").strip()
            if line:
                rows.append([float(t) for t in line.replace(";", " ").split()])
        width = max(len(r) for r in rows)
        out[key] = np.array([r + [np.nan] * (width - len(r)) for r in rows])
    return out
