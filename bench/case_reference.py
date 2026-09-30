"""Reference optima for the case studies, computed by HiGHS (comparison only; highspy is an optional
`bench` dependency and is imported lazily).

    python bench/case_reference.py            # rebuild every case, export MPS, solve, write CSV
    python bench/case_reference.py unit_commitment

HiGHS reads the exported MPS file (not an in-memory copy), so the reference value is the optimum of
exactly the file in data/cases/.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nirnay.cases import CASES, build, info  # noqa: E402
from nirnay.io.mps_write import write_mps  # noqa: E402

CASE_DIR = Path(__file__).resolve().parents[1] / "data" / "cases"
REFERENCE_CSV = CASE_DIR / "reference.csv"
FIELDS = ["case", "class", "rows", "cols", "integer_cols", "nnz", "q_nnz", "highs_status",
          "objective", "mip_rel_gap_target", "mip_gap", "mip_bound", "highs_time_s", "highs_version",
          "mps_file"]


def solve_mps(path, time_limit: float = 3600.0, mip_rel_gap: float = 1e-6, threads: int | None = None):
    import highspy
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", float(mip_rel_gap))
    if threads:
        h.setOptionValue("threads", int(threads))
    h.readModel(str(path))
    t = time.perf_counter()
    h.run()
    el = time.perf_counter() - t
    inf = h.getInfo()
    status = h.modelStatusToString(h.getModelStatus())
    gap = getattr(inf, "mip_gap", float("nan"))
    bound = getattr(inf, "mip_dual_bound", float("nan"))
    return {"status": status, "objective": inf.objective_function_value, "time": el,
            "mip_gap": gap, "mip_bound": bound, "version": h.version(),
            "x": list(h.getSolution().col_value)}


def options(name: str) -> dict:
    """Per-case HiGHS settings: a case module may define REFERENCE = {"mip_rel_gap": ..., ...}."""
    from nirnay.cases import module
    opts = {"time_limit": 3600.0, "mip_rel_gap": 1e-6}
    opts.update(getattr(module(name), "REFERENCE", {}))
    return opts


def run(names):
    rows = {}
    if REFERENCE_CSV.exists():
        with open(REFERENCE_CSV, newline="") as fh:
            rows = {r["case"]: r for r in csv.DictReader(fh)}
    for name in names:
        model = build(name)
        mps = CASE_DIR / f"{name}.mps"
        write_mps(model, mps)
        opts = options(name)
        r = solve_mps(mps, opts["time_limit"], opts["mip_rel_gap"])
        rows[name] = {
            "case": name, "class": info(name)["class"], "rows": model.m, "cols": model.n,
            "integer_cols": int(model.integer.sum()), "nnz": model.A.nnz,
            "q_nnz": model.Q.nnz if model.is_qp else 0, "highs_status": r["status"],
            "objective": repr(r["objective"]),
            "mip_rel_gap_target": "" if not model.is_mip else repr(opts["mip_rel_gap"]),
            "mip_gap": "" if not model.is_mip else f"{r['mip_gap']:.3g}",
            "mip_bound": "" if not model.is_mip else repr(r["mip_bound"]),
            "highs_time_s": f"{r['time']:.2f}", "highs_version": r["version"],
            "mps_file": f"data/cases/{mps.name}"}
        print(model.summary(), "->", r["status"], r["objective"], f"{r['time']:.2f}s", flush=True)
    with open(REFERENCE_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for name in CASES:
            if name in rows:
                w.writerow(rows[name])


if __name__ == "__main__":
    run(sys.argv[1:] or CASES)
