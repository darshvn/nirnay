"""Benchmark NIRNAY's branch-and-bound against HiGHS on a directory of MIP files.

Same process isolation as run_lp.py; HiGHS gets the same time limit and supplies the reference.
An instance counts as solved when NIRNAY proves optimality within the 1e-4 relative gap and its
objective agrees with the reference to 1e-4.

usage: python bench/run_mip.py data/miplib --limit 300 --out results/miplib_bnb.csv
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from run_lp import _highs


def _nirnay(path, limit, q):
    try:
        from nirnay.io.mps import read_mps
        from nirnay import solve
        warm = Path(__file__).resolve().parent.parent / "data" / "miplib" / "p0033.mps.gz"
        if warm.exists():
            solve(read_mps(warm), method="bnb", time_limit=30)
        m = read_mps(path)
        t = time.perf_counter()
        r = solve(m, method="bnb", time_limit=limit)
        viol = m.violation(r.x) if r.x is not None else {"row": np.nan, "bound": np.nan, "integrality": np.nan}
        q.put({"status": r.status, "obj": r.objective, "bound": r.bound, "time": time.perf_counter() - t,
               "nodes": r.nodes, "iters": r.iterations, "row_viol": viol["row"], "int_viol": viol["integrality"],
               "m": m.m, "n": m.n, "nint": int(m.integer.sum()), "nnz": m.A.nnz})
    except Exception as e:
        import traceback
        q.put({"status": f"error: {type(e).__name__}: {e}"[:160], "tb": traceback.format_exc()[-400:]})


def run_one(path, limit):
    q = mp.Queue()
    p = mp.Process(target=_nirnay, args=(str(path), limit, q))
    t = time.perf_counter()
    p.start()
    p.join(limit + 60)
    if p.is_alive():
        p.terminate()
        return {"status": "timeout", "time": time.perf_counter() - t}
    return q.get() if not q.empty() else {"status": "crashed"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--limit", type=float, default=300)
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    files = sorted(Path(a.folder).glob("*.mps*"))
    if a.only:
        files = [f for f in files if f.name.split(".")[0] in a.only]
    out = Path(a.out or f"results/{Path(a.folder).name}_bnb.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["name", "m", "n", "nint", "nnz", "status", "obj", "bound", "gap", "ref_status", "ref_obj",
              "rel_err", "time", "ref_time", "nodes", "iters", "row_viol", "int_viol"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for f in files:
            name = f.name.split(".")[0]
            res = run_one(f, a.limit)
            try:
                rst, robj, rtime = _highs(f, a.limit)
            except Exception as e:
                rst, robj, rtime = f"error {e}", np.nan, np.nan
            obj, bnd = res.get("obj", np.nan), res.get("bound", np.nan)
            ok_num = lambda v: v is not None and np.isfinite(v)
            rel = abs(obj - robj) / max(1.0, abs(robj)) if ok_num(obj) and ok_num(robj) else np.nan
            gap = abs(obj - bnd) / max(1.0, abs(obj)) if ok_num(obj) and ok_num(bnd) else np.nan
            row = dict(res, name=name, obj=obj, bound=bnd, gap=gap, ref_status=rst, ref_obj=robj,
                       rel_err=rel, ref_time=rtime)
            w.writerow(row)
            fh.flush()
            ok = "OK " if res["status"] == "optimal" and rel == rel and rel <= 1e-4 else "-- "
            print(f"{ok}{name:12s} {res['status'][:40]:16s} obj {obj:+.8e} ref {robj:+.8e} gap {gap:.1e} "
                  f"nodes {res.get('nodes', 0):7d} t {res.get('time', float('nan')):7.2f}s ref {rtime:6.2f}s "
                  f"{res.get('tb', '')}", flush=True)


if __name__ == "__main__":
    main()
