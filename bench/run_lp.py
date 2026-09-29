"""Benchmark an LP method of NIRNAY against HiGHS on a directory of MPS files.

HiGHS is the comparator only: it supplies the reference objective and a reference time. Each
instance runs in its own process with a wall-clock limit, so one hard or pathological instance
cannot stall the sweep.

usage: python bench/run_lp.py data/netlib --method ipm --limit 300 --out results/netlib_ipm.csv
"""
from __future__ import annotations

import argparse
import csv
import gzip
import multiprocessing as mp
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def _nirnay(path, method, limit, q):
    try:
        from nirnay.io.mps import read_mps
        from nirnay import solve
        # warm up: the first call in a fresh process loads Numba-compiled code; that load is not
        # solving time, so it is paid on a tiny instance before the clock starts
        root = Path(__file__).resolve().parent.parent / "data"
        warm = root / ("qp/mm/QAFIRO.QPS" if method == "qp-ipm" else "netlib/afiro.mps.gz")
        if warm.exists():
            solve(read_mps(warm), method=method, time_limit=30)
        m = read_mps(path)
        t = time.perf_counter()
        r = solve(m, method=method, time_limit=limit)
        viol = m.violation(r.x) if r.x is not None else {"row": np.nan, "bound": np.nan}
        q.put({"status": r.status, "obj": r.objective, "time": time.perf_counter() - t,
               "iters": r.iterations, "row_viol": viol["row"], "bound_viol": viol["bound"],
               "m": m.m, "n": m.n, "nnz": m.A.nnz})
    except Exception as e:                                # report, don't crash the sweep
        q.put({"status": f"error: {type(e).__name__}: {e}"[:160]})


def _highs(path, limit):
    import highspy
    # HiGHS on Windows reads neither .gz nor the .QPS extension: hand it a plain .mps copy
    stem = Path(path).name.replace(".gz", "")
    if stem.lower().endswith(".qps"):
        stem = stem[:-4] + ".mps"
    tmp = Path(tempfile.gettempdir()) / stem
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as a, open(tmp, "wb") as b:
            shutil.copyfileobj(a, b)
    elif Path(path).name != stem:
        shutil.copyfile(path, tmp)
    else:
        tmp = Path(path)
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(limit))
    h.readModel(str(tmp))
    t = time.perf_counter()
    h.run()
    el = time.perf_counter() - t
    st = h.modelStatusToString(h.getModelStatus())
    obj = h.getInfo().objective_function_value
    return st, obj, el


def run_one(path, method, limit):
    q = mp.Queue()
    p = mp.Process(target=_nirnay, args=(str(path), method, limit, q))
    t = time.perf_counter()
    p.start()
    p.join(limit + 30)
    if p.is_alive():
        p.terminate()
        res = {"status": "timeout", "time": time.perf_counter() - t}
    else:
        res = q.get() if not q.empty() else {"status": "crashed"}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--method", default="ipm")
    ap.add_argument("--limit", type=float, default=300)
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--max-nnz", type=int, default=10**9)
    a = ap.parse_args()
    files = sorted([*Path(a.folder).glob("*.mps*"), *Path(a.folder).glob("*.QPS"), *Path(a.folder).glob("*.qps")])
    if a.only:
        files = [f for f in files if f.name.split(".")[0] in a.only]
    out = Path(a.out or f"results/{Path(a.folder).name}_{a.method}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["name", "m", "n", "nnz", "status", "obj", "ref_status", "ref_obj", "rel_err", "time",
              "ref_time", "iters", "row_viol", "bound_viol"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for f in files:
            name = f.name.split(".")[0]
            res = run_one(f, a.method, a.limit)
            try:
                rst, robj, rtime = _highs(f, a.limit)
            except Exception as e:
                rst, robj, rtime = f"error {e}", np.nan, np.nan
            obj = res.get("obj", np.nan)
            rel = abs(obj - robj) / max(1.0, abs(robj)) if np.isfinite(obj) and np.isfinite(robj) else np.nan
            row = {"name": name, "m": res.get("m"), "n": res.get("n"), "nnz": res.get("nnz"),
                   "status": res["status"], "obj": obj, "ref_status": rst, "ref_obj": robj, "rel_err": rel,
                   "time": res.get("time"), "ref_time": rtime, "iters": res.get("iters"),
                   "row_viol": res.get("row_viol"), "bound_viol": res.get("bound_viol")}
            w.writerow(row)
            fh.flush()
            ok = "OK " if res["status"] == "optimal" and rel is not np.nan and rel < 1e-6 else "-- "
            print(f"{ok}{name:12s} {res['status']:16s} obj {obj:+.8e} ref {robj:+.8e} "
                  f"err {rel:.1e} t {res.get('time', float('nan')):7.2f}s ref {rtime:6.2f}s", flush=True)


if __name__ == "__main__":
    main()
