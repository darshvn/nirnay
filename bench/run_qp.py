"""Benchmark NIRNAY's QP interior point on the Maros-Meszaros set, with Clarabel as reference.

HiGHS's QP solver is active-set and times out on the larger instances, so the reference optimum
comes from Clarabel (an interior-point conic solver). The model Clarabel sees is read by HiGHS's
own parser, not by ours, so a reader bug in NIRNAY cannot hide by affecting both sides.

usage: python bench/run_qp.py data/qp/mm --limit 120 --out results/maros_qp.csv
"""
from __future__ import annotations

import argparse
import csv
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from run_lp import run_one


def reference(path, limit):
    import clarabel
    import highspy
    import scipy.sparse as sp
    tmp = Path(tempfile.gettempdir()) / (Path(path).stem + ".mps")
    shutil.copyfile(path, tmp)
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.readModel(str(tmp))
    mdl = h.getModel()
    lp, hess = mdl.lp_, mdl.hessian_
    n, m = lp.num_col_, lp.num_row_
    A = sp.csc_matrix((np.array(lp.a_matrix_.value_), np.array(lp.a_matrix_.index_),
                       np.array(lp.a_matrix_.start_)), shape=(m, n))
    c = np.array(lp.col_cost_)
    sense = -1.0 if lp.sense_ == highspy.ObjSense.kMaximize else 1.0
    if hess.dim_ > 0:
        H = sp.csc_matrix((np.array(hess.value_), np.array(hess.index_), np.array(hess.start_)), shape=(n, n))
        H = H + sp.triu(H.T, 1) if (sp.tril(H, -1).nnz and not sp.triu(H, 1).nnz) else H
        P = sp.triu(H + H.T - sp.diags(H.diagonal())) if not (sp.tril(H, -1).nnz and sp.triu(H, 1).nnz) else sp.triu(H)
    else:
        P = sp.csc_matrix((n, n))
    rl, ru = np.array(lp.row_lower_), np.array(lp.row_upper_)
    lb, ub = np.array(lp.col_lower_), np.array(lp.col_upper_)
    big = 1e20
    blocks, rhs, zero_n, nonneg_n = [], [], 0, 0
    I = sp.identity(n, format="csc")
    eq = (rl == ru) & (np.abs(rl) < big)
    feq = (lb == ub) & (np.abs(lb) < big)
    if eq.any():
        blocks.append(A[eq]); rhs.append(ru[eq]); zero_n += int(eq.sum())
    if feq.any():
        blocks.append(I[feq]); rhs.append(ub[feq]); zero_n += int(feq.sum())
    for M_, lo, up, skip in ((A, rl, ru, eq), (I, lb, ub, feq)):
        u = (up < big) & ~skip
        l = (lo > -big) & ~skip
        if u.any():
            blocks.append(M_[u]); rhs.append(up[u]); nonneg_n += int(u.sum())
        if l.any():
            blocks.append(-M_[l]); rhs.append(-lo[l]); nonneg_n += int(l.sum())
    Aall = sp.vstack(blocks).tocsc() if blocks else sp.csc_matrix((0, n))
    b = np.concatenate(rhs) if rhs else np.zeros(0)
    cones = []
    if zero_n:
        cones.append(clarabel.ZeroConeT(zero_n))
    if nonneg_n:
        cones.append(clarabel.NonnegativeConeT(nonneg_n))
    st = clarabel.DefaultSettings()
    st.verbose = False
    st.time_limit = float(limit)
    t = time.perf_counter()
    sol = clarabel.DefaultSolver(sp.csc_matrix(sense * P), sense * c, Aall, b, cones, st).solve()
    el = time.perf_counter() - t
    obj = sense * sol.obj_val + lp.offset_ if sol.x is not None else np.nan
    return str(sol.status), obj, el


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--limit", type=float, default=120)
    ap.add_argument("--out", default="results/maros_qp.csv")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    files = sorted({p.resolve() for p in Path(a.folder).iterdir() if p.suffix.lower() == ".qps"})
    if a.only:
        files = [f for f in files if f.stem in a.only]
    out = Path(a.out)
    fields = ["name", "m", "n", "nnz", "status", "obj", "ref_status", "ref_obj", "rel_err", "time",
              "ref_time", "iters", "row_viol", "bound_viol"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for f in files:
            res = run_one(f, "qp-ipm", a.limit)
            try:
                rst, robj, rtime = reference(f, a.limit)
            except Exception as e:
                rst, robj, rtime = f"error {type(e).__name__}: {e}"[:80], np.nan, np.nan
            obj = res.get("obj", np.nan)
            rel = abs(obj - robj) / max(1.0, abs(robj)) if np.isfinite(obj) and np.isfinite(robj) else np.nan
            w.writerow(dict(res, name=f.stem, obj=obj, ref_status=rst, ref_obj=robj, rel_err=rel, ref_time=rtime))
            fh.flush()
            ok = "OK " if res["status"] == "optimal" and rel == rel and rel < 1e-6 else "-- "
            print(f"{ok}{f.stem:10s} {res['status'][:30]:16s} obj {obj:+.8e} ref {robj:+.8e} ({rst}) err {rel:.1e} "
                  f"t {res.get('time', float('nan')):7.2f}s ref {rtime:6.2f}s", flush=True)


if __name__ == "__main__":
    main()
