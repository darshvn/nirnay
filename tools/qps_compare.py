"""Compare NIRNAY's QPS reading with HiGHS's on every file: A, row and column bounds, c, Q."""
import shutil, sys, tempfile
from pathlib import Path

import highspy
import numpy as np
import scipy.sparse as sp

from nirnay.io.mps import read_mps


def highs_model(p):
    tmp = Path(tempfile.gettempdir()) / (Path(p).stem + ".mps")
    shutil.copyfile(p, tmp)
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.readModel(str(tmp))
    return h.getModel()


def compare(p):
    m = read_mps(p)
    M = highs_model(p)
    lp, hs = M.lp_, M.hessian_
    out = []
    A = sp.csc_matrix((np.array(lp.a_matrix_.value_), np.array(lp.a_matrix_.index_),
                       np.array(lp.a_matrix_.start_)), shape=(lp.num_row_, lp.num_col_))
    if A.shape != (m.m, m.n):
        return [f"shape {A.shape} vs {(m.m, m.n)}"]
    ours = sp.csc_matrix((m.A.vals, m.A.rowidx, m.A.colptr), shape=(m.m, m.n))
    if abs(A - ours).max() > 1e-12:
        out.append("A")
    big = 1e20
    fix = lambda v: np.where(np.abs(v) >= big, np.sign(v) * np.inf, v)
    sense = -1 if lp.sense_ == highspy.ObjSense.kMaximize else 1
    for name, a, b in [("rl", fix(np.array(lp.row_lower_)), m.rl), ("ru", fix(np.array(lp.row_upper_)), m.ru),
                       ("lb", fix(np.array(lp.col_lower_)), m.lb), ("ub", fix(np.array(lp.col_upper_)), m.ub),
                       ("c", sense * np.array(lp.col_cost_), m.c)]:
        d = np.flatnonzero(~((a == b) | (np.abs(a - b) <= 1e-12 * (1 + np.abs(b)))))
        if len(d):
            out.append(f"{name}[{len(d)}] e.g. {int(d[0])}: highs {a[d[0]]} ours {b[d[0]]}")
    if hs.dim_ > 0:
        H = sp.csc_matrix((np.array(hs.value_), np.array(hs.index_), np.array(hs.start_)), shape=(hs.dim_, hs.dim_))
        Hf = (H + sp.tril(H, -1).T) * sense
        Q = sp.csc_matrix((m.Q.vals, m.Q.rowidx, m.Q.colptr), shape=(m.n, m.n)) if m.Q is not None else sp.csc_matrix((m.n, m.n))
        if abs(Hf - Q).max() > 1e-12:
            out.append("Q")
    if abs(sense * lp.offset_ - m.c0) > 1e-9 * (1 + abs(m.c0)):
        out.append(f"c0 highs {lp.offset_} ours {m.c0}")
    return out


if __name__ == "__main__":
    bad = 0
    for p in sys.argv[1:]:
        try:
            d = compare(p)
        except Exception as e:
            d = [f"error {type(e).__name__}: {e}"]
        if d:
            bad += 1
            print(f"{Path(p).stem:12s} " + "; ".join(d))
    print(f"{bad} files differ out of {len(sys.argv) - 1}")
