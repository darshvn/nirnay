"""Command line:  nirnay solve model.mps [--method auto|simplex|ipm|pdlp|pdlp-gpu|bnb|qp-ipm]

    nirnay solve  afiro.mps.gz                 solve and print the result
    nirnay solve  p0033.mps --time-limit 60 --solution p0033.sol
    nirnay info   model.mps                    size and structure of a model
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def _cmd_info(a):
    from .io.mps import read_mps
    m = read_mps(a.model)
    print(m.summary())
    inf = np.isinf
    print(f"  free columns {int(np.sum(inf(m.lb) & inf(m.ub)))}, boxed {int(np.sum(~inf(m.lb) & ~inf(m.ub)))}, "
          f"fixed {int(np.sum(m.lb == m.ub))}")
    print(f"  ranged rows {int(np.sum(~inf(m.rl) & ~inf(m.ru) & (m.rl != m.ru)))}, "
          f"objective sense {'max' if m.sense < 0 else 'min'}")
    if m.A.nnz:
        a_abs = np.abs(m.A.vals)
        print(f"  |A| range [{a_abs.min():.1e}, {a_abs.max():.1e}]")
    return 0


def _cmd_solve(a):
    from . import solve
    from .io.mps import read_mps
    t = time.perf_counter()
    m = read_mps(a.model)
    t_read = time.perf_counter() - t
    print(m.summary())
    print(f"read in {t_read:.2f}s; method {a.method}")
    opts = {"time_limit": a.time_limit, "verbose": a.verbose}
    r = solve(m, method=a.method, **opts)
    print(r)
    if r.x is not None:
        v = m.violation(r.x)
        print(f"max violation: rows {v['row']:.1e}, bounds {v['bound']:.1e}, integrality {v['integrality']:.1e}")
    if a.solution and r.x is not None:
        names = m.col_names or [f"x{j}" for j in range(m.n)]
        with open(a.solution, "w") as fh:
            fh.write(f"# status {r.status}\n# objective {r.objective:.15g}\n")
            for nm, val in zip(names, r.x):
                if val != 0.0:
                    fh.write(f"{nm} {val:.15g}\n")
        print(f"solution written to {a.solution}")
    return 0 if r.status in ("optimal", "feasible") else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nirnay", description="NIRNAY optimisation solver (LP, MILP, QP)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("solve", help="solve an MPS/QPS model")
    s.add_argument("model")
    s.add_argument("--method", default="auto",
                   choices=["auto", "simplex", "ipm", "pdlp", "pdlp-gpu", "bnb", "qp-ipm"])
    s.add_argument("--time-limit", type=float, default=np.inf)
    s.add_argument("--solution", help="write the nonzero values of the solution to this file")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(fn=_cmd_solve)
    i = sub.add_parser("info", help="print a model's size and structure")
    i.add_argument("model")
    i.set_defaults(fn=_cmd_info)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
