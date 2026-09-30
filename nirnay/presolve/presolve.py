"""Presolve: shrink a model before the solver sees it, and postsolve the answer back.

Only reductions with an exact primal *and* dual postsolve are used, so the duals and reduced
costs of the original model come back as well as x. Applied to a fixpoint:

  empty row            drop (infeasible if 0 violates its bounds); y_i = 0
  singleton row        a_ij x_j in [rl, ru] becomes a bound on x_j; row dropped. Postsolve: if
                       that bound is the active one, the reduced cost moves to the row dual,
                       y_i = z_j / a_ij and z_j = 0
  fixed column         x_j = l_j moves into the right-hand side; z_j = c_j - a_j' y at the end
  empty column         fixed at the bound its cost prefers (unbounded if none); z_j = c_j
  redundant row        activity bounds inside [rl, ru]: dropped, y_i = 0
  free column singleton on an equality row
                       x_j appears only in row i, an equality, and its bounds cannot bind
                       (free, or implied free by the row): x_j = (b_i - sum_k a_ik x_k) / a_ij
                       and y_i = c_j / a_ij, so the row and the column both go and every other
                       column in the row gets c_k -= a_ik c_j / a_ij
  integer bounds       for MILP, bounds of integer columns are rounded inward and tightened by
                       row-activity propagation (the LP dual is not needed for MILP)

References: Andersen & Andersen, "Presolving in linear programming", Math. Prog. 71, 1995;
Gondzio, "Presolve analysis of linear programs prior to applying an interior point method",
INFORMS J. Comput. 9, 1997; Achterberg et al., "Presolve reductions in MIP", INFORMS J. Comput.
32, 2020 (survey).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..model import CSC, INF, Model

FEAS = 1e-9


@dataclass
class Presolved:
    model: Model | None            # None when presolve proved the status
    status: str                    # "reduced" | "infeasible" | "unbounded" | "solved"
    stack: list = field(default_factory=list)
    keep_rows: np.ndarray | None = None
    keep_cols: np.ndarray | None = None
    x_fixed: np.ndarray | None = None
    orig: Model | None = None
    stats: dict = field(default_factory=dict)


def presolve(model: Model, max_passes: int = 20) -> Presolved:
    m, n = model.m, model.n
    A, AT = model.A, model.AT
    rl, ru = model.rl.copy(), model.ru.copy()
    lb, ub = model.lb.copy(), model.ub.copy()
    c = model.c.copy()
    c0 = model.c0
    is_int = model.integer
    row_alive = np.ones(m, dtype=bool)
    col_alive = np.ones(n, dtype=bool)
    x_fixed = np.zeros(n)                  # values of removed columns
    stack: list = []
    stats = {k: 0 for k in ("empty_row", "singleton_row", "fixed_col", "empty_col", "redundant_row",
                            "free_singleton", "int_bounds")}
    if model.is_qp:
        return Presolved(model=model, status="reduced", orig=model, keep_rows=np.arange(m),
                         keep_cols=np.arange(n), x_fixed=x_fixed, stats=stats)

    col_cnt = np.diff(A.colptr).astype(np.int64)
    row_cnt = np.diff(AT.colptr).astype(np.int64)

    def row_entries(i):
        p, q = AT.colptr[i], AT.colptr[i + 1]
        cols, vals = AT.rowidx[p:q], AT.vals[p:q]
        keep = col_alive[cols]
        return cols[keep], vals[keep]

    def col_entries(j):
        p, q = A.colptr[j], A.colptr[j + 1]
        rows, vals = A.rowidx[p:q], A.vals[p:q]
        keep = row_alive[rows]
        return rows[keep], vals[keep]

    if is_int.any():
        lb[is_int] = np.ceil(lb[is_int] - 1e-9)
        ub[is_int] = np.floor(ub[is_int] + 1e-9)
        if np.any(lb > ub + 1e-9):
            return Presolved(model=None, status="infeasible", orig=model)

    for _ in range(max_passes):
        changed = False
        # ---- columns: fixed and empty
        for j in np.flatnonzero(col_alive):
            if lb[j] == ub[j] or col_cnt[j] == 0:
                if lb[j] == ub[j]:
                    v = lb[j]
                    kind = "fixed_col"
                elif c[j] > 0:
                    if not np.isfinite(lb[j]):
                        return Presolved(model=None, status="unbounded", orig=model)
                    v, kind = lb[j], "empty_col"
                elif c[j] < 0:
                    if not np.isfinite(ub[j]):
                        return Presolved(model=None, status="unbounded", orig=model)
                    v, kind = ub[j], "empty_col"
                else:
                    v, kind = (lb[j] if np.isfinite(lb[j]) else (ub[j] if np.isfinite(ub[j]) else 0.0)), "empty_col"
                rows, vals = col_entries(j)
                rl[rows] -= vals * v
                ru[rows] -= vals * v
                row_cnt[rows] -= 1
                c0 += c[j] * v
                x_fixed[j] = v
                col_alive[j] = False
                stack.append(("col", j))
                stats[kind] += 1
                changed = True
        # ---- rows: empty, singleton, redundant
        for i in np.flatnonzero(row_alive):
            cols, vals = row_entries(i)
            if len(cols) == 0:
                if rl[i] > FEAS or ru[i] < -FEAS:
                    return Presolved(model=None, status="infeasible", orig=model)
                row_alive[i] = False
                stack.append(("row", i))
                stats["empty_row"] += 1
                changed = True
                continue
            if len(cols) == 1:
                j, a = int(cols[0]), float(vals[0])
                lo, hi = (rl[i] / a, ru[i] / a) if a > 0 else (ru[i] / a, rl[i] / a)
                if is_int[j]:
                    lo, hi = np.ceil(lo - 1e-9) if np.isfinite(lo) else lo, np.floor(hi + 1e-9) if np.isfinite(hi) else hi
                old_lb, old_ub = lb[j], ub[j]
                new_lb, new_ub = max(lb[j], lo), min(ub[j], hi)
                if new_lb > new_ub + 1e-9 * (1 + abs(new_ub)):
                    return Presolved(model=None, status="infeasible", orig=model)
                if new_lb > new_ub:
                    new_lb = new_ub = 0.5 * (new_lb + new_ub)
                lb[j], ub[j] = new_lb, new_ub
                # for the dual postsolve: which bounds this row supplied, and column j as the
                # reduced problem sees it at this moment (current cost, rows still alive)
                row_alive[i] = False
                jr, jv = col_entries(j)
                stack.append(("srow", i, j, a, new_lb if new_lb > old_lb else None,
                              new_ub if new_ub < old_ub else None, c[j], jr, jv))
                col_cnt[j] -= 1
                stats["singleton_row"] += 1
                changed = True
                continue
            # activity bounds
            pos = vals > 0
            lo_act = np.sum(np.where(pos, vals * lb[cols], vals * ub[cols]))
            hi_act = np.sum(np.where(pos, vals * ub[cols], vals * lb[cols]))
            if np.isfinite(lo_act) and np.isfinite(hi_act):
                if lo_act >= rl[i] - FEAS * (1 + abs(rl[i])) and hi_act <= ru[i] + FEAS * (1 + abs(ru[i])):
                    row_alive[i] = False
                    col_cnt[cols] -= 1
                    stack.append(("row", i))
                    stats["redundant_row"] += 1
                    changed = True
                    continue
                if lo_act > ru[i] + 1e-7 * (1 + abs(ru[i])) or hi_act < rl[i] - 1e-7 * (1 + abs(rl[i])):
                    return Presolved(model=None, status="infeasible", orig=model)
            # free column singleton on an equality row
            if rl[i] == ru[i]:
                for k, (j, a) in enumerate(zip(cols, vals)):
                    if col_cnt[j] != 1 or is_int[j]:
                        continue
                    # implied free? bounds of the others give the range of x_j through the row
                    others = np.arange(len(cols)) != k
                    oc, ov = cols[others], vals[others]
                    opos = ov > 0
                    o_lo = np.sum(np.where(opos, ov * lb[oc], ov * ub[oc]))
                    o_hi = np.sum(np.where(opos, ov * ub[oc], ov * lb[oc]))
                    b = rl[i]
                    xj_lo, xj_hi = ((b - o_hi) / a, (b - o_lo) / a) if a > 0 else ((b - o_lo) / a, (b - o_hi) / a)
                    if not (xj_lo >= lb[j] - FEAS * (1 + abs(lb[j])) and xj_hi <= ub[j] + FEAS * (1 + abs(ub[j]))):
                        continue
                    # x_j = (b - sum a_k x_k) / a ;  y_i = c_j / a ;  c_k -= a_k c_j / a ;  c0 += c_j b / a
                    yi = c[j] / a
                    c[oc] -= ov * yi
                    c0 += yi * b
                    stack.append(("fsing", i, j, a, oc.copy(), ov.copy(), b, c[j]))
                    row_alive[i] = False
                    col_alive[j] = False
                    col_cnt[oc] -= 1
                    stats["free_singleton"] += 1
                    changed = True
                    break
        # ---- integer bound propagation (MILP only)
        if is_int.any():
            from ..mip.propagate import propagate
            rows_alive = np.flatnonzero(row_alive)
            if len(rows_alive):
                # removed columns are already folded into rl/ru: they must count as zero here
                lb2 = np.where(col_alive, lb, 0.0)
                ub2 = np.where(col_alive, ub, 0.0)
                st, ntight = propagate(AT.colptr, AT.rowidx, AT.vals, np.where(row_alive, rl, -INF),
                                       np.where(row_alive, ru, INF), lb2, ub2, is_int, 3, 1e-6)
                if st == 1:
                    return Presolved(model=None, status="infeasible", orig=model)
                moved = (is_int & col_alive) & ((lb2 != lb) | (ub2 != ub))
                if moved.any():
                    lb[moved], ub[moved] = lb2[moved], ub2[moved]
                    stats["int_bounds"] += int(moved.sum())
                    changed = True
        if not changed:
            break

    keep_rows = np.flatnonzero(row_alive)
    keep_cols = np.flatnonzero(col_alive)
    if len(keep_cols) == 0:
        return Presolved(model=None, status="solved", stack=stack, keep_rows=keep_rows, keep_cols=keep_cols,
                         x_fixed=x_fixed, orig=model, stats=dict(stats, c0=c0))
    # build the reduced model
    cols_of = A.colidx
    mask = row_alive[A.rowidx] & col_alive[cols_of]
    row_new = np.full(m, -1, dtype=np.int64)
    row_new[keep_rows] = np.arange(len(keep_rows))
    col_new = np.full(n, -1, dtype=np.int64)
    col_new[keep_cols] = np.arange(len(keep_cols))
    A2 = CSC.from_triplets(len(keep_rows), len(keep_cols), row_new[A.rowidx[mask]], col_new[cols_of[mask]], A.vals[mask])
    red = Model(name=model.name, c=c[keep_cols], A=A2, rl=rl[keep_rows], ru=ru[keep_rows], lb=lb[keep_cols],
                ub=ub[keep_cols], integer=is_int[keep_cols], Q=None, c0=c0, sense=model.sense,
                col_names=[model.col_names[j] for j in keep_cols] if model.col_names else [],
                row_names=[model.row_names[i] for i in keep_rows] if model.row_names else [])
    return Presolved(model=red, status="reduced", stack=stack, keep_rows=keep_rows, keep_cols=keep_cols,
                     x_fixed=x_fixed, orig=model, stats=stats)


def postsolve(P: Presolved, x_red, y_red=None, z_red=None):
    """Map a solution of the reduced model back to the original: returns (x, y, z)."""
    orig = P.orig
    m, n = orig.m, orig.n
    A, AT = orig.A, orig.AT
    x = P.x_fixed.copy()
    y = np.zeros(m)
    if x_red is not None and len(P.keep_cols):
        x[P.keep_cols] = x_red
    if y_red is not None and len(P.keep_rows):
        y[P.keep_rows] = y_red
    # reduced costs of every column under the duals known so far; kept updated as row duals
    # are assigned while the reductions are undone
    z = orig.c - A.rmatvec(y)

    def set_row_dual(i, yi):
        y[i] = yi
        p, q = AT.colptr[i], AT.colptr[i + 1]
        z[AT.rowidx[p:q]] -= AT.vals[p:q] * yi

    for item in reversed(P.stack):
        kind = item[0]
        if kind == "fsing":
            _, i, j, a, oc, ov, b, cj = item
            x[j] = (b - float(ov @ x[oc])) / a
            # the reduction's own dual: y_i = c_j / a_ij with c_j the cost at removal time (it
            # already carries the substitutions done before it, whose rows are restored later)
            set_row_dual(i, cj / a)
        elif kind == "srow":
            _, i, j, a, new_lb, new_ub, cj, jr, jv = item
            # reduced cost of x_j in the problem as it was when the row went: every row still
            # alive then is either kept or removed later, so its dual is already final here
            zj = cj - float(jv @ y[jr])
            # if the bound this row created is the active one, the reduced cost belongs to the row
            tol = 1e-7 * (1 + abs(x[j]))
            at_lb = new_lb is not None and abs(x[j] - new_lb) <= tol and zj > 0
            at_ub = new_ub is not None and abs(x[j] - new_ub) <= tol and zj < 0
            if at_lb or at_ub:
                set_row_dual(i, zj / a)
        # "row" and "col": y_i = 0; nothing to do
    return x, y, z
