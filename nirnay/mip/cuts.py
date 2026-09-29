"""Gomory mixed-integer (GMI) cuts from the optimal simplex tableau.

Take a row of the optimal tableau whose basic variable is an integer column with a fractional
value. Written in the nonbasic variables' distances from their bounds, t_k >= 0,

    x_B + sum_k a_k t_k = b,        f0 = frac(b) in (0, 1),

every integer-feasible point satisfies the GMI inequality (Gomory 1960; in this form Balas,
Ceria, Cornuejols, Natraj, "Gomory cuts revisited", Oper. Res. Letters 19, 1996)

    sum_{k int, f_k <= f0} f_k / f0 t_k + sum_{k int, f_k > f0} (1 - f_k) / (1 - f0) t_k
  + sum_{k cont, a_k > 0}  a_k / f0 t_k + sum_{k cont, a_k < 0} -a_k / (1 - f0) t_k   >=  1,

with f_k = frac(a_k). The current LP point has t = 0, so it violates the cut by exactly 1.

Numerical safety follows the practice of Cook, Dash, Fukasawa, Goycoolea ("Numerically safe
Gomory mixed-integer cuts", INFORMS J. Comput. 21, 2009) in spirit: skip rows with f0 near 0
or 1, drop tiny coefficients by relaxing against a bound, and reject cuts with a large ratio
between their largest and smallest coefficients.
"""
from __future__ import annotations

import numpy as np

from ..lp.simplex import AT_LOWER, AT_UPPER, AT_ZERO, BASIC, FIXED, SimplexLP
from ..model import CSC, Model


def gmi_cuts(lp: SimplexLP, model: Model, max_cuts: int = 100, away: float = 0.01,
             max_dyn: float = 1e6, min_efficacy: float = 1e-5):
    """Cuts (g, h) with g'x >= h, violated by the current LP solution of `lp`."""
    n, m = lp.n, lp.m
    C, R = lp.C, lp.R
    is_int = model.integer
    # unscaled bounds of every variable in the computational form (structurals, then rows)
    lo_u = np.concatenate([lp.lo[:n] * C, lp.lo[n:] / R])
    up_u = np.concatenate([lp.up[:n] * C, lp.up[n:] / R])
    # which nonbasic variables are integer with integral bounds (their t_k is an integer)
    int_var = np.zeros(n + m, dtype=bool)
    int_var[:n] = is_int
    st = lp.status
    x = lp.primal()
    AT = model.AT
    cands = []
    for r, j in enumerate(lp.head):
        if j < n and is_int[j]:
            f = x[j] - np.floor(x[j])
            if away < f < 1 - away:
                cands.append((min(f, 1 - f), r, j))
    cands.sort(reverse=True)            # most fractional first
    cuts = []
    for _, r, j in cands[: 3 * max_cuts]:
        e = np.zeros(m)
        e[r] = 1.0
        alpha = lp._row(lp.factor_.btran(e))
        nz = np.flatnonzero(alpha)
        if len(nz) == 0:
            continue
        # unscaled tableau coefficients: x_j + sum coef_k v_k = 0
        coef = alpha[nz] * C[j]
        coef = np.where(nz < n, coef / C[np.minimum(nz, n - 1)], coef * R[np.maximum(nz - n, 0)])
        stk = st[nz]
        if np.any((stk == AT_ZERO) & (np.abs(coef) > 1e-12)):
            continue                     # a free nonbasic variable: no valid cut from this row
        at_up = stk == AT_UPPER
        bnd = np.where(at_up, up_u[nz], lo_u[nz])
        if not np.all(np.isfinite(bnd[stk != FIXED])):
            continue
        movable = (stk == AT_LOWER) | (stk == AT_UPPER)
        b = x[j]
        f0 = b - np.floor(b)
        if not (away < f0 < 1 - away):
            continue
        a = np.where(at_up, -coef, coef)[movable]
        k_idx = nz[movable]
        kint = int_var[k_idx] & (np.abs(bnd[movable] - np.round(bnd[movable])) < 1e-9)
        fk = a - np.floor(a)
        pi = np.where(kint,
                      np.where(fk <= f0, fk / f0, (1 - fk) / (1 - f0)),
                      np.where(a > 0, a / f0, -a / (1 - f0)))
        # back to the variables: t_k = sigma_k (v_k - bnd_k)
        sigma = np.where(at_up[movable], -1.0, 1.0)
        gv = pi * sigma                                   # coefficient on v_k
        h = 1.0 + float(np.sum(gv * bnd[movable]))
        g = np.zeros(n)
        s_mask = k_idx < n
        np.add.at(g, k_idx[s_mask], gv[s_mask])
        # a row's logical v = a_i x: expand into the structurals
        for kk, gk in zip(k_idx[~s_mask], gv[~s_mask]):
            i = kk - n
            cols = AT.rowidx[AT.colptr[i]:AT.colptr[i + 1]]
            vals = AT.vals[AT.colptr[i]:AT.colptr[i + 1]]
            np.add.at(g, cols, gk * vals)
        cut = _clean(g, h, model.lb, model.ub, max_dyn)
        if cut is None:
            continue
        g, h = cut
        viol = h - float(g @ x)
        norm = float(np.linalg.norm(g))
        if norm == 0 or viol / norm < min_efficacy:
            continue
        cuts.append((g, h, viol / norm))
        if len(cuts) >= max_cuts:
            break
    return cuts


def _clean(g, h, lb, ub, max_dyn):
    """Remove tiny coefficients (relaxing the cut with a bound) and reject badly scaled cuts."""
    amax = np.max(np.abs(g), initial=0.0)
    if amax == 0.0 or not np.isfinite(amax) or not np.isfinite(h):
        return None
    small = (np.abs(g) < 1e-9 * amax) & (g != 0)
    for k in np.flatnonzero(small):
        # g_k x_k <= max over the bounds; move that bound to the right-hand side
        bound = ub[k] if g[k] > 0 else lb[k]
        if not np.isfinite(bound):
            return None
        h -= g[k] * bound
        g[k] = 0.0
    nzv = np.abs(g[g != 0])
    if len(nzv) == 0 or nzv.max() / nzv.min() > max_dyn:
        return None
    scale = 1.0 / nzv.max()
    return g * scale, h * scale


def add_cuts(model: Model, cuts) -> Model:
    """A new Model with the cuts appended as rows g'x >= h."""
    if not cuts:
        return model
    A = model.A
    cols = np.repeat(np.arange(A.n), np.diff(A.colptr))
    rows, cc, vv = [A.rowidx], [cols], [A.vals]
    rl, ru = [model.rl], [model.ru]
    for k, (g, h, _) in enumerate(cuts):
        nz = np.flatnonzero(g)
        rows.append(np.full(len(nz), A.m + k))
        cc.append(nz)
        vv.append(g[nz])
    rl.append(np.array([h for _, h, _ in cuts]))
    ru.append(np.full(len(cuts), np.inf))
    A2 = CSC.from_triplets(A.m + len(cuts), A.n, np.concatenate(rows), np.concatenate(cc), np.concatenate(vv))
    names = list(model.row_names) + [f"gmi{k}" for k in range(len(cuts))] if model.row_names else []
    return Model(name=model.name, c=model.c, A=A2, rl=np.concatenate(rl), ru=np.concatenate(ru),
                 lb=model.lb, ub=model.ub, integer=model.integer, Q=None, c0=model.c0,
                 sense=model.sense, col_names=model.col_names, row_names=names)
