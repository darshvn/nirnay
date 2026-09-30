"""Row and column scaling: A_hat = R A C with R, C diagonal and positive.

Industrial models mix units (tonnes, kilolitres, rupees, percentages), so coefficients span many
orders of magnitude and every factorisation downstream loses digits to it. Scaling is the
cheapest robustness there is.

Two stages, both standard:
  * geometric mean scaling (Fourer 1982): repeatedly divide each row and column by
    sqrt(max|a| * min|a|), which pulls the entries of every line towards 1 from both ends;
  * equilibration: a final pass making the largest entry of every column exactly 1.

Factors are rounded to powers of two, so scaling multiplies by exact binary exponents and adds
no rounding error of its own.
"""
from __future__ import annotations

import numpy as np

from ..model import CSC, Model


def _pow2(x: np.ndarray) -> np.ndarray:
    return np.exp2(np.round(np.log2(x)))


def _line_extremes(A: CSC, by_col: bool):
    absv = np.abs(A.vals)
    if by_col:
        idx = np.repeat(np.arange(A.n), np.diff(A.colptr))
        size = A.n
    else:
        idx = A.rowidx
        size = A.m
    mx = np.zeros(size)
    mn = np.full(size, np.inf)
    np.maximum.at(mx, idx, absv)
    np.minimum.at(mn, idx, absv)
    mn[~np.isfinite(mn)] = 1.0
    mx[mx == 0] = 1.0
    return mx, mn


def compute_scaling(A: CSC, passes: int = 8, tol: float = 0.9):
    R = np.ones(A.m)
    C = np.ones(A.n)
    if A.nnz == 0:
        return R, C
    cols = np.repeat(np.arange(A.n), np.diff(A.colptr))
    base = np.abs(A.vals)

    def current():
        return base * R[A.rowidx] * C[cols]

    # entries below 1e-12 of the largest in their row or column carry no information at double
    # precision, but a geometric mean would let them dominate: KSIP has 1.2e-30 next to 1.0,
    # and scaling by it wrecks every other entry. They are left out of the min.
    def extremes(v, idx, size):
        mx = np.zeros(size)
        np.maximum.at(mx, idx, v)
        mn = np.full(size, np.inf)
        keep = v >= 1e-12 * mx[idx]
        np.minimum.at(mn, idx[keep], v[keep])
        return mx, mn

    prev = np.inf
    for _ in range(passes):
        v = current()
        mxr, _ = extremes(v, A.rowidx, A.m)
        sig = v[v >= 1e-12 * mxr[A.rowidx]]
        spread = sig.max() / sig.min()
        if spread > tol * prev and np.isfinite(prev):
            break
        prev = spread
        # rows
        mx, mn = extremes(v, A.rowidx, A.m)
        has = mx > 0
        R[has] /= np.sqrt(mx[has] * mn[has])
        v = current()
        mx, mn = extremes(v, cols, A.n)
        has = mx > 0
        C[has] /= np.sqrt(mx[has] * mn[has])
    # equilibrate columns
    v = current()
    mx = np.zeros(A.n)
    np.maximum.at(mx, cols, v)
    C[mx > 0] /= mx[mx > 0]
    return _pow2(R), _pow2(C)


def scale_model(model: Model, R: np.ndarray, C: np.ndarray) -> Model:
    A = model.A
    cols = np.repeat(np.arange(A.n), np.diff(A.colptr))
    As = CSC(A.m, A.n, A.colptr.copy(), A.rowidx.copy(), A.vals * R[A.rowidx] * C[cols])
    Q = None
    if model.Q is not None:
        q = model.Q
        qcols = np.repeat(np.arange(q.n), np.diff(q.colptr))
        Q = CSC(q.m, q.n, q.colptr.copy(), q.rowidx.copy(), q.vals * C[q.rowidx] * C[qcols])
    with np.errstate(invalid="ignore"):
        return Model(name=model.name, c=model.c * C, A=As, rl=model.rl * R, ru=model.ru * R,
                     lb=model.lb / C, ub=model.ub / C, integer=model.integer.copy(), Q=Q,
                     c0=model.c0, sense=model.sense, col_names=model.col_names, row_names=model.row_names)


def unscale(x, y, z, R, C):
    """Map a solution of the scaled model back: x = C x_hat, y = R y_hat, z = z_hat / C."""
    return (None if x is None else x * C,
            None if y is None else y * R,
            None if z is None else z / C)
