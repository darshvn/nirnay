"""Sparse LDL' factorisation for symmetric quasi-definite matrices.

    K = [ -H   A' ]      H positive definite (n x n), R positive definite (m x m)
        [  A   R  ]

A quasi-definite matrix has an LDL' factorisation with D diagonal (n negative entries, m
positive) under *any* symmetric permutation, so the fill-reducing ordering can be chosen for
sparsity alone, exactly as for Cholesky (Vanderbei, "Symmetric quasi-definite matrices", SIAM J.
Optim. 5, 1995). Interior-point methods for QP get such a K by regularising the augmented
system: H = Q + X^-1 Z + rho I and R = delta I (Friedlander & Orban, Math. Prog. Comp. 4, 2012).

The numeric kernel is the up-looking LDL' of Davis ("Algorithm 849: a concise sparse Cholesky
factorization package", ACM TOMS 31, 2005), sharing the analysis (ordering, elimination tree,
column counts) with cholesky.py. A pivot of the wrong sign or below `delta` in magnitude is
replaced by sign * delta (dynamic regularisation); the caller's iterative refinement recovers
the accuracy lost.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .cholesky import _etree, _permute_upper, _row_patterns_count
from .ordering import minimum_degree


@njit(cache=True)
def _numeric_ldl(n, colptr, rowidx, vals, parent, sign, Lp, Li, Lx, D, delta):
    Lnz = np.zeros(n, dtype=np.int64)
    y = np.zeros(n)
    flag = np.full(n, -1, dtype=np.int64)
    pattern = np.empty(n, dtype=np.int64)
    fixed = 0
    for k in range(n):
        top = n
        flag[k] = k
        for p in range(colptr[k], colptr[k + 1]):
            i = rowidx[p]
            if i > k:
                continue
            y[i] += vals[p]
            length = 0
            while flag[i] != k:
                pattern[length] = i
                length += 1
                flag[i] = k
                i = parent[i]
            while length > 0:
                top -= 1
                length -= 1
                pattern[top] = pattern[length]
        d = y[k]
        y[k] = 0.0
        for t in range(top, n):
            i = pattern[t]
            yi = y[i]
            y[i] = 0.0
            p2 = Lp[i] + Lnz[i]
            for p in range(Lp[i], p2):
                y[Li[p]] -= Lx[p] * yi
            lki = yi / D[i]
            d -= lki * yi
            Li[p2] = k
            Lx[p2] = lki
            Lnz[i] += 1
        if sign[k] * d < delta:
            d = sign[k] * delta
            fixed += 1
        D[k] = d
    return fixed


@njit(cache=True)
def _ldl_solve(n, Lp, Li, Lx, D, x):
    for j in range(n):
        xj = x[j]
        for p in range(Lp[j], Lp[j + 1]):
            x[Li[p]] -= Lx[p] * xj
    for j in range(n):
        x[j] /= D[j]
    for j in range(n - 1, -1, -1):
        s = x[j]
        for p in range(Lp[j], Lp[j + 1]):
            s -= Lx[p] * x[Li[p]]
        x[j] = s


class LDL:
    """Analyse once, factor many times; pattern given as a full symmetric CSC."""

    def __init__(self, n, colptr, rowidx, perm=None, delta=1e-12):
        self.n = n
        self.delta = delta
        colptr = np.asarray(colptr, dtype=np.int64)
        rowidx = np.asarray(rowidx, dtype=np.int64)
        self.perm = minimum_degree(n, colptr, rowidx) if perm is None else np.asarray(perm, dtype=np.int64)
        self.pinv = np.empty(n, dtype=np.int64)
        self.pinv[self.perm] = np.arange(n)
        cols = np.repeat(np.arange(n), np.diff(colptr))
        nnz_upper = int(np.sum(self.pinv[rowidx] <= self.pinv[cols]))
        self.Cp = np.empty(n + 1, dtype=np.int64)
        self.Ci = np.empty(nnz_upper, dtype=np.int64)
        self.Cmap = np.empty(nnz_upper, dtype=np.int64)
        _permute_upper(n, colptr, rowidx, self.pinv, self.Cp, self.Ci, self.Cmap)
        self.parent = _etree(n, self.Cp, self.Ci)
        counts = _row_patterns_count(n, self.Cp, self.Ci, self.parent)
        self.Lp = np.zeros(n + 1, dtype=np.int64)
        self.Lp[1:] = np.cumsum(counts)
        self.Li = np.empty(self.Lp[-1], dtype=np.int64)
        self.Lx = np.empty(self.Lp[-1])
        self.D = np.empty(n)
        self.fixed_pivots = 0

    @property
    def nnz(self) -> int:
        return int(self.Lp[-1]) + self.n

    def factor(self, vals: np.ndarray, sign: np.ndarray) -> int:
        """vals in the order of the constructor's pattern; sign[i] = expected sign of pivot i."""
        cvals = np.asarray(vals)[self.Cmap]
        self.fixed_pivots = _numeric_ldl(self.n, self.Cp, self.Ci, cvals, self.parent,
                                         np.asarray(sign, dtype=np.float64)[self.perm],
                                         self.Lp, self.Li, self.Lx, self.D, self.delta)
        return self.fixed_pivots

    def solve(self, b: np.ndarray) -> np.ndarray:
        x = np.asarray(b, dtype=np.float64)[self.perm].copy()
        _ldl_solve(self.n, self.Lp, self.Li, self.Lx, self.D, x)
        out = np.empty_like(x)
        out[self.perm] = x
        return out
