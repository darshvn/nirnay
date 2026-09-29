"""Sparse Cholesky factorisation L L' = P M P' for symmetric positive (semi)definite M.

The interior-point method solves one of these every iteration, always with the same sparsity
pattern, so the work is split the classic way:

  analyse(M)    fill-reducing ordering, elimination tree, column counts, the pattern of L
  factor(v)     numeric factorisation of new values on that fixed pattern
  solve(b)      forward and back substitution

The numeric kernel is the up-looking algorithm: row k of L is found by a sparse triangular solve
whose pattern is the reach of row k in the elimination tree (Davis, "Direct Methods for Sparse
Linear Systems", SIAM 2006, ch. 4). Everything runs in Numba-compiled loops over plain arrays.

Interior-point normal equations become singular as the iterates converge (dependent rows, tiny
slacks). A pivot below `tiny` is replaced by `huge`, which zeroes that component of the solution
instead of blowing it up: the standard remedy (Wright, "Primal-Dual Interior-Point Methods",
SIAM 1997, sec. 11.1). The number of such pivots is reported so the caller can see it happening.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .ordering import minimum_degree


@njit(cache=True)
def _etree(n, colptr, rowidx):
    """Elimination tree of a symmetric matrix given by its upper triangle in CSC (row < col)."""
    parent = np.full(n, -1, dtype=np.int64)
    ancestor = np.full(n, -1, dtype=np.int64)
    for k in range(n):
        for p in range(colptr[k], colptr[k + 1]):
            i = rowidx[p]
            while i != -1 and i < k:
                nxt = ancestor[i]
                ancestor[i] = k
                if nxt == -1:
                    parent[i] = k
                i = nxt
    return parent


@njit(cache=True)
def _row_patterns_count(n, colptr, rowidx, parent):
    """Number of off-diagonal nonzeros in each column of L, by walking every row's reach."""
    counts = np.zeros(n, dtype=np.int64)
    flag = np.full(n, -1, dtype=np.int64)
    for k in range(n):
        flag[k] = k
        for p in range(colptr[k], colptr[k + 1]):
            i = rowidx[p]
            if i >= k:
                continue
            while flag[i] != k:
                counts[i] += 1          # L[k, i] is nonzero
                flag[i] = k
                i = parent[i]
    return counts


@njit(cache=True)
def _numeric(n, colptr, rowidx, vals, parent, Lp, Li, Lx, tiny, huge):
    """Up-looking Cholesky on a fixed pattern. Upper triangle of the permuted matrix in CSC."""
    c = Lp[:-1].copy()            # next free slot in each column of L
    x = np.zeros(n)
    flag = np.full(n, -1, dtype=np.int64)
    stack = np.empty(n, dtype=np.int64)
    fixed = 0
    for k in range(n):
        # pattern of row k of L via the elimination tree
        top = n
        flag[k] = k
        for p in range(colptr[k], colptr[k + 1]):
            i = rowidx[p]
            if i > k:
                continue
            length = 0
            while flag[i] != k:
                stack[length] = i
                length += 1
                flag[i] = k
                i = parent[i]
            while length > 0:
                top -= 1
                length -= 1
                stack[top] = stack[length]
        # scatter column k of the upper triangle into x
        d = 0.0
        for p in range(colptr[k], colptr[k + 1]):
            i = rowidx[p]
            if i < k:
                x[i] = vals[p]
            elif i == k:
                d = vals[p]
        # sparse triangular solve for row k
        for t in range(top, n):
            i = stack[t]
            lki = x[i] / Lx[Lp[i]]
            x[i] = 0.0
            for p in range(Lp[i] + 1, c[i]):
                x[Li[p]] -= Lx[p] * lki
            d -= lki * lki
            pos = c[i]
            c[i] += 1
            Li[pos] = k
            Lx[pos] = lki
        if d <= tiny:
            d = huge
            fixed += 1
        pos = c[k]
        c[k] += 1
        Li[pos] = k
        Lx[pos] = np.sqrt(d)
    return fixed


@njit(cache=True)
def _lsolve(n, Lp, Li, Lx, x):
    for j in range(n):
        x[j] /= Lx[Lp[j]]
        xj = x[j]
        for p in range(Lp[j] + 1, Lp[j + 1]):
            x[Li[p]] -= Lx[p] * xj


@njit(cache=True)
def _ltsolve(n, Lp, Li, Lx, x):
    for j in range(n - 1, -1, -1):
        s = x[j]
        for p in range(Lp[j] + 1, Lp[j + 1]):
            s -= Lx[p] * x[Li[p]]
        x[j] = s / Lx[Lp[j]]


@njit(cache=True)
def _permute_upper(n, colptr, rowidx, pinv, out_colptr, out_rowidx, out_map):
    """Pattern of the upper triangle of P M P' from the full pattern of M, plus a value map."""
    counts = np.zeros(n + 1, dtype=np.int64)
    for j in range(n):
        for p in range(colptr[j], colptr[j + 1]):
            i = rowidx[p]
            a, b = pinv[i], pinv[j]
            if a <= b:
                counts[b + 1] += 1
    for j in range(n):
        counts[j + 1] += counts[j]
    out_colptr[:] = counts
    nxt = counts[:-1].copy()
    for j in range(n):
        for p in range(colptr[j], colptr[j + 1]):
            i = rowidx[p]
            a, b = pinv[i], pinv[j]
            if a <= b:
                q = nxt[b]
                nxt[b] += 1
                out_rowidx[q] = a
                out_map[q] = p


class Cholesky:
    """Analyse once, factor many times, on the pattern of a symmetric matrix held in full CSC."""

    def __init__(self, n, colptr, rowidx, perm=None, tiny=1e-30, huge=1e128):
        self.n = n
        self.tiny, self.huge = tiny, huge
        colptr = np.asarray(colptr, dtype=np.int64)
        rowidx = np.asarray(rowidx, dtype=np.int64)
        self.perm = minimum_degree(n, colptr, rowidx) if perm is None else np.asarray(perm, dtype=np.int64)
        self.pinv = np.empty(n, dtype=np.int64)
        self.pinv[self.perm] = np.arange(n)
        nnz_upper = int(np.sum(self.pinv[rowidx] <= self.pinv[np.repeat(np.arange(n), np.diff(colptr))]))
        self.Cp = np.empty(n + 1, dtype=np.int64)
        self.Ci = np.empty(nnz_upper, dtype=np.int64)
        self.Cmap = np.empty(nnz_upper, dtype=np.int64)
        _permute_upper(n, colptr, rowidx, self.pinv, self.Cp, self.Ci, self.Cmap)
        # sort row indices within each column (the tree walks assume nothing, but tidy is cheaper)
        self.parent = _etree(n, self.Cp, self.Ci)
        counts = _row_patterns_count(n, self.Cp, self.Ci, self.parent)
        self.Lp = np.zeros(n + 1, dtype=np.int64)
        self.Lp[1:] = np.cumsum(counts + 1)
        self.Li = np.empty(self.Lp[-1], dtype=np.int64)
        self.Lx = np.empty(self.Lp[-1])
        self.fixed_pivots = 0

    @property
    def nnz(self) -> int:
        return int(self.Lp[-1])

    def factor(self, vals: np.ndarray) -> int:
        """vals are the values of M in the same order as the pattern given to the constructor."""
        cvals = np.asarray(vals)[self.Cmap]
        self.fixed_pivots = _numeric(self.n, self.Cp, self.Ci, cvals, self.parent,
                                     self.Lp, self.Li, self.Lx, self.tiny, self.huge)
        return self.fixed_pivots

    def solve(self, b: np.ndarray) -> np.ndarray:
        x = np.asarray(b, dtype=np.float64)[self.perm].copy()
        _lsolve(self.n, self.Lp, self.Li, self.Lx, x)
        _ltsolve(self.n, self.Lp, self.Li, self.Lx, x)
        out = np.empty_like(x)
        out[self.perm] = x
        return out
