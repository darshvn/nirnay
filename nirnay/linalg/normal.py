"""Normal-equations matrix M = A diag(theta) A' + diag(extra), assembled on a fixed pattern.

The interior-point method rebuilds M every iteration with new weights theta but the same
sparsity, so the pattern of A A' and a map from every product a_ij * a_kj to its slot in M are
computed once. Each rebuild is then a single pass over that map: sum_j nnz(a_j)^2 multiply-adds,
with no searching and no allocation.

M is held in full (both triangles) because the Cholesky analysis takes a full pattern and picks
its own triangle after ordering.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _pattern(m, Ap, Ai, Rp, Rj):
    """Full pattern of A A' in CSC (column k = row k of A A'), with sorted row indices."""
    mark = np.full(m, -1, dtype=np.int64)
    counts = np.zeros(m + 1, dtype=np.int64)
    for k in range(m):
        mark[k] = k                    # the diagonal is always present
        cnt = 1
        for t in range(Rp[k], Rp[k + 1]):
            j = Rj[t]
            for p in range(Ap[j], Ap[j + 1]):
                i = Ai[p]
                if mark[i] != k:
                    mark[i] = k
                    cnt += 1
        counts[k + 1] = cnt
    for k in range(m):
        counts[k + 1] += counts[k]
    Mi = np.empty(counts[m], dtype=np.int64)
    mark[:] = -1
    for k in range(m):
        pos = counts[k]
        mark[k] = k
        Mi[pos] = k
        pos += 1
        for t in range(Rp[k], Rp[k + 1]):
            j = Rj[t]
            for p in range(Ap[j], Ap[j + 1]):
                i = Ai[p]
                if mark[i] != k:
                    mark[i] = k
                    Mi[pos] = i
                    pos += 1
        Mi[counts[k]:counts[k + 1]].sort()
    return counts, Mi


@njit(cache=True)
def _build_map(m, Ap, Ai, Rp, Rj, Mp, Mi):
    """For the loop order (row k, column j in row k, entry i of column j): the slot of M[i, k]."""
    total = 0
    for k in range(m):
        for t in range(Rp[k], Rp[k + 1]):
            j = Rj[t]
            total += Ap[j + 1] - Ap[j]
    slot = np.empty(total, dtype=np.int64)
    pos_of = np.full(m, -1, dtype=np.int64)
    diag = np.empty(m, dtype=np.int64)
    q = 0
    for k in range(m):
        for p in range(Mp[k], Mp[k + 1]):
            pos_of[Mi[p]] = p
        diag[k] = pos_of[k]
        for t in range(Rp[k], Rp[k + 1]):
            j = Rj[t]
            for p in range(Ap[j], Ap[j + 1]):
                slot[q] = pos_of[Ai[p]]
                q += 1
        for p in range(Mp[k], Mp[k + 1]):
            pos_of[Mi[p]] = -1
    return slot, diag


@njit(cache=True)
def _assemble(m, Ap, Ai, Ax, Rp, Rj, Rx, slot, diag, theta, extra, out):
    out[:] = 0.0
    q = 0
    for k in range(m):
        for t in range(Rp[k], Rp[k + 1]):
            j = Rj[t]
            w = theta[j] * Rx[t]
            for p in range(Ap[j], Ap[j + 1]):
                out[slot[q]] += w * Ax[p]
                q += 1
        out[diag[k]] += extra[k]


class NormalMatrix:
    def __init__(self, A, AT):
        """A in CSC and AT = A' in CSC (i.e. A by rows)."""
        self.A, self.AT = A, AT
        self.m = A.m
        self.Mp, self.Mi = _pattern(A.m, A.colptr, A.rowidx, AT.colptr, AT.rowidx)
        self.slot, self.diag = _build_map(A.m, A.colptr, A.rowidx, AT.colptr, AT.rowidx, self.Mp, self.Mi)
        self.vals = np.zeros(len(self.Mi))

    @property
    def nnz(self) -> int:
        return len(self.Mi)

    def assemble(self, theta: np.ndarray, extra: np.ndarray) -> np.ndarray:
        _assemble(self.m, self.A.colptr, self.A.rowidx, self.A.vals, self.AT.colptr, self.AT.rowidx,
                  self.AT.vals, self.slot, self.diag, theta, extra, self.vals)
        return self.vals
