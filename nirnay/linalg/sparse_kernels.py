"""Numba kernels for compressed sparse matrices held as plain arrays.

These are the only sparse operations a first-order method needs: products with A and A', and
the per-line norms that equilibration reads. A matrix is passed as (ptr, idx, val) of either
orientation: the CSC arrays of A are also the CSR arrays of A', so one row-wise product kernel
serves both A x (on CSR(A)) and A' y (on CSR(A') = CSC(A)). Row-wise products write each output
entry exactly once, so they parallelise over rows without atomics or races.

Two compiled variants of the product exist. The parallel one pays a fork/join of the thread pool
on every call (microseconds), which dominates on small matrices, so callers pick by size.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True, fastmath={"reassoc", "contract"})
def csr_matvec_par(ptr, idx, val, x, out):
    """out = M x for M in CSR form (row i is val[ptr[i]:ptr[i+1]] at columns idx[...])."""
    for i in prange(ptr.shape[0] - 1):
        s = 0.0
        for p in range(ptr[i], ptr[i + 1]):
            s += val[p] * x[idx[p]]
        out[i] = s


@njit(cache=True, fastmath={"reassoc", "contract"})
def csr_matvec_ser(ptr, idx, val, x, out):
    for i in range(ptr.shape[0] - 1):
        s = 0.0
        for p in range(ptr[i], ptr[i + 1]):
            s += val[p] * x[idx[p]]
        out[i] = s


@njit(cache=True)
def line_absmax(colptr, rowidx, vals, rs, cs, rmax, cmax):
    """Largest |r_i a_ij c_j| of every row and every column of diag(r) A diag(c), A in CSC.

    One pass over the nonzeros serves both orientations. Serial: a column-parallel loop would
    race on the row maxima, and this runs a handful of times per solve.
    """
    rmax[:] = 0.0
    cmax[:] = 0.0
    for j in range(colptr.shape[0] - 1):
        cj = cs[j]
        for p in range(colptr[j], colptr[j + 1]):
            i = rowidx[p]
            a = abs(vals[p]) * rs[i] * cj
            if a > rmax[i]:
                rmax[i] = a
            if a > cmax[j]:
                cmax[j] = a


@njit(cache=True)
def line_abssum(colptr, rowidx, vals, rs, cs, rsum, csum):
    """Sum of |r_i a_ij c_j| over every row and every column (the l1 norms), A in CSC."""
    rsum[:] = 0.0
    csum[:] = 0.0
    for j in range(colptr.shape[0] - 1):
        cj = cs[j]
        for p in range(colptr[j], colptr[j + 1]):
            i = rowidx[p]
            a = abs(vals[p]) * rs[i] * cj
            rsum[i] += a
            csum[j] += a


@njit(cache=True)
def scale_values(colptr, rowidx, vals, rs, cs, out):
    """out = values of diag(r) A diag(c), A in CSC (same pattern)."""
    for j in range(colptr.shape[0] - 1):
        cj = cs[j]
        for p in range(colptr[j], colptr[j + 1]):
            out[p] = vals[p] * rs[rowidx[p]] * cj


def matvec(ptr: np.ndarray, idx: np.ndarray, val: np.ndarray, x: np.ndarray,
           out: np.ndarray | None = None, parallel: bool = True) -> np.ndarray:
    if out is None:
        out = np.empty(ptr.shape[0] - 1)
    (csr_matvec_par if parallel else csr_matvec_ser)(ptr, idx, val, x, out)
    return out
