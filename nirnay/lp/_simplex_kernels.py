"""Numba kernels for the simplex method: the loops that run every iteration.

Status codes for every variable (structural or logical):
    BASIC = 0, AT_LOWER = 1, AT_UPPER = 2, AT_ZERO = 3 (nonbasic free, value 0), FIXED = 4
"""
from __future__ import annotations

import numpy as np
from numba import njit

BASIC, AT_LOWER, AT_UPPER, AT_ZERO, FIXED = 0, 1, 2, 3, 4


@njit(cache=True)
def pivot_row(n, m, ATp, ATi, ATx, rho, status, out):
    """alpha_r = rho' [A, -I] for nonbasic columns only (basic entries are left at 0).

    Walks the rows of A (A' in CSC) touched by the nonzeros of rho, so a sparse rho is cheap."""
    for j in range(n + m):
        out[j] = 0.0
    for i in range(m):
        ri = rho[i]
        if ri == 0.0:
            continue
        for p in range(ATp[i], ATp[i + 1]):
            out[ATi[p]] += ri * ATx[p]
        out[n + i] = -ri
    for j in range(n + m):
        if status[j] == BASIC:
            out[j] = 0.0


@njit(cache=True)
def choose_leaving(xB, lB, uB, weights, ptol):
    """Dual steepest edge: largest infeasibility^2 / weight among basic variables."""
    best = -1
    bestscore = 0.0
    for r in range(len(xB)):
        v = xB[r]
        if v < lB[r] - ptol:
            inf = lB[r] - v
        elif v > uB[r] + ptol:
            inf = v - uB[r]
        else:
            continue
        score = inf * inf / weights[r]
        if score > bestscore:
            bestscore = score
            best = r
    return best


@njit(cache=True)
def dual_ratio_test(alpha, d, status, lower, upper, sgn, delta, dtol, pivtol):
    """Bound-flipping ratio test with a Harris two-pass choice of the entering column.

    sgn = +1 when the leaving variable goes to its upper bound, -1 to its lower bound.
    delta = the size of its primal infeasibility (the initial slope of the dual objective).
    Returns (entering column, number of flips, flip list). Entering = -1 means no candidate
    (the dual is unbounded along this row: the primal is infeasible).

    The pivot tolerance is relative to the largest |alpha_j| in the row, so a badly scaled row
    cannot hand in a pivot that is tiny next to its neighbours. The Harris bound is the smallest
    tolerance-relaxed ratio (d_j + dtol) / |alpha_j|: stepping to any candidate below it leaves
    every reduced cost within dtol of feasibility.

    References: Maros 2003 ("A generalized dual phase-2 simplex algorithm", EJOR 149);
    Koberstein 2005 (PhD thesis, Paderborn) sec. 3.2.3 and 6.2.2; Harris 1973.
    """
    nn = len(alpha)
    amax = 0.0
    for j in range(nn):
        st = status[j]
        if st != BASIC and st != FIXED:
            a = abs(alpha[j])
            if a > amax:
                amax = a
    ptol = max(pivtol, 1e-9 * amax)
    cand = np.empty(nn, dtype=np.int64)
    ratio = np.empty(nn)
    relax = np.empty(nn)
    nc = 0
    for j in range(nn):
        st = status[j]
        if st == BASIC or st == FIXED:
            continue
        a = sgn * alpha[j]
        if st == AT_LOWER:
            if a > ptol:
                cand[nc] = j
                ratio[nc] = max(d[j], 0.0) / a
                relax[nc] = (max(d[j], 0.0) + dtol) / a
                nc += 1
        elif st == AT_UPPER:
            if a < -ptol:
                cand[nc] = j
                ratio[nc] = min(d[j], 0.0) / a
                relax[nc] = (min(d[j], 0.0) - dtol) / a
                nc += 1
        else:  # AT_ZERO: a free nonbasic variable blocks at ratio |d|/|a|
            if abs(a) > ptol:
                cand[nc] = j
                ratio[nc] = abs(d[j]) / abs(a)
                relax[nc] = (abs(d[j]) + dtol) / abs(a)
                nc += 1
    if nc == 0:
        return -1, 0, np.empty(0, dtype=np.int64)
    order = np.argsort(ratio[:nc])
    flips = np.empty(nc, dtype=np.int64)
    nf = 0
    slope = delta
    k = 0
    while k < nc - 1:
        j = cand[order[k]]
        rng = upper[j] - lower[j]
        drop = abs(alpha[j]) * rng
        if status[j] != AT_ZERO and np.isfinite(rng) and slope - drop > 0.0:
            flips[nf] = j
            nf += 1
            slope -= drop
            k += 1
            continue
        break
    # Harris pass 1: the smallest relaxed ratio among the remaining candidates
    bound = np.inf
    for kk in range(k, nc):
        r = relax[order[kk]]
        if r < bound:
            bound = r
    # pass 2: among candidates with ratio <= bound, the largest |alpha|
    best = cand[order[k]]
    besta = abs(alpha[best])
    kk = k + 1
    while kk < nc and ratio[order[kk]] <= bound:
        j = cand[order[kk]]
        a = abs(alpha[j])
        if a > besta:
            besta = a
            best = j
        kk += 1
    return best, nf, flips[:nf].copy()


@njit(cache=True)
def update_duals(d, alpha, theta, status):
    for j in range(len(d)):
        if status[j] != BASIC and alpha[j] != 0.0:
            d[j] -= theta * alpha[j]


@njit(cache=True)
def update_dse(weights, alpha_q, tau, r, beta_r):
    """Forrest & Goldfarb 1992 dual steepest-edge weight update (exact form)."""
    ar = alpha_q[r]
    wr = weights[r]
    for i in range(len(weights)):
        if i == r:
            continue
        ai = alpha_q[i]
        if ai == 0.0:
            continue
        ratio = ai / ar
        w = weights[i] + ratio * (ratio * beta_r - 2.0 * tau[i])
        weights[i] = max(w, 1e-4 * (1.0 + ratio * ratio))
    weights[r] = max(beta_r / (ar * ar), 1e-8)
