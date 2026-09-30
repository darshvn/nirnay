"""Numba kernels for the simplex method: the loops that run every iteration.

Status codes for every variable (structural or logical):
    BASIC = 0, AT_LOWER = 1, AT_UPPER = 2, AT_ZERO = 3 (nonbasic free, value 0), FIXED = 4
"""
from __future__ import annotations

import numpy as np
from numba import njit

from ..linalg.lu import _btran_lu_sparse, _eta_btran, _eta_ftran, _ftran_lu

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


@njit(cache=True)
def choose_leaving_head(x, head, lo, up, weights, skip, ptol):
    """choose_leaving without gathering x, lo, up by basis position first: reads through head."""
    best = -1
    bestscore = 0.0
    for r in range(len(head)):
        if skip[r]:
            continue
        j = head[r]
        v = x[j]
        if v < lo[j] - ptol:
            inf = lo[j] - v
        elif v > up[j] + ptol:
            inf = v - up[j]
        else:
            continue
        score = inf * inf / weights[r]
        if score > bestscore:
            bestscore = score
            best = r
    return best


@njit(cache=True)
def apply_flips(flips, status, x, lo, up, n, colptr, rowidx, vals, rhs):
    """Move each flipped nonbasic variable to its other bound; accumulate a_j * change into rhs
    (row-indexed), so one FTRAN updates the basic variables for all flips together."""
    for f in range(len(flips)):
        j = flips[f]
        old = x[j]
        if status[j] == AT_LOWER:
            status[j] = AT_UPPER
            x[j] = up[j]
        else:
            status[j] = AT_LOWER
            x[j] = lo[j]
        delta = x[j] - old
        if j < n:
            for p in range(colptr[j], colptr[j + 1]):
                rhs[rowidx[p]] += vals[p] * delta
        else:
            rhs[j - n] -= delta


@njit(cache=True)
def sub_scaled_head(x, head, v, t):
    """x[head] -= t * v, without the temporary arrays numpy would make."""
    for r in range(len(head)):
        if v[r] != 0.0:
            x[head[r]] -= t * v[r]


# ---------------------------------------------------------------------------------------------
# The whole dual simplex iteration in one compiled call. It runs until something needs the
# Python driver (optimality, a refactorisation, an unstable pivot, a possible infeasibility,
# a full eta file or the iteration budget) and says which with a code.
RUN_OPTIMAL, RUN_NO_ENTERING, RUN_UNSTABLE, RUN_REFACTOR, RUN_GROW, RUN_BUDGET, RUN_SKIP_EXHAUSTED = \
    0, 1, 2, 3, 4, 5, 6


@njit(cache=True)
def dual_run(n, m, Acp, Ari, Avl, ATp, ATi, ATx,
             x, d, c, status, head, weights, lo, up, skip,
             Lp, Li, Lx, Up, Ui, Ux, Udiag, prow, q, URp, URj, URx, LRp, LRc, LRx,
             Ep, Er, Ei, Ex, n_eta, refactor_every, lu_nnz,
             ptol, dtol, pivtol, budget):
    """Returns (code, iterations done, n_eta, row r involved)."""
    nm = n + m
    e = np.zeros(m)
    rho = np.empty(m)
    wrk = np.empty(m)
    alpha_r = np.empty(nm)
    alpha_q = np.empty(m)
    tau = np.empty(m)
    rhs = np.zeros(m)
    any_skip = False
    for t in range(m):
        if skip[t]:
            any_skip = True
    its = 0
    while its < budget:
        # room in the eta file for one more (dense, worst case) update
        if Ep[n_eta] + 1 + m > len(Ei) or n_eta + 2 >= len(Er):
            return RUN_GROW, its, n_eta, -1
        r = choose_leaving_head(x, head, lo, up, weights, skip, ptol)
        if r < 0:
            if any_skip:
                return RUN_SKIP_EXHAUSTED, its, n_eta, -1
            return RUN_OPTIMAL, its, n_eta, -1
        jr = head[r]
        if x[jr] < lo[jr]:
            sgn = -1.0
            target = lo[jr]
            delta = lo[jr] - x[jr]
        else:
            sgn = 1.0
            target = up[jr]
            delta = x[jr] - up[jr]
        # BTRAN: rho = B^-T e_r
        for t in range(m):
            e[t] = 0.0
        e[r] = 1.0
        if n_eta:
            _eta_btran(e, Ep, Er, Ei, Ex, n_eta)
        _btran_lu_sparse(m, URp, URj, URx, LRp, LRc, LRx, Udiag, prow, q, e, rho)
        pivot_row(n, m, ATp, ATi, ATx, rho, status, alpha_r)
        qe, nflip, flips = dual_ratio_test(alpha_r, d, status, lo, up, sgn, delta, dtol, pivtol)
        if qe < 0:
            return RUN_NO_ENTERING, its, n_eta, r
        dq = d[qe]
        stq = status[qe]
        if (stq == AT_LOWER and dq < 0.0) or (stq == AT_UPPER and dq > 0.0):
            c[qe] -= dq
            d[qe] = 0.0
        # FTRAN: alpha_q = B^-1 a_q
        for t in range(m):
            wrk[t] = 0.0
        if qe < n:
            for p in range(Acp[qe], Acp[qe + 1]):
                wrk[Ari[p]] = Avl[p]
        else:
            wrk[qe - n] = -1.0
        _ftran_lu(m, Lp, Li, Lx, Up, Ui, Ux, Udiag, prow, q, wrk, alpha_q)
        if n_eta:
            _eta_ftran(alpha_q, Ep, Er, Ei, Ex, n_eta)
        apiv = alpha_q[r]
        if abs(apiv - alpha_r[qe]) > 1e-6 * (1.0 + abs(apiv)) or abs(apiv) < 1e-9:
            return RUN_UNSTABLE, its, n_eta, r
        # bound flips move the basic variables once, before the pivot
        if nflip:
            for t in range(m):
                rhs[t] = 0.0
            apply_flips(flips, status, x, lo, up, n, Acp, Ari, Avl, rhs)
            _ftran_lu(m, Lp, Li, Lx, Up, Ui, Ux, Udiag, prow, q, rhs, wrk)
            if n_eta:
                _eta_ftran(wrk, Ep, Er, Ei, Ex, n_eta)
            sub_scaled_head(x, head, wrk, 1.0)
        # dual update
        theta_d = d[qe] / alpha_r[qe]
        update_duals(d, alpha_r, theta_d, status)
        leaving = head[r]
        d[leaving] = -theta_d
        d[qe] = 0.0
        # primal update
        theta_p = (x[leaving] - target) / apiv
        sub_scaled_head(x, head, alpha_q, theta_p)
        x[qe] += theta_p
        x[leaving] = target
        # dual steepest-edge weights: tau = B^-1 rho
        beta = 0.0
        for t in range(m):
            wrk[t] = rho[t]
            beta += rho[t] * rho[t]
        _ftran_lu(m, Lp, Li, Lx, Up, Ui, Ux, Udiag, prow, q, wrk, tau)
        if n_eta:
            _eta_ftran(tau, Ep, Er, Ei, Ex, n_eta)
        update_dse(weights, alpha_q, tau, r, beta)
        # basis change
        if lo[leaving] == up[leaving]:
            status[leaving] = FIXED
        elif target == lo[leaving]:
            status[leaving] = AT_LOWER
        else:
            status[leaving] = AT_UPPER
        head[r] = qe
        status[qe] = BASIC
        its += 1
        if any_skip:
            for t in range(m):
                skip[t] = False
            any_skip = False
        # eta update (product form): alpha_q stored sparsely, its pivot first
        start = Ep[n_eta]
        Ei[start] = r
        Ex[start] = apiv
        k = start + 1
        for t in range(m):
            if t != r and abs(alpha_q[t]) > 1e-14:
                Ei[k] = t
                Ex[k] = alpha_q[t]
                k += 1
        Er[n_eta] = r
        n_eta += 1
        Ep[n_eta] = k
        if n_eta >= refactor_every or k > 8 * (lu_nnz + m):
            return RUN_REFACTOR, its, n_eta, r
    return RUN_BUDGET, its, n_eta, -1
