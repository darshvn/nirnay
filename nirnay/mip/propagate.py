"""Activity-based bound propagation (domain propagation) for linear rows.

For a row  rl <= sum_j a_j x_j <= ru  the smallest and largest possible activities over the
current bounds give, for every variable in the row, a bound implied by the others:

    a_j > 0:   x_j <= (ru - minact_{-j}) / a_j,     x_j >= (rl - maxact_{-j}) / a_j
    a_j < 0:   the same with the inequalities reversed

where minact_{-j} is the minimum activity without x_j. Integer variables round the implied bound
inward. Infinite contributions are counted, so a row with exactly one infinite contributor can
still bound that one variable.

Reference: T. Achterberg, "Constraint Integer Programming", PhD thesis, TU Berlin 2007, ch. 7.1;
           M. W. P. Savelsbergh, "Preprocessing and probing techniques for MIP", ORSA JoC 6, 1994.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def propagate(Rp, Rj, Rv, rl, ru, lb, ub, is_int, max_passes, feastol):
    """Tighten lb/ub in place. Returns (status, number of tightenings); status 1 = infeasible."""
    m = len(rl)
    total = 0
    for _ in range(max_passes):
        changed = 0
        for i in range(m):
            if rl[i] == -np.inf and ru[i] == np.inf:
                continue
            minact = 0.0
            maxact = 0.0
            nminf = 0
            nmaxf = 0
            jminf = -1
            jmaxf = -1
            for p in range(Rp[i], Rp[i + 1]):
                j = Rj[p]
                a = Rv[p]
                if a > 0:
                    if lb[j] == -np.inf:
                        nminf += 1
                        jminf = j
                    else:
                        minact += a * lb[j]
                    if ub[j] == np.inf:
                        nmaxf += 1
                        jmaxf = j
                    else:
                        maxact += a * ub[j]
                else:
                    if ub[j] == np.inf:
                        nminf += 1
                        jminf = j
                    else:
                        minact += a * ub[j]
                    if lb[j] == -np.inf:
                        nmaxf += 1
                        jmaxf = j
                    else:
                        maxact += a * lb[j]
            # row infeasibility
            if nminf == 0 and minact > ru[i] + feastol * (1.0 + abs(ru[i])):
                return 1, total
            if nmaxf == 0 and maxact < rl[i] - feastol * (1.0 + abs(rl[i])):
                return 1, total
            if nminf > 1 and nmaxf > 1:
                continue
            for p in range(Rp[i], Rp[i + 1]):
                j = Rj[p]
                a = Rv[p]
                # minimum / maximum activity of the rest of the row
                if a > 0:
                    cmin = a * lb[j] if lb[j] != -np.inf else 0.0
                    cmax = a * ub[j] if ub[j] != np.inf else 0.0
                    own_min_inf = lb[j] == -np.inf
                    own_max_inf = ub[j] == np.inf
                else:
                    cmin = a * ub[j] if ub[j] != np.inf else 0.0
                    cmax = a * lb[j] if lb[j] != -np.inf else 0.0
                    own_min_inf = ub[j] == np.inf
                    own_max_inf = lb[j] == -np.inf
                rest_min_ok = (nminf == 0) or (nminf == 1 and own_min_inf and jminf == j)
                rest_max_ok = (nmaxf == 0) or (nmaxf == 1 and own_max_inf and jmaxf == j)
                rest_min = minact - cmin
                rest_max = maxact - cmax
                new_lb = -np.inf
                new_ub = np.inf
                if ru[i] != np.inf and rest_min_ok:
                    v = (ru[i] - rest_min) / a
                    if a > 0:
                        new_ub = v
                    else:
                        new_lb = v
                if rl[i] != -np.inf and rest_max_ok:
                    v = (rl[i] - rest_max) / a
                    if a > 0:
                        if v > new_lb:
                            new_lb = v
                    else:
                        if v < new_ub:
                            new_ub = v
                if is_int[j]:
                    if new_ub != np.inf:
                        new_ub = np.floor(new_ub + 1e-6)
                    if new_lb != -np.inf:
                        new_lb = np.ceil(new_lb - 1e-6)
                # accept only clear improvements, so round-off cannot creep into the bounds
                if new_ub < ub[j]:
                    gain = ub[j] - new_ub
                    if is_int[j] or gain > 1e-3 * max(1.0, abs(new_ub)):
                        if new_ub < lb[j] - feastol * (1.0 + abs(lb[j])):
                            return 1, total
                        ub[j] = max(new_ub, lb[j])
                        changed += 1
                if new_lb > lb[j]:
                    gain = new_lb - lb[j]
                    if is_int[j] or gain > 1e-3 * max(1.0, abs(new_lb)):
                        if new_lb > ub[j] + feastol * (1.0 + abs(ub[j])):
                            return 1, total
                        lb[j] = min(new_lb, ub[j])
                        changed += 1
        total += changed
        if changed == 0:
            break
    return 0, total
