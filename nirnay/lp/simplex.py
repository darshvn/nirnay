"""Bounded dual simplex method, with a primal simplex for the final cleanup.

References
  A. Koberstein, "The dual simplex method, techniques for a fast and stable implementation",
      PhD thesis, Universität Paderborn, 2005 (the overall design followed here).
  J. J. Forrest, D. Goldfarb, "Steepest-edge simplex algorithms for linear programming",
      Mathematical Programming 57, 1992 (dual steepest-edge pricing).
  I. Maros, "A generalized dual phase-2 simplex algorithm", EJOR 149, 2003 (bound flipping).
  P. M. J. Harris, "Pivot selection methods of the Devex LP code", Math. Programming 5, 1973.
  R. Fourer, "Notes on the dual simplex method", 1994 (the auxiliary problem for dual phase 1).

Computational form: every row gets a logical variable, so all constraints are equalities

      A x - s = 0,     lb <= x <= ub,    rl <= s <= ru,

over n + m variables. The basis starts as the m logicals (B = -I), which needs no factorisation.

Dual phase 1 solves the auxiliary problem of Fourer (Koberstein sec. 4.1.2): the same costs with
every bound replaced by a box — free -> [-1, 1], lower only -> [0, 1], upper only -> [-1, 0],
boxed -> [0, 0]. Every variable is boxed there, so any basis is dual feasible after choosing
bounds, and its optimal basis is dual feasible for the real problem whenever the real problem is
dual feasible at all. Phase 2 then runs from that basis on the real bounds.

Degeneracy: costs are perturbed by small random amounts in the direction that keeps the start
dual feasible (Koberstein sec. 6.2). The perturbation is removed at the end and any dual
infeasibility it leaves is cleaned up by the primal simplex.

The class keeps its factorisation and basis between calls, so a branch-and-bound node that only
changes bounds re-solves from its parent's basis with a few dual iterations.
"""
from __future__ import annotations

import time

import numpy as np

from ..linalg.lu import BasisFactor
from ..model import INF, Model, Result
from ..presolve.scaling import compute_scaling, scale_model
from . import _simplex_kernels as K

BASIC, AT_LOWER, AT_UPPER, AT_ZERO, FIXED = K.BASIC, K.AT_LOWER, K.AT_UPPER, K.AT_ZERO, K.FIXED


class SimplexLP:
    def __init__(self, model: Model, scale: bool = True, perturb: bool = True, verbose: bool = False,
                 ptol: float = 1e-7, dtol: float = 1e-7, pivtol: float = 1e-7, seed: int = 0):
        self.model = model
        self.verbose = verbose
        self.ptol, self.dtol, self.pivtol = ptol, dtol, pivtol
        self.rng = np.random.default_rng(seed)
        if scale:
            self.R, self.C = compute_scaling(model.A)
        else:
            self.R, self.C = np.ones(model.m), np.ones(model.n)
        S = scale_model(model, self.R, self.C)
        self.A = S.A
        self.AT = S.A.transpose()
        self.n, self.m = S.n, S.m
        n, m = self.n, self.m
        self.cost = np.concatenate([S.c, np.zeros(m)])
        self.c0 = model.c0
        self.lo = np.concatenate([S.lb, S.rl])           # real bounds, scaled
        self.up = np.concatenate([S.ub, S.ru])
        self.perturb = perturb
        self.head = np.arange(n, n + m, dtype=np.int64)
        self.status = np.empty(n + m, dtype=np.int64)
        self.x = np.zeros(n + m)
        self.factor_ = BasisFactor(m)
        self.iterations = 0
        self.phase1_done = False
        self._init_nonbasic(self.lo, self.up, self.cost)
        self.status[self.head] = BASIC
        self.weights = np.ones(m)

    # ---------------------------------------------------------------- helpers
    def _init_nonbasic(self, lo, up, c):
        idx = np.arange(self.n + self.m)
        self.status[:] = self._statuses(idx, lo, up, c)
        self.x[:] = self._values(idx, lo, up)

    @staticmethod
    def _statuses(idx, lo, up, d):
        """Vectorised _bound_status for the variables idx (d: their reduced costs)."""
        l, u = lo[idx], up[idx]
        fl, fu = np.isfinite(l), np.isfinite(u)
        dj = d[idx] if np.ndim(d) else np.full(len(idx), float(d))
        st = np.where(fl & fu, np.where(dj >= 0, AT_LOWER, AT_UPPER),
                      np.where(fl, AT_LOWER, np.where(fu, AT_UPPER, AT_ZERO)))
        st[l == u] = FIXED
        return st

    def _values(self, idx, lo, up):
        """Vectorised _bound_value: the value each nonbasic variable in idx sits at."""
        st = self.status[idx]
        return np.where((st == AT_LOWER) | (st == FIXED), lo[idx], np.where(st == AT_UPPER, up[idx], 0.0))

    @staticmethod
    def _bound_status(j, lo, up, dj):
        l, u = lo[j], up[j]
        if l == u:
            return FIXED
        if np.isfinite(l) and np.isfinite(u):
            return AT_LOWER if dj >= 0 else AT_UPPER
        if np.isfinite(l):
            return AT_LOWER
        if np.isfinite(u):
            return AT_UPPER
        return AT_ZERO

    def _bound_value(self, j, lo, up):
        st = self.status[j]
        if st == AT_LOWER or st == FIXED:
            return lo[j]
        if st == AT_UPPER:
            return up[j]
        return 0.0

    def _column(self, j):
        if j < self.n:
            return self.A.col(j)
        return np.array([j - self.n]), np.array([-1.0])

    def _dense_column(self, j):
        v = np.zeros(self.m)
        r, a = self._column(j)
        v[r] = a
        return v

    def _refactor(self, lo, up, c):
        m, n = self.m, self.n
        head = self.head
        is_log = head >= n
        # column lengths: A's for structurals, 1 for logicals (-e_i)
        hs = np.where(is_log, 0, head)
        lens = np.where(is_log, 1, self.A.colptr[hs + 1] - self.A.colptr[hs])
        Bp = np.zeros(m + 1, dtype=np.int64)
        np.cumsum(lens, out=Bp[1:])
        # gather every entry in one pass: position p of B maps to entry start[k] + offset in A
        k_of = np.repeat(np.arange(m), lens)
        off = np.arange(Bp[-1]) - Bp[k_of]
        src = self.A.colptr[hs][k_of] + off
        log_k = is_log[k_of]
        src_safe = np.where(log_k, 0, src)
        Bi = np.where(log_k, head[k_of] - n, self.A.rowidx[src_safe] if self.A.nnz else 0).astype(np.int64)
        Bx = np.where(log_k, -1.0, self.A.vals[src_safe] if self.A.nnz else 0.0)
        repairs = self.factor_.factor(Bp, Bi, Bx, is_log)
        self.factor_valid = True
        if repairs:
            for pos, row in repairs:
                old = self.head[pos]
                new = self.n + row
                self.head[pos] = new
                self.status[new] = BASIC
                self.status[old] = self._bound_status(old, lo, up, 0.0)
                self.x[old] = self._bound_value(old, lo, up)
            self.repairs = getattr(self, "repairs", 0) + len(repairs)
            return self._refactor(lo, up, c)
        self._recompute(lo, up, c)

    def _recompute(self, lo, up, c):
        """Primal basic values and duals from scratch."""
        nb = self.status != BASIC
        rhs = -(self.A.matvec(self.x[: self.n] * nb[: self.n]))
        # logical columns are -e_i: their contribution to -N x_N is +x_{n+i}
        rhs += self.x[self.n:] * nb[self.n:]
        xB = self.factor_.ftran(rhs)
        self.x[self.head] = xB
        y = self.factor_.btran(c[self.head])
        self.y = y
        d = c.copy()
        d[: self.n] -= self.A.rmatvec(y)
        d[self.n:] += y
        d[self.head] = 0.0
        self.d = d

    def _row(self, rho):
        alpha = np.empty(self.n + self.m)
        K.pivot_row(self.n, self.m, self.AT.colptr, self.AT.rowidx, self.AT.vals, rho, self.status, alpha)
        return alpha

    def _dual_feasibility_fix(self, lo, up, c):
        """Restore dual feasibility after a recompute: flip boxed variables, shift other costs."""
        st, d = self.status, self.d
        bad = ((st == AT_LOWER) & (d < -self.dtol)) | ((st == AT_UPPER) & (d > self.dtol)) |               ((st == AT_ZERO) & (np.abs(d) > self.dtol))
        idx = np.flatnonzero(bad)
        if len(idx) == 0:
            return 0
        boxed = np.isfinite(lo[idx]) & np.isfinite(up[idx])
        flip = idx[boxed]                       # boxed: move to the other bound
        to_up = st[flip] == AT_LOWER
        st[flip] = np.where(to_up, AT_UPPER, AT_LOWER)
        self.x[flip] = np.where(to_up, up[flip], lo[flip])
        shift = idx[~boxed]                     # otherwise shift the cost so d_j = 0
        c[shift] -= d[shift]
        d[shift] = 0.0
        self._recompute(lo, up, c)
        return len(idx)

    # ---------------------------------------------------------------- dual simplex loop
    def _ensure_factor(self, lo, up, c):
        """Refactor only when the basis changed outside the simplex loops (a new basis was
        loaded); otherwise the current LU + eta file is still B^-1 and a recompute suffices."""
        if getattr(self, "factor_valid", False):
            self._recompute(lo, up, c)
        else:
            self._refactor(lo, up, c)

    def _dual_loop(self, lo, up, c, deadline, max_iter):
        m = self.m
        self._ensure_factor(lo, up, c)
        self._dual_feasibility_fix(lo, up, c)
        skip = np.zeros(m, dtype=bool)
        fresh = True
        while True:
            if self.iterations >= max_iter:
                return "iteration_limit"
            if time.perf_counter() > deadline:
                return "time_limit"
            xB = self.x[self.head]
            lB, uB = lo[self.head], up[self.head]
            w = np.where(skip, np.inf, self.weights)
            r = K.choose_leaving(xB, lB, uB, w, self.ptol)
            if r < 0:
                if skip.any():
                    skip[:] = False
                    self._refactor(lo, up, c)
                    continue
                return "optimal"
            if xB[r] < lB[r]:
                sgn, target, delta = -1.0, lB[r], lB[r] - xB[r]
            else:
                sgn, target, delta = 1.0, uB[r], xB[r] - uB[r]
            e = np.zeros(m)
            e[r] = 1.0
            rho = self.factor_.btran(e)
            alpha_r = self._row(rho)
            q, nflip, flips = K.dual_ratio_test(alpha_r, self.d, self.status, lo, up, sgn, delta,
                                                self.dtol, self.pivtol)
            if q < 0:
                if self.factor_.n_eta or not fresh:
                    # confirm on a fresh factorisation with recomputed duals before concluding
                    self._refactor(lo, up, c)
                    self._dual_feasibility_fix(lo, up, c)
                    fresh = True
                    continue
                return "infeasible"
            fresh = False
            # a reduced cost with the wrong sign (inside the tolerance) is shifted to zero, so
            # the step never moves the dual objective backwards (Koberstein sec. 6.2.2.2)
            dq, stq = self.d[q], self.status[q]
            if (stq == AT_LOWER and dq < 0) or (stq == AT_UPPER and dq > 0):
                c[q] -= dq
                self.d[q] = 0.0
            alpha_q = self.factor_.ftran(self._dense_column(q))
            # the pivot seen from the row and from the column must agree
            if abs(alpha_q[r] - alpha_r[q]) > 1e-6 * (1 + abs(alpha_q[r])) or abs(alpha_q[r]) < 1e-9:
                if self.factor_.n_eta == 0:
                    skip[r] = True         # fresh factor and still unstable: try another row
                self._refactor(lo, up, c)
                continue
            # bound flips move the basic variables once, before the pivot
            if nflip:
                rhs = np.zeros(m)
                for j in flips:
                    old = self.x[j]
                    if self.status[j] == AT_LOWER:
                        self.status[j], self.x[j] = AT_UPPER, up[j]
                    else:
                        self.status[j], self.x[j] = AT_LOWER, lo[j]
                    rr, aa = self._column(j)
                    rhs[rr] += aa * (self.x[j] - old)
                dxB = self.factor_.ftran(rhs)
                self.x[self.head] -= dxB
            # dual update
            theta_d = self.d[q] / alpha_r[q]
            K.update_duals(self.d, alpha_r, theta_d, self.status)
            leaving = self.head[r]
            self.d[leaving] = -theta_d
            self.d[q] = 0.0
            # primal update
            xr = self.x[leaving]
            theta_p = (xr - target) / alpha_q[r]
            self.x[self.head] -= theta_p * alpha_q
            self.x[q] += theta_p
            self.x[leaving] = target
            # steepest-edge weights
            tau = self.factor_.ftran(rho)
            K.update_dse(self.weights, alpha_q, tau, r, float(rho @ rho))
            # basis change
            self.status[leaving] = AT_LOWER if target == lo[leaving] else AT_UPPER
            if lo[leaving] == up[leaving]:
                self.status[leaving] = FIXED
            self.head[r] = q
            self.status[q] = BASIC
            self.iterations += 1
            skip[:] = False
            if self.factor_.update(r, alpha_q):
                self._refactor(lo, up, c)
                self._dual_feasibility_fix(lo, up, c)
            if self.verbose and self.iterations % 200 == 0:
                inf = np.maximum(0, np.maximum(lo[self.head] - self.x[self.head], self.x[self.head] - up[self.head]))
                print(f"  dual {self.iterations:7d}  obj {c @ self.x:+.10e}  pinf {inf.sum():.2e}")

    # ---------------------------------------------------------------- primal simplex (cleanup)
    def _primal_loop(self, lo, up, c, deadline, max_iter):
        m = self.m
        self._ensure_factor(lo, up, c)
        while True:
            if self.iterations >= max_iter:
                return "iteration_limit"
            if time.perf_counter() > deadline:
                return "time_limit"
            d, st = self.d, self.status
            viol = np.where(st == AT_LOWER, -d, 0.0) + np.where(st == AT_UPPER, d, 0.0) + \
                np.where(st == AT_ZERO, np.abs(d), 0.0)
            viol[(st == BASIC) | (st == FIXED)] = 0.0
            q = int(np.argmax(viol))
            if viol[q] <= self.dtol:
                return "optimal"
            direction = 1.0 if (st[q] == AT_LOWER or (st[q] == AT_ZERO and d[q] < 0)) else -1.0
            alpha_q = self.factor_.ftran(self._dense_column(q))
            rate = -direction * alpha_q                     # d x_B / d theta
            xB, lB, uB = self.x[self.head], lo[self.head], up[self.head]
            # Harris two-pass ratio test over the basic variables
            with np.errstate(divide="ignore", invalid="ignore"):
                lim = np.where(rate < -self.pivtol, (xB - lB + self.ptol) / -rate,
                               np.where(rate > self.pivtol, (uB - xB + self.ptol) / rate, np.inf))
            tmax = lim.min(initial=np.inf)
            own = up[q] - lo[q]
            if not np.isfinite(tmax) and not np.isfinite(own):
                return "unbounded"
            if own <= tmax:
                # the entering variable reaches its other bound first: a flip, no pivot
                self.x[self.head] -= direction * own * alpha_q
                self.x[q] = up[q] if direction > 0 else lo[q]
                st[q] = AT_UPPER if direction > 0 else AT_LOWER
                self.iterations += 1
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                exact = np.where(rate < -self.pivtol, (xB - lB) / -rate,
                                 np.where(rate > self.pivtol, (uB - xB) / rate, np.inf))
            cand = np.flatnonzero(exact <= tmax)
            r = int(cand[np.argmax(np.abs(alpha_q[cand]))])
            theta = max(exact[r], 0.0)
            e = np.zeros(m)
            e[r] = 1.0
            rho = self.factor_.btran(e)
            alpha_r = self._row(rho)
            theta_d = d[q] / alpha_q[r]
            K.update_duals(self.d, alpha_r, theta_d, self.status)
            leaving = self.head[r]
            self.d[leaving] = -theta_d
            self.d[q] = 0.0
            self.x[self.head] -= direction * theta * alpha_q
            self.x[q] += direction * theta
            to_lower = rate[r] < 0
            self.x[leaving] = lB[r] if to_lower else uB[r]
            st[leaving] = FIXED if lo[leaving] == up[leaving] else (AT_LOWER if to_lower else AT_UPPER)
            self.head[r] = q
            st[q] = BASIC
            self.iterations += 1
            if self.factor_.update(r, alpha_q):
                self._refactor(lo, up, c)

    # ---------------------------------------------------------------- driver
    def set_bounds(self, j: int, lb: float, ub: float):
        """Change a structural variable's bounds (unscaled values), keeping the basis."""
        self.lo[j] = lb / self.C[j]
        self.up[j] = ub / self.C[j]
        if self.status[j] != BASIC:
            self.status[j] = self._bound_status(j, self.lo, self.up, self.d[j] if hasattr(self, "d") else 0.0)
            self.x[j] = self._bound_value(j, self.lo, self.up)

    def _perturbed(self, c, lo, up):
        """Costs moved by small random amounts in the direction each nonbasic variable's bound
        already favours, so the start stays dual feasible and ties between ratios are broken
        (Koberstein sec. 6.2.2.3). Without it the dual simplex can cycle on degenerate LPs."""
        if not self.perturb:
            return c.copy()
        nm = self.n + self.m
        if getattr(self, "_pert_u", None) is None:
            self._pert_u = self.rng.random(nm)      # drawn once: repeated solves reuse it
        mag = 5e-7 * (1 + np.abs(c)) * (1 + self._pert_u)
        sign = np.where(self.status == AT_UPPER, -1.0, 1.0)
        pert = np.where(self.status == BASIC, 0.0, sign * mag)
        pert[(lo == up) | (self.status == AT_ZERO)] = 0.0
        return c + pert

    def set_structural_bounds(self, lb: np.ndarray, ub: np.ndarray):
        """Replace the bounds of every structural variable (unscaled), keeping the basis.

        Only the variables whose bounds actually change are touched, so a branch-and-bound node
        that differs from the last one in a few bounds costs a few updates."""
        n = self.n
        nlo, nup = lb / self.C, ub / self.C
        changed = np.flatnonzero((nlo != self.lo[:n]) | (nup != self.up[:n]))
        self.lo[:n] = nlo
        self.up[:n] = nup
        d = getattr(self, "d", None)
        nb = changed[self.status[changed] != BASIC]
        if len(nb):
            self.status[nb] = self._statuses(nb, self.lo, self.up, d if d is not None else 0.0)
            self.x[nb] = self._values(nb, self.lo, self.up)
        return len(changed)

    def get_basis(self) -> np.ndarray:
        """The basis as one status byte per variable (the basic set is where status == BASIC)."""
        return self.status.astype(np.int8)

    def load_basis(self, status: np.ndarray):
        """Install a basis saved by get_basis (from any node of the same model)."""
        st = status.astype(np.int64)
        if int(np.sum(st == BASIC)) != self.m:
            return False
        same = bool(np.array_equal(st == BASIC, self.status == BASIC))
        self.status[:] = st
        if not same:
            # a different basic set: the factorisation no longer describes it
            self.head = np.flatnonzero(st == BASIC).astype(np.int64)
            self.factor_valid = False
        nb = np.flatnonzero(st != BASIC)
        lo, up, s = self.lo[nb], self.up[nb], st[nb]
        # statuses that no longer fit the current bounds are recomputed from the bounds
        bad = ((s == FIXED) & (lo != up)) | ((s == AT_LOWER) & ~np.isfinite(lo)) |               ((s == AT_UPPER) & ~np.isfinite(up))
        s = np.where(bad, self._statuses(nb, self.lo, self.up, 0.0), s)
        s = np.where((s != FIXED) & (lo == up), FIXED, s)
        self.status[nb] = s
        self.x[nb] = self._values(nb, self.lo, self.up)
        return True

    def primal(self) -> np.ndarray:
        """Current structural values, unscaled."""
        return self.x[: self.n] * self.C

    def solve(self, time_limit: float = np.inf, max_iter: int = 10**7) -> Result:
        t0 = time.perf_counter()
        deadline = t0 + time_limit
        if not self.phase1_done:
            # --- dual phase 1 on Fourer's auxiliary boxes ---
            has_l, has_u = np.isfinite(self.lo), np.isfinite(self.up)
            alo = np.where(has_l & has_u, 0.0, np.where(has_l, 0.0, -1.0))
            aup = np.where(has_l & has_u, 0.0, np.where(has_u, 0.0, 1.0))
            fixed = self.lo == self.up
            alo[fixed] = aup[fixed] = 0.0
            self._init_nonbasic(alo, aup, self.cost)
            self.status[self.head] = BASIC
            st = self._dual_loop(alo, aup, self._perturbed(self.cost, alo, aup), deadline, max_iter)
            if st not in ("optimal",):
                if st == "infeasible":
                    st = "dual_infeasible"
                return self._result(st, t0)
            self.phase1_done = True
            nb = np.flatnonzero(self.status != BASIC)
            self.status[nb] = self._statuses(nb, self.lo, self.up, self.d)
            self.x[nb] = self._values(nb, self.lo, self.up)
        # --- phase 2 on perturbed costs ---
        c = self._perturbed(self.cost, self.lo, self.up)
        st = self._dual_loop(self.lo, self.up, c, deadline, max_iter)
        if st == "optimal":
            # remove the perturbation; clean up any dual infeasibility with the primal simplex
            c = self.cost.copy()
            self._recompute(self.lo, self.up, c)
            # the cleanup polishes to a tighter dual tolerance: on badly scaled models (pilot) a
            # reduced cost of 1e-8 times a wide variable range still moves the objective by 1e-5
            dtol, self.dtol = self.dtol, min(self.dtol, 1e-9)
            try:
                st = self._primal_loop(self.lo, self.up, c, deadline, max_iter)
            finally:
                self.dtol = dtol
            if st == "optimal":
                inf = np.maximum(0, np.maximum(self.lo[self.head] - self.x[self.head],
                                               self.x[self.head] - self.up[self.head]))
                if inf.max(initial=0.0) > 10 * self.ptol:     # the cleanup lost primal feasibility
                    st = self._dual_loop(self.lo, self.up, c, deadline, max_iter)
        return self._result(st, t0)

    def _result(self, status, t0):
        n = self.n
        x = self.x[:n] * self.C
        if status == "optimal" and getattr(self, "factor_valid", False):
            # the loops update reduced costs incrementally but y only at refactorisation, and
            # the dual loop may have shifted costs: recompute y exactly for the true costs
            self.y = self.factor_.btran(self.cost[self.head])
        y = getattr(self, "y", np.zeros(self.m)) * self.R
        model = self.model
        z = model.c - model.A.rmatvec(y)
        obj = model.user_objective(x) if status == "optimal" else np.nan
        if status == "infeasible" or status == "dual_infeasible":
            obj = np.nan
        return Result(status=status, x=x, y=y * model.sense, z=z * model.sense, objective=obj,
                      bound=obj, iterations=self.iterations, time=time.perf_counter() - t0,
                      method="dual-simplex",
                      info={"basis": (self.head.copy(), self.status.copy()),
                            "repairs": getattr(self, "repairs", 0)})


def solve(model: Model, time_limit: float = np.inf, max_iter: int = 10**7, verbose: bool = False, **kw) -> Result:
    if model.is_mip:
        from .ipm import _relax
        model = _relax(model)
    return SimplexLP(model, verbose=verbose, **kw).solve(time_limit=time_limit, max_iter=max_iter)
