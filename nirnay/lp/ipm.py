"""Primal-dual interior-point method for LP: Mehrotra's predictor-corrector.

References
  S. Mehrotra, "On the implementation of a primal-dual interior point method",
      SIAM J. Optimization 2(4), 1992.
  S. J. Wright, "Primal-Dual Interior-Point Methods", SIAM 1997 (ch. 10-11: the practical method,
      free variables, the normal equations and their numerical issues).
  I. Lustig, R. Marsten, D. Shanno, "Interior point methods for linear programming: computational
      state of the art", ORSA J. Computing 6(1), 1994 (the bounded standard form used here).

Internal form. Rows that are equalities stay as B v = b; every other row gets a slack column,
a_i x - s_i = 0 with rl_i <= s_i <= ru_i. Every variable is then shifted and, if it only has an
upper bound, reflected, so that each one is either free, x >= 0, or 0 <= x <= U:

    min c'x   s.t.  B x = b,   x_j >= 0 (j in L),   x_j + w_j = U_j, w_j >= 0 (j in U),   rest free

with duals y (rows), z >= 0 on x >= 0 and s >= 0 on w >= 0. The Newton system reduces to the
normal equations  B Theta B' dy = r  with  Theta^-1 = z/x + s/w,  solved by the sparse Cholesky
in nirnay.linalg. Free variables have no barrier term, so a small primal regularisation keeps
Theta finite; a small dual regularisation keeps M positive definite when rows are dependent.
"""
from __future__ import annotations

import time

import numpy as np

from ..linalg.cholesky import Cholesky
from ..linalg.normal import NormalMatrix
from ..model import CSC, INF, Model, Result
from ..presolve.scaling import compute_scaling, scale_model


class _Form:
    """The bounded standard form of a model, and the maps back to it."""

    def __init__(self, model: Model):
        A = model.A
        m, n = A.m, A.n
        eq = np.isfinite(model.rl) & (model.rl == model.ru)
        free_row = ~np.isfinite(model.rl) & ~np.isfinite(model.ru)
        slack_rows = np.flatnonzero(~eq & ~free_row)
        keep_rows = np.flatnonzero(~free_row)          # a free row constrains nothing
        self.keep_rows = keep_rows
        row_new = np.full(m, -1, dtype=np.int64)
        row_new[keep_rows] = np.arange(len(keep_rows))

        fixed = np.isfinite(model.lb) & (model.lb == model.ub)
        self.fixed = fixed
        self.x_fixed = np.where(fixed, model.lb, 0.0)
        keep_cols = np.flatnonzero(~fixed)
        self.keep_cols = keep_cols

        # triplets of B = [A(keep rows, keep cols), -I on slack rows]
        cols_of = np.repeat(np.arange(n), np.diff(A.colptr))
        mask = (row_new[A.rowidx] >= 0) & ~fixed[cols_of]
        col_new = np.full(n, -1, dtype=np.int64)
        col_new[keep_cols] = np.arange(len(keep_cols))
        r = row_new[A.rowidx[mask]]
        c = col_new[cols_of[mask]]
        v = A.vals[mask]
        nx = len(keep_cols)
        ns = len(slack_rows)
        r = np.concatenate([r, row_new[slack_rows]])
        c = np.concatenate([c, nx + np.arange(ns)])
        v = np.concatenate([v, -np.ones(ns)])
        self.mB = len(keep_rows)
        self.nB = nx + ns
        self.nx = nx
        self.slack_rows = slack_rows

        # right-hand side: equalities give b; slack rows give 0; fixed columns move to the rhs
        b = np.zeros(self.mB)
        b[row_new[np.flatnonzero(eq)]] = model.rl[eq]
        if fixed.any():
            act = A.matvec(self.x_fixed)
            b -= act[keep_rows]

        lo = np.concatenate([model.lb[keep_cols], model.rl[slack_rows]])
        up = np.concatenate([model.ub[keep_cols], model.ru[slack_rows]])
        cost = np.concatenate([model.c[keep_cols], np.zeros(ns)])
        self.c0 = model.c0 + float(model.c @ self.x_fixed)

        has_lo, has_up = np.isfinite(lo), np.isfinite(up)
        self.flip = ~has_lo & has_up                   # x = u - x'
        self.shift = np.where(has_lo, lo, np.where(self.flip, up, 0.0))
        sign = np.where(self.flip, -1.0, 1.0)
        self.sign = sign
        self.lower = has_lo | has_up                    # x' >= 0
        self.boxed = has_lo & has_up
        self.U = np.where(self.boxed, up - lo, INF)
        self.free = ~self.lower

        v_signed = v * sign[c]
        self.B = CSC.from_triplets(self.mB, self.nB, r, c, v_signed)
        self.c = cost * sign
        # B_orig v = b with v = shift + sign x'  and  B = B_orig diag(sign)  =>  B x' = b - B (sign shift)
        self.b = b - self.B.matvec(sign * self.shift)
        self.c0 += float(cost @ self.shift)

    def recover(self, xp: np.ndarray, n: int) -> np.ndarray:
        v = self.shift + self.sign * xp
        x = self.x_fixed.copy()
        x[self.keep_cols] = v[: self.nx]
        return x


def _max_step(v, dv):
    """Largest alpha keeping v + alpha dv >= 0 (inf if dv never decreases)."""
    neg = dv < 0
    if not neg.any():
        return np.inf
    return float(np.min(-v[neg] / dv[neg]))


def solve(model: Model, tol: float = 1e-8, max_iter: int = 200, scale: bool = True,
          time_limit: float = np.inf, verbose: bool = False) -> Result:
    t0 = time.perf_counter()
    if model.is_mip:
        model = _relax(model)
    if scale:
        R, C = compute_scaling(model.A)
        work = scale_model(model, R, C)
    else:
        R, C = np.ones(model.m), np.ones(model.n)
        work = model
    F = _Form(work)
    B, BT = F.B, F.B.transpose()
    mB, nB = F.mB, F.nB
    c, b, U = F.c, F.b, F.U
    L = F.lower
    X = F.boxed
    free = F.free

    NM = NormalMatrix(B, BT)
    chol = Cholesky(mB, NM.Mp, NM.Mi)
    reg_p, reg_free, reg_d = 1e-10, 1e-8, 1e-10

    def factor(theta, extra):
        vals = NM.assemble(theta, extra)
        chol.factor(vals)
        return vals

    def normal_solve(rhs, vals, extra_vec, theta):
        dy = chol.solve(rhs)
        # two steps of iterative refinement against the assembled matrix
        for _ in range(2):
            r = rhs - _spmv_sym(NM.Mp, NM.Mi, vals, dy)
            if np.linalg.norm(r, np.inf) <= 1e-14 * (1 + np.linalg.norm(rhs, np.inf)):
                break
            dy += chol.solve(r)
        return dy

    # ---- starting point (Mehrotra's heuristic, adapted to bounds) ----
    theta0 = np.ones(nB)
    vals = factor(theta0, np.full(mB, reg_d))
    x = BT.matvec(chol.solve(b))
    y = chol.solve(B.matvec(c))
    zc = c - BT.matvec(y)
    z = np.where(L, np.maximum(zc, 0.0), 0.0)
    s = np.where(X, np.maximum(-zc, 0.0), 0.0)
    w = np.where(X, U - x, 0.0)
    xl = x[L]
    dx_shift = max(-1.5 * min(xl.min(initial=np.inf), w[X].min(initial=np.inf)), 0.0)
    x = np.where(L, x + dx_shift, x)
    w = np.where(X, w + dx_shift, 0.0)
    dz_shift = max(-1.5 * min(z[L].min(initial=np.inf), s[X].min(initial=np.inf)), 0.0)
    z = np.where(L, z + dz_shift, 0.0)
    s = np.where(X, s + dz_shift, 0.0)
    xz = x[L] @ z[L] + w[X] @ s[X]
    sx = x[L].sum() + w[X].sum()
    sz = z[L].sum() + s[X].sum()
    if sx > 0 and sz > 0:
        x = np.where(L, x + 0.5 * xz / sz, x)
        w = np.where(X, w + 0.5 * xz / sz, 0.0)
        z = np.where(L, z + 0.5 * xz / sx, 0.0)
        s = np.where(X, s + 0.5 * xz / sx, 0.0)
    # guard against an all-zero start
    x = np.where(L & (x <= 0), 1.0, x)
    z = np.where(L & (z <= 0), 1.0, z)
    w = np.where(X & (w <= 0), 1.0, w)
    s = np.where(X & (s <= 0), 1.0, s)

    ncomp = int(L.sum() + X.sum())
    # relative residuals in the infinity norm, as HiGHS/IPX and most IPM codes report them
    inf_norm = lambda v: float(np.max(np.abs(v), initial=0.0))
    nb, nc, nU = 1 + inf_norm(b), 1 + inf_norm(c), 1 + inf_norm(U[X])
    status, it = "iteration_limit", 0
    history = []
    best_err, stall = np.inf, 0
    loose = max(tol, 1e-6)          # accepted when progress stops: the normal equations' accuracy floor
    for it in range(1, max_iter + 1):
        rb = b - B.matvec(x)
        ru = np.where(X, U - x - w, 0.0)
        rc = c - BT.matvec(y) - z + s
        mu = (x[L] @ z[L] + w[X] @ s[X]) / max(ncomp, 1)
        pobj = c @ x
        dobj = b @ y - U[X] @ s[X]
        pinf = inf_norm(rb) / nb + inf_norm(ru) / nU
        dinf = inf_norm(rc) / nc
        gap = abs(pobj - dobj) / (1 + abs(pobj))
        history.append((it, pobj, dobj, pinf, dinf, mu))
        if verbose:
            print(f"{it:4d}  {pobj + F.c0:+.10e} {dobj + F.c0:+.10e}  p {pinf:.1e} d {dinf:.1e} g {gap:.1e} mu {mu:.1e}")
        if pinf < tol and dinf < tol and gap < tol:
            status = "optimal"
            break
        # stall: the three measures stopped improving. If all are within the loose tolerance the
        # point is optimal to that accuracy; continuing only drives mu to 0 and z/x to overflow.
        err = max(pinf, dinf, gap)
        if err < 0.5 * best_err:
            best_err, stall = err, 0
        else:
            stall += 1
        if stall >= 4 or mu < 1e-14 * (1 + abs(pobj)):
            if pinf < loose and dinf < loose and gap < loose:
                status = "optimal"
                break
            if stall >= 30:
                status = "numerical_error"
                break
        if time.perf_counter() - t0 > time_limit:
            status = "time_limit"
            break
        if not np.isfinite(pobj) or not np.isfinite(dobj):
            status = "numerical_error"
            break
        # infeasibility heuristics: the iterates run off to infinity while the other side converges
        if it > 20 and mu < 1e-12 and pinf > 1e-6 and np.linalg.norm(y, np.inf) > 1e12:
            status = "infeasible"
            break
        if it > 20 and dinf > 1e-6 and np.linalg.norm(x, np.inf) > 1e14:
            status = "unbounded"
            break

        xs = np.where(L, x, 1.0)
        ws = np.where(X, w, 1.0)
        dinv = np.where(L, z / xs, 0.0) + np.where(X, s / ws, 0.0)
        dinv = dinv + np.where(free, reg_free, reg_p)
        theta = 1.0 / dinv
        vals = factor(theta, np.full(mB, reg_d))

        def direction(rxz, rws):
            rhat = rc - np.where(L, rxz / xs, 0.0) + np.where(X, (rws - s * ru) / ws, 0.0)
            rhs = rb + B.matvec(theta * rhat)
            dy = normal_solve(rhs, vals, None, theta)
            dx = theta * (BT.matvec(dy) - rhat)
            dz = np.where(L, (rxz - z * dx) / xs, 0.0)
            dw = np.where(X, ru - dx, 0.0)
            ds = np.where(X, (rws - s * dw) / ws, 0.0)
            return dx, dy, dz, dw, ds

        # predictor (affine scaling)
        dx, dy, dz, dw, ds = direction(np.where(L, -x * z, 0.0), np.where(X, -w * s, 0.0))
        ap = min(1.0, _max_step(x[L], dx[L]), _max_step(w[X], dw[X]))
        ad = min(1.0, _max_step(z[L], dz[L]), _max_step(s[X], ds[X]))
        mu_aff = ((x[L] + ap * dx[L]) @ (z[L] + ad * dz[L]) + (w[X] + ap * dw[X]) @ (s[X] + ad * ds[X])) / max(ncomp, 1)
        sigma = (mu_aff / mu) ** 3 if mu > 0 else 0.0
        sigma = min(max(sigma, 0.0), 1.0)
        # corrector: centring plus the second-order term of the predictor
        rxz = np.where(L, sigma * mu - x * z - dx * dz, 0.0)
        rws = np.where(X, sigma * mu - w * s - dw * ds, 0.0)
        dx, dy, dz, dw, ds = direction(rxz, rws)
        ap = min(_max_step(x[L], dx[L]), _max_step(w[X], dw[X]))
        ad = min(_max_step(z[L], dz[L]), _max_step(s[X], ds[X]))
        eta = 0.9995                       # stay this fraction of the way from the boundary
        ap, ad = min(1.0, eta * ap), min(1.0, eta * ad)
        x = x + ap * dx
        w = np.where(X, w + ap * dw, 0.0)
        y = y + ad * dy
        z = np.where(L, z + ad * dz, 0.0)
        s = np.where(X, s + ad * ds, 0.0)

    # ---- back to the model's variables ----
    xs_model = F.recover(x, work.n)
    ys = np.zeros(work.m)
    ys[F.keep_rows] = y
    x_orig = xs_model * C
    y_orig = ys * R
    base = model
    z_orig = base.c - base.A.rmatvec(y_orig)
    obj = base.user_objective(x_orig)
    return Result(status=status, x=x_orig, y=y_orig * base.sense, z=z_orig * base.sense,
                  objective=obj, bound=obj if status == "optimal" else np.nan, iterations=it,
                  time=time.perf_counter() - t0, method="ipm",
                  info={"history": history, "nnz_L": chol.nnz, "nnz_M": NM.nnz,
                        "fixed_pivots": chol.fixed_pivots, "rows": mB, "cols": nB})


def _spmv_sym(Mp, Mi, Mx, x):
    """y = M x for M held in full CSC (symmetric)."""
    cols = np.repeat(np.arange(len(Mp) - 1), np.diff(Mp))
    y = np.zeros(len(x))
    np.add.at(y, Mi, Mx * x[cols])
    return y


def _relax(model: Model) -> Model:
    return Model(name=model.name, c=model.c, A=model.A, rl=model.rl, ru=model.ru, lb=model.lb,
                 ub=model.ub, integer=np.zeros(model.n, dtype=bool), Q=model.Q, c0=model.c0,
                 sense=model.sense, col_names=model.col_names, row_names=model.row_names)
