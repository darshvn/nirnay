"""Primal-dual interior-point method for convex QP (and LP), on the augmented system.

    min  c'x + 1/2 x'Qx    s.t.  B x = b,  bounds as in lp/ipm.py (same bounded standard form)

With Q present the normal equations B (Q + D)^-1 B' lose their sparsity, so every Newton step
solves the regularised augmented system instead,

    [ -(Q + D + rho I)   B'      ] [dx]   [ rhat ]         D = Z/X + S/W  (barrier terms)
    [        B          delta I  ] [dy] = [ rb   ]

which is symmetric quasi-definite and is factorised by our sparse LDL' (linalg/ldl.py). The
regularisations rho and delta are tiny and their effect is removed by iterative refinement
against the unregularised matrix (Friedlander & Orban, Math. Prog. Comp. 4, 2012; Altman &
Gondzio, Optim. Methods Softw. 11, 1999).

The outer iteration is Mehrotra's predictor-corrector, as for LP (Mehrotra 1992; Wright 1997,
ch. 10; for QP: Gertz & Wright, "Object-oriented software for quadratic programming", ACM TOMS
29, 2003, sec. 3).
"""
from __future__ import annotations

import time

import numpy as np

from ..linalg.ldl import LDL
from ..lp.ipm import _Form, _max_step, _relax
from ..model import Model, Result
from ..presolve.scaling import compute_scaling, scale_model


class _KKT:
    """The augmented matrix on a fixed pattern: raw triplets sorted once, summed per iteration."""

    def __init__(self, nB, mB, B, Qr, Qc, Qv):
        N = nB + mB
        cols_B = np.repeat(np.arange(B.n), np.diff(B.colptr))
        diag_x = np.arange(nB)
        diag_y = nB + np.arange(mB)
        r = np.concatenate([Qr, diag_x, nB + B.rowidx, cols_B, diag_y])
        c = np.concatenate([Qc, diag_x, cols_B, nB + B.rowidx, diag_y])
        self.nQ, self.nB, self.mB, self.nnzB = len(Qv), nB, mB, B.nnz
        self.Qv, self.Bv = Qv, B.vals
        order = np.lexsort((r, c))
        r, c = r[order], c[order]
        key = c * N + r
        keep = np.ones(len(key), dtype=bool)
        keep[1:] = key[1:] != key[:-1]
        self.order = order
        self.starts = np.flatnonzero(keep)
        self.rowidx = r[self.starts]
        cc = c[self.starts]
        self.colptr = np.zeros(N + 1, dtype=np.int64)
        np.add.at(self.colptr, cc + 1, 1)
        np.cumsum(self.colptr, out=self.colptr)
        self.cols = cc
        self.N = N

    def values(self, dx_diag, dy_diag):
        raw = np.concatenate([-self.Qv, -dx_diag, self.Bv, self.Bv, dy_diag])
        return np.add.reduceat(raw[self.order], self.starts)

    def matvec(self, vals, v):
        out = np.zeros(self.N)
        np.add.at(out, self.rowidx, vals * v[self.cols])
        return out


def solve(model: Model, tol: float = 1e-8, max_iter: int = 200, scale: bool = True,
          time_limit: float = np.inf, verbose: bool = False, start_floor: float = 1.0,
          nb_scale: float = 1e12, common_step: bool = True) -> Result:
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
    mB, nB, nx = F.mB, F.nB, F.nx
    c, b, U = F.c.copy(), F.b, F.U
    c0 = F.c0
    L, X, free = F.lower, F.boxed, F.free

    # ---- Q in the standard-form variables: x_model = x0 + E (sign * x') ----
    Qv = np.zeros(0)
    Qr = np.zeros(0, dtype=np.int64)
    Qc = np.zeros(0, dtype=np.int64)
    if work.Q is not None and work.Q.nnz:
        Qm = work.Q
        x0 = F.x_fixed.copy()
        x0[F.keep_cols] = F.shift[:nx]
        Qx0 = Qm.matvec(x0)
        c[:nx] += F.sign[:nx] * Qx0[F.keep_cols]
        c0 += 0.5 * float(x0 @ Qx0)
        colnew = np.full(work.n, -1, dtype=np.int64)
        colnew[F.keep_cols] = np.arange(nx)
        qc = np.repeat(np.arange(Qm.n), np.diff(Qm.colptr))
        qr = Qm.rowidx
        mask = (colnew[qr] >= 0) & (colnew[qc] >= 0)
        Qr, Qc = colnew[qr[mask]], colnew[qc[mask]]
        Qv = Qm.vals[mask] * F.sign[Qr] * F.sign[Qc]
    nQ = len(Qv)

    def Qmul(v):
        out = np.zeros(nB)
        if nQ:
            np.add.at(out, Qr, Qv * v[Qc])
        return out

    K = _KKT(nB, mB, B, Qr, Qc, Qv)
    ldl = LDL(K.N, K.colptr, K.rowidx, delta=1e-11)
    sign = np.concatenate([-np.ones(nB), np.ones(mB)])
    reg_p, reg_free, reg_d = 1e-10, 1e-8, 1e-10

    solve_err = [0.0]

    def kkt_solve(vals_reg, vals_true, rhs):
        sol = ldl.solve(rhs)
        scale = 1 + np.max(np.abs(rhs))
        for _ in range(3):
            r = rhs - K.matvec(vals_true, sol)
            if np.max(np.abs(r)) <= 1e-13 * scale:
                break
            sol += ldl.solve(r)
        r = rhs - K.matvec(vals_true, sol)
        solve_err[0] = max(solve_err[0], float(np.max(np.abs(r), initial=0.0)) / scale)
        return sol

    # ---- starting point ----
    vals0 = K.values(np.ones(nB), np.full(mB, 1e-8))
    ldl.factor(vals0, sign)
    x = kkt_solve(vals0, vals0, np.concatenate([np.zeros(nB), b]))[:nB]
    y = kkt_solve(vals0, vals0, np.concatenate([c, np.zeros(mB)]))[nB:]
    zc = c + Qmul(x) - BT.matvec(y)
    z = np.where(L, np.maximum(zc, 0.0), 0.0)
    s = np.where(X, np.maximum(-zc, 0.0), 0.0)
    w = np.where(X, U - x, 0.0)
    dx_shift = max(-1.5 * min(x[L].min(initial=np.inf), w[X].min(initial=np.inf)), 0.0)
    x = np.where(L, x + dx_shift, x)
    w = np.where(X, w + dx_shift, 0.0)
    dz_shift = max(-1.5 * min(z[L].min(initial=np.inf), s[X].min(initial=np.inf)), 0.0)
    z = np.where(L, z + dz_shift, 0.0)
    s = np.where(X, s + dz_shift, 0.0)
    xz = x[L] @ z[L] + w[X] @ s[X]
    sx, sz = x[L].sum() + w[X].sum(), z[L].sum() + s[X].sum()
    if sx > 0 and sz > 0:
        x = np.where(L, x + 0.5 * xz / sz, x)
        w = np.where(X, w + 0.5 * xz / sz, 0.0)
        z = np.where(L, z + 0.5 * xz / sx, 0.0)
        s = np.where(X, s + 0.5 * xz / sx, 0.0)
    # keep the start well inside the cone: when Mehrotra's heuristic lands on a point with tiny
    # complementarity but large residuals (YAO: mu 2e-6 against dual infeasibility 0.5) the
    # iterates hug the boundary and crawl. A floor proportional to the residual scale fixes it.
    floor = start_floor * max(1.0, float(np.max(np.abs(b), initial=0.0)) / nb_scale) if start_floor else 0.0
    x = np.where(L, np.maximum(x, floor), x)
    z = np.where(L, np.maximum(z, floor), z)
    w = np.where(X, np.maximum(w, floor), 0.0)
    s = np.where(X, np.maximum(s, floor), 0.0)
    x = np.where(L & (x <= 0), 1.0, x)
    z = np.where(L & (z <= 0), 1.0, z)
    w = np.where(X & (w <= 0), 1.0, w)
    s = np.where(X & (s <= 0), 1.0, s)

    ncomp = int(L.sum() + X.sum())
    inf_norm = lambda v: float(np.max(np.abs(v), initial=0.0))
    nb, nc, nU = 1 + inf_norm(b), 1 + inf_norm(c), 1 + inf_norm(U[X])
    status, it = "iteration_limit", 0
    history = []
    best_err, stall = np.inf, 0
    loose = max(tol, 1e-6)
    for it in range(1, max_iter + 1):
        Qx = Qmul(x)
        rb = b - B.matvec(x)
        ru = np.where(X, U - x - w, 0.0)
        rc = c + Qx - BT.matvec(y) - z + s
        mu = (x[L] @ z[L] + w[X] @ s[X]) / max(ncomp, 1)
        xQx = float(x @ Qx)
        pobj = c @ x + 0.5 * xQx
        dobj = b @ y - U[X] @ s[X] - 0.5 * xQx
        pinf = inf_norm(rb) / nb + inf_norm(ru) / nU
        dinf = inf_norm(rc) / nc
        # relative to the objective the user sees (constant included): with a large constant and
        # an optimum near zero (HS268, GOULDQP3) a gap relative to pobj alone is far too loose
        gap = abs(pobj - dobj) / (1 + min(abs(pobj), abs(pobj + c0)))
        history.append((it, pobj, dobj, pinf, dinf, mu))
        if verbose:
            print(f"{it:4d}  {pobj + c0:+.10e} {dobj + c0:+.10e}  p {pinf:.1e} d {dinf:.1e} g {gap:.1e} mu {mu:.1e}")
        if pinf < tol and dinf < tol and gap < tol:
            status = "optimal"
            break
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
        if it > 20 and mu < 1e-12 and pinf > 1e-6 and np.linalg.norm(y, np.inf) > 1e12:
            status = "infeasible"
            break
        if it > 20 and dinf > 1e-6 and np.linalg.norm(x, np.inf) > 1e14:
            status = "unbounded"
            break

        xs = np.where(L, x, 1.0)
        ws = np.where(X, w, 1.0)
        dbar = np.where(L, z / xs, 0.0) + np.where(X, s / ws, 0.0)
        vals_true = K.values(dbar + np.where(free, reg_free, 0.0), np.zeros(mB))

        def direction(rxz, rws):
            rhat = rc - np.where(L, rxz / xs, 0.0) + np.where(X, (rws - s * ru) / ws, 0.0)
            sol = kkt_solve(vals_reg, vals_true, np.concatenate([rhat, rb]))
            dx, dy = sol[:nB], sol[nB:]
            dz = np.where(L, (rxz - z * dx) / xs, 0.0)
            dw = np.where(X, ru - dx, 0.0)
            ds = np.where(X, (rws - s * dw) / ws, 0.0)
            return dx, dy, dz, dw, ds

        # factor; if the direction comes back non-finite, regularise harder and refactor
        # a factorisation that lost too much to pivot flooring gives a direction that is finite
        # but wrong (QRECIPE: |dx| ~ 1e46); the refined residual shows it, so check it too
        boost = 1.0
        for _attempt in range(5):
            vals_reg = K.values(dbar + np.where(free, reg_free, reg_p) * boost, np.full(mB, reg_d * boost))
            ldl.factor(vals_reg, sign)
            solve_err[0] = 0.0
            dx, dy, dz, dw, ds = direction(np.where(L, -x * z, 0.0), np.where(X, -w * s, 0.0))
            if np.all(np.isfinite(dx)) and np.all(np.isfinite(dy)) and solve_err[0] < 1e-6:
                break
            boost *= 1e2
        else:
            status = "numerical_error"
            break
        ap = min(1.0, _max_step(x[L], dx[L]), _max_step(w[X], dw[X]))
        ad = min(1.0, _max_step(z[L], dz[L]), _max_step(s[X], ds[X]))
        mu_aff = ((x[L] + ap * dx[L]) @ (z[L] + ad * dz[L]) + (w[X] + ap * dw[X]) @ (s[X] + ad * ds[X])) / max(ncomp, 1)
        sigma = min(max((mu_aff / mu) ** 3 if mu > 0 else 0.0, 0.0), 1.0)
        rxz = np.where(L, sigma * mu - x * z - dx * dz, 0.0)
        rws = np.where(X, sigma * mu - w * s - dw * ds, 0.0)
        dx, dy, dz, dw, ds = direction(rxz, rws)
        ap = min(_max_step(x[L], dx[L]), _max_step(w[X], dw[X]))
        ad = min(_max_step(z[L], dz[L]), _max_step(s[X], ds[X]))
        if nQ and common_step:
            # with Q the primal and dual steps are coupled through Q dx; take one common step
            ap = ad = min(ap, ad)
        eta = 0.9995
        ap, ad = min(1.0, eta * ap), min(1.0, eta * ad)
        x = x + ap * dx
        w = np.where(X, w + ap * dw, 0.0)
        y = y + ad * dy
        z = np.where(L, z + ad * dz, 0.0)
        s = np.where(X, s + ad * ds, 0.0)

    xs_model = F.recover(x, work.n)
    ys = np.zeros(work.m)
    ys[F.keep_rows] = y
    x_orig = xs_model * C
    y_orig = ys * R
    z_orig = model.c - model.A.rmatvec(y_orig)
    if model.is_qp:
        z_orig = z_orig + model.Q.matvec(x_orig)
    obj = model.user_objective(x_orig)
    return Result(status=status, x=x_orig, y=y_orig * model.sense, z=z_orig * model.sense,
                  objective=obj, bound=obj if status == "optimal" else np.nan, iterations=it,
                  time=time.perf_counter() - t0, method="qp-ipm",
                  info={"history": history, "nnz_L": ldl.nnz, "rows": mB, "cols": nB})
