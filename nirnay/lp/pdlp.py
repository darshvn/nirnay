"""PDLP: restarted, preconditioned primal-dual hybrid gradient for linear programming.

References
  D. Applegate, M. Diaz, O. Hinder, H. Lu, M. Lubin, B. O'Donoghue, W. Schudy, "Practical
      large-scale linear programming using primal-dual hybrid gradient", NeurIPS 2021
      (arXiv 2106.04756). The method: adaptive steps, primal weight, restarts, averaging,
      diagonal preconditioning.
  D. Applegate, O. Hinder, H. Lu, M. Lubin, "Faster first-order primal-dual methods for linear
      programming using restarts and sharpness", Math. Programming 201, 2023 (why restarts work).
  D. Applegate, M. Diaz, H. Lu, M. Lubin, "Infeasibility detection with primal-dual hybrid
      gradient for large-scale linear programming", SIAM J. Optim. 34(1), 2024 (the difference of
      iterates converges to the infimal displacement vector, whose parts are Farkas certificates).
  H. Lu, J. Yang, "cuPDLP.jl: a GPU implementation of restarted primal-dual hybrid gradient for
      linear programming in Julia", arXiv 2311.12180 (KKT-error restart criteria, device-resident
      iteration, termination checks every 64 iterations).
  H. Lu, J. Yang et al., "cuPDLP-C: a strengthened implementation of cuPDLP for linear
      programming by C language", arXiv 2312.14832 (bound and objective rescaling).
  T. Pock, A. Chambolle, "Diagonal preconditioning for first order primal-dual algorithms in
      convex optimization", ICCV 2011.  D. Ruiz, "A scaling algorithm to equilibrate both rows and
      columns norms in matrices", RAL-TR-2001-034.

Problem. The model's own form, with no conversion to equalities and no slack columns:

    min c'x   s.t.   rl <= A x <= ru,   lb <= x <= ub       (entries may be infinite)

as the saddle point  min_{x in X} max_{y in Y}  c'x - y'Ax + p(y),  p(y) = rl'y+ - ru'y-,
where X is the box and Y the sign cone of the rows (y_i >= 0 if ru_i = +inf, y_i <= 0 if
rl_i = -inf, y_i = 0 for a free row). One PDHG step with tau = eta/omega, sigma = eta*omega:

    x' = proj_X( x - tau (c - A'y) )
    y' = argmax_y  -y'A(2x'-x) + p(y) - |y - y_k|^2 / (2 sigma)

The dual step has a closed form. With t = y - sigma A(2x'-x), Moreau's identity gives
y' = t + sigma proj_[rl,ru](-t/sigma) = t - clip(t, -sigma ru, -sigma rl): an equality row gets
y - sigma(A x_bar - b), a >= row gets max(0, t + sigma rl), and a free row gets 0.

Termination (relative, measured on the unscaled problem, as PDLP):
    |r_primal|_2 <= tol (1 + |b|_2),  |r_dual|_2 <= tol (1 + |c|_2),
    |c'x - dual objective| <= tol (1 + |c'x| + |dual objective|),
with b the vector of finite row bounds and r_dual the part of c - A'y that the variable bounds
cannot absorb (a positive reduced cost needs a finite lower bound, a negative one an upper bound).

Backends. The algorithm below is written once over an array module xp (NumPy or CuPy). The
inner loop - `check_every` PDHG attempts between two checks - belongs to the backend:
  * CPU: one Numba call runs the whole block, with the SpMV fused into the proximal steps and
    the step-size reductions (three passes over the data per attempt);
  * GPU: four CUDA kernels per attempt (commit+primal step, rows = A x' fused with the dual step,
    columns = A'y' fused with the reductions, a one-thread step-size kernel). The adaptive step
    decision stays on the device (a rejected step is simply not committed), so the block has no
    host round trip and is captured once as a CUDA graph and replayed.
"""
from __future__ import annotations

import contextlib
import time
import types

import numpy as np
from numba import njit, prange

from ..linalg import sparse_kernels as sk
from ..model import Model, Result

# Restart parameters of cuPDLP.jl / PDLP (KKT-error based adaptive restarts).
BETA_SUFFICIENT = 0.2
BETA_NECESSARY = 0.8
BETA_ARTIFICIAL = 0.36
PRIMAL_WEIGHT_SMOOTHING = 0.5
# PDLP tests for certificates at every check. Here: every 4th check (256 attempts). The test costs
# two extra products and ~20 vector passes, which on small models on the GPU cost more than the
# 64-attempt block itself; a certificate only gets better with time, so waiting costs little.
RAY_CHECK_EVERY = 4


# =============================================================================================
# Preconditioning
# =============================================================================================

@njit(cache=True)
def _transpose(m, colptr, rowidx, vals):
    """CSR arrays of a CSC matrix (equivalently: CSC of its transpose), by counting sort."""
    n = colptr.shape[0] - 1
    nnz = colptr[n]
    rowptr = np.zeros(m + 1, np.int64)
    for p in range(nnz):
        rowptr[rowidx[p] + 1] += 1
    for i in range(m):
        rowptr[i + 1] += rowptr[i]
    pos = rowptr[:m].copy()
    colidx = np.empty(nnz, np.int64)
    tv = np.empty(nnz, np.float64)
    for j in range(n):
        for p in range(colptr[j], colptr[j + 1]):
            i = rowidx[p]
            q = pos[i]
            colidx[q] = j
            tv[q] = vals[p]
            pos[i] = q + 1
    return rowptr, colidx, tv


def _combined_bounds(rl, ru):
    """Per row, the largest finite bound in absolute value (PDLP's 'combined bounds' vector)."""
    a = np.where(np.isfinite(rl), np.abs(rl), 0.0)
    b = np.where(np.isfinite(ru), np.abs(ru), 0.0)
    return np.maximum(a, b)


class _Scaled:
    """The preconditioned problem  A~ = diag(R) A diag(C) / (nothing),  c~ = C c / beta_c,
    rows R [rl, ru] / beta_b,  bounds [lb, ub] / (C beta_b),  and the maps back:
    x = C beta_b x~,  y = R beta_c y~."""

    def __init__(self, model: Model, ruiz_iters: int, pock_chambolle: bool, rescale: bool):
        A = model.A
        m, n = A.m, A.n
        R, C = np.ones(m), np.ones(n)
        rbuf, cbuf = np.empty(m), np.empty(n)
        # Ruiz: divide each row and column by sqrt of its largest entry, both from the same matrix.
        for _ in range(ruiz_iters):
            sk.line_absmax(A.colptr, A.rowidx, A.vals, R, C, rbuf, cbuf)
            R /= np.sqrt(np.where(rbuf > 0, rbuf, 1.0))
            C /= np.sqrt(np.where(cbuf > 0, cbuf, 1.0))
        # Pock-Chambolle with alpha = 1: rows and columns by sqrt of their l1 norms.
        if pock_chambolle:
            sk.line_abssum(A.colptr, A.rowidx, A.vals, R, C, rbuf, cbuf)
            R /= np.sqrt(np.where(rbuf > 0, rbuf, 1.0))
            C /= np.sqrt(np.where(cbuf > 0, cbuf, 1.0))
        vals = np.empty_like(A.vals)
        sk.scale_values(A.colptr, A.rowidx, A.vals, R, C, vals)
        c = model.c * C
        with np.errstate(invalid="ignore"):
            rl, ru = model.rl * R, model.ru * R
            lb, ub = model.lb / C, model.ub / C
        # cuPDLP-C's bound and objective rescaling: both vectors to norm about one, so that the
        # primal weight starts near 1 and the relative tolerances mean the same on every model.
        beta_b = beta_c = 1.0
        if rescale:
            beta_b = 1.0 + float(np.linalg.norm(_combined_bounds(rl, ru)))
            beta_c = 1.0 + float(np.linalg.norm(c))
        self.c = c / beta_c
        self.rl, self.ru = rl / beta_b, ru / beta_b
        self.lb, self.ub = lb / beta_b, ub / beta_b
        self.m, self.n = m, n
        self.colptr, self.rowidx, self.vals = A.colptr, A.rowidx, vals      # CSC(A~) = CSR(A~')
        self.rowptr, self.colidx, self.tvals = _transpose(m, A.colptr, A.rowidx, vals)  # CSR(A~)
        self.R, self.C, self.beta_b, self.beta_c = R, C, beta_b, beta_c
        self.amax = float(np.abs(vals).max()) if len(vals) else 0.0
        cn = np.linalg.norm(self.c)
        bn = np.linalg.norm(_combined_bounds(self.rl, self.ru))
        self.omega0 = float(cn / bn) if cn > 1e-10 and bn > 1e-10 else 1.0


# =============================================================================================
# CPU backend: one Numba call per block of PDHG attempts
# =============================================================================================

def _pdhg_block_src(nsteps, k0, rp, rj, rv, cp, ci, cv, c, lb, ub, rl, ru,
                    x, y, ax, aty, xn, yn, axn, atyn, sx, sy, eta, omega, wsum):
    """`nsteps` attempts of adaptive-step PDHG (PDLP Algorithm 3, AdaptiveStepPDHG).

    (x, y, ax = A x, aty = A'y) is the committed iterate; (xn, yn, axn, atyn) the candidate.
    An accepted step swaps the two sets; its weight (the step size) enters the running sums
    sx, sy of the average lazily, fused into the next attempt's first two passes.
    """
    n = c.shape[0]
    m = rl.shape[0]
    nrej = 0
    pend = 0.0
    for s in range(nsteps):
        k = k0 + s + 1
        tau = eta / omega
        sigma = eta * omega
        # pass 1: primal step (and the pending average update of x)
        for j in prange(n):
            xj = x[j]
            if pend != 0.0:
                sx[j] += pend * xj
            v = xj - tau * (c[j] - aty[j])
            xn[j] = min(max(v, lb[j]), ub[j])
        # pass 2: rows of A x', each followed at once by its dual proximal step
        dy2 = 0.0
        for i in prange(m):
            yi = y[i]
            if pend != 0.0:
                sy[i] += pend * yi
            acc = 0.0
            for p in range(rp[i], rp[i + 1]):
                acc += rv[p] * xn[rj[p]]
            axn[i] = acc
            t = yi - sigma * (2.0 * acc - ax[i])
            v = t - min(max(t, -sigma * ru[i]), -sigma * rl[i])
            yn[i] = v
            d = v - yi
            dy2 += d * d
        pend = 0.0
        # pass 3: columns of A'y', fused with the two reductions the step-size rule needs
        dx2 = 0.0
        inter = 0.0
        for j in prange(n):
            acc = 0.0
            for p in range(cp[j], cp[j + 1]):
                acc += cv[p] * yn[ci[p]]
            atyn[j] = acc
            d = xn[j] - x[j]
            dx2 += d * d
            inter += d * (acc - aty[j])
        # PDLP step-size rule: eta_bar = |dz|_omega^2 / (2 |dx' A' dy|)
        num = omega * dx2 + dy2 / omega
        if inter != 0.0:
            ebar = num / (2.0 * abs(inter))
        else:
            ebar = np.inf
        enew = min((1.0 - (k + 1.0) ** -0.3) * ebar, (1.0 + (k + 1.0) ** -0.6) * eta)
        if eta <= ebar:
            x, xn = xn, x
            y, yn = yn, y
            ax, axn = axn, ax
            aty, atyn = atyn, aty
            pend = eta
            wsum += eta
        else:
            nrej += 1
        eta = enew
    if pend != 0.0:
        for j in prange(n):
            sx[j] += pend * x[j]
        for i in prange(m):
            sy[i] += pend * y[i]
    return x, y, ax, aty, xn, yn, axn, atyn, eta, wsum, nrej


def _twin(f, suffix):
    """A copy of f under another qualified name. Numba's disk cache is keyed by qualname, so the
    serial and the parallel compilations of one source must not share a name."""
    g = types.FunctionType(f.__code__, f.__globals__, f.__name__ + suffix, f.__defaults__, f.__closure__)
    g.__qualname__ = f.__qualname__ + suffix
    return g


# Reassociation lets LLVM keep several partial sums in the SpMV dot products instead of one
# serial chain of dependent adds. No 'nnan'/'ninf': bounds are infinite and clips must stay exact.
_FM = {"reassoc", "contract"}
_block_par = njit(cache=True, parallel=True, fastmath=_FM)(_twin(_pdhg_block_src, "_par"))
_block_ser = njit(cache=True, fastmath=_FM)(_twin(_pdhg_block_src, "_ser"))


@njit(cache=True)
def _kkt_cpu(c, rl, ru, lb, ub, x, y, ax, aty, ux, uy, uax, uaty):
    """KKT sums of (x*ux, y*uy, ax*uax, aty*uaty) in one pass, no temporaries: see _finish_kkt.

    y is assumed to lie in the sign cone Y (true of every iterate and of their averages).
    """
    rp2 = 0.0
    dobj = 0.0
    for i in range(rl.shape[0]):
        a = ax[i] * uax[i]
        r = 0.0
        if a < rl[i]:
            r = rl[i] - a
        elif a > ru[i]:
            r = a - ru[i]
        rp2 += r * r
        yi = y[i] * uy[i]
        if yi > 0.0 and rl[i] > -np.inf:
            dobj += rl[i] * yi
        elif yi < 0.0 and ru[i] < np.inf:
            dobj += ru[i] * yi
    pobj = 0.0
    rd2 = 0.0
    for j in range(c.shape[0]):
        pobj += c[j] * (x[j] * ux[j])
        lam = c[j] - aty[j] * uaty[j]
        if lam > 0.0:
            if lb[j] > -np.inf:
                dobj += lb[j] * lam
            else:
                rd2 += lam * lam
        elif lam < 0.0:
            if ub[j] < np.inf:
                dobj += ub[j] * lam
            else:
                rd2 += lam * lam
    out = np.empty(4)
    out[0] = pobj
    out[1] = dobj
    out[2] = rp2
    out[3] = rd2
    return out


@njit(cache=True)
def _sqdist(a, b):
    s = 0.0
    for i in range(a.shape[0]):
        d = a[i] - b[i]
        s += d * d
    return s


class _CPU:
    xp = np

    def __init__(self, S: _Scaled, parallel: bool):
        self.S = S
        self.par = parallel
        self.name = "cpu-numba" + ("-parallel" if parallel else "")
        self.c, self.lb, self.ub, self.rl, self.ru = S.c, S.lb, S.ub, S.rl, S.ru
        m, n = S.m, S.n
        self.bufs = [np.zeros(k) for k in (n, m, m, n, n, m, m, n)]
        self.sx, self.sy = np.zeros(n), np.zeros(m)
        self.eta = 1.0
        self.omega = 1.0
        self.wsum = 0.0
        self.nrej = 0
        self.ones = (np.ones(n), np.ones(m), np.ones(m), np.ones(n))

    def context(self):
        return contextlib.nullcontext()

    @property
    def x(self):
        return self.bufs[0]

    @property
    def y(self):
        return self.bufs[1]

    @property
    def Ax(self):
        return self.bufs[2]

    @property
    def ATy(self):
        return self.bufs[3]

    def asarray(self, a):
        return np.asarray(a, dtype=np.float64)

    def to_host(self, a):
        return np.asarray(a)

    def matvec(self, v):
        S = self.S
        return sk.matvec(S.rowptr, S.colidx, S.tvals, v, parallel=self.par)

    def rmatvec(self, v):
        S = self.S
        return sk.matvec(S.colptr, S.rowidx, S.vals, v, parallel=self.par)

    def set_iterate(self, x, y, Ax, ATy):
        for buf, v in zip(self.bufs[:4], (x, y, Ax, ATy)):
            buf[...] = v

    def reset_average(self):
        self.sx.fill(0.0)
        self.sy.fill(0.0)
        self.wsum = 0.0

    def set_step(self, eta):
        self.eta = float(eta)

    def set_omega(self, omega):
        self.omega = float(omega)

    def run(self, steps: int, k0: int):
        S = self.S
        f = _block_par if self.par else _block_ser
        out = f(steps, k0, S.rowptr, S.colidx, S.tvals, S.colptr, S.rowidx, S.vals,
                self.c, self.lb, self.ub, self.rl, self.ru, *self.bufs, self.sx, self.sy,
                self.eta, self.omega, self.wsum)
        self.bufs = list(out[:8])
        self.eta, self.wsum = float(out[8]), float(out[9])
        self.nrej += int(out[10])

    def scalars(self):
        return self.eta, self.wsum, self.nrej

    def kkt(self, P, x, y, Ax, ATy, U=None):
        if U is None:
            U = self.ones
        return _kkt_cpu(P.c, P.rl, P.ru, P.lb, P.ub, x, y, Ax, ATy, *U)

    def dist2(self, a, ar, b, br):
        return np.array([_sqdist(a, ar), _sqdist(b, br)])

    def averages(self):
        return self.sx / self.wsum, self.sy / self.wsum


# =============================================================================================
# GPU backend: CuPy arrays, fused CUDA kernels, device-side step-size control, CUDA graph
# =============================================================================================

# Device scalar slots. Everything the step-size rule touches lives here, so no attempt needs
# the host: the host reads this array once per check.
_ETA, _OMEGA, _K, _ACC, _PEND, _DX2, _INTER, _DY2, _WSUM, _NREJ = range(10)

_CUDA_SRC = r"""
#define FULL 0xffffffffu

__device__ __forceinline__ double warp_sum(double v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(FULL, v, o);
    return v;
}

// Sum a and b over the block, then one atomicAdd per block for each. blockDim.x % 32 == 0.
__device__ __forceinline__ void block_add2(double a, double b, double* da, double* db) {
    __shared__ double sa[32], sb[32];
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    a = warp_sum(a); b = warp_sum(b);
    if (lane == 0) { sa[w] = a; sb[w] = b; }
    __syncthreads();
    if (w == 0) {
        int nw = blockDim.x >> 5;
        a = lane < nw ? sa[lane] : 0.0;
        b = lane < nw ? sb[lane] : 0.0;
        a = warp_sum(a); b = warp_sum(b);
        if (lane == 0) {
            if (a != 0.0) atomicAdd(da, a);
            if (db && b != 0.0) atomicAdd(db, b);
        }
    }
}

// Commit the previous attempt if it was accepted (and add it, weighted, to the average), then
// take the primal step x' = proj(x - tau (c - A'y)). Threads [0, n) are columns, [n, n+m) rows.
extern "C" __global__ void pdhg_commit_primal(
        int n, int m, int step, const double* __restrict__ c, const double* __restrict__ lb,
        const double* __restrict__ ub, double* x, double* xn, double* aty,
        const double* __restrict__ atyn, double* sx, double* y, const double* __restrict__ yn,
        double* ax, const double* __restrict__ axn, double* sy, const double* __restrict__ sc) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    bool acc = sc[3] != 0.0;
    double pend = sc[4];
    if (t < n) {
        double xj, aj;
        if (acc) { xj = xn[t]; aj = atyn[t]; x[t] = xj; aty[t] = aj; sx[t] += pend * xj; }
        else     { xj = x[t];  aj = aty[t]; }
        if (step) {
            double tau = sc[0] / sc[1];
            double v = xj - tau * (c[t] - aj);
            xn[t] = fmin(fmax(v, lb[t]), ub[t]);
        }
    } else if (t < (long long)n + m) {
        int r = (int)(t - n);
        if (acc) { double yi = yn[r]; y[r] = yi; ax[r] = axn[r]; sy[r] += pend * yi; }
    }
}

// Rows: (A x')_i by a group of G lanes, then the dual proximal step of row i, and |y'-y|^2.
template<int G>
__global__ void pdhg_rows(int m, const int* __restrict__ p, const int* __restrict__ j,
        const double* __restrict__ v, const double* __restrict__ xn, const double* __restrict__ y,
        const double* __restrict__ ax, const double* __restrict__ rl, const double* __restrict__ ru,
        double* __restrict__ axn, double* __restrict__ yn, double* sc) {
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int row = (int)(gid / G), lane = (int)(gid % G);
    double s = 0.0;
    if (row < m) {
        int e = p[row + 1];
        for (int q = p[row] + lane; q < e; q += G) s += v[q] * xn[j[q]];
    }
    for (int o = G / 2; o > 0; o >>= 1) s += __shfl_down_sync(FULL, s, o, G);
    double d2 = 0.0;
    if (row < m && lane == 0) {
        double sigma = sc[0] * sc[1];
        double yi = y[row];
        double t = yi - sigma * (2.0 * s - ax[row]);
        double yv = t - fmin(fmax(t, -sigma * ru[row]), -sigma * rl[row]);
        axn[row] = s; yn[row] = yv;
        double d = yv - yi; d2 = d * d;
    }
    block_add2(d2, 0.0, sc + 7, 0);
}

// Columns: (A'y')_j by a group of G lanes, then |x'-x|^2 and (x'-x)'(A'y' - A'y).
template<int G>
__global__ void pdhg_cols(int n, const int* __restrict__ p, const int* __restrict__ i,
        const double* __restrict__ v, const double* __restrict__ yn, const double* __restrict__ x,
        const double* __restrict__ xn, const double* __restrict__ aty, double* __restrict__ atyn,
        double* sc) {
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int col = (int)(gid / G), lane = (int)(gid % G);
    double s = 0.0;
    if (col < n) {
        int e = p[col + 1];
        for (int q = p[col] + lane; q < e; q += G) s += v[q] * yn[i[q]];
    }
    for (int o = G / 2; o > 0; o >>= 1) s += __shfl_down_sync(FULL, s, o, G);
    double a = 0.0, b = 0.0;
    if (col < n && lane == 0) {
        atyn[col] = s;
        double d = xn[col] - x[col];
        a = d * d; b = d * (s - aty[col]);
    }
    block_add2(a, b, sc + 5, sc + 6);
}

// PDLP's adaptive step-size rule, on one thread: accept or reject, next eta.
extern "C" __global__ void pdhg_stepsize(double* sc) {
    double eta = sc[0], w = sc[1];
    double k = sc[2] + 1.0;
    double num = w * sc[5] + sc[7] / w;
    double inter = fabs(sc[6]);
    double ebar = inter != 0.0 ? num / (2.0 * inter) : __longlong_as_double(0x7ff0000000000000LL);  // +inf (NVRTC has no INFINITY)
    double enew = fmin((1.0 - pow(k + 1.0, -0.3)) * ebar, (1.0 + pow(k + 1.0, -0.6)) * eta);
    bool acc = eta <= ebar;
    sc[2] = k;
    sc[3] = acc ? 1.0 : 0.0;
    sc[4] = acc ? eta : 0.0;
    if (acc) sc[8] += eta; else sc[9] += 1.0;
    sc[0] = enew;
    sc[5] = 0.0; sc[6] = 0.0; sc[7] = 0.0;
}

extern "C" __global__ void pdhg_clear(double* sc) { sc[3] = 0.0; sc[4] = 0.0; }

// KKT sums (see _kkt_cpu): out += [c'x, dual objective, |r_primal|^2, |r_dual|^2] of
// (x*ux, y*uy, ax*uax, aty*uaty) if scaled, else of (x, y, ax, aty). Threads [0,m) rows, then columns.
extern "C" __global__ void kkt_sums(int n, int m, const double* __restrict__ c,
        const double* __restrict__ rl, const double* __restrict__ ru, const double* __restrict__ lb,
        const double* __restrict__ ub, const double* __restrict__ x, const double* __restrict__ y,
        const double* __restrict__ ax, const double* __restrict__ aty, const double* __restrict__ ux,
        const double* __restrict__ uy, const double* __restrict__ uax, const double* __restrict__ uaty,
        int scaled, double* out) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    double po = 0.0, du = 0.0, rp = 0.0, rd = 0.0;
    if (t < m) {
        int i = (int)t;
        double a = scaled ? ax[i] * uax[i] : ax[i];
        double r = a < rl[i] ? rl[i] - a : (a > ru[i] ? a - ru[i] : 0.0);
        rp = r * r;
        double yi = scaled ? y[i] * uy[i] : y[i];
        if (yi > 0.0 && isfinite(rl[i])) du = rl[i] * yi;
        else if (yi < 0.0 && isfinite(ru[i])) du = ru[i] * yi;
    } else if (t < (long long)n + m) {
        int j = (int)(t - m);
        po = c[j] * (scaled ? x[j] * ux[j] : x[j]);
        double lam = c[j] - (scaled ? aty[j] * uaty[j] : aty[j]);
        if (lam > 0.0) { if (isfinite(lb[j])) du = lb[j] * lam; else rd = lam * lam; }
        else if (lam < 0.0) { if (isfinite(ub[j])) du = ub[j] * lam; else rd = lam * lam; }
    }
    block_add2(po, du, out, out + 1);
    __syncthreads();
    block_add2(rp, rd, out + 2, out + 3);
}

// out += [|a - ar|^2, |b - br|^2] with a, ar of length n and b, br of length m.
extern "C" __global__ void dist2(int n, int m, const double* __restrict__ a,
        const double* __restrict__ ar, const double* __restrict__ b, const double* __restrict__ br,
        double* out) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    double u = 0.0, v = 0.0;
    if (t < n) { double d = a[t] - ar[t]; u = d * d; }
    else if (t < (long long)n + m) { double d = b[t - n] - br[t - n]; v = d * d; }
    block_add2(u, v, out, out + 1);
}

// Plain product out = M v, M in CSR, G lanes per row.
template<int G>
__global__ void csr_spmv(int m, const int* __restrict__ p, const int* __restrict__ j,
        const double* __restrict__ v, const double* __restrict__ x, double* __restrict__ out) {
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int row = (int)(gid / G), lane = (int)(gid % G);
    double s = 0.0;
    if (row < m) {
        int e = p[row + 1];
        for (int q = p[row] + lane; q < e; q += G) s += v[q] * x[j[q]];
    }
    for (int o = G / 2; o > 0; o >>= 1) s += __shfl_down_sync(FULL, s, o, G);
    if (row < m && lane == 0) out[row] = s;
}
"""

_TPB = 256


def _group(avg_len: float) -> int:
    """Lanes per row for CSR-vector: the power of two nearest below the mean row length, <= 32."""
    g = 1
    while g * 2 <= avg_len and g < 32:
        g *= 2
    return g


class _GPU:
    def __init__(self, S: _Scaled, use_graph: bool = True):
        import warnings
        with warnings.catch_warnings():            # CuPy warns when CUDA_PATH is unset even though
            warnings.simplefilter("ignore")        # its pip-installed toolkit libraries are found
            import cupy as cp
        self.cp = cp
        self.xp = cp
        self.S = S
        self.use_graph = use_graph
        self.stream = cp.cuda.Stream(non_blocking=True)
        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        self.name = "gpu-cupy (" + props["name"].decode() + ")"
        m, n = S.m, S.n
        self.m, self.n = m, n
        nnz = len(S.vals)
        self.gr = _group(nnz / max(m, 1))
        self.gc = _group(nnz / max(n, 1))
        with self.stream:
            f64 = lambda a: cp.asarray(np.ascontiguousarray(a, dtype=np.float64))
            i32 = lambda a: cp.asarray(np.ascontiguousarray(a, dtype=np.int32))
            self.c, self.lb, self.ub, self.rl, self.ru = (f64(v) for v in (S.c, S.lb, S.ub, S.rl, S.ru))
            self.rp, self.rj, self.rv = i32(S.rowptr), i32(S.colidx), f64(S.tvals)      # CSR(A)
            self.cpp, self.ci, self.cv = i32(S.colptr), i32(S.rowidx), f64(S.vals)      # CSR(A')
            self.x, self.xn, self.aty, self.atyn, self.sx = (cp.zeros(n) for _ in range(5))
            self.y, self.yn, self.ax, self.axn, self.sy = (cp.zeros(m) for _ in range(5))
            self.sc = cp.zeros(16)
            names = [f"pdhg_rows<{self.gr}>", f"pdhg_cols<{self.gc}>",
                     f"csr_spmv<{self.gr}>", f"csr_spmv<{self.gc}>"]
            mod = cp.RawModule(code=_CUDA_SRC, options=("--std=c++14",), name_expressions=names)
            self.k_commit = mod.get_function("pdhg_commit_primal")
            self.k_step = mod.get_function("pdhg_stepsize")
            self.k_clear = mod.get_function("pdhg_clear")
            self.k_kkt = mod.get_function("kkt_sums")
            self.k_dist = mod.get_function("dist2")
            self.k_rows = mod.get_function(names[0])
            self.k_cols = mod.get_function(names[1])
            self.k_spmv_r = mod.get_function(names[2])
            self.k_spmv_c = mod.get_function(names[3])
        self.graphs = {}
        i = np.int32
        self.args_commit1 = (i(n), i(m), i(1), self.c, self.lb, self.ub, self.x, self.xn, self.aty,
                             self.atyn, self.sx, self.y, self.yn, self.ax, self.axn, self.sy, self.sc)
        self.args_commit0 = self.args_commit1[:2] + (i(0),) + self.args_commit1[3:]
        self.args_rows = (i(m), self.rp, self.rj, self.rv, self.xn, self.y, self.ax, self.rl,
                          self.ru, self.axn, self.yn, self.sc)
        self.args_cols = (i(n), self.cpp, self.ci, self.cv, self.yn, self.x, self.xn, self.aty,
                          self.atyn, self.sc)
        self.grid_nm = ((n + m + _TPB - 1) // _TPB,)
        self.grid_r = ((m * self.gr + _TPB - 1) // _TPB,)
        self.grid_c = ((n * self.gc + _TPB - 1) // _TPB,)

    def context(self):
        return self.stream

    Ax = property(lambda self: self.ax)
    ATy = property(lambda self: self.aty)

    def asarray(self, a):
        return self.cp.asarray(np.asarray(a, dtype=np.float64))

    def to_host(self, a):
        return self.cp.asnumpy(a, stream=self.stream)

    def matvec(self, v):
        out = self.cp.empty(self.m)
        if self.m:
            self.k_spmv_r(self.grid_r, (_TPB,), (np.int32(self.m), self.rp, self.rj, self.rv, v, out))
        return out

    def rmatvec(self, v):
        out = self.cp.empty(self.n)
        if self.n:
            self.k_spmv_c(self.grid_c, (_TPB,), (np.int32(self.n), self.cpp, self.ci, self.cv, v, out))
        return out

    def set_iterate(self, x, y, Ax, ATy):
        self.x[...] = x
        self.y[...] = y
        self.ax[...] = Ax
        self.aty[...] = ATy

    def reset_average(self):
        self.sx.fill(0.0)
        self.sy.fill(0.0)
        self.sc[_WSUM] = 0.0

    def set_step(self, eta):
        self.sc[_ETA] = float(eta)

    def set_omega(self, omega):
        self.sc[_OMEGA] = float(omega)

    def _attempt(self):
        if self.n + self.m:
            self.k_commit(self.grid_nm, (_TPB,), self.args_commit1)
        if self.m:
            self.k_rows(self.grid_r, (_TPB,), self.args_rows)
        if self.n:
            self.k_cols(self.grid_c, (_TPB,), self.args_cols)
        self.k_step((1,), (1,), (self.sc,))

    def _flush(self):
        if self.n + self.m:
            self.k_commit(self.grid_nm, (_TPB,), self.args_commit0)
        self.k_clear((1,), (1,), (self.sc,))

    def _block(self, steps):
        for _ in range(steps):
            self._attempt()
        self._flush()

    def run(self, steps: int, k0: int):
        # The attempt counter k lives on the device (sc[_K]); k0 is only the host's copy.
        with self.stream:                  # kernels must go to (and be captured on) this stream
            if not self.use_graph:
                self._block(steps)
                return
            g = self.graphs.get(steps)
            if g is None:
                self.stream.begin_capture()
                self._block(steps)
                g = self.stream.end_capture()
                self.graphs[steps] = g
            g.launch(self.stream)

    def scalars(self):
        s = self.sc.get(stream=self.stream)
        return float(s[_ETA]), float(s[_WSUM]), int(s[_NREJ])

    def kkt(self, P, x, y, Ax, ATy, U=None):
        out = self.cp.zeros(4)
        scaled = U is not None
        U = U if scaled else (x, y, Ax, ATy)          # unused when not scaled
        self.k_kkt(self.grid_nm, (_TPB,), (np.int32(self.n), np.int32(self.m), P.c, P.rl, P.ru,
                                           P.lb, P.ub, x, y, Ax, ATy, *U, np.int32(scaled), out))
        return out

    def dist2(self, a, ar, b, br):
        out = self.cp.zeros(2)
        self.k_dist(self.grid_nm, (_TPB,), (np.int32(self.n), np.int32(self.m), a, ar, b, br, out))
        return out

    def averages(self):
        w = self.sc[_WSUM]
        return self.sx / w, self.sy / w


# =============================================================================================
# KKT error and infeasibility certificates (written once, for NumPy and CuPy)
# =============================================================================================

class _Vecs:
    """Bounds and costs of one problem (scaled or original) on the backend's device."""

    def __init__(self, xp, c, rl, ru, lb, ub, c0=0.0):
        asd = lambda a: xp.asarray(np.asarray(a, dtype=np.float64))
        self.c, self.rl, self.ru, self.lb, self.ub = (asd(v) for v in (c, rl, ru, lb, ub))
        fin = lambda a: xp.asarray(np.isfinite(a))
        self.has_rl, self.has_ru, self.has_lb, self.has_ub = (fin(v) for v in (rl, ru, lb, ub))
        z = lambda a: asd(np.where(np.isfinite(a), a, 0.0))
        self.rl0, self.ru0, self.lb0, self.ub0 = (z(v) for v in (rl, ru, lb, ub))
        self.c0 = float(c0)


def _finish_kkt(v, c0):
    """[primal objective, dual objective, |primal residual|_2, |dual residual|_2] from the sums."""
    return np.array([v[0] + c0, v[1] + c0, np.sqrt(max(v[2], 0.0)), np.sqrt(max(v[3], 0.0))])


def _rays(xp, P: _Vecs, dx, dy, Adx, ATdy):
    """Farkas-certificate quality of a candidate primal ray dx and dual ray dy (original units).

    Returns [dual ray objective, max dual ray infeasibility, primal ray objective,
    max primal ray infeasibility]. dy certifies primal infeasibility if its objective is > 0 and
    its infeasibility is small relative to it; dx certifies dual infeasibility if c'dx < 0 and
    its infeasibility is small relative to -c'dx (PDLP's tests, which are scale invariant).
    dx and dy must already lie in the recession cones of the bounds and of Y.
    """
    lam = -ATdy
    lp = xp.maximum(lam, 0.0)
    lm = xp.maximum(-lam, 0.0)
    dobj = P.rl0 @ xp.maximum(dy, 0.0) - P.ru0 @ xp.maximum(-dy, 0.0) + P.lb0 @ lp - P.ub0 @ lm
    dres = xp.max(xp.where(P.has_lb, 0.0, lp) + xp.where(P.has_ub, 0.0, lm)) if lam.size else xp.asarray(0.0)
    pobj = P.c @ dx
    viol = xp.where(P.has_rl, xp.maximum(-Adx, 0.0), 0.0) + xp.where(P.has_ru, xp.maximum(Adx, 0.0), 0.0)
    pres = xp.max(viol) if viol.size else xp.asarray(0.0)
    return xp.stack([xp.asarray(dobj), xp.asarray(dres), xp.asarray(pobj), xp.asarray(pres)])


def _cone_x(xp, P: _Vecs, dx):
    """Project a primal direction onto the recession cone of the box."""
    dx = xp.where(P.has_lb, xp.maximum(dx, 0.0), dx)
    return xp.where(P.has_ub, xp.minimum(dx, 0.0), dx)


def _cone_y(xp, P: _Vecs, dy):
    """Project a dual direction onto Y: rows without a lower bound cannot have y > 0, etc."""
    dy = xp.where(P.has_rl, dy, xp.minimum(dy, 0.0))
    return xp.where(P.has_ru, dy, xp.maximum(dy, 0.0))


# =============================================================================================
# The method
# =============================================================================================

def _relax(model: Model) -> Model:
    return Model(name=model.name, c=model.c, A=model.A, rl=model.rl, ru=model.ru, lb=model.lb,
                 ub=model.ub, integer=np.zeros(model.n, dtype=bool), Q=model.Q, c0=model.c0,
                 sense=model.sense, col_names=model.col_names, row_names=model.row_names)


def solve(model: Model, gpu: bool = False, tol: float = 1e-4, max_iter: int = 1_000_000,
          time_limit: float = np.inf, verbose: bool = False, check_every: int = 64,
          ruiz_iters: int = 10, pock_chambolle: bool = True, rescale: bool = True,
          restarts: bool = True, eps_infeasible: float = 1e-8, parallel: bool | None = None,
          threads: int | None = None, use_graph: bool = True) -> Result:
    """Solve the LP (or the LP relaxation of a MIP) by restarted PDHG.

    tol          relative KKT tolerance (primal residual, dual residual, gap; see module doc)
    max_iter     PDHG attempts (each costs one A x and one A'y); rejected steps count
    check_every  attempts between termination / restart / infeasibility checks (cuPDLP: 64)
    parallel     CPU only: thread-parallel kernels; default by size (fork/join costs microseconds)
    threads      CPU only: Numba thread count (default: Numba's, i.e. all logical cores)
    use_graph    GPU only: replay each block of attempts as a CUDA graph
    """
    t0 = time.perf_counter()
    if model.is_qp:
        raise ValueError("pdlp solves linear programs; the model has a quadratic objective")
    if model.is_mip:
        model = _relax(model)
    S = _Scaled(model, ruiz_iters, pock_chambolle, rescale)
    if gpu:
        be = _GPU(S, use_graph=use_graph)
    else:
        if parallel is None:
            # measured crossover on a 6-core laptop: below ~30k nonzeros the thread pool's
            # fork/join per pass costs more than it saves
            parallel = model.A.nnz >= 30_000
        if threads:
            import numba
            numba.set_num_threads(int(threads))
        be = _CPU(S, parallel)
    xp = be.xp
    with be.context():
        return _run(model, S, be, xp, t0, tol, max_iter, time_limit, verbose, check_every,
                    restarts, eps_infeasible)


def _run(model, S, be, xp, t0, tol, max_iter, time_limit, verbose, check_every, restarts, eps_inf):
    Ps = _Vecs(xp, S.c, S.rl, S.ru, S.lb, S.ub)
    Po = _Vecs(xp, model.c, model.rl, model.ru, model.lb, model.ub, model.c0)
    # maps from the scaled iterate to the original one (x = C beta_b x~, A x = A~x~ beta_b / R)
    ux = be.asarray(S.C * S.beta_b)
    uy = be.asarray(S.R * S.beta_c)
    uax = be.asarray(S.beta_b / S.R)
    uaty = be.asarray(S.beta_c / S.C)
    bnorm = float(np.linalg.norm(_combined_bounds(model.rl, model.ru)))
    cnorm = float(np.linalg.norm(model.c))

    U = (ux, uy, uax, uaty)

    def rel(v):
        """(relative primal residual, relative dual residual, relative gap) of a _kkt vector."""
        p, d, rp, rd = v
        return rp / (1 + bnorm), rd / (1 + cnorm), abs(p - d) / (1 + abs(p) + abs(d))

    def wkkt(v, w):
        """The primal-weighted KKT error cuPDLP restarts on (scaled problem)."""
        p, d, rp, rd = v
        return float(np.sqrt(w * rp * rp + rd * rd / w + (p - d) ** 2))

    # ---- start: x0 = projection of 0 onto the box, y0 = 0 ----
    x0 = be.asarray(np.clip(0.0, S.lb, S.ub))
    y0 = be.asarray(np.zeros(S.m))
    be.set_iterate(x0, y0, be.matvec(x0), be.asarray(np.zeros(S.n)))
    be.reset_average()
    omega = S.omega0
    be.set_omega(omega)
    be.set_step(1.0 / S.amax if S.amax > 0 else 1.0)       # PDLP: eta_0 = 1 / |A|_inf
    xr, yr = be.x.copy(), be.y.copy()                     # last restart point
    xref, yref = be.x.copy(), be.y.copy()                 # iterate at the last check (rays)
    kkt_last = wkkt(_finish_kkt(be.to_host(be.kkt(Ps, be.x, be.y, be.Ax, be.ATy)), 0.0), omega)
    kkt_prev = np.inf
    setup = time.perf_counter() - t0

    k = since = nrest = 0
    status = "iteration_limit"
    best = None                     # (label, host scaled x, y) of the returned iterate
    final_rel = (np.nan, np.nan, np.nan)
    ray = None
    last_print = -np.inf
    nchk = 0
    n_row_rejects = 0
    while True:
        steps = min(check_every, max_iter - k)
        if steps <= 0:
            break
        be.run(steps, k)
        k += steps
        since += steps
        nchk += 1
        eta, wsum, nrej = be.scalars()
        x, y, Ax, ATy = be.x, be.y, be.Ax, be.ATy
        have_avg = wsum > 0.0
        parts = [be.kkt(Ps, x, y, Ax, ATy), be.kkt(Po, x, y, Ax, ATy, U),
                 be.dist2(x, xr, y, yr)]
        if have_avg:
            xa, ya = be.averages()
            Axa, ATya = be.matvec(xa), be.rmatvec(ya)
            parts += [be.kkt(Ps, xa, ya, Axa, ATya), be.kkt(Po, xa, ya, Axa, ATya, U),
                      be.dist2(xa, xr, ya, yr)]
        # infeasibility: the difference of iterates since the last ray check (or restart)
        check_rays = nchk % RAY_CHECK_EVERY == 0
        if check_rays:
            dx = _cone_x(xp, Ps, x - xref)
            dy = _cone_y(xp, Ps, y - yref)
            parts.append(_rays(xp, Po, dx * ux, dy * uy, be.matvec(dx) * uax, be.rmatvec(dy) * uaty))
        h = be.to_host(xp.concatenate(parts))
        ks_cur, ko_cur = _finish_kkt(h[0:4], 0.0), _finish_kkt(h[4:8], Po.c0)
        dist_cur = np.sqrt(h[8:10])
        if have_avg:
            ks_avg, ko_avg = _finish_kkt(h[10:14], 0.0), _finish_kkt(h[14:18], Po.c0)
            dist_avg = np.sqrt(h[18:20])
        rays = h[-4:] if check_rays else np.array([0.0, np.inf, 0.0, np.inf])
        elapsed = time.perf_counter() - t0

        cands = [("current", ko_cur)] + ([("average", ko_avg)] if have_avg else [])
        scored = sorted(((max(rel(v)), lab, v) for lab, v in cands), key=lambda t: t[0])
        if verbose and (elapsed - last_print > 1.0 or verbose > 1):
            last_print = elapsed
            rp_, rd_, rg_ = rel(scored[0][2])
            print(f"{k:8d} {elapsed:8.2f}s  obj {scored[0][2][0]:+.8e}  rel p {rp_:.1e} d {rd_:.1e} "
                  f"g {rg_:.1e}  eta {eta:.2e} w {omega:.2e} restarts {nrest}")
        if not np.all(np.isfinite(h[:20 if have_avg else 10])):
            status = "numerical_error"
            best = ("current", x, y)
            final_rel = rel(ko_cur)
            break
        err, lab, v = scored[0]
        if err <= tol:
            # the paper's test is relative to ||b||_2: a few huge right-hand sides (s250r10,
            # ||b|| ~ 2e5) make it accept rows violated by ~20 in absolute terms. Accept only if
            # every row also meets the tolerance relative to its own bound (infinity norm).
            xs_c = x if lab == "current" else xa
            x_host = be.to_host(xs_c * ux)
            ax = model.A.matvec(x_host)
            viol = np.maximum(0.0, np.maximum(model.rl - ax, ax - model.ru))
            bmag = np.maximum(np.where(np.isfinite(model.rl), np.abs(model.rl), 0.0),
                              np.where(np.isfinite(model.ru), np.abs(model.ru), 0.0))
            row_err = float(np.max(viol / (1.0 + bmag), initial=0.0))
            if row_err <= 10 * tol:
                status = "optimal"
                best = (lab, x, y) if lab == "current" else (lab, xa, ya)
                final_rel = rel(v)
                break
            n_row_rejects += 1
        # Farkas certificates, PDLP's relative tests
        d_obj, d_res, p_obj, p_res = rays
        if d_obj > 0 and d_res <= eps_inf * d_obj:
            status = "infeasible"
            ray = be.to_host(dy * uy)
        elif p_obj < 0 and p_res <= eps_inf * (-p_obj):
            status = "unbounded"
            ray = be.to_host(dx * ux)
        if status in ("infeasible", "unbounded"):
            best = ("current", x, y)
            final_rel = rel(ko_cur)
            break
        if elapsed >= time_limit:
            status = "time_limit"
            break

        # ---- adaptive restart (cuPDLP's KKT criteria) ----
        use_avg = have_avg and wkkt(ks_avg, omega) < wkkt(ks_cur, omega)
        ks_c = ks_avg if use_avg else ks_cur
        kc = wkkt(ks_c, omega)
        do_restart = restarts and (
            kc <= BETA_SUFFICIENT * kkt_last
            or (kc <= BETA_NECESSARY * kkt_last and kc > kkt_prev)
            or since >= BETA_ARTIFICIAL * k)
        kkt_prev = kc
        if do_restart:
            if use_avg:
                be.set_iterate(xa, ya, Axa, ATya)
            be.reset_average()
            ddx, ddy = dist_avg if use_avg else dist_cur
            # PDLP primal weight update: omega <- exp(theta log(dy/dx) + (1-theta) log omega)
            if ddx > 1e-10 and ddy > 1e-10:
                th = PRIMAL_WEIGHT_SMOOTHING
                omega = float(np.exp(th * np.log(ddy / ddx) + (1 - th) * np.log(omega)))
                be.set_omega(omega)
            xr, yr = be.x.copy(), be.y.copy()
            kkt_last = wkkt(ks_c, omega)
            kkt_prev = np.inf
            since = 0
            nrest += 1
        if do_restart or check_rays:
            xref, yref = be.x.copy(), be.y.copy()

    # ---- the iterate to return ----
    if best is None:
        # a limit was hit: return whichever of current / average has the smaller relative KKT,
        # evaluated afresh (the last check may have restarted, moving the current iterate)
        _, wsum, _ = be.scalars()
        cands = [("current", be.x, be.y, be.Ax, be.ATy)]
        if wsum > 0:
            xa, ya = be.averages()
            cands.append(("average", xa, ya, be.matvec(xa), be.rmatvec(ya)))
        scored = []
        for lab, xx, yy, axx, atyy in cands:
            v = _finish_kkt(be.to_host(be.kkt(Po, xx, yy, axx, atyy, U)), Po.c0)
            scored.append((max(rel(v)), lab, xx, yy, v))
        scored.sort(key=lambda t: t[0])
        _, lab, xx, yy, v = scored[0]
        best = (lab, xx, yy)
        final_rel = rel(v)
    lab, xs, ys = best
    x_o = be.to_host(xs * ux)
    y_o = be.to_host(ys * uy)
    z_o = model.c - model.A.rmatvec(y_o)
    eta, wsum, nrej = be.scalars()
    obj = model.user_objective(x_o)
    info = {"backend": be.name, "rel_primal": float(final_rel[0]), "rel_dual": float(final_rel[1]),
            "rel_gap": float(final_rel[2]), "restarts": nrest, "rejected_steps": nrej,
            "primal_weight": omega, "step_size": eta, "solution": lab, "setup_time": setup,
            "tol": tol, "checks": nchk, "row_check_rejects": n_row_rejects}
    if ray is not None:
        info["ray"] = ray
    ok = status == "optimal"
    return Result(status=status, x=None if status == "infeasible" else x_o,
                  y=y_o * model.sense, z=z_o * model.sense,
                  objective=obj if status not in ("infeasible", "unbounded") else np.nan,
                  bound=obj if ok else np.nan, iterations=k, time=time.perf_counter() - t0,
                  method="pdlp-gpu" if isinstance(be, _GPU) else "pdlp", info=info)
