"""Sparse LU factorisation of a simplex basis, with product-form updates.

Factorisation: left-looking LU with threshold partial pivoting (Gilbert & Peierls, "Sparse
partial pivoting in time proportional to arithmetic operations", SIAM J. Sci. Stat. Comput. 9(5),
1988). Column k is found by a sparse triangular solve with the columns of L already computed;
the nonzero pattern comes from a depth-first search in the graph of L, so the work is
proportional to the flops, not to m.

Column order: logical (slack) columns first, since they are unit vectors and factor for free,
then structural columns by increasing length, a cheap stand-in for a fill-reducing order that
works well on simplex bases, which are mostly slack.

Pivot choice: among rows within a factor `threshold` of the largest entry of the column, take the
one whose row is shortest, trading a little stability for much less fill (Markowitz's idea
applied row-wise). threshold = 0.1 is the usual compromise.

Singular bases (dependent columns chosen by the simplex, or rounding) are repaired rather than
rejected: a column with no acceptable pivot is reported together with an unpivoted row, and the
caller swaps that basic variable for the row's logical. This is how industrial simplex codes keep
going on degenerate models.

Updates: after each simplex pivot the new inverse is E B^-1, with E an elementary eta matrix
(product form of the inverse, Dantzig & Orchard-Hays 1954). Refactorise every `refactor_every`
updates, or earlier if the eta file grows large, to bound both work and error growth.

Conventions: B Q = Lt U, where Lt is unit lower triangular up to the row permutation (column k of
Lt has a 1 at pivot row p_k), U is upper triangular in pivot order, and Q is the column order.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _grow_i(a, need):
    if need <= len(a):
        return a
    b = np.empty(max(need, 2 * len(a)), dtype=a.dtype)
    b[: len(a)] = a
    return b


@njit(cache=True)
def _grow_f(a, need):
    if need <= len(a):
        return a
    b = np.empty(max(need, 2 * len(a)), dtype=a.dtype)
    b[: len(a)] = a
    return b


@njit(cache=True)
def _factor(m, Bp, Bi, Bx, q, row_count, threshold, tiny, pref):
    """Left-looking LU of the m x m matrix B (CSC), columns taken in the order q."""
    cap = max(4 * Bp[m] + m, 16)
    Lp = np.zeros(m + 1, dtype=np.int64)
    Li = np.empty(cap, dtype=np.int64)
    Lx = np.empty(cap)
    Up = np.zeros(m + 1, dtype=np.int64)
    Ui = np.empty(cap, dtype=np.int64)
    Ux = np.empty(cap)
    Udiag = np.zeros(m)
    pinv = np.full(m, -1, dtype=np.int64)
    prow = np.full(m, -1, dtype=np.int64)          # pivot row of step k
    x = np.zeros(m)
    mark = np.full(m, -1, dtype=np.int64)
    stack = np.empty(m, dtype=np.int64)
    pstack = np.empty(m, dtype=np.int64)
    topo = np.empty(m, dtype=np.int64)
    singular_steps = np.full(m, -1, dtype=np.int64)
    nsing = 0
    lnz = 0
    unz = 0
    for k in range(m):
        j = q[k]
        # --- reach: rows touched by solving Lt x = B[:, j], in topological order ---
        top = m
        for p in range(Bp[j], Bp[j + 1]):
            i = Bi[p]
            if mark[i] == k:
                continue
            # iterative DFS from i
            head = 0
            stack[0] = i
            while head >= 0:
                r = stack[head]
                if mark[r] != k:
                    mark[r] = k
                    s = pinv[r]
                    pstack[head] = Lp[s] if s >= 0 else 0
                done = True
                s = pinv[r]
                if s >= 0:
                    pend = Lp[s + 1]
                    pp = pstack[head]
                    while pp < pend:
                        rr = Li[pp]
                        pp += 1
                        if mark[rr] != k:
                            pstack[head] = pp
                            head += 1
                            stack[head] = rr
                            done = False
                            break
                    if done:
                        pstack[head] = pend
                if done:
                    head -= 1
                    top -= 1
                    topo[top] = r
        # --- numeric solve ---
        for p in range(Bp[j], Bp[j + 1]):
            x[Bi[p]] = Bx[p]
        for t in range(top, m):
            r = topo[t]
            s = pinv[r]
            if s < 0:
                continue
            xr = x[r]
            if xr == 0.0:
                continue
            for pp in range(Lp[s] + 1, Lp[s + 1]):   # entry 0 of each L column is its unit pivot
                x[Li[pp]] -= Lx[pp] * xr
        # --- U column: pivoted rows ---
        Ui = _grow_i(Ui, unz + (m - top) + 1)
        Ux = _grow_f(Ux, unz + (m - top) + 1)
        amax = 0.0
        for t in range(top, m):
            r = topo[t]
            if pinv[r] >= 0:
                if x[r] != 0.0:
                    Ui[unz] = pinv[r]
                    Ux[unz] = x[r]
                    unz += 1
            else:
                a = abs(x[r])
                if a > amax:
                    amax = a
        # --- pivot choice among unpivoted rows ---
        piv = -1
        pr = pref[k]
        if pr >= 0 and pinv[pr] < 0 and abs(x[pr]) > tiny and abs(x[pr]) >= threshold * amax:
            piv = pr                    # the singleton row the ordering assigned to this column
        elif amax > tiny:
            best = 1 << 60
            bestval = 0.0
            for t in range(top, m):
                r = topo[t]
                if pinv[r] < 0:
                    a = abs(x[r])
                    if a >= threshold * amax:
                        c = row_count[r]
                        if c < best or (c == best and a > bestval):
                            best = c
                            bestval = a
                            piv = r
        Up[k + 1] = unz
        Lp_start = lnz
        Li = _grow_i(Li, lnz + (m - top) + 1)
        Lx = _grow_f(Lx, lnz + (m - top) + 1)
        if piv < 0:
            # dependent column: leave the step unpivoted and let the caller repair the basis
            singular_steps[nsing] = k
            nsing += 1
            Udiag[k] = 0.0
            Li[lnz] = -1
            Lx[lnz] = 1.0
            lnz += 1
        else:
            d = x[piv]
            Udiag[k] = d
            pinv[piv] = k
            prow[k] = piv
            Li[lnz] = piv
            Lx[lnz] = 1.0
            lnz += 1
            for t in range(top, m):
                r = topo[t]
                if pinv[r] < 0 and x[r] != 0.0:
                    Li[lnz] = r
                    Lx[lnz] = x[r] / d
                    lnz += 1
        Lp[k + 1] = lnz
        for t in range(top, m):
            x[topo[t]] = 0.0
    return Lp, Li[:lnz].copy(), Lx[:lnz].copy(), Up, Ui[:unz].copy(), Ux[:unz].copy(), Udiag, pinv, prow, singular_steps[:nsing].copy()


@njit(cache=True)
def _ftran_lu(m, Lp, Li, Lx, Up, Ui, Ux, Udiag, prow, q, b, out):
    """Solve B x = b with B Q = Lt U. b is indexed by row and is overwritten; out by basis position."""
    w = np.empty(m)
    for k in range(m):
        p = prow[k]
        wk = b[p]
        w[k] = wk
        if wk != 0.0:
            for pp in range(Lp[k] + 1, Lp[k + 1]):
                b[Li[pp]] -= Lx[pp] * wk
    for k in range(m - 1, -1, -1):
        vk = w[k] / Udiag[k]
        w[k] = vk
        if vk != 0.0:
            for pp in range(Up[k], Up[k + 1]):
                w[Ui[pp]] -= Ux[pp] * vk
    for k in range(m):
        out[q[k]] = w[k]


@njit(cache=True)
def _btran_lu(m, Lp, Li, Lx, Up, Ui, Ux, Udiag, prow, q, b, out):
    """Solve B' y = b. b is indexed by basis position; out by row."""
    t = np.empty(m)
    for k in range(m):
        s = b[q[k]]
        for pp in range(Up[k], Up[k + 1]):
            s -= Ux[pp] * t[Ui[pp]]
        t[k] = s / Udiag[k]
    for k in range(m - 1, -1, -1):
        s = t[k]
        for pp in range(Lp[k] + 1, Lp[k + 1]):
            s -= Lx[pp] * out[Li[pp]]
        out[prow[k]] = s


@njit(cache=True)
def _triangular_order(m, Bp, Bi):
    """Column order for the LU of a simplex basis (Suhl & Suhl, ORSA J. Comput. 2, 1990).

    Column singletons of the active submatrix are taken first, repeatedly (logical columns are
    the first of them); then row singletons, repeatedly, placed last in reverse order; the rest
    (the "bump") goes in between, shortest columns first. In the left-looking factorisation the
    front and back columns then have exactly one candidate pivot row each, so the triangular
    parts of the basis are factorised with no fill at all; only the bump can fill."""
    # row -> columns
    rcnt = np.zeros(m + 1, dtype=np.int64)
    for p in range(Bp[m]):
        rcnt[Bi[p] + 1] += 1
    for i in range(m):
        rcnt[i + 1] += rcnt[i]
    Rp = rcnt.copy()
    Rc = np.empty(Bp[m], dtype=np.int64)
    nxt = rcnt[:-1].copy()
    for j in range(m):
        for p in range(Bp[j], Bp[j + 1]):
            i = Bi[p]
            Rc[nxt[i]] = j
            nxt[i] += 1
    col_cnt = np.empty(m, dtype=np.int64)
    for j in range(m):
        col_cnt[j] = Bp[j + 1] - Bp[j]
    row_cnt = np.empty(m, dtype=np.int64)
    for i in range(m):
        row_cnt[i] = Rp[i + 1] - Rp[i]
    col_done = np.zeros(m, dtype=np.bool_)
    row_done = np.zeros(m, dtype=np.bool_)
    front = np.empty(m, dtype=np.int64)
    front_row = np.empty(m, dtype=np.int64)
    nf = 0
    stack = np.empty(m, dtype=np.int64)
    ns = 0
    for j in range(m):
        if col_cnt[j] == 1:
            stack[ns] = j
            ns += 1
    while ns > 0:
        ns -= 1
        j = stack[ns]
        if col_done[j] or col_cnt[j] != 1:
            continue
        r = -1
        for p in range(Bp[j], Bp[j + 1]):
            if not row_done[Bi[p]]:
                r = Bi[p]
                break
        col_done[j] = True
        front[nf] = j
        front_row[nf] = r
        nf += 1
        row_done[r] = True
        for p in range(Bp[j], Bp[j + 1]):          # column j leaves every row it touches
            row_cnt[Bi[p]] -= 1
        for p in range(Rp[r], Rp[r + 1]):          # row r leaves every column it touches
            c = Rc[p]
            if not col_done[c]:
                col_cnt[c] -= 1
                if col_cnt[c] == 1:
                    stack[ns] = c
                    ns += 1
    back = np.empty(m, dtype=np.int64)
    back_row = np.empty(m, dtype=np.int64)
    nb = 0
    ns = 0
    for i in range(m):
        if not row_done[i] and row_cnt[i] == 1:
            stack[ns] = i
            ns += 1
    while ns > 0:
        ns -= 1
        i = stack[ns]
        if row_done[i] or row_cnt[i] != 1:
            continue
        c = -1
        for p in range(Rp[i], Rp[i + 1]):
            if not col_done[Rc[p]]:
                c = Rc[p]
                break
        col_done[c] = True
        row_done[i] = True
        back[nb] = c
        back_row[nb] = i
        nb += 1
        for p in range(Rp[i], Rp[i + 1]):
            cc = Rc[p]
            if not col_done[cc]:
                col_cnt[cc] -= 1
        for p in range(Bp[c], Bp[c + 1]):          # column c leaves its other rows
            r = Bi[p]
            if not row_done[r]:
                row_cnt[r] -= 1
                if row_cnt[r] == 1:
                    stack[ns] = r
                    ns += 1
    # the bump: remaining columns, fewest active entries first
    bump = np.empty(m - nf - nb, dtype=np.int64)
    keys = np.empty(m - nf - nb, dtype=np.int64)
    k = 0
    for j in range(m):
        if not col_done[j]:
            bump[k] = j
            keys[k] = col_cnt[j]
            k += 1
    bump = bump[np.argsort(keys, kind="mergesort")]
    # left-looking order: column singletons, then row singletons in the order found (each
    # pivots on its own row, and no later column touches that row, so its L column is never
    # used again), then the bump, where all the fill is
    order = np.empty(m, dtype=np.int64)
    pref = np.full(m, -1, dtype=np.int64)
    order[:nf] = front[:nf]
    pref[:nf] = front_row[:nf]
    order[nf:nf + nb] = back[:nb]
    pref[nf:nf + nb] = back_row[:nb]
    order[nf + nb:] = bump
    # row counts inside the bump: the tie-break for the bump's pivot rows (closer to Markowitz
    # than counts over the whole basis, most of which is already triangular)
    bump_rc = np.zeros(m, dtype=np.int64)
    for t in range(len(bump)):
        j = bump[t]
        for p in range(Bp[j], Bp[j + 1]):
            if not row_done[Bi[p]]:
                bump_rc[Bi[p]] += 1
    return order, pref, nf, nb, bump_rc


@njit(cache=True)
def _transpose_factors(m, Lp, Li, Lx, Up, Ui, Ux, pinv):
    """Row-wise copies of L and U for the sparse (axpy) form of BTRAN.

    Returns, for U, the entries of each row k (columns j > k, values U[k, j]); for L, for each
    pivot step k the entries L[prow[k], c] of the columns c < k that have a nonzero in row
    prow[k] (stored against the step k of that row)."""
    # U: column j holds rows Ui (steps < j); transpose into rows
    ucnt = np.zeros(m + 1, dtype=np.int64)
    for pp in range(Up[m]):
        ucnt[Ui[pp] + 1] += 1
    for k in range(m):
        ucnt[k + 1] += ucnt[k]
    URp = ucnt.copy()
    URj = np.empty(Up[m], dtype=np.int64)
    URx = np.empty(Up[m])
    nxt = ucnt[:-1].copy()
    for j in range(m):
        for pp in range(Up[j], Up[j + 1]):
            i = Ui[pp]
            q = nxt[i]
            nxt[i] += 1
            URj[q] = j
            URx[q] = Ux[pp]
    # L: column c holds (after its unit diagonal) original rows r with pinv[r] > c
    nL = Lp[m]
    lcnt = np.zeros(m + 1, dtype=np.int64)
    for c in range(m):
        for pp in range(Lp[c] + 1, Lp[c + 1]):
            r = Li[pp]
            if r >= 0:
                lcnt[pinv[r] + 1] += 1
    for k in range(m):
        lcnt[k + 1] += lcnt[k]
    LRp = lcnt.copy()
    LRc = np.empty(max(lcnt[m], 1), dtype=np.int64)
    LRx = np.empty(max(lcnt[m], 1))
    nxt = lcnt[:-1].copy()
    for c in range(m):
        for pp in range(Lp[c] + 1, Lp[c + 1]):
            r = Li[pp]
            if r >= 0:
                k = pinv[r]
                q = nxt[k]
                nxt[k] += 1
                LRc[q] = c
                LRx[q] = Lx[pp]
    return URp, URj, URx, LRp, LRc, LRx


@njit(cache=True)
def _btran_lu_sparse(m, URp, URj, URx, LRp, LRc, LRx, Udiag, prow, q, b, out):
    """B' y = b in axpy form over the row-wise factors: a zero is skipped, not multiplied."""
    t = np.empty(m)
    for k in range(m):
        t[k] = b[q[k]]
    for k in range(m):                      # U' t = b, forward
        tk = t[k]
        if tk != 0.0:
            tk /= Udiag[k]
            t[k] = tk
            for pp in range(URp[k], URp[k + 1]):
                t[URj[pp]] -= URx[pp] * tk
    for k in range(m - 1, -1, -1):          # L' y = t, backward
        yk = t[k]
        out[prow[k]] = yk
        if yk != 0.0:
            for pp in range(LRp[k], LRp[k + 1]):
                t[LRc[pp]] -= LRx[pp] * yk


@njit(cache=True)
def _eta_ftran(x, Ep, Er, Ei, Ex, n_eta):
    for e in range(n_eta):
        r = Er[e]
        xr = x[r] / Ex[Ep[e]]           # first entry of each eta is its pivot alpha_r
        x[r] = xr
        if xr != 0.0:
            for pp in range(Ep[e] + 1, Ep[e + 1]):
                x[Ei[pp]] -= Ex[pp] * xr


@njit(cache=True)
def _eta_btran(b, Ep, Er, Ei, Ex, n_eta):
    for e in range(n_eta - 1, -1, -1):
        r = Er[e]
        s = b[r]
        for pp in range(Ep[e] + 1, Ep[e + 1]):
            s -= Ex[pp] * b[Ei[pp]]
        b[r] = s / Ex[Ep[e]]


class BasisFactor:
    """B^-1 as an LU factorisation plus a file of eta updates."""

    def __init__(self, m: int, threshold: float = 0.1, tiny: float = 1e-11, refactor_every: int = 100):
        self.m = m
        self.threshold, self.tiny = threshold, tiny
        self.refactor_every = refactor_every
        self.n_eta = 0
        self._Ep = np.zeros(refactor_every + 2, dtype=np.int64)
        self._Er = np.zeros(refactor_every + 1, dtype=np.int64)
        self._Ei = np.zeros(16 * m + 64, dtype=np.int64)
        self._Ex = np.zeros(16 * m + 64)

    def factor(self, Bp, Bi, Bx, is_logical):
        """Factor the basis matrix given column-wise. Returns [(position, row)] repairs needed."""
        m = self.m
        # column singletons first, row singletons last, the bump between (see _triangular_order)
        q, pref, self.n_front, self.n_back, row_count = _triangular_order(m, Bp, Bi)
        (self.Lp, self.Li, self.Lx, self.Up, self.Ui, self.Ux, self.Udiag, self.pinv, self.prow,
         sing) = _factor(m, Bp, Bi, Bx, q, row_count, self.threshold, self.tiny, pref)
        self.q = q
        (self.URp, self.URj, self.URx, self.LRp, self.LRc, self.LRx) = _transpose_factors(
            m, self.Lp, self.Li, self.Lx, self.Up, self.Ui, self.Ux, self.pinv)
        self.n_eta = 0
        self._Ep[0] = 0
        repairs = []
        if len(sing):
            free_rows = np.flatnonzero(self.pinv < 0)
            for k, r in zip(sing, free_rows):
                repairs.append((int(q[k]), int(r)))
        return repairs

    @property
    def nnz(self) -> int:
        return len(self.Li) + len(self.Ui) + self.m

    def ftran(self, rhs_by_row: np.ndarray) -> np.ndarray:
        b = np.array(rhs_by_row, dtype=np.float64)
        out = np.empty(self.m)
        _ftran_lu(self.m, self.Lp, self.Li, self.Lx, self.Up, self.Ui, self.Ux, self.Udiag,
                  self.prow, self.q, b, out)
        if self.n_eta:
            _eta_ftran(out, self._Ep, self._Er, self._Ei, self._Ex, self.n_eta)
        return out

    def btran(self, rhs_by_pos: np.ndarray) -> np.ndarray:
        b = np.array(rhs_by_pos, dtype=np.float64)
        if self.n_eta:
            _eta_btran(b, self._Ep, self._Er, self._Ei, self._Ex, self.n_eta)
        out = np.empty(self.m)
        _btran_lu_sparse(self.m, self.URp, self.URj, self.URx, self.LRp, self.LRc, self.LRx,
                         self.Udiag, self.prow, self.q, b, out)
        return out

    def grow(self):
        """Make room in the eta file (the compiled simplex loop asks when it is nearly full)."""
        size = max(2 * len(self._Ei), int(self._Ep[self.n_eta]) + 2 * self.m + 64)
        self._Ei = np.resize(self._Ei, size)
        self._Ex = np.resize(self._Ex, size)
        if self.n_eta + 3 >= len(self._Er):
            self._Er = np.resize(self._Er, 2 * len(self._Er) + 4)
            self._Ep = np.resize(self._Ep, 2 * len(self._Ep) + 4)

    def update(self, r: int, alpha: np.ndarray, drop: float = 1e-14) -> bool:
        """Record the pivot at basis position r with FTRANed entering column alpha.
        Returns True when a refactorisation is due."""
        nz = np.flatnonzero(np.abs(alpha) > drop)
        nz = nz[nz != r]
        need = self._Ep[self.n_eta] + 1 + len(nz)
        if need > len(self._Ei):
            grow = max(need, 2 * len(self._Ei))
            self._Ei = np.resize(self._Ei, grow)
            self._Ex = np.resize(self._Ex, grow)
        start = self._Ep[self.n_eta]
        self._Ei[start] = r
        self._Ex[start] = alpha[r]
        self._Ei[start + 1: start + 1 + len(nz)] = nz
        self._Ex[start + 1: start + 1 + len(nz)] = alpha[nz]
        self._Er[self.n_eta] = r
        self.n_eta += 1
        self._Ep[self.n_eta] = start + 1 + len(nz)
        return self.n_eta >= self.refactor_every or self._Ep[self.n_eta] > 8 * (self.nnz + self.m)
