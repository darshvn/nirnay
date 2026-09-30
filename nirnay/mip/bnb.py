"""Branch-and-bound for mixed-integer linear programmes, on the warm-started dual simplex.

Every node is the LP relaxation with some integer bounds tightened. After a branching change
the parent's optimal basis stays dual feasible, so the dual simplex re-optimises a child in a
few pivots. That is why MILP codes are built on the dual simplex, and so is this one.

Components (with the references each follows):
  * node selection: best bound, with depth-first plunging into the last child while it looks
    promising (Achterberg 2007, sec. 6.3)
  * branching: pseudocost branching with reliability initialisation by strong branching
    (Achterberg, Koch, Martin, "Branching rules revisited", Oper. Res. Letters 33, 2005)
  * domain propagation on the rows at every node (propagate.py)
  * primal heuristics: simple rounding at every node; at the root and periodically, fix the
    integers to rounded LP values and re-solve the LP for the continuous part
  * Gomory mixed-integer cuts in rounds at the root (cuts.py); cuts that end up slack are dropped
  * pruning by bound, with the bound rounded up when the objective is integral on integer points

Tolerances follow common practice: a value within 1e-6 of an integer is integral; the search
stops at a relative gap of 1e-4 (the HiGHS and SCIP default).
"""
from __future__ import annotations

import heapq
import itertools
import time

import numpy as np

from ..lp.simplex import SimplexLP
from ..model import Model, Result
from .cuts import add_cuts, gmi_cuts
from .propagate import propagate

INT_TOL = 1e-6
FEAS_TOL = 1e-6


class _Node:
    __slots__ = ("bound", "depth", "lb", "ub", "basis", "branch")

    def __init__(self, bound, depth, lb, ub, basis, branch):
        self.bound = bound          # parent LP value: a valid lower bound for this subtree
        self.depth = depth
        self.lb = lb                # full bound vectors for the integer columns
        self.ub = ub
        self.basis = basis          # parent's basis (int8 status), or None
        self.branch = branch        # (column, direction, fractional part) that created it


class BranchAndBound:
    def __init__(self, model: Model, time_limit=np.inf, node_limit=10**9, gap=1e-4,
                 verbose=False, strong_candidates=8, strong_iters=60, reliability=4, cut_rounds=20):
        self.model = model
        self.orig = model
        self.cut_rounds = cut_rounds
        self.n_cuts = 0
        self.time_limit = time_limit
        self.node_limit = node_limit
        self.gap_tol = gap
        self.verbose = verbose
        self.strong_candidates = strong_candidates
        self.strong_iters = strong_iters
        self.reliability = reliability
        self.int_idx = np.flatnonzero(model.integer)
        n = model.n
        # pseudocosts: objective gain per unit change, down and up
        self.pc_sum = np.zeros((2, n))
        self.pc_cnt = np.zeros((2, n))
        # objective integral on integer points? then bounds can be rounded up
        c = model.c
        cont = ~model.integer
        self.int_obj = bool(np.all(c[cont] == 0) and np.all(np.abs(c - np.round(c)) < 1e-9)
                            and abs(model.c0 - round(model.c0)) < 1e-9)
        AT = model.AT
        self.Rp, self.Rj, self.Rv = AT.colptr, AT.rowidx, AT.vals
        self.incumbent = None
        self.inc_obj = np.inf
        self.nodes = 0
        self.lp_iters = 0
        self.lp = SimplexLP(_relaxed(model))

    # ---------------------------------------------------------------- utilities
    def _bound_up(self, v):
        """Round a bound up when the objective is integral at integer points."""
        if self.int_obj and np.isfinite(v):
            return np.ceil(v - 1e-6)
        return v

    def _cutoff(self):
        if not np.isfinite(self.inc_obj):
            return np.inf
        return self.inc_obj - max(1e-9, 1e-9 * abs(self.inc_obj)) if not self.int_obj else self.inc_obj - 1 + 1e-6

    def _try_incumbent(self, x, source):
        m = self.orig
        xi = x.copy()
        xi[self.int_idx] = np.round(xi[self.int_idx])
        v = m.violation(xi)
        if v["row"] > FEAS_TOL * 10 or v["bound"] > FEAS_TOL or v["integrality"] > INT_TOL:
            return False
        obj = m.objective(xi)
        if obj < self.inc_obj - 1e-9 * max(1.0, abs(obj)):
            self.incumbent, self.inc_obj = xi, obj
            if self.verbose:
                print(f"  incumbent {m.sense * obj:+.10g} from {source} at node {self.nodes}")
            return True
        return False

    def _solve_lp(self, lb, ub, basis=None, iter_limit=None):
        lp = self.lp
        lp.set_structural_bounds(lb, ub)
        if basis is not None:
            lp.load_basis(basis)
        it0 = lp.iterations
        limit = it0 + (iter_limit if iter_limit is not None else 10**7)
        r = lp.solve(time_limit=max(1.0, self.deadline - time.perf_counter()), max_iter=limit)
        self.lp_iters += lp.iterations - it0
        if r.status == "optimal":
            return "optimal", self.model.objective(r.x), r.x
        return r.status, np.inf, r.x

    def _fractional(self, x):
        xi = x[self.int_idx]
        f = xi - np.floor(xi)
        mask = (f > INT_TOL) & (f < 1 - INT_TOL)
        return self.int_idx[mask], f[mask]

    def _propagate(self, lb, ub):
        is_int = self.model.integer
        st, _ = propagate(self.Rp, self.Rj, self.Rv, self.model.rl, self.model.ru, lb, ub,
                          is_int, 5, FEAS_TOL)
        return st == 0

    # ---------------------------------------------------------------- branching
    def _pc(self, j, direction):
        cnt = self.pc_cnt[direction, j]
        if cnt > 0:
            return self.pc_sum[direction, j] / cnt
        tot = self.pc_cnt[direction].sum()
        return self.pc_sum[direction].sum() / tot if tot else 1.0

    def _record_pc(self, j, direction, frac, gain):
        if not np.isfinite(gain):
            return
        dist = frac if direction == 0 else 1 - frac
        self.pc_sum[direction, j] += max(gain, 0.0) / max(dist, 1e-6)
        self.pc_cnt[direction, j] += 1

    @staticmethod
    def _score(down, up):
        return max(down, 1e-6) * max(up, 1e-6)

    def _choose_branch(self, x, obj, frac_idx, frac, lb, ub):
        scores = np.array([self._score(self._pc(j, 0) * f, self._pc(j, 1) * (1 - f))
                           for j, f in zip(frac_idx, frac)])
        unreliable = [k for k, j in enumerate(frac_idx)
                      if min(self.pc_cnt[0, j], self.pc_cnt[1, j]) < self.reliability]
        if unreliable and self.strong_candidates > 0:
            # strong branching on the most promising unreliable candidates
            order = sorted(unreliable, key=lambda k: -scores[k])[: self.strong_candidates]
            basis = self.lp.get_basis()
            for k in order:
                j, f = frac_idx[k], frac[k]
                gains = []
                for direction in (0, 1):
                    l2, u2 = lb.copy(), ub.copy()
                    if direction == 0:
                        u2[j] = np.floor(x[j])
                    else:
                        l2[j] = np.ceil(x[j])
                    st, o, xc = self._solve_lp(l2, u2, basis, self.strong_iters)
                    if st == "infeasible":
                        g = np.inf
                    elif st == "optimal":
                        g = o - obj
                        self._record_pc(j, direction, f, g)
                        if xc is not None and not len(self._fractional(xc)[0]):
                            self._try_incumbent(xc, "strong branching")
                    else:
                        g = self._pc(j, direction) * (f if direction == 0 else 1 - f)
                    gains.append(g)
                if np.isinf(gains[0]) and np.isinf(gains[1]):
                    scores[k] = np.inf            # both children infeasible: prune this node
                else:
                    scores[k] = self._score(min(gains[0], 1e12), min(gains[1], 1e12))
                    if np.isinf(gains[0]) or np.isinf(gains[1]):
                        scores[k] = 1e30          # one side infeasible: branching fixes it
            self.lp.set_structural_bounds(lb, ub)
            self.lp.load_basis(basis)
        k = int(np.argmax(scores))
        return frac_idx[k], frac[k], scores[k]

    # ---------------------------------------------------------------- heuristics
    def _fix_and_solve(self, x, lb, ub):
        """Fix every integer to its rounded LP value and solve for the continuous part."""
        l2, u2 = lb.copy(), ub.copy()
        r = np.clip(np.round(x[self.int_idx]), lb[self.int_idx], ub[self.int_idx])
        l2[self.int_idx] = r
        u2[self.int_idx] = r
        if not self._propagate(l2, u2):
            return
        basis = self.lp.get_basis()
        st, o, xc = self._solve_lp(l2, u2, basis, 5000)
        if st == "optimal":
            self._try_incumbent(xc, "fix-and-solve")
        self.lp.set_structural_bounds(lb, ub)
        self.lp.load_basis(basis)

    def _dive(self, x, lb, ub, max_depth=None, lp_budget=None, up=False):
        """Diving with propagation and a single backtrack: repeatedly round one fractional
        integer by tightening one of its bounds (x_j <= floor or x_j >= ceil), propagate and
        re-solve the LP, until the LP is integral. When a step makes the LP infeasible, try the
        other direction once before giving up (Berthold, "Primal heuristics for mixed integer
        programs", ZIB diploma thesis 2006; Achterberg 2007, sec. 9.2)."""
        basis = self.lp.get_basis()
        l2, u2 = lb.copy(), ub.copy()
        is_int = self.model.integer
        budget = lp_budget if lp_budget is not None else max(1000, 2 * self.lp_iters)
        it0 = self.lp_iters
        found = False
        for _ in range(max_depth or 10 * len(self.int_idx) + 10):
            frac_idx, frac = self._fractional(x)
            if len(frac_idx) == 0:
                found = self._try_incumbent(x, "diving")
                break
            if self.lp_iters - it0 > budget or time.perf_counter() > self.deadline:
                break
            if up:
                # "up" diving: round the variable closest to its ceiling upward
                k = int(np.argmax(frac))
                go_up = True
            else:
                # fractional diving: round the least fractional variable to its nearest integer
                dist = np.minimum(frac, 1 - frac)
                k = int(np.argmin(dist))
                go_up = frac[k] >= 0.5
            j = frac_idx[k]
            ok = False
            for direction in (go_up, not go_up):         # single backtrack
                l3, u3 = l2.copy(), u2.copy()
                if direction:
                    l3[j] = np.ceil(x[j])
                else:
                    u3[j] = np.floor(x[j])
                if not self._propagate(l3, u3):
                    continue
                st, o, x3 = self._solve_lp(np.where(is_int, l3, self.orig.lb), np.where(is_int, u3, self.orig.ub))
                if st == "optimal" and o < self._cutoff():
                    l2, u2, x, ok = l3, u3, x3, True
                    break
            if not ok:
                break
        self.lp.set_structural_bounds(np.where(is_int, lb, self.orig.lb), np.where(is_int, ub, self.orig.ub))
        self.lp.load_basis(basis)
        return found

    # ---------------------------------------------------------------- root cutting planes
    def _root_cuts(self, lb, ub, obj, x):
        base = self.model
        m0 = base.m
        history = [obj]
        st = "optimal"
        for rnd in range(self.cut_rounds):
            if time.perf_counter() > self.deadline or not len(self._fractional(x)[0]):
                break
            cuts = gmi_cuts(self.lp, self.model, max_cuts=max(20, min(200, base.m // 2)))
            if not cuts:
                break
            model = add_cuts(self.model, cuts)
            lp = SimplexLP(_relaxed(model))
            lp.set_structural_bounds(lb, ub)
            r = lp.solve(time_limit=max(1.0, self.deadline - time.perf_counter()))
            self.lp_iters += lp.iterations
            if r.status != "optimal":
                break                       # keep the last good relaxation
            self.model, self.lp = model, lp
            obj, x = self.orig.objective(r.x), r.x
            history.append(obj)
            self._try_incumbent(x, f"cut round {rnd + 1}")
            if self.verbose:
                print(f"  cut round {rnd + 1}: {len(cuts)} cuts, bound {self.orig.sense * obj:+.10g}")
            # stop when three rounds together move the bound by less than 0.1% of its size
            if len(history) > 3 and history[-1] - history[-4] < 1e-3 * max(1.0, abs(history[-1])):
                break
            if self.model.m > 4 * m0 + 1000:
                break
        # drop the cuts that are slack at the final root point; they only slow the nodes down
        if self.model.m > m0:
            act = self.model.A.matvec(x)
            rows = np.arange(self.model.m)
            rl = np.where(rows < m0, 0.0, self.model.rl)
            keep = (rows < m0) | (act <= rl + 1e-6 * (1 + np.abs(rl)))
            if not keep.all():
                cuts = [(_row_dense(self.model, i), self.model.rl[i], 0.0) for i in np.flatnonzero(keep & (rows >= m0))]
                self.model = add_cuts(base, cuts)
                self.lp = SimplexLP(_relaxed(self.model))
                self.lp.set_structural_bounds(lb, ub)
                r = self.lp.solve(time_limit=max(1.0, self.deadline - time.perf_counter()))
                self.lp_iters += self.lp.iterations
                st, obj, x = r.status, self.orig.objective(r.x), r.x
            self.n_cuts = self.model.m - m0
        AT = self.model.AT
        self.Rp, self.Rj, self.Rv = AT.colptr, AT.rowidx, AT.vals
        return st, obj, x

    # ---------------------------------------------------------------- main loop
    def solve(self) -> Result:
        t0 = time.perf_counter()
        self.deadline = t0 + self.time_limit
        m = self.model
        lb0, ub0 = m.lb.copy(), m.ub.copy()
        if not self._propagate(lb0, ub0):
            return self._result("infeasible", t0, -np.inf)
        # continuous bounds found by propagation help the search but stay out of the LP
        is_int = m.integer
        lb0 = np.where(is_int, lb0, m.lb)
        ub0 = np.where(is_int, ub0, m.ub)
        st, obj, x = self._solve_lp(lb0, ub0)
        if st == "infeasible":
            return self._result("infeasible", t0, np.inf)
        if st == "dual_infeasible" or st == "unbounded":
            return self._result("unbounded", t0, -np.inf)
        if st != "optimal":
            return self._result(st, t0, -np.inf)
        if self.cut_rounds:
            self._try_incumbent(x, "root LP")
            st, obj, x = self._root_cuts(lb0, ub0, obj, x)
            if st != "optimal":
                return self._result("infeasible" if st == "infeasible" else st, t0, np.inf)
        root_bound = self._bound_up(obj)
        if self.verbose:
            print(f"  root LP {m.sense * obj:+.10g}, {len(self._fractional(x)[0])} fractional")
        self._try_incumbent(x, "root LP")
        self._fix_and_solve(x, lb0, ub0)
        self._dive(x, lb0.copy(), ub0.copy())
        if self.incumbent is not None:
            # a second dive rounding the other way often finds a different, better point
            self._dive(x, lb0.copy(), ub0.copy(), up=True)

        counter = itertools.count()
        heap = []          # (bound, -depth, tie, node): ties go to the deepest node
        node = _Node(root_bound, 0, lb0, ub0, None, None)
        current = (node, st, obj, x)        # a node whose LP is already solved
        best_bound = root_bound
        status = "optimal"
        while True:
            if time.perf_counter() > self.deadline:
                status = "time_limit"
                break
            if self.nodes >= self.node_limit:
                status = "node_limit"
                break
            if current is None:
                # drop nodes that can no longer beat the incumbent
                while heap and heap[0][0] >= self._cutoff():
                    heapq.heappop(heap)
                if not heap:
                    break
                bnd, _, _, node = heapq.heappop(heap)
                lb, ub = node.lb, node.ub
                if not self._propagate(lb, ub):
                    self.nodes += 1
                    continue
                st, obj, x = self._solve_lp(np.where(is_int, lb, m.lb), np.where(is_int, ub, m.ub),
                                            node.basis)
                self.nodes += 1
                if node.branch is not None and st in ("optimal", "infeasible"):
                    j, direction, f = node.branch
                    self._record_pc(j, direction, f, (obj if st == "optimal" else np.inf) - node.bound)
            else:
                node, st, obj, x = current
                current = None
                self.nodes += 1
            if st != "optimal":
                continue                     # infeasible (or failed) subtree
            obj_b = self._bound_up(obj)
            if obj_b >= self._cutoff():
                continue
            frac_idx, frac = self._fractional(x)
            if len(frac_idx) == 0:
                self._try_incumbent(x, "LP")
                continue
            if self._try_incumbent(np.where(np.isin(np.arange(m.n), self.int_idx), np.round(x), x), "rounding"):
                pass
            if (self.incumbent is None and self.nodes % 50 == 0) or self.nodes % 400 == 0:
                self._dive(x, node.lb.copy(), node.ub.copy(), up=bool(self.nodes % 800))
            if self.nodes % 100 == 0:
                self._fix_and_solve(x, np.where(is_int, node.lb, m.lb), np.where(is_int, node.ub, m.ub))
            lb, ub = node.lb, node.ub
            j, f, score = self._choose_branch(x, obj, frac_idx, frac, np.where(is_int, lb, m.lb),
                                              np.where(is_int, ub, m.ub))
            if np.isinf(score) and score > 0 and score != 1e30:
                continue                     # strong branching proved both children infeasible
            basis = self.lp.get_basis() if len(heap) < 20000 else None
            down_ub = ub.copy()
            down_ub[j] = np.floor(x[j])
            up_lb = lb.copy()
            up_lb[j] = np.ceil(x[j])
            down = _Node(obj_b, node.depth + 1, lb.copy(), down_ub, basis, (j, 0, f))
            up = _Node(obj_b, node.depth + 1, up_lb, ub.copy(), basis, (j, 1, f))
            # plunge into the child the pseudocosts favour; queue the other
            first, second = (up, down) if f >= 0.5 else (down, up)
            heapq.heappush(heap, (second.bound, -second.depth, next(counter), second))
            if not self._propagate(first.lb, first.ub):
                self.nodes += 1
                continue
            st2, obj2, x2 = self._solve_lp(np.where(is_int, first.lb, m.lb),
                                           np.where(is_int, first.ub, m.ub))
            if first.branch is not None and st2 in ("optimal", "infeasible"):
                jj, dd, ff = first.branch
                self._record_pc(jj, dd, ff, (obj2 if st2 == "optimal" else np.inf) - first.bound)
            # keep plunging only while the child stays close to the best open bound
            open_best = heap[0][0] if heap else np.inf
            if st2 == "optimal" and self._bound_up(obj2) <= open_best + 0.1 * abs(self._cutoff() - open_best
                                                                               if np.isfinite(self._cutoff()) else np.inf):
                current = (first, st2, obj2, x2)
            else:
                self.nodes += 1
                if st2 == "optimal":
                    first.bound = self._bound_up(obj2)
                    first.basis = self.lp.get_basis() if len(heap) < 20000 else None
                    heapq.heappush(heap, (first.bound, -first.depth, next(counter), first))
            if self.verbose and self.nodes % (25 if self.verbose > 1 else 200) == 0:
                lo_b = min(heap[0][0] if heap else np.inf, self._bound_up(obj))
                print(f"  nodes {self.nodes:7d}  open {len(heap):6d}  bound {m.sense * lo_b:+.10g}  "
                      f"incumbent {m.sense * self.inc_obj:+.10g}")
            # stop once the gap closes
            lo_b = min(heap[0][0] if heap else np.inf, self._bound_up(obj2) if current else np.inf)
            if np.isfinite(self.inc_obj) and lo_b < np.inf:
                if (self.inc_obj - lo_b) / max(1.0, abs(self.inc_obj)) <= self.gap_tol:
                    heap.clear()
                    current = None
                    break
        # final bound
        open_bounds = [h[0] for h in heap]
        if current is not None:
            open_bounds.append(self._bound_up(current[2]))
        best_bound = min(open_bounds) if open_bounds else self.inc_obj
        best_bound = min(best_bound, self.inc_obj)
        if status == "optimal" and self.incumbent is None:
            status = "infeasible"
        elif status in ("time_limit", "node_limit") and self.incumbent is not None:
            status = status if (self.inc_obj - best_bound) / max(1.0, abs(self.inc_obj)) > self.gap_tol else "optimal"
        return self._result(status, t0, best_bound)

    def _result(self, status, t0, bound):
        m = self.model
        x = self.incumbent
        obj = m.sense * self.inc_obj if x is not None else np.nan
        return Result(status=status, x=x, objective=obj, bound=m.sense * bound if np.isfinite(bound) else
                      m.sense * bound, iterations=self.lp_iters, nodes=self.nodes,
                      time=time.perf_counter() - t0, method="branch-and-bound",
                      info={"lp_iterations": self.lp_iters, "cuts": self.n_cuts})


def _row_dense(model: Model, i: int) -> np.ndarray:
    AT = model.AT
    g = np.zeros(model.n)
    g[AT.rowidx[AT.colptr[i]:AT.colptr[i + 1]]] = AT.vals[AT.colptr[i]:AT.colptr[i + 1]]
    return g


def _relaxed(model: Model) -> Model:
    return Model(name=model.name, c=model.c, A=model.A, rl=model.rl, ru=model.ru, lb=model.lb.copy(),
                 ub=model.ub.copy(), integer=np.zeros(model.n, dtype=bool), Q=None, c0=model.c0,
                 sense=model.sense, col_names=model.col_names, row_names=model.row_names)


def solve(model: Model, time_limit: float = np.inf, node_limit: int = 10**9, gap: float = 1e-4,
          verbose: bool = False, **kw) -> Result:
    if not model.is_mip:
        from ..lp import simplex
        return simplex.solve(model, time_limit=time_limit, verbose=verbose)
    return BranchAndBound(model, time_limit=time_limit, node_limit=node_limit, gap=gap,
                          verbose=verbose, **kw).solve()
