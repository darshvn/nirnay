"""The problem every algorithm in NIRNAY reads.

    minimise    c'x + 1/2 x'Qx + c0
    subject to  rl <= A x <= ru
                lb <=   x <= ub
                x_j integer for j in `integer`

A maximisation problem is stored negated (`sense = -1` remembers it, so reported objectives
and duals come back in the user's sense). Infinite bounds are +-inf. A row with rl == ru is an
equality. Q is symmetric and stored in full (both triangles); it is None for an LP or MILP.

A is kept in compressed sparse column form (CSC) because the simplex method, the LU factor and
presolve all walk columns. Row-wise access (CSR) is built once on demand and cached.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

INF = np.inf


@dataclass
class CSC:
    """Compressed sparse column matrix. Plain arrays, so Numba kernels can take them directly."""
    m: int
    n: int
    colptr: np.ndarray      # int64, length n + 1
    rowidx: np.ndarray      # int64, length nnz
    vals: np.ndarray        # float64, length nnz

    @property
    def nnz(self) -> int:
        return int(self.colptr[-1])

    @staticmethod
    def from_triplets(m: int, n: int, rows, cols, vals, sum_duplicates: bool = True) -> "CSC":
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        vals = np.asarray(vals, dtype=np.float64)
        order = np.lexsort((rows, cols))
        rows, cols, vals = rows[order], cols[order], vals[order]
        if sum_duplicates and len(rows):
            key = cols * max(m, 1) + rows
            keep = np.ones(len(key), dtype=bool)
            keep[1:] = key[1:] != key[:-1]
            if not keep.all():
                starts = np.flatnonzero(keep)
                vals = np.add.reduceat(vals, starts)
                rows, cols = rows[starts], cols[starts]
        nz = vals != 0.0
        rows, cols, vals = rows[nz], cols[nz], vals[nz]
        colptr = np.zeros(n + 1, dtype=np.int64)
        np.add.at(colptr, cols + 1, 1)
        np.cumsum(colptr, out=colptr)
        return CSC(m, n, colptr, rows, vals)

    @staticmethod
    def empty(m: int, n: int) -> "CSC":
        return CSC(m, n, np.zeros(n + 1, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0))

    def col(self, j: int):
        a, b = self.colptr[j], self.colptr[j + 1]
        return self.rowidx[a:b], self.vals[a:b]

    def transpose(self) -> "CSC":
        """A' in CSC form, which is A in CSR form: row i of A is column i of the result."""
        cols = np.repeat(np.arange(self.n, dtype=np.int64), np.diff(self.colptr))
        return CSC.from_triplets(self.n, self.m, cols, self.rowidx, self.vals, sum_duplicates=False)

    @property
    def colidx(self) -> np.ndarray:
        """Column index of every stored entry (cached; the pattern never changes in place)."""
        c = self.__dict__.get("_colidx")
        if c is None or len(c) != self.nnz:
            c = np.repeat(np.arange(self.n, dtype=np.int64), np.diff(self.colptr))
            self.__dict__["_colidx"] = c
        return c

    def matvec(self, x: np.ndarray) -> np.ndarray:
        return np.bincount(self.rowidx, weights=self.vals * x[self.colidx], minlength=self.m)

    def rmatvec(self, y: np.ndarray) -> np.ndarray:
        """A' y, one dot product per column (empty columns give 0)."""
        return np.bincount(self.colidx, weights=self.vals * y[self.rowidx], minlength=self.n)

    def to_dense(self) -> np.ndarray:
        D = np.zeros((self.m, self.n))
        for j in range(self.n):
            r, v = self.col(j)
            D[r, j] = v
        return D

    def copy(self) -> "CSC":
        return CSC(self.m, self.n, self.colptr.copy(), self.rowidx.copy(), self.vals.copy())


@dataclass
class Model:
    name: str
    c: np.ndarray
    A: CSC
    rl: np.ndarray
    ru: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    integer: np.ndarray                      # bool per column
    Q: CSC | None = None
    c0: float = 0.0
    sense: int = 1                           # +1 minimise, -1 the file asked to maximise
    col_names: list = field(default_factory=list)
    row_names: list = field(default_factory=list)
    _AT: CSC | None = field(default=None, repr=False)

    @property
    def m(self) -> int:
        return self.A.m

    @property
    def n(self) -> int:
        return self.A.n

    @property
    def is_mip(self) -> bool:
        return bool(self.integer.any())

    @property
    def is_qp(self) -> bool:
        return self.Q is not None and self.Q.nnz > 0

    @property
    def AT(self) -> CSC:
        """Row-wise access to A, built once."""
        if self._AT is None:
            self._AT = self.A.transpose()
        return self._AT

    def objective(self, x: np.ndarray) -> float:
        val = float(self.c @ x) + self.c0
        if self.is_qp:
            val += 0.5 * float(x @ self.Q.matvec(x))
        return val

    def user_objective(self, x: np.ndarray) -> float:
        """Objective in the sense the file asked for (a maximisation is reported as a maximum)."""
        return self.sense * self.objective(x)

    def violation(self, x: np.ndarray) -> dict:
        """Largest absolute violations of rows, bounds and integrality."""
        Ax = self.A.matvec(x)
        row = np.maximum(0.0, np.maximum(self.rl - Ax, Ax - self.ru))
        bnd = np.maximum(0.0, np.maximum(self.lb - x, x - self.ub))
        integ = np.abs(x[self.integer] - np.round(x[self.integer])) if self.is_mip else np.zeros(0)
        return {
            "row": float(row.max(initial=0.0)),
            "bound": float(bnd.max(initial=0.0)),
            "integrality": float(integ.max(initial=0.0)),
        }

    def summary(self) -> str:
        kind = "MIQP" if self.is_qp and self.is_mip else "QP" if self.is_qp else "MILP" if self.is_mip else "LP"
        n_int = int(self.integer.sum())
        n_eq = int(np.sum(self.rl == self.ru))
        return (f"{self.name}: {kind}, {self.m} rows ({n_eq} equalities), {self.n} columns"
                f"{f' ({n_int} integer)' if n_int else ''}, {self.A.nnz} nonzeros"
                f"{f', Q with {self.Q.nnz} nonzeros' if self.is_qp else ''}")

    def validate(self) -> None:
        assert self.c.shape == (self.n,)
        assert self.rl.shape == self.ru.shape == (self.m,)
        assert self.lb.shape == self.ub.shape == self.integer.shape == (self.n,)
        if np.any(self.rl > self.ru):
            bad = int(np.argmax(self.rl > self.ru))
            raise ValueError(f"row {self.row_names[bad] if self.row_names else bad}: lower > upper")
        if np.any(self.lb > self.ub):
            bad = int(np.argmax(self.lb > self.ub))
            raise ValueError(f"column {self.col_names[bad] if self.col_names else bad}: lower > upper")


# ---------------------------------------------------------------------------------------------

@dataclass
class Result:
    """What every solver returns. Values are in the user's objective sense."""
    status: str                              # optimal | infeasible | unbounded | feasible | time_limit
    #                                          | iteration_limit | node_limit | numerical_error
    x: np.ndarray | None = None
    y: np.ndarray | None = None              # row duals
    z: np.ndarray | None = None              # reduced costs
    objective: float = np.nan
    bound: float = np.nan                    # best proven bound (MIP); equals objective at optimality
    iterations: int = 0
    nodes: int = 0
    time: float = 0.0
    method: str = ""
    info: dict = field(default_factory=dict)

    @property
    def gap(self) -> float:
        if not np.isfinite(self.objective) or not np.isfinite(self.bound):
            return np.inf
        return abs(self.objective - self.bound) / max(1.0, abs(self.objective))

    def __str__(self) -> str:
        parts = [f"status {self.status}", f"objective {self.objective:.10g}"]
        if self.nodes:
            parts += [f"bound {self.bound:.10g}", f"gap {100 * self.gap:.3g}%", f"{self.nodes} nodes"]
        parts += [f"{self.iterations} iterations", f"{self.time:.3f}s", self.method]
        return ", ".join(parts)
