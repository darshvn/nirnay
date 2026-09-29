"""A small helper for writing case-study generators: named variable blocks, rows added as
(column indices, coefficients) and a final `to_model()` that returns a nirnay Model.

It is deliberately plain: arrays and triplet lists, no expression overloading.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..model import CSC, INF, Model

RAW = Path(__file__).resolve().parents[2] / "data" / "cases" / "raw"


class Builder:
    def __init__(self, name: str, sense: int = 1):
        self.name = name
        self.sense = sense              # +1 minimise, -1 maximise (objective given in user sense)
        self.c: list[float] = []
        self.lb: list[float] = []
        self.ub: list[float] = []
        self.integer: list[bool] = []
        self.col_names: list[str] = []
        self.rl: list[float] = []
        self.ru: list[float] = []
        self.row_names: list[str] = []
        self._r: list[np.ndarray] = []
        self._c: list[np.ndarray] = []
        self._v: list[np.ndarray] = []
        self._q: list[tuple[int, int, float]] = []
        self.c0 = 0.0

    # ---- columns -----------------------------------------------------------------------------
    def var(self, name: str, lb: float = 0.0, ub: float = INF, cost: float = 0.0,
            integer: bool = False) -> int:
        j = len(self.c)
        self.c.append(float(cost))
        self.lb.append(float(lb))
        self.ub.append(float(ub))
        self.integer.append(bool(integer))
        self.col_names.append(name)
        return j

    def binary(self, name: str, cost: float = 0.0) -> int:
        return self.var(name, 0.0, 1.0, cost, integer=True)

    def set_bounds(self, j: int, lb: float | None = None, ub: float | None = None) -> None:
        if lb is not None:
            self.lb[j] = float(lb)
        if ub is not None:
            self.ub[j] = float(ub)

    def add_cost(self, j: int, cost: float) -> None:
        self.c[j] += float(cost)

    def add_quad(self, i: int, j: int, v: float) -> None:
        """Adds v * x_i * x_j to the objective (i == j gives v * x_i^2)."""
        if i == j:
            self._q.append((i, i, 2.0 * v))          # 1/2 x'Qx convention
        else:
            self._q.append((i, j, v))
            self._q.append((j, i, v))

    # ---- rows --------------------------------------------------------------------------------
    def row(self, name: str, cols, coefs, lo: float = -INF, up: float = INF) -> int:
        i = len(self.rl)
        cols = np.asarray(cols, dtype=np.int64)
        coefs = np.asarray(coefs, dtype=np.float64)
        if coefs.ndim == 0:
            coefs = np.full(len(cols), float(coefs))
        self._r.append(np.full(len(cols), i, dtype=np.int64))
        self._c.append(cols)
        self._v.append(coefs)
        self.rl.append(float(lo))
        self.ru.append(float(up))
        self.row_names.append(name)
        return i

    def le(self, name, cols, coefs, rhs):
        return self.row(name, cols, coefs, -INF, rhs)

    def ge(self, name, cols, coefs, rhs):
        return self.row(name, cols, coefs, rhs, INF)

    def eq(self, name, cols, coefs, rhs):
        return self.row(name, cols, coefs, rhs, rhs)

    # ---- output ------------------------------------------------------------------------------
    def to_model(self) -> Model:
        n, m = len(self.c), len(self.rl)
        if self._r:
            rows = np.concatenate(self._r)
            cols = np.concatenate(self._c)
            vals = np.concatenate(self._v)
        else:
            rows = cols = np.zeros(0, dtype=np.int64)
            vals = np.zeros(0)
        A = CSC.from_triplets(m, n, rows, cols, vals)
        c = np.array(self.c) * self.sense
        Q = None
        if self._q:
            qi, qj, qv = zip(*self._q)
            Q = CSC.from_triplets(n, n, qi, qj, np.array(qv) * self.sense)
        model = Model(name=self.name, c=c, A=A, rl=np.array(self.rl), ru=np.array(self.ru),
                      lb=np.array(self.lb), ub=np.array(self.ub),
                      integer=np.array(self.integer, dtype=bool), Q=Q, c0=self.sense * self.c0,
                      sense=self.sense, col_names=list(self.col_names),
                      row_names=list(self.row_names))
        model.validate()
        return model
