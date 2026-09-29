"""Capacitated facility (warehouse) location, MILP, OR-Library instances of J.E. Beasley.

Decision: which depots to open (fixed cost, capacity) and how to split each customer's demand
among the open depots, to minimise fixed plus allocation cost. This is the core of distribution
network design for petroleum-product depots and terminals.

    min   sum_i f_i y_i + sum_ij c_ij x_ij
    s.t.  sum_i x_ij = 1                      every customer j (x_ij = fraction of j's demand from i)
          sum_j d_j x_ij <= s_i y_i           every depot i
          x_ij <= y_i                         every pair (strong linking; valid, tightens the LP)
          0 <= x_ij,  y_i in {0,1}

c_ij is, as in the data files, the cost of allocating ALL of customer j's demand to depot i, so a
fraction x_ij costs c_ij x_ij. Customer demand may be split (this is the problem the published
optimal values in capopt refer to).

Default instance: capb with capacity 5000 (100 depots, 1000 customers): 100 binaries and
100 000 continuous variables. The optimal value 13656379.578 is published in OR-Library's capopt;
HiGHS 1.15.1 proves 13656379.5776 in about 9 minutes. (capa with capacity 8000 is harder: HiGHS
finds the published optimum 19240822.449 but does not close the gap within 30 minutes.)
"""
from __future__ import annotations

import numpy as np

from ._builder import RAW, Builder

CASE_INFO = {
    "title": "Capacitated facility location, OR-Library capb (100 depots x 1000 customers)",
    "sector": "Supply chain / logistics network design",
    "class": "MILP",
    "sources": [
        {"what": "Depot capacities and fixed costs, customer demands, allocation costs",
         "name": "OR-Library, capacitated warehouse location, file capb",
         "url": "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/capb.txt",
         "licence": "MIT (OR-Library legal page, https://people.brunel.ac.uk/~mastjjb/jeb/orlib/legal.html)"},
        {"what": "Published optimal values (capb, capacity 5000: 13656379.578)",
         "name": "OR-Library capopt",
         "url": "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/capopt.txt",
         "licence": "MIT"},
        {"what": "Problem definition and the capacity values for capa/capb/capc",
         "name": "J.E. Beasley, An algorithm for solving large capacitated warehouse location "
                 "problems, European Journal of Operational Research 33 (1988) 314-325",
         "url": "https://doi.org/10.1016/0377-2217(88)90175-0", "licence": "citation only"},
    ],
    "real": ["all numbers from the OR-Library file; the capacity 5000 for capb is one of the four "
             "values listed in capopt (and Table 1 of Beasley 1988)"],
    "assumed": ["the data are a standard benchmark (derived from the Akinc-Khumawala test "
                "problems), not an Indian network; no Indian depot-level dataset with costs "
                "and capacities is public"],
}


def read_orlib_cap(path, capacity: float | None = None):
    tok = open(path).read().split()
    m, n = int(tok[0]), int(tok[1])
    k = 2
    cap, fixed = np.zeros(m), np.zeros(m)
    for i in range(m):
        if tok[k].lower() == "capacity":
            if capacity is None:
                raise ValueError("this file needs a capacity value (see capopt)")
            cap[i] = capacity
        else:
            cap[i] = float(tok[k])
        fixed[i] = float(tok[k + 1])
        k += 2
    dem = np.zeros(n)
    cost = np.zeros((m, n))
    for j in range(n):
        dem[j] = float(tok[k])
        cost[:, j] = [float(v) for v in tok[k + 1:k + 1 + m]]
        k += 1 + m
    return cap, fixed, dem, cost


# published optima (OR-Library capopt) for the instances used in tests
PUBLISHED = {("cap41", None): 1040444.375, ("cap131", None): 793439.562,
             ("capa", 8000.0): 19240822.449, ("capb", 5000.0): 13656379.578}


def build(instance: str = "capb", capacity: float | None = 5000.0, strong: bool = True):
    path = RAW / "supply" / f"orlib_{instance}.txt"
    cap, fixed, dem, cost = read_orlib_cap(path, capacity if instance.startswith("capa")
                                           or instance in ("capb", "capc") else None)
    m, n = len(cap), len(dem)
    B = Builder(f"orlib_{instance}" + (f"_{int(capacity)}" if instance in ("capa", "capb", "capc") else ""))
    y = [B.binary(f"open_{i + 1}", cost=fixed[i]) for i in range(m)]
    x = np.empty((m, n), dtype=np.int64)
    for i in range(m):
        for j in range(n):
            x[i, j] = B.var(f"x_{i + 1}_{j + 1}", 0.0, 1.0, cost=cost[i, j])
    for j in range(n):
        B.eq(f"serve_{j + 1}", x[:, j], 1.0, 1.0)
    for i in range(m):
        B.le(f"cap_{i + 1}", list(x[i, :]) + [y[i]], list(dem) + [-cap[i]], 0.0)
    if strong:
        for i in range(m):
            for j in range(n):
                B.le(f"link_{i + 1}_{j + 1}", [x[i, j], y[i]], [1.0, -1.0], 0.0)
    return B.to_model()
