"""Economic dispatch with a DC network (DC optimal power flow) as a convex QP.

Data: IEEE PES Power Grid Library, PGLib-OPF v23.07, case pglib_opf_case2000_goc (a synthetic
2000-bus Texas-footprint grid from the ARPA-E Grid Optimization Competition), which carries
quadratic generator cost curves c2 p^2 + c1 p + c0 in MATPOWER gencost format.

Formulation (the DC model used by PowerModels.jl `DCPPowerModel`, which produced the DC column
of pglib-opf's BASELINE.md, so the optimum can be checked against a published value):

    min   sum_g  c2_g p_g^2 + c1_g p_g + c0_g                      ($/h, p in MW)
    s.t.  sum_{g at i} p_g - sum_{branch (i,j)} f_ij + sum_{branch (j,i)} f_ji = Pd_i + Gs_i
          f_ij = baseMVA * b_ij * (theta_i - theta_j)              for every in-service branch
          -rateA_ij <= f_ij <= rateA_ij                             (when rateA > 0)
          angmin_ij <= theta_i - theta_j <= angmax_ij
          Pmin_g <= p_g <= Pmax_g,   theta_ref = 0

with b_ij = -Im(1 / (r_ij + j x_ij)) = x / (r^2 + x^2), PowerModels' series susceptance. As in
PowerModels' DC model the tap ratio is not applied and the phase shift is. With these conventions the
HiGHS optimum is 943042.2 $/h, which matches the published baseline 9.4304e+05 $/h.
"""
from __future__ import annotations

import numpy as np

from ..model import INF
from ._builder import RAW, Builder
from ._matpower import read_matpower

CASE_INFO = {
    "title": "Economic dispatch with DC network constraints (DC-OPF), 2000-bus grid",
    "sector": "Power system dispatch",
    "class": "QP",
    "sources": [
        {"what": "Network, loads, generator limits and quadratic cost curves",
         "name": "IEEE PES PGLib-OPF v23.07, pglib_opf_case2000_goc.m",
         "url": "https://raw.githubusercontent.com/power-grid-lib/pglib-opf/master/pglib_opf_case2000_goc.m",
         "licence": "CC BY 4.0 (file header and repository LICENSE)"},
        {"what": "Reference DC-OPF objective 9.4304e+05 $/h for this case (PowerModels.jl v0.19.9)",
         "name": "pglib-opf BASELINE.md",
         "url": "https://github.com/power-grid-lib/pglib-opf/blob/master/BASELINE.md",
         "licence": "CC BY 4.0"},
        {"what": "Original synthetic network",
         "name": "Birchfield et al., Grid Structural Characteristics as Validation Criteria for "
                 "Synthetic Networks, IEEE Trans. Power Systems 32(4), 2017",
         "url": "https://doi.org/10.1109/TPWRS.2016.2616385", "licence": "citation only"},
    ],
    "real": ["every bus load, generator limit, cost coefficient, branch impedance, thermal "
             "rating and angle limit is read unchanged from the PGLib file"],
    "assumed": ["DC power-flow approximation (lossless, flat voltage) - a modelling choice, "
                "the standard one for economic dispatch",
                "single period (the file is one operating snapshot)"],
}

DEFAULT_FILE = RAW / "supply" / "pglib-opf_pglib_opf_case2000_goc.m"


def build(path=None, angle_limits: bool = True):
    d = read_matpower(path or DEFAULT_FILE)
    base = d["baseMVA"]
    bus, gen, cost, br = d["bus"], d["gen"], d["gencost"], d["branch"]
    nb = len(bus)
    idx = {int(b): k for k, b in enumerate(bus[:, 0])}
    ref = [k for k in range(nb) if int(bus[k, 1]) == 3]

    B = Builder("pglib_opf_case2000_goc_dcopf")
    theta = [B.var(f"va_{int(bus[k, 0])}", -INF, INF) for k in range(nb)]
    for k in ref:
        B.set_bounds(theta[k], 0.0, 0.0)

    inj: list[list[tuple[int, float]]] = [[] for _ in range(nb)]
    on = gen[:, 7] > 0
    for g in np.flatnonzero(on):
        pmin, pmax = gen[g, 9], gen[g, 8]
        p = B.var(f"pg_{g + 1}_bus{int(gen[g, 0])}", pmin, pmax)
        model, ncost = int(cost[g, 0]), int(cost[g, 3])
        if model != 2:
            raise ValueError("only polynomial gencost is handled")
        coef = cost[g, 4:4 + ncost][::-1]          # c0, c1, c2 ...
        if ncost > 3 and np.any(coef[3:] != 0):
            raise ValueError("cost polynomial above degree 2")
        c0 = coef[0] if ncost >= 1 else 0.0
        c1 = coef[1] if ncost >= 2 else 0.0
        c2 = coef[2] if ncost >= 3 else 0.0
        B.add_cost(p, c1)
        if c2:
            B.add_quad(p, p, c2)
        B.c0 += c0
        inj[idx[int(gen[g, 0])]].append((p, 1.0))

    for l in range(len(br)):
        if br[l, 10] <= 0:
            continue
        f, t = idx[int(br[l, 0])], idx[int(br[l, 1])]
        r, x = br[l, 2], br[l, 3]
        b = x / (r * r + x * x)
        shift = np.deg2rad(br[l, 9])
        rate = br[l, 5]
        lo, up = (-rate, rate) if rate > 0 else (-INF, INF)
        fl = B.var(f"pf_{l + 1}_{int(br[l, 0])}_{int(br[l, 1])}", lo, up)
        # f - base*b*(theta_f - theta_t) = -base*b*shift
        B.eq(f"ohm_{l + 1}", [fl, theta[f], theta[t]], [1.0, -base * b, base * b], -base * b * shift)
        inj[f].append((fl, -1.0))
        inj[t].append((fl, 1.0))
        if angle_limits:
            amin, amax = br[l, 11], br[l, 12]
            if np.isfinite(amin) and np.isfinite(amax) and (amin > -360 or amax < 360):
                B.row(f"angdiff_{l + 1}", [theta[f], theta[t]], [1.0, -1.0],
                      np.deg2rad(amin), np.deg2rad(amax))

    for k in range(nb):
        load = bus[k, 2] + bus[k, 4]                      # Pd + Gs (shunt at 1 p.u. voltage)
        cols = [c for c, _ in inj[k]]
        vals = [v for _, v in inj[k]]
        B.eq(f"bal_{int(bus[k, 0])}", cols, vals, load)
    return B.to_model()
