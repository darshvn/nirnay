"""Thermal unit commitment (MILP) on a PGLib-UC instance.

Data: IEEE PES Power Grid Library - Unit Commitment (pglib-uc), instance rts_gmlc/2020-01-27.json
(the RTS-GMLC system: thermal units with piecewise-linear production costs, three startup cost
categories, ramp limits, minimum up/down times and initial conditions; renewable units with hourly
availability; 48 hourly periods of demand and spinning-reserve requirement).

Formulation: the reference model shipped with pglib-uc (MODEL.tex / uc_model.py), equations
(1)-(24): the tight and compact formulation of Morales-Espana, Latorre and Ramos (2013) with the
piecewise production cost of Sridhar, Linderoth and Luedtke (2013). Equation numbers in the
comments below are those of uc_model.py.

    min  sum_g sum_t  c_g(t) + CP_g^1 u_g(t) + sum_s CS_g^s delta_g^s(t)                     (1)
    s.t. sum_g (p_g(t) + Pmin_g u_g(t)) + sum_w p_w(t) = D(t)                                   (2)
         sum_g r_g(t) >= R(t)                                                                    (3)
         initial up/down time, logical, startup-category and ramp conditions at t = 1        (4)-(10)
         must-run, u_g(t) - u_g(t-1) = v_g(t) - w_g(t), minimum up/down time             (11)-(14)
         startup category selection and link                                              (15)-(16)
         generation limits with startup/shutdown capability, ramp up/down                  (17)-(20)
         piecewise production: p = sum (P^l - P^1) lambda^l, c = sum (CP^l - CP^1) lambda^l,
         u = sum lambda^l                                                                  (21)-(23)
         Pmin_w(t) <= p_w(t) <= Pmax_w(t)                                                        (24)
"""
from __future__ import annotations

import json
from pathlib import Path

from ..model import INF
from ._builder import RAW, Builder

CASE_INFO = {
    "title": "Thermal unit commitment, RTS-GMLC system, 48 hours",
    "sector": "Power system scheduling",
    "class": "MILP",
    "sources": [
        {"what": "All generator, demand, reserve and renewable data",
         "name": "IEEE PES PGLib-UC, rts_gmlc/2020-01-27.json",
         "url": "https://raw.githubusercontent.com/power-grid-lib/pglib-uc/master/rts_gmlc/2020-01-27.json",
         "licence": "CC BY 4.0 (repository LICENSE)"},
        {"what": "Reference formulation, equations (1)-(24)",
         "name": "pglib-uc MODEL.tex and uc_model.py",
         "url": "https://github.com/power-grid-lib/pglib-uc/blob/master/uc_model.py",
         "licence": "CC BY 4.0"},
        {"what": "Formulation references",
         "name": "G. Morales-Espana, J.M. Latorre, A. Ramos, Tight and compact MILP formulation for "
                 "the thermal unit commitment problem, IEEE Trans. Power Systems 28(4):4897-4908, "
                 "2013; S. Sridhar, J. Linderoth, J. Luedtke, Locally ideal formulations for "
                 "piecewise linear functions with indicator variables, Oper. Res. Lett. "
                 "41(6):627-632, 2013 (doi:10.1016/j.orl.2013.08.010)",
         "url": "https://doi.org/10.1109/TPWRS.2013.2251373", "licence": "citation only"},
        {"what": "Underlying test system",
         "name": "RTS-GMLC, Reliability Test System - Grid Modernization Lab Consortium",
         "url": "https://github.com/GridMod/RTS-GMLC", "licence": "see repository"},
    ],
    "real": ["every parameter (demand, reserves, unit limits, ramp rates, min up/down times, "
             "piecewise cost points, startup cost categories, initial status, renewable "
             "availability) is read unchanged from the pglib-uc JSON file"],
    "assumed": ["none in the data; the formulation is the one published with the benchmark",
                "the HiGHS reference is solved to a 1% relative gap, the setting of pglib-uc's "
                "reference script; it is not a proven optimum (HiGHS 1.15.1 reaches only a 0.52% "
                "gap in 600 s with a 1e-6 target)"],
}

# pglib-uc's reference script solves to a 1% relative gap; with a 1e-6 target HiGHS 1.15 stalls at
# a 0.52% gap after 600 s on this instance, so the reference optimum is the 1%-gap one.
REFERENCE = {"mip_rel_gap": 0.01, "time_limit": 3600.0}

DEFAULT_FILE = RAW / "supply" / "pglib-uc_rts_gmlc_2020-01-27.json"


def build(path=None, periods: int | None = None):
    data = json.load(open(path or DEFAULT_FILE))
    T = data["time_periods"] if periods is None else min(periods, data["time_periods"])
    thermal = data["thermal_generators"]
    renew = data["renewable_generators"]
    stem = Path(path or DEFAULT_FILE).stem.replace("pglib-uc_", "")
    B = Builder(f"pglib_uc_{stem}" + (f"_T{T}" if periods else ""))
    TT = range(1, T + 1)

    cg, pg, rg, ug, vg, wg, dg, lg = {}, {}, {}, {}, {}, {}, {}, {}
    for g, gen in thermal.items():
        cp1 = gen["piecewise_production"][0]["cost"]
        for t in TT:
            cg[g, t] = B.var(f"c_{g}_{t}", -INF, INF, cost=1.0)
            pg[g, t] = B.var(f"p_{g}_{t}")
            rg[g, t] = B.var(f"r_{g}_{t}")
            ug[g, t] = B.binary(f"u_{g}_{t}", cost=cp1)
            vg[g, t] = B.binary(f"v_{g}_{t}")
            wg[g, t] = B.binary(f"w_{g}_{t}")
            for s, st in enumerate(gen["startup"]):
                dg[g, s, t] = B.binary(f"d_{g}_{s}_{t}", cost=st["cost"])
            for l in range(len(gen["piecewise_production"])):
                lg[g, l, t] = B.var(f"l_{g}_{l}_{t}", 0.0, 1.0)
    pw = {}
    for w, rgen in renew.items():
        for t in TT:
            pw[w, t] = B.var(f"pw_{w}_{t}", rgen["power_output_minimum"][t - 1],
                             rgen["power_output_maximum"][t - 1])          # (24)

    for t in TT:
        cols, vals = [], []
        for g, gen in thermal.items():
            cols += [pg[g, t], ug[g, t]]
            vals += [1.0, gen["power_output_minimum"]]
        for w in renew:
            cols.append(pw[w, t])
            vals.append(1.0)
        B.eq(f"demand_{t}", cols, vals, data["demand"][t - 1])            # (2)
        B.ge(f"reserves_{t}", [rg[g, t] for g in thermal], 1.0, data["reserves"][t - 1])  # (3)

    for g, gen in thermal.items():
        on0 = gen["unit_on_t0"]
        pmin, pmax = gen["power_output_minimum"], gen["power_output_maximum"]
        p0 = gen["power_output_t0"]
        su = max(pmax - gen["ramp_startup_limit"], 0.0)
        sd = max(pmax - gen["ramp_shutdown_limit"], 0.0)
        ru, rd = gen["ramp_up_limit"], gen["ramp_down_limit"]
        starts = gen["startup"]
        pieces = gen["piecewise_production"]

        if on0 == 1:
            k = min(gen["time_up_minimum"] - gen["time_up_t0"], T)
            if gen["time_up_minimum"] - gen["time_up_t0"] >= 1:       # (4): u = 1 for t <= k
                B.eq(f"uptimet0_{g}", [ug[g, t] for t in range(1, k + 1)], 1.0, float(k))
        else:
            k = min(gen["time_down_minimum"] - gen["time_down_t0"], T)
            if gen["time_down_minimum"] - gen["time_down_t0"] >= 1:   # (5)
                B.eq(f"downtimet0_{g}", [ug[g, t] for t in range(1, k + 1)], 1.0, 0.0)
        B.eq(f"logicalt0_{g}", [ug[g, 1], vg[g, 1], wg[g, 1]], [1.0, -1.0, 1.0], float(on0))  # (6)

        cols = []                                                           # (7)
        for s in range(len(starts) - 1):
            lo = max(1, starts[s + 1]["lag"] - gen["time_down_t0"] + 1)
            hi = min(starts[s + 1]["lag"] - 1, T)
            cols += [dg[g, s, t] for t in range(lo, hi + 1)]
        if cols:
            B.eq(f"startupt0_{g}", cols, 1.0, 0.0)

        prev = on0 * (p0 - pmin)
        B.le(f"rampupt0_{g}", [pg[g, 1], rg[g, 1]], [1.0, 1.0], ru + prev)          # (8)
        B.le(f"rampdownt0_{g}", [pg[g, 1]], [-1.0], rd - prev)                       # (9)
        # (10): prev <= on0 (pmax - pmin) - sd w(1)
        if sd != 0.0:
            B.le(f"shutdownt0_{g}", [wg[g, 1]], [sd], on0 * (pmax - pmin) - prev)
        elif prev > on0 * (pmax - pmin):
            raise ValueError(f"generator {g}: infeasible initial condition (10)")

        UT = min(gen["time_up_minimum"], T)
        DT = min(gen["time_down_minimum"], T)
        for t in TT:
            if gen["must_run"]:                                               # (11)
                B.set_bounds(ug[g, t], lb=1.0)
            if t > 1:                                                         # (12)
                B.eq(f"logical_{g}_{t}", [ug[g, t], ug[g, t - 1], vg[g, t], wg[g, t]],
                     [1.0, -1.0, -1.0, 1.0], 0.0)
            if t >= UT:                                                       # (13)
                B.le(f"uptime_{g}_{t}", [vg[g, i] for i in range(t - UT + 1, t + 1)] + [ug[g, t]],
                     [1.0] * UT + [-1.0], 0.0)
            if t >= DT:                                                       # (14)
                B.le(f"downtime_{g}_{t}", [wg[g, i] for i in range(t - DT + 1, t + 1)] + [ug[g, t]],
                     [1.0] * DT + [1.0], 1.0)
            B.eq(f"startup_select_{g}_{t}", [vg[g, t]] + [dg[g, s, t] for s in range(len(starts))],
                 [1.0] + [-1.0] * len(starts), 0.0)                           # (16)
            B.le(f"gen_limit1_{g}_{t}", [pg[g, t], rg[g, t], ug[g, t], vg[g, t]],
                 [1.0, 1.0, -(pmax - pmin), su], 0.0)                         # (17)
            if t < T:                                                         # (18)
                B.le(f"gen_limit2_{g}_{t}", [pg[g, t], rg[g, t], ug[g, t], wg[g, t + 1]],
                     [1.0, 1.0, -(pmax - pmin), sd], 0.0)
            if t > 1:
                B.le(f"ramp_up_{g}_{t}", [pg[g, t], rg[g, t], pg[g, t - 1]], [1.0, 1.0, -1.0], ru)  # (19)
                B.le(f"ramp_down_{g}_{t}", [pg[g, t - 1], pg[g, t]], [1.0, -1.0], rd)               # (20)
            L = range(len(pieces))
            B.eq(f"power_select_{g}_{t}", [pg[g, t]] + [lg[g, l, t] for l in L],
                 [1.0] + [-(pieces[l]["mw"] - pieces[0]["mw"]) for l in L], 0.0)       # (21)
            B.eq(f"cost_select_{g}_{t}", [cg[g, t]] + [lg[g, l, t] for l in L],
                 [1.0] + [-(pieces[l]["cost"] - pieces[0]["cost"]) for l in L], 0.0)   # (22)
            B.eq(f"on_select_{g}_{t}", [ug[g, t]] + [lg[g, l, t] for l in L],
                 [1.0] + [-1.0] * len(pieces), 0.0)                                     # (23)

        for s in range(len(starts) - 1):                                      # (15)
            for t in TT:
                if t >= starts[s + 1]["lag"]:
                    ws = [wg[g, t - i] for i in range(starts[s]["lag"], starts[s + 1]["lag"])]
                    B.le(f"startup_allowed_{g}_{s}_{t}", [dg[g, s, t]] + ws,
                         [1.0] + [-1.0] * len(ws), 0.0)
    return B.to_model()
