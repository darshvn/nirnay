"""Industrial case studies built from public data.

Each module exposes `build(**params) -> Model` and a `CASE_INFO` dict (title, sector, problem
class, sources with URLs and licences, and a list of what is real and what is assumed).
docs/CASE_STUDIES.md describes every case; data/cases/reference.csv holds the HiGHS optimum of
the default instance of each case, and data/cases/<name>.mps the model itself.

    from nirnay.cases import CASES, build
    model = build("unit_commitment")
"""
from __future__ import annotations

import importlib

CASES = [
    "refinery_planning",       # LP: crude selection, CDU cuts, conversion units, product blending
    "refinery_multiperiod",    # MILP: 12-month planning with inventories and unit mode changeovers
    "unit_commitment",         # MILP: pglib-uc RTS-GMLC unit commitment
    "economic_dispatch",       # QP: DC optimal power flow with quadratic costs (pglib-opf)
    "facility_location",       # MILP: OR-Library capacitated warehouse location
    "product_distribution",    # LP: Indian petroleum-product distribution, refineries to states
    "crude_scheduling",        # MILP: crude oil unloading and blending schedule
]


def module(name: str):
    return importlib.import_module(f"{__name__}.{name}")


def build(name: str, **params):
    return module(name).build(**params)


def info(name: str) -> dict:
    return module(name).CASE_INFO
