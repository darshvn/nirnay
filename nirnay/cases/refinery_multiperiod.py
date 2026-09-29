"""Twelve-month refinery production plan with crude parcels, product inventories and FCC mode
changeovers (MILP). FY 2024-25, April to March.

Each month t contains the whole network of refinery_planning (8 SPR crudes by default, 3 CDUs, reformer, FCC,
BS-VI blending). On top of it:

  * crude is bought in whole parcels:  charge_ct = parcel_kt_c * n_ct,  n_ct integer
    (parcel = 1 million barrels; parcel_kt_c = 158.987 m3 x density_c)
  * at most K crudes per month (tank segregation):  n_ct <= N_max * w_ct,  sum_c w_ct <= K,  w binary
  * CDU capacity scaled by days in the month
  * FCC runs in at most one severity per month:  F_t,s <= Fmax * y_t,s,  sum_s y_t,s <= 1,  y binary
  * severity changeovers:  z_t >= y_t,s - y_t-1,s,  sum_t z_t <= Z_max
  * finished-product inventories for MS 91, MS 95, HSD and ATF:
        I_pt = I_p,t-1 + production_pt - sales_pt,   I_p0 = 0,  I_p,12 = 0
        sum_p I_pt / rho_min_p <= storage (thousand m3)
  * monthly market limit:  sales_pt <= MRPL FY24-25 output_p x (India consumption_p,t / India total_p)

Revenue is booked at production at FY 2024-25 average prices; with constant prices and zero closing
stock this equals revenue at sale.
"""
from __future__ import annotations

from ._builder import Builder
from .refinery_planning import (CASE_INFO as _LP_INFO, CDU_CAPACITY, M3_PER_BBL, MRPL_OUTPUT_2024_25,
                                PRICE, SPEC, add_refinery, crude_data)

MONTHS = ["APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC", "JAN", "FEB", "MAR"]
DAYS = [30, 31, 30, 31, 31, 30, 31, 30, 31, 31, 28, 31]            # FY 2024-25 (Feb 2025: 28 days)

# PPAC, Domestic consumption of petroleum products, April-24 to March-25 ('000 t), Annexure-I
INDIA_CONSUMPTION = {
    "MS": [3285, 3463, 3296, 3297, 3360, 3149, 3412, 3428, 3320, 3308, 3174, 3512],
    "HSD": [7925, 8412, 7982, 7193, 6501, 6369, 7645, 8166, 8054, 7738, 7346, 8075],
    "ATF": [742, 744, 707, 727, 732, 726, 757, 748, 782, 784, 735, 801],
    "LPG": [2373, 2410, 2320, 2649, 2664, 2610, 2721, 2665, 2772, 2835, 2583, 2729],
}
STOCKED = ("MS91", "MS95", "HSD", "ATF")
# MRPL Annual Report 2024-25 (Devangonthi marketing terminal): gross storage 81 thousand kL of
# finished products (MS, HSD, ATF). Used here as the finished-product storage limit.
STORAGE_TKL = 81.0
PARCEL_KBBL = 1000.0            # ASSUMPTION: 1 million barrel parcel (Suezmax class)
MAX_CRUDES_PER_MONTH = 3        # ASSUMPTION: tank segregation limit
MAX_FCC_CHANGEOVERS = 2         # ASSUMPTION: severity changes allowed per year

CASE_INFO = {
    "title": "12-month refinery production plan with crude parcels, inventories and FCC "
             "severity changeovers (MRPL-sized)",
    "sector": "Refinery production planning",
    "class": "MILP",
    "sources": _LP_INFO["sources"] + [
        {"what": "Monthly all-India consumption of MS, HSD, ATF, LPG, FY 2024-25 (seasonality)",
         "name": "PPAC, Consumption of petroleum products (product-wise, monthly), 2024-25, Annexure-I",
         "url": "https://ppac.gov.in/consumption/products-wise",
         "licence": "Government of India publication (GODL - India)"},
        {"what": "Finished-product storage 81 TKL; crude bought as Suezmax / VLCC parcels",
         "name": "MRPL 37th Annual Report 2024-25 (Devangonthi terminal; risk management table)",
         "url": "https://admin.mrpl.co.in/img/UploadedFiles/AnnualReport/Files/161636e409d7464996e216b0ffbb5bba.pdf", "licence": "public company annual report"},
    ],
    "real": _LP_INFO["real"] + ["monthly national consumption profile (PPAC)",
                                "81 TKL finished-product storage (MRPL AR)", "days per month"],
    "assumed": _LP_INFO["assumed"] + [
        "monthly market limit = MRPL annual output x national monthly share of that product",
        "parcel size 1 million barrels (the AR confirms Suezmax parcels but not their size); "
        "each parcel is processed in the month it is bought (no crude tank inventory)",
        "at most 3 crudes per month; at most 2 FCC severity changes per year",
        "the 81 TKL terminal storage is used as the finished-product storage limit, with each "
        "product's volume taken at its minimum specification density (a conservative bound)",
        "no inventory holding cost; opening and closing stocks are zero",
    ],
}


def build(months: int = 12, crudes=None, max_crudes: int = MAX_CRUDES_PER_MONTH,
          max_changeovers: int = MAX_FCC_CHANGEOVERS, include_exxonmobil: bool = False):
    cd = crude_data(crudes, include_exxonmobil)
    B = Builder("refinery_multiperiod_mrpl")
    T = list(range(months))
    tot_cap = sum(CDU_CAPACITY.values())
    parcel_kt = {c: PARCEL_KBBL * M3_PER_BBL * d["_density"] for c, d in cd.items()}
    share = {p: [v / sum(vals) for v in vals] for p, vals in INDIA_CONSUMPTION.items()}
    prod_key = {"MS91": "MS", "MS95": "MS", "HSD": "HSD", "ATF": "ATF", "LPG": "LPG"}
    rho_min = {"MS91": SPEC["MS91"]["density"][0], "MS95": SPEC["MS95"]["density"][0],
               "HSD": SPEC["HSD"]["density"][0], "ATF": SPEC["ATF_A"]["density"][0]}

    inv_prev = {p: None for p in STOCKED}
    y_prev = None
    z_cols = []
    for t in T:
        tag = f"_{MONTHS[t]}"
        scale = DAYS[t] / 365.0
        n, w = {}, {}
        nmax = {}
        for c in cd:
            nmax[c] = int(tot_cap * scale // parcel_kt[c])
            n[c] = B.var(f"parcels{tag}_{c}".replace(" ", "_"), 0.0, nmax[c],
                         cost=PRICE["crude_usd_bbl"] * PARCEL_KBBL, integer=True)
            w[c] = B.binary(f"use{tag}_{c}".replace(" ", "_"))
            B.le(f"use_link{tag}_{c}".replace(" ", "_"), [n[c], w[c]], [1.0, -nmax[c]], 0.0)
        B.le(f"max_crudes{tag}", [w[c] for c in cd], 1.0, float(max_crudes))
        cols = add_refinery(B, cd, tag=tag, cdu_scale=scale,
                            market_scale=share["LPG"][t], crude_cols={c: (n[c], parcel_kt[c]) for c in cd},
                            stocked=STOCKED)
        # LPG market limit uses the LPG monthly share (set through market_scale above); stocked
        # products get inventories and their own monthly market limits here
        for p in STOCKED:
            prod = cols["sales"][p]
            I = B.var(f"stock{tag}_{p}")
            S = B.var(f"sales{tag}_{p}", 0.0, MRPL_OUTPUT_2024_25[p] * share[prod_key[p]][t])
            prev = [] if inv_prev[p] is None else [inv_prev[p]]
            B.eq(f"stock_bal{tag}_{p}", [I, S] + prod + prev,
                 [1.0, 1.0] + [-1.0] * len(prod) + [-1.0] * len(prev), 0.0)
            inv_prev[p] = I
            if t == T[-1]:
                B.set_bounds(I, ub=0.0)
        B.le(f"storage{tag}", [inv_prev[p] for p in STOCKED], [1.0 / rho_min[p] for p in STOCKED], STORAGE_TKL)

        # FCC severity mode
        fmax = tot_cap * scale
        y = {}
        for sev in ("low", "high"):
            y[sev] = B.binary(f"fcc_mode{tag}_{sev}")
            F = B.col_names.index(f"fcc_{sev}_feed{tag}")
            B.le(f"fcc_on{tag}_{sev}", [F, y[sev]], [1.0, -fmax], 0.0)
        B.le(f"fcc_one_mode{tag}", [y["low"], y["high"]], 1.0, 1.0)
        if y_prev is not None:
            z = B.binary(f"fcc_change{tag}")
            z_cols.append(z)
            for sev in ("low", "high"):
                B.ge(f"fcc_change_def{tag}_{sev}", [z, y[sev], y_prev[sev]], [1.0, -1.0, 1.0], 0.0)
        y_prev = y
    if z_cols:
        B.le("fcc_changeovers", z_cols, 1.0, float(max_changeovers))
    return B.to_model()
