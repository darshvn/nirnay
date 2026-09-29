"""Refinery crude selection and product blending (planning LP), MRPL-sized.

Decision: which crudes to run on the three crude distillation units, how to route each crude's
distillation cuts through the reformer and the fluid catalytic cracker (FCC), and how to blend
BS-VI petrol (MS 91 and MS 95), BS-VI diesel (HSD), jet fuel (ATF), LPG, naphtha and fuel oil to
the Indian specifications, to maximise gross margin (product value minus crude cost).

Streams are kept crude-segregated (cut s of crude c has the assay's own properties), so every
quality constraint is linear:
  * volume-based specs (density, RON, cetane index, smoke point) use volume = mass / density,
    e.g. RON:  sum_i (m_i / rho_i) (RON_i - 91) >= 0;  density: sum_i m_i <= rho_max sum_i m_i / rho_i
  * sulphur is blended by mass.
Units: flows in thousand tonnes per year (kt); prices in US$; objective in thousand US$.

    max  sum_products price_p * (volume or mass sold)  -  crude price * sum_c bbl(x_c)
    s.t. sum_c x_cu <= CDU capacity_u                         (3 CDUs)
         cut balance: yield_cs * sum_u x_cu = sum_dest flow(c, s, dest)
         reformer: 0.90 reformate + 0.02 butane + 0.08 fuel per tonne of heavy naphtha feed
         FCC low / high severity yields per tonne of vacuum gas oil feed
         product blends meet BIS specs; sales <= market limit
"""
from __future__ import annotations

from ..model import INF
from ._assays import load as load_assays
from ._builder import Builder

M3_PER_BBL = 0.158987294928            # exact: 42 US gal x 3.785411784 L

# ---- data ---------------------------------------------------------------------------------------
# PPAC "Snapshot of India's Oil & Gas data", Aug 2026, table 25: international FOB prices, FY 2024-25
PRICE = {"crude_usd_bbl": 78.56,                # Indian basket
         "MS_usd_bbl": 85.50, "HSD_usd_bbl": 89.41, "ATF_usd_bbl": 88.77,   # ATF uses kerosene quote
         "LPG_usd_t": 605.15, "NAPHTHA_usd_t": 620.99, "FO_usd_t": 451.73}

# MRPL PACE project brief (MoEFCC environmental clearance document): present CDU capacities, kt/yr
CDU_CAPACITY = {"CDU1": 5000.0, "CDU2": 7200.0, "CDU3": 3300.0}

# MRPL Annual Report 2024-25, Directors' report section 6 "Products" (tonnes -> kt): FY 2024-25
# production, used as market / evacuation limits for each product
MRPL_OUTPUT_2024_25 = {"HSD": 6679.905, "ATF": 2721.671, "MS91": 2006.812, "MS95": 482.841,
                       "LPG": 1187.054}

# Bureau of Indian Standards (BS-VI columns of Table 1)
SPEC = {
    # IS 2796:2017 Table 1: density 720-775 kg/m3, RON >= 91 / 95, sulphur <= 10 mg/kg
    "MS91": {"density": (0.720, 0.775), "ron_min": 91.0},
    "MS95": {"density": (0.720, 0.775), "ron_min": 95.0},
    # IS 1460:2017 Table 1 (BS VI): density 810-845 kg/m3, cetane index >= 46, sulphur <= 10 mg/kg
    "HSD": {"density": (0.810, 0.845), "ci_min": 46.0},
    # IS 1571:2018 Table 1: density 775-840 kg/m3, total sulphur <= 0.30 % mass, and either
    # smoke point >= 25.0 mm (pool ATF_A) or smoke point >= 18.0 mm with naphthalenes <= 3.00 % v/v
    # (pool ATF_B). The two pools share the ATF market limit and price.
    "ATF_A": {"density": (0.775, 0.840), "sulphur_max": 0.30, "smoke_min": 25.0},
    "ATF_B": {"density": (0.775, 0.840), "sulphur_max": 0.30, "smoke_min": 18.0, "naph_max": 3.00},
}

# Exxon Platoform example refinery (K.H. Palmer, A Model Management Framework for Mathematical
# Programming, Exxon Monograph Series, Wiley 1984), as published in the GAMS Model Library model
# FAWLEY (SEQ=65), tables ap(c,p) and prop(c,*): yields as weight fractions of unit feed.
REFORMER = {"reformate": 0.90, "butane": 0.02, "fuel": 0.08}
FCC = {"low": {"butane": 0.05, "ccnaph": 0.325, "ccdist": 0.585, "fuel": 0.040},
       "high": {"butane": 0.06, "ccnaph": 0.45, "ccdist": 0.44, "fuel": 0.050}}
COMPONENT = {  # RON and density (t/m3) of converted streams, FAWLEY prop table
    "reformate": {"ron": 102.5, "density": 0.865},
    "butane": {"ron": 101.6, "density": 0.570},
    "ccnaph_low": {"ron": 94.9, "density": 0.730},
    "ccnaph_high": {"ron": 99.1, "density": 0.750},
    "ccdist": {"sulphur": 1.5},
}

CASE_INFO = {
    "title": "Refinery crude selection and BS-VI product blending LP (MRPL-sized CDUs, 8 SPR crudes)",
    "sector": "Refinery planning / crude blending",
    "class": "LP",
    "sources": [
        {"what": "Crude assays: cut yields, density, sulphur, RON, cetane index, smoke point "
                 "(8 Strategic Petroleum Reserve streams)",
         "name": "US DOE Strategic Petroleum Reserve, Crude Oil Analysis (assay files 2024/2026)",
         "url": "https://www.spr.doe.gov/reports/crude_oil_assays.html",
         "licence": "US Government work, public domain (17 U.S.C. 105)"},
        {"what": "OPTIONAL (include_exxonmobil=True): assays of Upper Zakum, Azeri BTC, CPC Blend, "
                 "Erha, Qua Iboe",
         "name": "ExxonMobil, Assays available for download",
         "url": "https://corporate.exxonmobil.com/what-we-do/energy-supply/crude-trading/crude-oil-assays",
         "licence": "ExxonMobil terms and conditions: redistribution only of complete, unaltered "
                    "documents; not an open licence. Not used in the default instance"},
        {"what": "BS-VI motor gasoline limits (density, RON)",
         "name": "BIS IS 2796:2017 Motor Gasoline - Specification, Table 1 (amendments 1-4 checked)",
         "url": "https://archive.org/details/gov.in.is.2796.2017",
         "licence": "BIS copyright; copy made public by Public.Resource.Org; limits cited as facts"},
        {"what": "BS-VI automotive diesel limits (density, cetane index)",
         "name": "BIS IS 1460:2017 Automotive Diesel Fuel - Specification, Table 1 (amendments 1-2 checked)",
         "url": "https://archive.org/details/gov.in.is.1460.2017",
         "licence": "as above"},
        {"what": "Jet A-1 limits (density, sulphur, smoke point, naphthalenes)",
         "name": "BIS IS 1571:2018 Aviation Turbine Fuels, Kerosine Type, Jet A-1, Table 1 "
                 "(amendments 1-4 checked)",
         "url": "https://archive.org/details/gov.in.is.1571.2018",
         "licence": "as above"},
        {"what": "Cross-check of MS and HSD BS-VI limits",
         "name": "HPCL, MS / HSD BS-VI specifications sheet",
         "url": "https://www.hindustanpetroleum.com/images/pdf/MS_HSD_BS-VI_SPECS.pdf",
         "licence": "public document of HPCL"},
        {"what": "Crude and product prices, FY 2024-25 averages",
         "name": "PPAC, Snapshot of India's Oil & Gas data, August 2026, table 25",
         "url": "https://ppac.gov.in/download.php?file=rep_studies/1790418308_Final_Snapshot_of_Indias_Oil_Gas_data_augpages.pdf",
         "licence": "Government of India publication (GODL - India)"},
        {"what": "CDU capacities 5.0 / 7.2 / 3.3 MMTPA",
         "name": "MRPL, PACE project brief summary (environmental clearance submission)",
         "url": "https://environmentclearance.nic.in/DownloadPfdFile.aspx?FileName=ME2hD5LK2d7ehBlLNo%2Fi1jG0gk46GQu53wzMRCyOnEsBrRNcPfxRnyv294r0VHaVrpVquU045MmQwsjxIQAa9FW98gf7Rm4B1VExHqd8gV7yectH9Sag5gTT3LgzAByZ&FilePath=93ZZBm8LWEXfg+HAlQix2fE2t8z%2FpgnoBhDlYdZCxzXmG8GlihX6H9UP1HygCn3pCkAF2zPFXFQNqA4krKa1Aw%3D%3D",
         "licence": "public government filing"},
        {"what": "FY 2024-25 product output (market limits) and crude processed",
         "name": "MRPL 37th Annual Report 2024-25, Directors' report, sections 4 and 6 (pp. 60-61)",
         "url": "https://admin.mrpl.co.in/img/UploadedFiles/AnnualReport/Files/161636e409d7464996e216b0ffbb5bba.pdf",
         "licence": "public company annual report"},
        {"what": "Reformer and FCC yields and component RON/density",
         "name": "GAMS Model Library, FAWLEY (Platoform example refinery, Palmer 1984)",
         "url": "https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_fawley.html",
         "licence": "GAMS Development Corp. copyright; no licence stated. Illustrative data from "
                    "an Exxon monograph; used as published numbers with citation"},
    ],
    "real": ["8 SPR crude assays (cut yields and qualities)", "BIS BS-VI limits used",
             "PPAC FY 2024-25 international prices", "MRPL CDU capacities and FY 2024-25 output"],
    "assumed": [
        "all crudes are priced at the Indian basket average (no public grade differentials)",
        "ATF is valued at PPAC's kerosene FOB quote",
        "conversion-unit yields are the FAWLEY example-refinery values (not MRPL's; MRPL's unit "
        "yields are not public), applied to every crude's heavy naphtha / vacuum gas oil",
        "hydrotreaters are not modelled: MS and HSD components are taken to be treated to the "
        "10 mg/kg sulphur limit with no change in mass, density, RON or cetane index",
        "reformer and FCC capacities are not public and are left unbounded; refinery fuel has "
        "zero value; operating costs are not included",
        "cetane index, RON, smoke point and naphthalenes blend linearly by volume (a standard "
        "approximation); ATF freezing point (max -47 C) is not modelled because it does not blend "
        "linearly and no public blending index was found",
        "fuel oil has no sulphur limit (IS 1593 is only available as a scanned image)",
        "market limits for MS, HSD, ATF and LPG are MRPL's FY 2024-25 output of each",
        "assay cut yields are normalised to 100 % by mass",
    ],
}


def crude_data(crudes=None, include_exxonmobil: bool = False):
    data = load_assays(include_exxonmobil)
    names = list(data) if crudes is None else list(crudes)
    out = {}
    for c in names:
        d = data[c]
        tot = sum(d[s]["wt_pct"] for s in d if s != "CRUDE")
        out[c] = {s: dict(d[s], frac=d[s]["wt_pct"] / tot) for s in d if s != "CRUDE"}
        out[c]["_density"] = d["CRUDE"]["density"]
    return out


def add_refinery(B: Builder, crudes: dict, tag: str = "", cdu_scale: float = 1.0,
                 market_scale: float = 1.0, crude_cols=None, stocked=()):
    """Adds one period of the refinery network to builder B; returns the dict of key columns.
    cdu_scale scales CDU capacities (e.g. days/365 for a month); market_scale the market limits.
    If crude_cols is given ({crude: (column, kt per unit)}), the crude charged in this period
    equals column x kt-per-unit and carries no cost here (the caller prices the column).
    Products in `stocked` get no market limit here: the caller adds inventories and sales.
    Revenue is booked on production (exact when prices are constant and final stock is zero)."""
    t = tag
    P = PRICE
    cols = {"crude": {}, "sales": {}}
    # crude charge per CDU
    x = {}
    for c, d in crudes.items():
        bbl_per_kt = 1000.0 / (d["_density"] * M3_PER_BBL) / 1000.0     # thousand bbl per kt
        for u in CDU_CAPACITY:
            x[c, u] = B.var(f"crude{t}_{c}_{u}".replace(" ", "_"),
                            cost=0.0 if crude_cols else P["crude_usd_bbl"] * bbl_per_kt)
    for u, cap in CDU_CAPACITY.items():
        B.le(f"cdu_cap{t}_{u}", [x[c, u] for c in crudes], 1.0, cap * cdu_scale)
    if crude_cols:
        for c in crudes:
            col, kt = crude_cols[c]
            B.eq(f"crude_link{t}_{c}".replace(" ", "_"), [x[c, u] for u in CDU_CAPACITY] + [col],
                 [1.0] * len(CDU_CAPACITY) + [-kt], 0.0)
    cols["crude"] = x

    # pools: list of (column, mass-based props) per product
    pool = {p: [] for p in ("MS91", "MS95", "HSD", "ATF_A", "ATF_B", "LPG", "NAPHTHA", "FO")}
    ref_feed, fcc_feed = [], {"low": [], "high": []}

    def route(c, s, dests):
        d = crudes[c][s]
        charge = [x[c, u] for u in CDU_CAPACITY]
        flows = []
        for dest in dests:
            j = B.var(f"{s}{t}_{c}_to_{dest}".replace(" ", "_"))
            flows.append(j)
            if dest in pool:
                pool[dest].append((j, d))
            elif dest == "REFORMER":
                ref_feed.append(j)
            elif dest.startswith("FCC_"):
                fcc_feed[dest[4:]].append(j)
        B.eq(f"cut{t}_{c}_{s}".replace(" ", "_"), flows + charge, [1.0] * len(flows) + [-d["frac"]] * len(charge), 0.0)

    for c, d in crudes.items():
        route(c, "LPG", ["LPG"])
        ln_dest = ["NAPHTHA"] + (["MS91", "MS95"] if d["LN"]["ron"] is not None else [])
        route(c, "LN", ln_dest)
        hn_dest = ["NAPHTHA", "REFORMER"] + (["MS91", "MS95"] if d["HN"]["ron"] is not None else [])
        route(c, "HN", hn_dest)
        kero_dest = ["FO"]
        if d["KERO"]["smoke_point_mm"] is not None and d["KERO"]["sulphur_wt_pct"] is not None:
            kero_dest.append("ATF_A")
            if d["KERO"]["naphthalenes_vol_pct"] is not None:
                kero_dest.append("ATF_B")
        if d["KERO"]["cetane_index"] is not None:
            kero_dest.append("HSD")
        route(c, "KERO", kero_dest)
        route(c, "GO", ["FO"] + (["HSD"] if d["GO"]["cetane_index"] is not None else []))
        route(c, "VGO", ["FO", "FCC_low", "FCC_high"])
        route(c, "VR", ["FO"])

    # reformer
    R = B.var(f"reformer_feed{t}")
    B.eq(f"reformer_in{t}", ref_feed + [R], [1.0] * len(ref_feed) + [-1.0], 0.0)
    ref_out = {}
    for k, frac in REFORMER.items():
        if k == "fuel":
            continue
        dests = ["MS91", "MS95"] + (["LPG"] if k == "butane" else [])
        js = []
        for dest in dests:
            j = B.var(f"ref_{k}{t}_to_{dest}")
            js.append(j)
            props = dict(COMPONENT[k])
            pool[dest].append((j, {"ron": props["ron"], "density": props["density"]}))
        B.eq(f"reformer_{k}{t}", js + [R], [1.0] * len(js) + [-frac], 0.0)
    # FCC, two severities
    for sev, y in FCC.items():
        F = B.var(f"fcc_{sev}_feed{t}")
        B.eq(f"fcc_{sev}_in{t}", fcc_feed[sev] + [F], [1.0] * len(fcc_feed[sev]) + [-1.0], 0.0)
        for k, frac in y.items():
            if k == "fuel":
                continue
            if k == "ccnaph":
                dests, props = ["MS91", "MS95"], COMPONENT[f"ccnaph_{sev}"]
            elif k == "butane":
                dests, props = ["MS91", "MS95", "LPG"], COMPONENT["butane"]
            else:
                dests, props = ["FO"], COMPONENT["ccdist"]
            js = []
            for dest in dests:
                j = B.var(f"fcc_{sev}_{k}{t}_to_{dest}")
                js.append(j)
                pool[dest].append((j, props))
            B.eq(f"fcc_{sev}_{k}{t}", js + [F], [1.0] * len(js) + [-frac], 0.0)

    # product blending, specs, sales
    for p, comps in pool.items():
        js = [j for j, _ in comps]
        spec = SPEC.get(p, {})
        if "density" in spec:
            lo, hi = spec["density"]
            inv = [1.0 / d["density"] for _, d in comps]
            B.le(f"dens_max{t}_{p}", js, [1.0 - hi * v for v in inv], 0.0)       # mass <= hi * volume
            B.ge(f"dens_min{t}_{p}", js, [1.0 - lo * v for v in inv], 0.0)       # mass >= lo * volume
        if "ron_min" in spec:
            B.ge(f"ron{t}_{p}", js, [(d["ron"] - spec["ron_min"]) / d["density"] for _, d in comps], 0.0)
        if "ci_min" in spec:
            B.ge(f"cetane{t}_{p}", js, [(d["cetane_index"] - spec["ci_min"]) / d["density"] for _, d in comps], 0.0)
        if "smoke_min" in spec:
            B.ge(f"smoke{t}_{p}", js, [(d["smoke_point_mm"] - spec["smoke_min"]) / d["density"] for _, d in comps], 0.0)
        if "naph_max" in spec:
            B.le(f"naphthalenes{t}_{p}", js, [(d["naphthalenes_vol_pct"] - spec["naph_max"]) / d["density"] for _, d in comps], 0.0)
        if "sulphur_max" in spec:
            B.le(f"sulphur{t}_{p}", js, [d["sulphur_wt_pct"] - spec["sulphur_max"] for _, d in comps], 0.0)
        # value: $/bbl products by volume, $/t products by mass (objective in thousand $)
        key = {"MS91": "MS_usd_bbl", "MS95": "MS_usd_bbl", "HSD": "HSD_usd_bbl", "ATF_A": "ATF_usd_bbl",
               "ATF_B": "ATF_usd_bbl"}.get(p)
        if key:
            coef = [-PRICE[key] / (d["density"] * M3_PER_BBL) for _, d in comps]   # k$ per kt
        else:
            coef = [-PRICE[{"LPG": "LPG_usd_t", "NAPHTHA": "NAPHTHA_usd_t", "FO": "FO_usd_t"}[p]]] * len(js)
        for j, v in zip(js, coef):
            B.add_cost(j, v)
        mkt = MRPL_OUTPUT_2024_25.get(p)
        if mkt is not None and p not in stocked:
            B.le(f"market{t}_{p}", js, 1.0, mkt * market_scale)
        cols["sales"][p] = js
    cols["sales"]["ATF"] = cols["sales"]["ATF_A"] + cols["sales"]["ATF_B"]
    if "ATF" not in stocked:
        B.le(f"market{t}_ATF", cols["sales"]["ATF"], 1.0, MRPL_OUTPUT_2024_25["ATF"] * market_scale)
    return cols


def build(crudes=None, include_exxonmobil: bool = False):
    """Default: the 8 public-domain SPR crudes. include_exxonmobil=True adds Upper Zakum, Azeri
    BTC, CPC Blend, Erha and Qua Iboe, read from the unaltered ExxonMobil files (needs openpyxl)."""
    B = Builder("refinery_planning_mrpl")          # minimise cost = crude cost - product value
    add_refinery(B, crude_data(crudes, include_exxonmobil))
    return B.to_model()
