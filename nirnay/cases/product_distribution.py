"""Indian petroleum-product distribution: refineries to states, BS-VI petrol (MS) and diesel (HSD), LP.

Decision: how much BS-VI MS and HSD each Indian refinery sends to each state and union territory
so that every state's annual sales are met, at least total transport work (tonne-km).

    min   sum_{p,r,s} dist(r,s) x_prs                          ('000 t x km)
    s.t.  sum_s x_prs <= supply_pr        every product p, refinery r
          sum_r x_prs  = demand_ps        every product p, state s
          x >= 0

Data (FY 2024-25 unless stated; all quantities in '000 metric tonnes):
  * demand_ps: PPAC "State-wise sales of MS / HSD" (sheets PT_Cons_Statewise MS / HSD, column
    2024-25) - 36 states and UTs.
  * supply_pr: PPAC "Production of petroleum products", all-India MS-VI = 42178.97 and
    HSD-VI = 92170.41, split across refineries in proportion to each refinery's crude processed
    (PPAC "Crude oil processed by refineries", FY 2024-25 TOTAL column).  ASSUMPTION: PPAC does
    not publish refinery-wise product output, so every refinery is given the national average
    BS-VI yield; RIL's SEZ refinery at Jamnagar (export-oriented) and CPCL Narimanam (zero
    throughput) are excluded.
  * dist(r,s): great-circle distance (haversine, mean Earth radius 6371.0088 km) from the refinery
    to the state capital. Coordinates from Wikidata (CC0) and English Wikipedia (CC BY-SA 4.0).
    ASSUMPTION: great-circle distance stands in for the rail/pipeline/road network distance, and
    the state capital stands in for the state's demand centre. No machine-readable public freight
    tariff was found (the Indian Railways rate circular obtained is a scanned image).
"""
from __future__ import annotations

import math

from ._builder import Builder

CASE_INFO = {
    "title": "Distribution of BS-VI petrol and diesel from 21 Indian refineries to 36 states/UTs",
    "sector": "Supply chain / petroleum-product logistics (India)",
    "class": "LP",
    "sources": [
        {"what": "State-wise annual sales of MS and HSD, 2024-25",
         "name": "PPAC, State-wise sales of petroleum products (PT_Cons_Statewise MS / HSD)",
         "url": "https://ppac.gov.in/uploads/page-images/1787137273_Statewise_Sales-POL_Consumption_Final.xlsx",
         "licence": "Government of India publication; reuse under the Government Open Data "
                    "Licence - India (GODL) as a public statistical release"},
        {"what": "Refinery-wise crude oil processed, FY 2024-25",
         "name": "PPAC, Crude oil processed by refineries (refinery-wise), 2024-25 (P)",
         "url": "https://ppac.gov.in/production/crude-processing",
         "licence": "as above (GODL - India)"},
        {"what": "All-India production of MS-VI and HSD-VI, 2024-25",
         "name": "PPAC, Production of petroleum products, 2024-25 (P)",
         "url": "https://ppac.gov.in/production/petroleum-products",
         "licence": "as above (GODL - India)"},
        {"what": "Refinery and state-capital coordinates",
         "name": "Wikidata (items listed in COORDS; SPARQL query results saved in data/cases/raw/supply)",
         "url": "https://query.wikidata.org/",
         "licence": "Wikidata CC0 1.0"},
        {"what": "Coordinates of Koyali, Jamnagar and Tatipaka refineries (Wikipedia API) and Kochi "
                 "refinery (infobox)",
         "name": "English Wikipedia",
         "url": "https://en.wikipedia.org/wiki/Kochi_Refinery",
         "licence": "CC BY-SA 4.0"},
    ],
    "real": ["state-wise MS and HSD sales (36 states/UTs)", "refinery-wise crude processed",
             "all-India MS-VI and HSD-VI production", "coordinates"],
    "assumed": ["refinery-wise product supply = national BS-VI yield x crude processed "
                "(PPAC does not publish refinery-wise product output); RIL SEZ excluded",
                "cost = great-circle tonne-km to the state capital (no public machine-readable "
                "freight tariff; network distance is longer than great-circle)",
                "imports, exports, inventories and inter-company product exchanges are not modelled"],
}

# PPAC crude processed FY 2024-25 (P), '000 t, TOTAL column of the refinery-wise sheet
CRUDE_PROCESSED = {
    "IOCL Barauni": 6513.258, "IOCL Koyali": 15283.045999999997, "IOCL Haldia": 6934.496,
    "IOCL Mathura": 8053.82, "IOCL Panipat": 15398.267999999998, "IOCL Guwahati": 1178.1999999999998,
    "IOCL Digboi": 774.113, "IOCL Bongaigaon": 2772.0820000000003, "IOCL Paradip": 14657.086000000003,
    "CPCL Manali": 10453.710478900002, "BPCL Mumbai": 15529.0846783548, "BPCL Kochi": 17193.122163,
    "BPCL Bina": 7708.855094, "NRL Numaligarh": 3065.5908095110003, "ONGC Tatipaka": 73.759834895,
    "MRPL Mangalore": 18043.580252691998, "HPCL Mumbai": 9957.797251, "HPCL Visakh": 15309.831773000002,
    "HMEL Bathinda": 13044.806655851478, "RIL Jamnagar (DTA)": 34990.492633,
    "NEL Vadinar": 20486.754414327996,
}
# excluded: "RIL-(SEZ), JAMNAGAR" 31191.571794999996 (export refinery), "CPCL-NARIMANAM" 0

# PPAC production of petroleum products 2024-25 (P), '000 t
NATIONAL_PRODUCTION = {"MS": 42178.97201107366, "HSD": 92170.4076250244}     # MS-VI, HSD-VI rows

# PPAC state-wise sales 2024-25, '000 t: (MS, HSD)
DEMAND = {
    "Chandigarh": (129.425, 257.11111568595044), "Delhi": (1035.327, 557.6637577107438),
    "Haryana": (1430.32, 4285.761430770099), "Himachal Pradesh": (356.864, 786.9708919785126),
    "Jammu & Kashmir": (412.931, 941.5539056413223), "Ladakh": (22.36, 103.36331011900826),
    "Punjab": (1252.426, 3347.1477604109914), "Rajasthan": (2035.678, 5814.202866005069),
    "Uttar Pradesh": (4832.791, 10636.321677065458), "Uttarakhand": (488.611, 821.7457924528925),
    "Arunachal Pradesh": (89.656, 253.05293791735537), "Assam": (720.135, 1277.770723146901),
    "Manipur": (83.287, 117.95839172727273), "Meghalaya": (146.804, 412.74973736363637),
    "Mizoram": (59.28, 119.29690650413224), "Nagaland": (59.778, 127.24956058677685),
    "Sikkim": (30.496, 86.8896908677686), "Tripura": (70.895, 118.09913203305783),
    "Andaman & Nicobar": (21.787, 178.71481762809918), "Bihar": (1161.232, 2444.93913375252),
    "Jharkhand": (756.305, 2020.3529043859257), "Odisha": (1156.326, 3390.545569479364),
    "West Bengal": (1378.713, 3546.4134158551583), "Chhattisgarh": (881.351, 2318.150800623347),
    "Dadra & Nagar Haveli and Daman & Diu": (47.894, 172.65131101652892),
    "Goa": (227.114, 281.1404251404959), "Gujarat": (2701.218, 7318.392166893447),
    "Madhya Pradesh": (1955.161, 4392.47681733878), "Maharashtra": (4370.547, 10450.487194559282),
    "Andhra Pradesh": (1627.316, 3512.23179604324), "Karnataka": (3054.319, 8258.82986350343),
    "Kerala": (1879.418, 2419.3431267314054), "Lakshadweep": (1.489, 21.1190819338843),
    "Puducherry": (192.318, 584.2836174049586), "Tamil Nadu": (3533.707, 6343.537354951301),
    "Telangana": (1801.248, 3646.8731792524254),
}

# (latitude, longitude, source)
COORDS = {
    "IOCL Barauni": (25.466, 85.987, "Wikidata Q4858511"),
    "IOCL Koyali": (22.3705, 73.1255, "Wikipedia 'Gujarat Refinery'"),
    "IOCL Haldia": (22.03333333, 88.13333333, "Wikidata Q5641276"),
    "IOCL Mathura": (27.378305555, 77.686444444, "Wikidata Q15982506"),
    "IOCL Panipat": (29.470944444, 76.87225, "Wikidata Q7131186"),
    "IOCL Guwahati": (26.18, 91.8, "Wikidata: Noonmati (refinery site)"),
    "IOCL Digboi": (27.39083333, 95.61861111, "Wikidata Q5275564"),
    "IOCL Bongaigaon": (26.51583333, 90.53166667, "Wikidata Q4941822"),
    "IOCL Paradip": (20.247638888, 86.598194444, "Wikidata Q7134170"),
    "CPCL Manali": (13.160444444, 80.277527777, "Wikidata Q113987261"),
    "BPCL Mumbai": (19.011939, 72.860086, "Wikidata Q6935320"),
    "BPCL Kochi": (9.9775, 76.375555556, "Wikipedia 'Kochi Refinery' infobox 09 58 39 N 76 22 32 E"),
    "BPCL Bina": (24.2515, 78.1587, "Wikidata Q17018647"),
    "NRL Numaligarh": (26.63333333, 93.75, "Wikidata: Numaligarh (town)"),
    "ONGC Tatipaka": (16.50361111, 81.87555556, "Wikipedia 'Tatipaka'"),
    "MRPL Mangalore": (12.9836731, 74.8261508, "Wikidata Q6748719"),
    "HPCL Mumbai": (19.011939, 72.860086, "Wikidata Q6935320 (adjacent BPCL Mahul site; ASSUMPTION)"),
    "HPCL Visakh": (17.693888888, 83.292222222, "Wikidata Q200016 Visakhapatnam city (ASSUMPTION)"),
    "HMEL Bathinda": (30.23, 74.951944444, "Wikidata Q5620369"),
    "RIL Jamnagar (DTA)": (22.34805556, 69.86888889, "Wikipedia 'Jamnagar refinery'"),
    "NEL Vadinar": (22.33166667, 69.74722222, "Wikidata Q12417189"),
    # state / UT capitals (Wikidata P36 capital, P625 coordinate)
    "Chandigarh": (30.733333333, 76.779722222, "Wikidata"), "Delhi": (28.613888888, 77.208888888, "Wikidata New Delhi"),
    "Haryana": (30.733333333, 76.779722222, "Wikidata Chandigarh"),
    "Himachal Pradesh": (31.103333333, 77.172222222, "Wikidata Shimla"),
    "Jammu & Kashmir": (34.091111111, 74.806111111, "Wikidata Srinagar"),
    "Ladakh": (34.164166666, 77.584722222, "Wikidata Leh"),
    "Punjab": (30.733333333, 76.779722222, "Wikidata Chandigarh"),
    "Rajasthan": (26.915, 75.82, "Wikidata Jaipur"), "Uttar Pradesh": (26.847, 80.947, "Wikidata Lucknow"),
    "Uttarakhand": (30.318, 78.029, "Wikidata Dehradun"),
    "Arunachal Pradesh": (27.1, 93.62, "Wikidata Itanagar"), "Assam": (26.15, 91.77, "Wikidata Dispur"),
    "Manipur": (24.82, 93.95, "Wikidata Imphal"), "Meghalaya": (25.574444444, 91.878888888, "Wikidata Shillong"),
    "Mizoram": (23.733333333, 92.716666666, "Wikidata Aizawl"), "Nagaland": (25.666666666, 94.119444444, "Wikidata Kohima"),
    "Sikkim": (27.33, 88.62, "Wikidata Gangtok"), "Tripura": (23.833333333, 91.266666666, "Wikidata Agartala"),
    "Andaman & Nicobar": (11.666666666, 92.75, "Wikidata Srivijayapuram"),
    "Bihar": (25.61, 85.141388888, "Wikidata Patna"), "Jharkhand": (23.355555555, 85.334722222, "Wikidata Ranchi"),
    "Odisha": (20.295, 85.825, "Wikidata Bhubaneswar"), "West Bengal": (22.5675, 88.37, "Wikidata Kolkata"),
    "Chhattisgarh": (21.2379468, 81.6336833, "Wikidata Raipur"),
    "Dadra & Nagar Haveli and Daman & Diu": (20.416938888, 72.834022222, "Wikidata Daman"),
    "Goa": (15.48, 73.83, "Wikidata Panaji"), "Gujarat": (23.223, 72.65, "Wikidata Gandhinagar"),
    "Madhya Pradesh": (23.258888888, 77.4125, "Wikidata Bhopal"), "Maharashtra": (19.075833333, 72.8775, "Wikidata Mumbai"),
    "Andhra Pradesh": (16.5131, 80.5165, "Wikidata Amaravati"), "Karnataka": (12.9791198, 77.5912997, "Wikidata Bengaluru"),
    "Kerala": (8.4875, 76.9525, "Wikidata Thiruvananthapuram"), "Lakshadweep": (10.566666666, 72.638888888, "Wikidata Kavaratti"),
    "Puducherry": (11.93, 79.83, "Wikidata Pondicherry"), "Tamil Nadu": (13.0825, 80.275, "Wikidata Chennai"),
    "Telangana": (17.361666666, 78.474722222, "Wikidata Hyderabad"),
}


def haversine_km(a, b) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(h))


def build(products=("MS", "HSD")):
    total = sum(CRUDE_PROCESSED.values())
    B = Builder("india_ms_hsd_distribution_2024_25")
    refs, states = list(CRUDE_PROCESSED), list(DEMAND)
    for p in products:
        k = 0 if p == "MS" else 1
        x = {}
        for r in refs:
            for s in states:
                d = haversine_km(COORDS[r], COORDS[s])
                x[r, s] = B.var(f"x_{p}_{r}_{s}".replace(" ", "_"), cost=d)
        for r in refs:
            supply = NATIONAL_PRODUCTION[p] * CRUDE_PROCESSED[r] / total
            B.le(f"supply_{p}_{r}".replace(" ", "_"), [x[r, s] for s in states], 1.0, supply)
        for s in states:
            B.eq(f"demand_{p}_{s}".replace(" ", "_"), [x[r, s] for r in refs], 1.0, DEMAND[s][k])
    return B.to_model()
