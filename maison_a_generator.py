# %% [markdown]
# # Maison A – synthetic dataset generator
#
# This notebook generates the simulated dataset used to demonstrate and evaluate the DSS (Chapter 5).
#
# **Design principles**
# 1. *Calibrated on published data.* Every structural parameter is anchored to a public source (company reports, Bindi et al. (2023), the online catalogues of five maisons, lead-time literature). Parameters with no source are declared as *hypotheses* and listed in `parameters_sources.csv`.
# 2. *No circularity.* Sales are generated from **product life-cycle profiles** with product-specific noise, **never** from the DSS thresholds. The DSS never sees the generating profile (`ground_truth.csv` is used only for verification).
# 3. *Realistic format.* Output looks like a company extract: a product master file and **weekly sales by article (style × material × colour)**. Aggregation to the unit of analysis (production version) and to the season is a step of the DSS pipeline.
# 4. *Clean and dirty versions*, with a log of every injected error.
# 5. *Reproducible*: fixed random seed. Seed 2026 produces the **development** dataset (`maison_a/`), seed 2027 the **test** dataset (`maison_a_test/`), generated with the same parameters but never inspected while the DSS was being designed.
#
# **Outputs** (folder `maison_a/`): `products.csv`, `sales.csv`, `kpi_measured.csv`, `ground_truth.csv`, `products_dirty.csv`, `sales_dirty.csv`, `dirty_log.csv`, `parameters_sources.csv`.

# %%
import numpy as np
import pandas as pd
from pathlib import Path

import os
# 2026 = development dataset (used to build and improve the DSS)
# 2027 = test dataset (used ONLY for the final evaluation, never looked at during development)
SEED = int(os.environ.get("MAISON_SEED", 2026))
# Stress tests (evaluation only): MAISON_STRESS = JSON with one changed assumption, e.g.
# {"out": "stress_S1", "profile_mix_co": {...}} / {"noise_mult": 1.5} / {"n_seasons": 6} / {"dirty_mult": 2}
import json
STRESS = json.loads(os.environ.get("MAISON_STRESS", "{}"))
rng = np.random.default_rng(SEED)
OUT = Path(STRESS.get("out") or ("maison_a" if SEED == 2026 else "maison_a_test"))
OUT.mkdir(exist_ok=True)

# %% [markdown]
# ## 1. Parameters and sources
# All parameters are collected here. Each one is recorded in `parameters_sources.csv` with its source or flagged as a hypothesis.

# %%
PARAMS = []  # (group, parameter, value, source)
def P(group, name, value, source):
    PARAMS.append((group, name, str(value), source))
    return value

# --- Time
SEASONS = [f"{t}{y}" for y in range(21, 26) for t in ("SS", "FW")][-int(STRESS.get("n_seasons", 10)):]   # SS21 ... FW25
N_S = P("Time", "number of seasons", len(SEASONS), "Design choice (D7): 10 seasons = 5 years")
CUR = N_S - 1                                                        # current season = FW25
WEEKS = P("Time", "weeks per season", 26, "Two-season calendar (SS: Jan-Jun, FW: Jul-Dec)")

# --- Portfolio (current season)
N_PORTFOLIO = P("Portfolio", "active styles in current season", 500, "Design choice (D7, option b); consistent with catalogue sizes (150-650 styles per category, catalogue_stats)")
CAT_MIX = P("Portfolio", "category mix", {"Bags": 0.45, "Ready-to-wear": 0.30, "Shoes": 0.25},
            "Revenue mix by category of listed maisons (annual reports), D7")
CONT_SHARE = P("Portfolio", "share of continuative lines (permanent + carry-over) by category",
               {"Bags": 0.40, "Shoes": 0.20, "Ready-to-wear": 0.05},
               "Bindi et al. (2023) for bags and footwear; RTW = hypothesis (overall ~23% vs ~20% in Bindi)")
SEAS_CAPS = P("Portfolio", "seasonal : capsule ratio among non-continuative", (45, 35), "Bindi et al. (2023): seasonal ~45%, capsule ~35%")
PERM_OF_CONT = P("Portfolio", "share of permanent within continuative", 0.25, "Hypothesis")

# --- Life-cycle profiles (latent; never seen by the DSS)
PROFILE_MIX_CO = P("Profiles", "carry-over label -> profile mix", {"stable": 0.45, "oscillating": 0.40, "declining": 0.15},
                   "Hypothesis anchored to Bindi et al. (2023): 5 of 9 carry-over cases not lean; subject to sensitivity analysis")
PROFILE_MIX_PERM = P("Profiles", "permanent label -> profile mix", {"stable": 0.80, "oscillating": 0.20}, "Hypothesis")
REPROPOSED_SHARE = P("Profiles", "share of current seasonal styles already sold in earlier seasons", 0.20, "Hypothesis")
SPECIAL_OF_CAPS = P("Profiles", "share of current capsules that are special editions of a continuative style", 0.20, "Hypothesis; bags and shoes only")
NOISE = P("Profiles", "season-to-season noise (lognormal sigma) range by profile",
          {"stable": (0.05, 0.25), "oscillating": (0.30, 0.70), "declining": (0.08, 0.25), "reproposed": (0.60, 1.00)},
          "Hypothesis; ranges overlap the DSS CV thresholds on purpose (no circularity)")
if "profile_mix_co" in STRESS:
    PROFILE_MIX_CO = STRESS["profile_mix_co"]
if "noise_mult" in STRESS:
    NOISE = {k: (a * STRESS["noise_mult"], b * STRESS["noise_mult"]) for k, (a, b) in NOISE.items()}
DECLINE_RATE = P("Profiles", "decline rate per season (declining profile)", (0.15, 0.35), "Hypothesis")
INTRO_FACTOR = P("Profiles", "demand in launch season relative to maturity", 0.6, "Hypothesis (introduction phase of the product life cycle)")

# --- Selling windows
SELL_WEEKS = P("Life cycle", "selling weeks per season", {"continuative": 26, "seasonal": 20, "capsule": 10},
               "Bindi et al. (2023): seasonal ~20 weeks, capsule ~10 weeks")
MARKDOWN = P("Life cycle", "markdown on last 6 weeks of seasonal products", 0.30, "Hypothesis (end-of-season sales)")

# --- Demand levels (units per season at maturity)
BASE_UNITS = P("Demand", "median units per style per season (lognormal)", {"continuative": 300, "seasonal": 120, "capsule": 60},
               "Hypothesis: only relative levels matter for the DSS")
BASE_SIGMA = P("Demand", "dispersion of style levels (lognormal sigma)", 0.9, "Hypothesis; ABC concentration emerges, it is not imposed")
CAT_UNITS = P("Demand", "category volume multiplier", {"Bags": 1.0, "Shoes": 1.0, "Ready-to-wear": 0.7}, "Hypothesis")
SEASON_EFFECT = P("Demand", "SS / FW demand multiplier by category",
                  {"Bags": (1.00, 1.05), "Shoes": (1.05, 0.95), "Ready-to-wear": (0.85, 1.15)}, "Hypothesis (FW includes holiday peak)")

# --- Prices and variants: catalogue statistics of five maisons (collected 2026-11-06)
CATALOGUE = pd.DataFrame([
    # brand, category, styles, colours/style, p25, median, p75, sizes
    ("Ferragamo", "Bags", 68, 3.44, 1450, 1900, 2550), ("Ferragamo", "Shoes", 82, 2.77, 780, 850, 995), ("Ferragamo", "Ready-to-wear", 165, 1.25, 1150, 1450, 2200),
    ("Prada", "Bags", 99, 1.82, 1950, 2400, 2850), ("Prada", "Shoes", 90, 2.10, 940, 990, 1190), ("Prada", "Ready-to-wear", 492, 1.19, 1400, 1850, 2800),
    ("Gucci", "Bags", 174, 2.57, 1400, 1750, 2500), ("Gucci", "Shoes", 225, 1.45, 780, 920, 1050), ("Gucci", "Ready-to-wear", 532, 1.22, 950, 1500, 2400),
    ("Balenciaga", "Bags", 74, 2.50, 1150, 1890, 2490), ("Balenciaga", "Shoes", 46, 1.41, 820, 925, 1190), ("Balenciaga", "Ready-to-wear", 115, 1.17, 950, 1500, 2900),
    ("Tod's", "Bags", 13, 3.92, 1400, 1800, 2200), ("Tod's", "Shoes", 28, 2.46, 750, 790, 890), ("Tod's", "Ready-to-wear", 70, 1.34, 890, 1980, 4100),
], columns=["brand", "category", "styles", "colours", "p25", "p50", "p75"])
CAT_STATS = CATALOGUE.groupby("category")[["colours", "p25", "p50", "p75"]].mean()
P("Catalogue", "mean colours per style / price quartiles by category", CAT_STATS.round(2).to_dict("index"),
  "Online catalogues of Ferragamo, Prada, Gucci, Balenciaga, Tod's (women, Italy), collected 2026-11-06 (catalogue_stats_2026-11-06.csv)")
SIZES = P("Catalogue", "sizes per style", {"Bags": (1, 1), "Shoes": (10, 14), "Ready-to-wear": (4, 6)}, "Product pages of Ferragamo (shoes 10-14, RTW 5 sizes)")
PRICE_GROWTH = P("Catalogue", "price increase per season (backwards from FW25)", 0.025, "Hypothesis (~5% per year, luxury price increases 2021-2025)")
COLOUR_ROTATION = P("Catalogue", "carry-over: prob. a colour is dropped / added each season", (0.25, 0.30), "Hypothesis (carry-over styles are re-coloured each season)")

# --- Supply side
LT_P = P("Supply", "production lead time P (weeks)", {"Bags": (8, 12), "Shoes": (8, 12), "Ready-to-wear": (10, 14)},
         "Thomassey (2010); Christopher et al. (2004); industry sources")
LT_D = P("Supply", "expected delivery time D (weeks)", {"continuative": (2, 4), "seasonal": (20, 30)},
         "Replenishment of continuative lines vs. sales-campaign lead time of seasonal collections (D7)")
N_SUPPLIERS = P("Supply", "suppliers per category", {"Bags": 12, "Shoes": 8, "Ready-to-wear": 15}, "Hypothesis")
MATERIALS = {
    "Bags": [("calf leather", 0.55), ("grained leather", 0.20), ("canvas", 0.15), ("exotic leather", 0.10)],
    "Shoes": [("calf leather", 0.60), ("suede", 0.20), ("fabric", 0.15), ("exotic leather", 0.05)],
    "Ready-to-wear": [("wool", 0.30), ("silk", 0.20), ("cotton", 0.30), ("cashmere", 0.10), ("leather", 0.10)],
}
P("Supply", "materials by category (probability)", MATERIALS, "Hypothesis; precious materials assigned to higher-price styles")

# --- KPIs currently measured by the company (by product line label)
KPI_LEAN = {
    "New product development": ["Total cost of new product development"],
    "Plan": ["Total cost of supply chain management"],
    "Source": ["Total supplier-related costs", "Total raw material cost", "% of orders received on time", "% of orders with delivery date variation",
               "% of compliant orders received", "Total inventory costs", "Obsolescence costs", "Returned transport cost", "Raw material shipping cost"],
    "Make": ["Total production cost", "Productive effectiveness", "% of non-conformity on total product",
             "% of orders with delivery date variation during production", "Capacity saturation"],
    "Deliver": ["% of shipments handled on time", "% of compliant orders filled", "Average time to receive a product at the store", "Service level", "New product success rate"],
    "Return": ["Number of products returned for non-compliance"],
}
KPI_SEASONAL = {
    "New product development": ["Total cost of new product development", "Total cost of the realised prototypes", "Total cost of samples produced",
                                "Cost of prototypes on total production cost (%)", "Industrialisation cost on production cost (%)"],
    "Plan": ["Total cost of supply chain management", "Forecast accuracy", "Forecast volatility"],
    "Source": ["% of orders received on time", "% of orders received with the correct quantity", "% of orders with delivery date variation",
               "Number of change order requests fulfilled", "% of compliant orders received", "Accessible information on total information", "Total raw material cost"],
    "Make": ["Productive effectiveness", "% of non-conformity on total product", "% of available capacity",
             "% of orders with delivery date variation during production", "Time to market"],
    "Deliver": KPI_LEAN["Deliver"],
    "Return": KPI_LEAN["Return"],
}
P("KPI", "KPIs measured by label", "permanent/carry-over -> lean carry-over set; seasonal/capsule -> seasonal set",
  "Bindi et al. (2023), KPI configurations (Table 2.5): companies configure KPIs by product line label")

# --- Dirty data
DIRTY = P("Dirty data", "error rates", {
    "missing_rows": 0.03, "spikes_rows": 0.005, "spike_factor": (5, 10), "code_change_styles": 0.03,
    "stockout_styles": 0.10, "duplicate_rows": 0.01, "return_rows": 0.01, "missing_lead_time": 0.05},
    "Hypothesis; typical data-quality problems of retail sales extracts")
if "dirty_mult" in STRESS:
    DIRTY = {k: (v * STRESS["dirty_mult"] if k != "spike_factor" else v) for k, v in DIRTY.items()}

# %% [markdown]
# ## 2. Helpers

# %%
def lognorm_noise(sigma, size=None):
    """Multiplicative noise with mean 1."""
    return np.exp(rng.normal(-sigma**2 / 2, sigma, size))

def price_draw(cat):
    s = CAT_STATS.loc[cat]
    mu, sig = np.log(s.p50), (np.log(s.p75) - np.log(s.p25)) / 1.349
    return float(np.clip(np.exp(rng.normal(mu, sig)), 0.3 * s.p50, 8 * s.p50))

def n_colours_draw(cat):
    return 1 + rng.poisson(CAT_STATS.loc[cat, "colours"] - 1)

_codes = set()
def new_model_code():
    while True:
        c = str(rng.integers(100000, 999999))
        if c not in _codes:
            _codes.add(c); return c

def material_code():
    return "".join(rng.choice(list("ABCDEFGHJKLMNPRSTUVWXYZ0123456789"), 5))

def colour_code():
    return str(rng.integers(1000, 9999))

def weekly_shape(kind, season_idx, start=0):
    """Share of season demand in each of the 26 weeks + discount per week."""
    w = np.zeros(WEEKS); disc = np.zeros(WEEKS)
    if kind == "continuative":
        w[:] = 1.0
        if SEASONS[season_idx].startswith("FW"):
            w[19:25] *= 1.4          # Nov-Dec holiday peak
    elif kind == "seasonal":
        t = np.arange(SELL_WEEKS["seasonal"])
        shape = (t + 1) ** 1.5 * np.exp(-(t + 1) / 3.0)   # fast ramp-up, peak ~week 4, decay
        shape[-6:] += shape.max() * 0.25                    # markdown bump
        w[:len(t)] = shape
        disc[len(t) - 6:len(t)] = MARKDOWN
    elif kind == "capsule":
        t = np.arange(SELL_WEEKS["capsule"])
        w[start:start + len(t)] = np.exp(-0.25 * t)
    return w / w.sum(), disc

# %% [markdown]
# ## 3. Product master (units of analysis = production versions)
# A unit is a **model + material** (production version). Colours and sizes are variants of the same unit (D1). Special editions share the model code of their parent but have a different material and supplier, so they are separate units.

# %%
units = []   # list of dicts (one per unit of analysis)

def make_unit(cat, line, profile, first, last, parent=None, active=True):
    price = price_draw(cat)
    mats, probs = zip(*MATERIALS[cat])
    # precious materials more likely for expensive styles
    q = (np.log(price) - np.log(CAT_STATS.loc[cat, "p25"])) / (np.log(CAT_STATS.loc[cat, "p75"]) - np.log(CAT_STATS.loc[cat, "p25"]))
    pr = np.array(probs) * np.where(np.array([("exotic" in m) or (m in ("cashmere", "leather")) for m in mats]), 0.5 + 2 * np.clip(q, 0, 1.5), 1.0)
    material = rng.choice(mats, p=pr / pr.sum())
    kind = "continuative" if line in ("Permanent", "Carry-over") else ("capsule" if line == "Capsule" else "seasonal")
    smin, smax = SIZES[cat]
    u = dict(
        unit_id=None, model_code=parent["model_code"] if parent else new_model_code(), material_code=material_code(),
        category=cat, product_line=line, material=material,
        supplier=f"S-{cat[0]}{rng.integers(1, N_SUPPLIERS[cat] + 1):02d}",
        price_fw25=round(price, -1), n_colours=n_colours_draw(cat), n_sizes=int(rng.integers(smin, smax + 1)),
        lt_production_weeks=int(rng.integers(*LT_P[cat]) + rng.integers(0, 2)),
        lt_delivery_weeks=int(rng.integers(*(LT_D["continuative"] if kind == "continuative" else LT_D["seasonal"])) + rng.integers(0, 2)),
        first_season=SEASONS[first], last_season=SEASONS[last], active_current=active,
        is_special_edition=parent is not None, parent_unit=parent["unit_id"] if parent else "",
        # latent
        _profile=profile, _kind=kind, _first=first, _last=last,
    )
    if parent is not None:   # special edition: different supplier, often precious material, higher price
        while u["supplier"] == parent["supplier"]:
            u["supplier"] = f"S-{cat[0]}{rng.integers(1, N_SUPPLIERS[cat] + 1):02d}"
        u["price_fw25"] = round(parent["price_fw25"] * rng.uniform(1.3, 2.5), -1)
        u["n_colours"] = 1
    u["unit_id"] = f'{u["model_code"]}-{u["material_code"]}'
    units.append(u)
    return u

# --- current portfolio
cat_counts = {c: int(round(N_PORTFOLIO * s)) for c, s in CAT_MIX.items()}
for cat, n in cat_counts.items():
    n_cont = int(round(n * CONT_SHARE[cat]))
    n_perm = int(round(n_cont * PERM_OF_CONT))
    n_rest = n - n_cont
    n_seas = int(round(n_rest * SEAS_CAPS[0] / sum(SEAS_CAPS)))
    n_caps = n_rest - n_seas
    for i in range(n_cont):
        line = "Permanent" if i < n_perm else "Carry-over"
        mix = PROFILE_MIX_PERM if line == "Permanent" else PROFILE_MIX_CO
        prof = rng.choice(list(mix), p=list(mix.values()))
        if line == "Permanent":
            H = N_S if rng.random() < 0.8 else int(rng.integers(5, N_S))
        elif prof == "declining":
            H = int(rng.integers(5, N_S + 1))
        else:
            H = int(rng.integers(1, N_S + 1))
        make_unit(cat, line, prof, CUR - H + 1, CUR)
    for i in range(n_seas):
        prof = "reproposed" if rng.random() < REPROPOSED_SHARE else "seasonal"
        make_unit(cat, "Seasonal", prof, CUR, CUR)      # first season fixed later for reproposed
    conts = [u for u in units if u["category"] == cat and u["_kind"] == "continuative"]
    for i in range(n_caps):
        if cat != "Ready-to-wear" and rng.random() < SPECIAL_OF_CAPS and conts:   # special editions of iconic bags and shoes
            make_unit(cat, "Capsule", "special_edition", CUR, CUR, parent=conts[rng.integers(len(conts))])
        else:
            make_unit(cat, "Capsule", "capsule", CUR, CUR)

# reproposed seasonal styles: 1-3 earlier selling seasons (gaps allowed)
for u in units:
    if u["_profile"] == "reproposed":
        past = sorted(rng.choice(np.arange(CUR), size=int(rng.integers(1, 4)), replace=False))
        u["_runs"] = list(past) + [CUR]
        u["_first"] = past[0]; u["first_season"] = SEASONS[past[0]]

# --- discontinued styles (history only, not in the decision set)
for s in range(CUR):
    for cat, n in cat_counts.items():
        n_rest = n - int(round(n * CONT_SHARE[cat]))
        n_seas = int(round(n_rest * SEAS_CAPS[0] / sum(SEAS_CAPS)))
        for _ in range(int(n_seas * rng.uniform(0.9, 1.1))):
            make_unit(cat, "Seasonal", "seasonal", s, s, active=False)
        for _ in range(int((n_rest - n_seas) * rng.uniform(0.9, 1.1))):
            make_unit(cat, "Capsule", "capsule", s, s, active=False)
for cat, n in cat_counts.items():   # continuative styles withdrawn before FW25
    for _ in range(int(round(n * CONT_SHARE[cat] * 0.15))):
        last = int(rng.integers(2, CUR))
        first = int(rng.integers(0, last - 1))
        make_unit(cat, "Carry-over", "declining", first, last, active=False)

print(f"units: {len(units)} total, {sum(u['active_current'] for u in units)} active in {SEASONS[CUR]}")

# %% [markdown]
# ## 4. Sales generation
# Season demand = base level × life-cycle shape × season effect × product-specific noise. Weekly sales are drawn around the season demand with a selling-curve shape, and split across the colours active in that season.

# %%
rows = []
truth = []
for u in units:
    cat, prof, kind = u["category"], u["_profile"], u["_kind"]
    key = "capsule" if kind == "capsule" else ("seasonal" if kind == "seasonal" else "continuative")
    base = BASE_UNITS[key] * CAT_UNITS[cat] * lognorm_noise(BASE_SIGMA)
    if prof == "special_edition":
        base = BASE_UNITS["capsule"] * CAT_UNITS[cat] * lognorm_noise(0.6) * 0.8
    sig = rng.uniform(*NOISE[prof]) if prof in NOISE else np.nan
    rate = rng.uniform(*DECLINE_RATE) if prof == "declining" else np.nan
    first, last = u["_first"], u["_last"]
    seasons = u.get("_runs", list(range(first, last + 1)))
    if prof == "declining":
        span = last - first
        d0 = first + int(rng.integers(1, max(2, span - 2)))   # decline starts here
    launched_in_window = not (u["product_line"] == "Permanent" and first == 0)

    # colours
    colours = [colour_code() for _ in range(u["n_colours"])]
    active_cols = list(colours)
    cap_start = int(rng.integers(0, WEEKS - SELL_WEEKS["capsule"] + 1))

    for s in seasons:
        # life-cycle shape
        f = 1.0
        if kind == "continuative" and s == first and launched_in_window and first > 0:
            f *= INTRO_FACTOR
        if prof == "declining" and s > d0:
            f *= np.exp(-rate * (s - d0))
        se = SEASON_EFFECT[cat][0 if SEASONS[s].startswith("SS") else 1]
        noise = lognorm_noise(sig) if not np.isnan(sig) else 1.0
        demand = base * f * se * noise

        # colour rotation for carry-over
        if u["product_line"] == "Carry-over" and s > first:
            active_cols = [c for c in active_cols if rng.random() > COLOUR_ROTATION[0]]
            if rng.random() < COLOUR_ROTATION[1] or not active_cols:
                active_cols.append(colour_code())
        col_share = rng.dirichlet([2.0] * len(active_cols))

        shape, disc = weekly_shape(kind, s, cap_start)
        price_s = u["price_fw25"] / (1 + PRICE_GROWTH) ** (CUR - s)
        for w in np.nonzero(shape)[0]:
            for c, cs in zip(active_cols, col_share):
                lam = demand * shape[w] * cs * rng.gamma(10, 0.1)
                q = rng.poisson(lam)
                if q > 0:
                    rows.append((SEASONS[s], int(w) + 1, f'{u["model_code"]}{u["material_code"]}{c}', u["model_code"], u["material_code"], c,
                                 q, round(q * price_s * (1 - disc[w]), 2)))
    truth.append(dict(unit_id=u["unit_id"], profile=prof, base_units=round(base, 1), noise_sigma=round(sig, 3) if not np.isnan(sig) else None,
                      decline_rate=round(rate, 3) if not np.isnan(rate) else None,
                      decline_start=SEASONS[d0] if prof == "declining" else None, seasons_on_market=len(seasons)))

sales = pd.DataFrame(rows, columns=["season", "week", "article_code", "model_code", "material_code", "colour_code", "units", "net_sales_eur"])
# calendar date of the week (SS starts first Monday of January, FW first Monday of July)
def week_date(season, week):
    y = 2000 + int(season[2:]); m = 1 if season.startswith("SS") else 7
    d = pd.Timestamp(y, m, 1); d += pd.Timedelta(days=(7 - d.weekday()) % 7)
    return d + pd.Timedelta(weeks=week - 1)
_dates = {(s, w): week_date(s, w) for s in SEASONS for w in range(1, WEEKS + 1)}
sales.insert(0, "week_start", [_dates[k].date() for k in zip(sales.season, sales.week)])

products = pd.DataFrame(units)
ground_truth = pd.DataFrame(truth)
products_out = products[[c for c in products.columns if not c.startswith("_")]]
print(f"sales rows: {len(sales):,}  |  revenue FW25: {sales[sales.season=='FW25'].net_sales_eur.sum()/1e6:.1f} M EUR")

# %% [markdown]
# ## 5. KPIs currently measured (by product line label)

# %%
kpi_rows = []
for line in ["Permanent", "Carry-over", "Seasonal", "Capsule"]:
    cfg = KPI_LEAN if line in ("Permanent", "Carry-over") else KPI_SEASONAL
    for proc, kpis in cfg.items():
        for k in kpis:
            kpi_rows.append((line, proc, k))
kpi_measured = pd.DataFrame(kpi_rows, columns=["product_line", "process", "kpi"])

# %% [markdown]
# ## 6. Dirty version
# Errors are injected into a copy of the data; every injection is logged in `dirty_log.csv` so that the robustness test can measure its effect.

# %%
log = []
sd = sales.copy()
pdirty = products_out.copy()
n = len(sd)

# missing rows
idx = rng.choice(sd.index, int(DIRTY["missing_rows"] * n), replace=False)
log += [("missing_row", sd.at[i, "article_code"], sd.at[i, "season"], int(sd.at[i, "week"]), "") for i in idx]
sd = sd.drop(idx)

# stock-outs on continuative styles: 3-6 consecutive weeks with no sales
cont_active = products[(products._kind == "continuative") & products.active_current]
for _, u in cont_active.sample(frac=DIRTY["stockout_styles"], random_state=SEED).iterrows():
    s = SEASONS[int(rng.integers(u._first, CUR + 1))]
    w0 = int(rng.integers(1, WEEKS - 6)); w1 = w0 + int(rng.integers(3, 7))
    m = (sd.model_code == u.model_code) & (sd.material_code == u.material_code) & (sd.season == s) & sd.week.between(w0, w1 - 1)
    log.append(("stockout", u.unit_id, s, w0, f"weeks {w0}-{w1-1}, {int(sd.loc[m,'units'].sum())} units lost"))
    sd = sd[~m]

# spikes
idx = rng.choice(sd.index, int(DIRTY["spikes_rows"] * n), replace=False)
for i in idx:
    k = int(rng.integers(*DIRTY["spike_factor"]))
    log.append(("spike", sd.at[i, "article_code"], sd.at[i, "season"], int(sd.at[i, "week"]), f"x{k}"))
    sd.at[i, "units"] *= k; sd.at[i, "net_sales_eur"] = round(sd.at[i, "net_sales_eur"] * k, 2)

# returns (negative lines)
idx = rng.choice(sd.index, int(DIRTY["return_rows"] * n), replace=False)
ret = sd.loc[idx].copy()
ret["units"] = -np.minimum(ret["units"], rng.integers(1, 4, len(ret)))
ret["net_sales_eur"] = (ret["net_sales_eur"] / sd.loc[idx, "units"] * ret["units"]).round(2)
log += [("return", a, s, int(w), "") for a, s, w in zip(ret.article_code, ret.season, ret.week)]

# duplicates
dup = sd.loc[rng.choice(sd.index, int(DIRTY["duplicate_rows"] * n), replace=False)]
log += [("duplicate", a, s, int(w), "") for a, s, w in zip(dup.article_code, dup.season, dup.week)]
sd = pd.concat([sd, ret, dup])

# code changes: from a given season onward the style is sold under a new model code
long_lived = products[products.active_current & (products._kind == "continuative") & ((CUR - products._first) >= 3)]
for _, u in long_lived.sample(frac=DIRTY["code_change_styles"] * len(products[products.active_current]) / len(long_lived), random_state=SEED + 1).iterrows():
    s_new = int(rng.integers(u._first + 2, CUR + 1))
    new_code = new_model_code()
    m = (sd.model_code == u.model_code) & (sd.material_code == u.material_code) & (sd.season.map(SEASONS.index) >= s_new)
    sd.loc[m, "model_code"] = new_code
    sd.loc[m, "article_code"] = new_code + sd.loc[m, "material_code"] + sd.loc[m, "colour_code"]
    newrow = pdirty[pdirty.unit_id == u.unit_id].copy()
    newrow["model_code"] = new_code; newrow["unit_id"] = f"{new_code}-{u.material_code}"; newrow["first_season"] = SEASONS[s_new]
    pdirty.loc[pdirty.unit_id == u.unit_id, ["last_season", "active_current"]] = [SEASONS[s_new - 1], False]
    pdirty = pd.concat([pdirty, newrow])
    log.append(("code_change", u.unit_id, SEASONS[s_new], None, f"new unit {new_code}-{u.material_code}"))

# missing lead times in the master file
pdirty = pdirty.reset_index(drop=True)
idx = rng.choice(pdirty.index, int(DIRTY["missing_lead_time"] * len(pdirty)), replace=False)
log += [("missing_lead_time", pdirty.at[i, "unit_id"], None, None, "") for i in idx]
pdirty.loc[idx, "lt_production_weeks"] = np.nan

sd = sd.sample(frac=1, random_state=SEED).sort_values(["season", "week"], kind="stable").reset_index(drop=True)
dirty_log = pd.DataFrame(log, columns=["error_type", "code", "season", "week", "detail"])
print(dirty_log.error_type.value_counts().to_string())

# %% [markdown]
# ## 7. Save

# %%
products_out.to_csv(OUT / "products.csv", index=False)
sales.to_csv(OUT / "sales.csv", index=False)
kpi_measured.to_csv(OUT / "kpi_measured.csv", index=False)
ground_truth.to_csv(OUT / "ground_truth.csv", index=False)
pdirty.to_csv(OUT / "products_dirty.csv", index=False)
sd.to_csv(OUT / "sales_dirty.csv", index=False)
dirty_log.to_csv(OUT / "dirty_log.csv", index=False)
pd.DataFrame(PARAMS, columns=["group", "parameter", "value", "source"]).to_csv(OUT / "parameters_sources.csv", index=False)
print("saved to", OUT.resolve())

# %% [markdown]
# ## 8. Sanity checks (descriptive only – this is not the DSS)
# Quick look at the generated data: example sales histories by profile, spread of season-to-season variability by profile, and revenue concentration.

# %%
import matplotlib.pyplot as plt
s = sales.copy(); s["unit_id"] = s.model_code + "-" + s.material_code
ss = s.groupby(["unit_id", "season"]).units.sum().unstack().reindex(columns=SEASONS).fillna(0)
act = products_out[products_out.active_current].merge(ground_truth[["unit_id", "profile"]], on="unit_id")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
for prof in ["stable", "oscillating", "declining", "reproposed"]:
    sub = act[act.profile == prof]
    uid = sub.loc[sub.first_season.map(SEASONS.index).idxmin(), "unit_id"]   # longest history
    ax[0].plot(SEASONS, ss.loc[uid], marker="o", label=prof)
ax[0].set_title("Seasonal sales of one style per profile"); ax[0].legend(); ax[0].tick_params(axis="x", rotation=45)

def cv_last6(row, first):
    v = row[SEASONS[max(first, CUR - 5):]]
    return v.std(ddof=0) / v.mean() if len(v) > 1 and v.mean() > 0 else np.nan
act["first_idx"] = act.first_season.map(SEASONS.index)
act["cv"] = [cv_last6(ss.loc[u], f) for u, f in zip(act.unit_id, act.first_idx)]
profs = ["stable", "oscillating", "declining", "reproposed"]
ax[1].boxplot([act[act.profile == p].cv.dropna() for p in profs])
ax[1].set_xticks(range(1, len(profs) + 1)); ax[1].set_xticklabels(profs)
for y, ls in [(0.25, ":"), (0.5, "--"), (1.0, "--")]:
    ax[1].axhline(y, color="grey", ls=ls, lw=0.8)
ax[1].set_title("CV of seasonal sales (last 6 seasons)\n(dashed: base thresholds, dotted: alternative)")

val = s[s.season == SEASONS[CUR]].groupby("unit_id").net_sales_eur.sum().sort_values(ascending=False)
ax[2].plot(np.arange(1, len(val) + 1) / len(val) * 100, val.cumsum() / val.sum() * 100)
ax[2].axhline(80, color="grey", ls="--", lw=0.8); ax[2].set_xlabel("% of styles"); ax[2].set_ylabel("% of revenue")
ax[2].set_title(f"Revenue concentration {SEASONS[CUR]}")
plt.tight_layout(); plt.savefig(OUT / "sanity_checks.png", dpi=150); plt.show()
print(act.groupby(["product_line", "profile"]).cv.describe()[["count", "50%", "min", "max"]].round(2))
