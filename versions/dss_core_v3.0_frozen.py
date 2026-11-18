"""
dss_core.py - Model component of the DSS (Chapter 5).

The same functions are used by the analysis notebook and by the Streamlit dashboard,
so that what the manager sees is exactly what is reported in the thesis.

Pipeline
    1. clean()            data cleaning + data-quality report        (R2, R8)
    2. aggregate()        weekly article sales -> unit x season      (D1)
    3. features()         V1-V6 + trend alert                       (D2)
    4. classify()         threshold rules, class S / V / I / N       (D3, D4)
    5. prescribe()        strategy + order penetration point         (D5)
    6. kpi_alignment()    required vs measured KPIs                  (D5, D6)
    7. cluster_check()    k-means check: silhouette + ARI            (D3)
    priority_list()       products to look at first
    run()                 runs 1-6 and returns one table per unit

Version 2 (after the first evaluation cycle on the development dataset):
    - CV computed on de-seasonalised sales (SS/FW index of the category estimated from the data)
    - confidence interval of the CV; products whose interval crosses a threshold are flagged "to review"
    - priority list for the manager
    Version 1 is reproduced with params = V1.

Version 3 (after the second evaluation cycle and the pre-test audit on the development dataset):
    - possible code changes are flagged (relaunch_check): the manager sees the class the product
      would have with the combined history and decides
    - products with declining sales get a separate check ("end of life cycle?"), kept
      distinct from the "class to confirm" list because the question for the manager differs
    - for products to review: how many more seasons would make the class reliable
    - cost-aware classification: the manager states how much more costly it is to treat a
      variable product as stable than the opposite (cost_ratio); default 1 = neutral
    Version 2 is reproduced with params = V2.
"""
import numpy as np
import pandas as pd

SEASONS = [f"{t}{y}" for y in range(21, 26) for t in ("SS", "FW")]

# ---------------------------------------------------------------------------
# Parameters (all editable by the manager in the dashboard)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    current_season="FW25",
    cv_window=6,             # V2: CV computed on the last 6 seasons at most
    min_history=3,           # H < 3 -> class N (insufficient history)
    cv_thresholds=(0.5, 1.0),  # base scenario; alternative scenario = (0.25, 0.5)
    abc_cuts=(0.80, 0.95),   # ABC 80 / 15 / 5 on value of the last 12 months
    pd_threshold=1.0,        # P/D <= 1 -> MTO
    high_quantile=2 / 3,     # V5, V6 "high" = top tercile within category
    trend_min_history=4,     # trend alert only if H >= 4
    trend_drop=0.30,         # last-2-season average > 30% below previous average
    spike_factor=4.0,        # cleaning: week > 4 x max(neighbouring weeks, season median) ...
    spike_min_excess=8,      # ... and at least 8 units above it = spike
    # --- version 2 improvements (set to False / None to reproduce version 1)
    deseasonalise=True,      # remove the SS/FW effect of the category before computing the CV
    review_confidence=0.80,  # confidence level of the CV interval; interval crossing a threshold -> "to review"
    # --- version 3 improvements
    cost_ratio=1.0,          # cost of treating a variable product as stable / cost of the opposite error (1 = neutral)
    decline_to_review=True,  # products with a trend alert get a separate "declining sales" check
    relaunch_check=True,     # flag new continuative products that may continue an older code (code change)
    relaunch_price_tol=0.10, # ... same category, line, material and supplier, price within +-10%,
    relaunch_recent=3,       # ... first sold in the last 3 seasons, right after the old code stopped selling
)
V1 = dict(DEFAULTS, deseasonalise=False, review_confidence=None, cost_ratio=1.0, decline_to_review=False, relaunch_check=False)
V2 = dict(DEFAULTS, cost_ratio=1.0, decline_to_review=False, relaunch_check=False)
LABEL_STRATEGY = {"Permanent": "Lean", "Carry-over": "Lean", "Seasonal": "Hybrid", "Capsule": "Hybrid"}
SCENARIOS = {"base": (0.5, 1.0), "alternative": (0.25, 0.5)}

CLASS_NAMES = {"S": "Stable", "V": "Variable", "I": "Irregular", "N": "Insufficient history"}
STRATEGY = {"S": "Lean", "V": "Hybrid", "I": "Agile", "N": "Hybrid (prudent)"}

# ---------------------------------------------------------------------------
# KPI configurations - Bindi et al. (2023), Table 2.5
# ---------------------------------------------------------------------------
_DELIVER = ["% of shipments handled on time", "% of compliant orders filled", "Average time to receive a product at the store",
            "Service level", "New product success rate"]
_RETURN = ["Number of products returned for non-compliance"]
KPI_SETS = {
    "carry-over lean": {
        "New product development": ["Total cost of new product development"],
        "Plan": ["Total cost of supply chain management"],
        "Source": ["Total supplier-related costs", "Total raw material cost", "% of orders received on time",
                   "% of orders with delivery date variation", "% of compliant orders received", "Total inventory costs",
                   "Obsolescence costs", "Returned transport cost", "Raw material shipping cost"],
        "Make": ["Total production cost", "Productive effectiveness", "% of non-conformity on total product",
                 "% of orders with delivery date variation during production", "Capacity saturation"],
        "Deliver": _DELIVER, "Return": _RETURN},
    "carry-over hybrid": {
        "New product development": ["Total cost of new product development"],
        "Plan": ["Total cost of supply chain management"],
        "Source": ["% of orders received on time", "% of orders received with the correct quantity",
                   "% of orders with delivery date variation", "Number of change order requests fulfilled",
                   "% of compliant orders received", "Accessible information on total information", "Total raw material cost"],
        "Make": ["Productive effectiveness", "% of non-conformity on total product", "% of available capacity",
                 "% of orders with delivery date variation during production", "Time to market"],
        "Deliver": _DELIVER, "Return": _RETURN},
    "seasonal/capsule": {
        "New product development": ["Total cost of new product development", "Total cost of the realised prototypes",
                                    "Total cost of samples produced", "Cost of prototypes on total production cost (%)",
                                    "Industrialisation cost on production cost (%)"],
        "Plan": ["Total cost of supply chain management", "Forecast accuracy", "Forecast volatility"],
        "Source": None, "Make": None, "Deliver": _DELIVER, "Return": _RETURN},
}
KPI_SETS["seasonal/capsule"]["Source"] = KPI_SETS["carry-over hybrid"]["Source"]
KPI_SETS["seasonal/capsule"]["Make"] = KPI_SETS["carry-over hybrid"]["Make"]
FORECAST_KPIS = {"Plan": ["Forecast accuracy", "Forecast volatility"]}


def required_kpis(cls):
    """KPI set required by a DSS class -> set of (process, kpi)."""
    if cls == "S":
        cfg = KPI_SETS["carry-over lean"]
    elif cls in ("V", "I"):
        cfg = {k: list(v) for k, v in KPI_SETS["carry-over hybrid"].items()}
        cfg["Plan"] = cfg["Plan"] + FORECAST_KPIS["Plan"]
    else:
        cfg = KPI_SETS["seasonal/capsule"]
    return {(p, k) for p, ks in cfg.items() for k in ks}


# Short explanation shown to the manager for each KPI that may be missing
KPI_WHY = {
    "Forecast accuracy": "demand of this product varies between seasons: planning on the average of past seasons produces errors that should be measured",
    "Forecast volatility": "the forecast of this product changes from season to season: its stability should be monitored",
    "% of orders received with the correct quantity": "a hybrid strategy relies on responsive suppliers: quantity errors directly affect availability",
    "Number of change order requests fulfilled": "with variable demand, orders are revised during the season: supplier flexibility should be measured",
    "Accessible information on total information": "a hybrid/agile strategy needs information shared with suppliers",
    "% of available capacity": "responsiveness depends on free production capacity rather than on saturation",
    "Time to market": "with variable demand, the speed of replenishment matters more than unit cost",
    "Total cost of the realised prototypes": "product with little history: development costs are still relevant",
    "Total cost of samples produced": "product with little history: development costs are still relevant",
    "Cost of prototypes on total production cost (%)": "product with little history: development costs are still relevant",
    "Industrialisation cost on production cost (%)": "product with little history: development costs are still relevant",
    "Total supplier-related costs": "stable demand: a lean strategy competes on cost",
    "Total inventory costs": "stable demand produced to stock: inventory cost is the main lever",
    "Obsolescence costs": "stable demand produced to stock: the risk of obsolete stock should be monitored",
    "Returned transport cost": "lean strategy: logistics costs should be monitored",
    "Raw material shipping cost": "lean strategy: logistics costs should be monitored",
    "Total production cost": "lean strategy: production cost is the main lever",
    "Capacity saturation": "lean strategy: capacity should be used efficiently",
}


# ---------------------------------------------------------------------------
# 1. Cleaning
# ---------------------------------------------------------------------------
def clean(sales, products, params=DEFAULTS):
    """Basic cleaning. Returns (clean sales, data-quality report).
    Stock-outs and code changes are NOT repaired: they need data (stock levels,
    code mapping) that a sales extract does not contain."""
    s = sales.copy()
    s["model_code"] = s["model_code"].astype(str)
    s["material_code"] = s["material_code"].astype(str)
    rep = {"rows_in": len(s)}
    # duplicates: same article in the same week
    dup = s.duplicated(["season", "week", "article_code", "units"], keep="first")
    rep["duplicates_removed"] = int(dup.sum()); s = s[~dup]
    # returns are not demand
    ret = s.units < 0
    rep["return_rows_excluded"] = int(ret.sum()); s = s[~ret]
    # spikes: week far above both its neighbouring weeks and the article's season median
    # -> replaced by that reference value (conservative rule: few false positives on clean data)
    key = s.set_index(["article_code", "season", "week"]).units
    key = key[~key.index.duplicated()]
    prev = key.reindex(pd.MultiIndex.from_arrays([s.article_code, s.season, s.week - 1])).fillna(0).values
    nxt = key.reindex(pd.MultiIndex.from_arrays([s.article_code, s.season, s.week + 1])).fillna(0).values
    med = s.groupby(["article_code", "season"]).units.transform("median").values
    ref = np.maximum(np.maximum(prev, nxt), med)
    spike = (s.units.values > params["spike_factor"] * ref) & (s.units.values > ref + params["spike_min_excess"])
    rep["spikes_smoothed"] = int(spike.sum())
    s.loc[spike, "net_sales_eur"] = s.loc[spike, "net_sales_eur"] / s.loc[spike, "units"] * ref[spike]
    s.loc[spike, "units"] = np.round(ref[spike])
    # master data completeness
    rep["units_missing_lead_time"] = int(products.lt_production_weeks.isna().sum() + products.lt_delivery_weeks.isna().sum())
    rep["rows_out"] = len(s)
    return s, rep


# ---------------------------------------------------------------------------
# 2. Aggregation: article x week -> unit (model + material) x season
# ---------------------------------------------------------------------------
def aggregate(sales):
    s = sales.copy()
    s["unit_id"] = s["model_code"].astype(str) + "-" + s["material_code"].astype(str)
    agg = s.groupby(["unit_id", "season"]).agg(units=("units", "sum"), value=("net_sales_eur", "sum"),
                                                 colours=("colour_code", "nunique")).reset_index()
    return agg


# ---------------------------------------------------------------------------
# 3. Features V1-V6 + trend alert
# ---------------------------------------------------------------------------
def season_index(units, products, params=DEFAULTS, min_units=10):
    """Demand index of SS and FW seasons for each category, estimated from the data.
    Estimated only on units sold continuously in the whole CV window (at least 4 seasons, launch
    season excluded), so that intermittent or new products do not distort it. Each season is
    divided by the unit's average; the index is the median of these ratios by category and
    season type, normalised so that SS and FW average to 1. Categories with fewer than
    `min_units` such units use the index of all categories together."""
    cur = SEASONS.index(params["current_season"])
    cat = products.set_index("unit_id").category
    rec = []
    for uid, u in units.iterrows():
        sold = u.values > 0
        if not sold.any():
            continue
        first = int(np.argmax(sold))
        w = u.iloc[max(first + 1, cur + 1 - params["cv_window"]):]
        if len(w) < 4 or (w <= 0).any():
            continue
        for s, v in (w / w.mean()).items():
            rec.append((uid, cat.get(uid), s[:2], v))
    rec = pd.DataFrame(rec, columns=["unit_id", "category", "type", "ratio"])
    pooled = rec.groupby("type").ratio.median()
    n = rec.groupby("category").unit_id.nunique()
    r = rec.groupby(["category", "type"]).ratio.median().unstack()
    for c in r.index:
        if n.get(c, 0) < min_units:
            r.loc[c] = pooled
    return r.div(r.mean(axis=1), axis=0)          # rows: category, columns: SS / FW


def cv_se(cv, n):
    """Large-sample standard error of a coefficient of variation estimated on n values."""
    return cv * np.sqrt((1 + 2 * cv ** 2) / (2 * n))


def seasons_needed(cv, n, threshold, confidence, n_max):
    """Smallest number of seasons for which the interval of the current CV would no longer
    contain the threshold (assuming the CV stays the same). None if more than n_max are needed."""
    from scipy.stats import norm
    z = norm.ppf(0.5 + confidence / 2)
    for m in range(int(n), int(n_max) + 1):
        if abs(cv - threshold) > z * cv_se(cv, m):
            return m
    return None


def cv_interval(cv, n, confidence):
    """Approximate confidence interval of a coefficient of variation estimated on n values
    (large-sample standard error: CV * sqrt((1 + 2 CV^2) / (2n)))."""
    from scipy.stats import norm
    z = norm.ppf(0.5 + confidence / 2)
    se = cv * np.sqrt((1 + 2 * cv ** 2) / (2 * n))
    return max(0.0, cv - z * se), cv + z * se


def features(agg, products, params=DEFAULTS):
    cur = SEASONS.index(params["current_season"])
    units = agg.pivot(index="unit_id", columns="season", values="units").reindex(columns=SEASONS[:cur + 1]).fillna(0)
    value = agg.pivot(index="unit_id", columns="season", values="value").reindex(columns=SEASONS[:cur + 1]).fillna(0)
    cat = products.set_index("unit_id").category
    sidx = season_index(units, products, params) if params.get("deseasonalise") else None
    rows = []
    for uid, u in units.iterrows():
        sold = u.values > 0
        if not sold.any():
            continue
        first = int(np.argmax(sold))
        H = int(sold.sum())                                             # V1
        start = max(first, cur + 1 - params["cv_window"])
        w = u.values[start:].astype(float)                              # zeros count from first appearance
        if sidx is not None and cat.get(uid) in sidx.index:
            w = w / np.array([sidx.loc[cat.get(uid), s[:2]] for s in SEASONS[start:cur + 1]])
        cv = float(w.std() / w.mean()) if len(w) >= 2 and w.mean() > 0 else np.nan   # V2
        lo, hi = (cv_interval(cv, len(w), params["review_confidence"])
                  if params.get("review_confidence") and not np.isnan(cv) else (cv, cv))
        hist = u.values[first:]
        trend = False
        if H >= params["trend_min_history"] and len(hist) >= 4:
            last2, prev = hist[-2:].mean(), hist[:-2][-(params["cv_window"] - 2):].mean()
            trend = bool(prev > 0 and last2 < (1 - params["trend_drop"]) * prev)
        # indicative error of "plan on the average of previous seasons" (WAPE)
        err = np.nan
        if len(hist) >= 3:
            a = hist[2:][-params["cv_window"]:]
            f = np.array([hist[:i].mean() for i in range(2, len(hist))])[-params["cv_window"]:]
            err = float(np.abs(a - f).sum() / a.sum()) if a.sum() > 0 else np.nan
        rows.append(dict(unit_id=uid, first_sale=SEASONS[first], H=H, n_cv=len(w), gaps=int((w == 0).sum()), cv=cv, cv_low=lo, cv_high=hi,
                         value_12m=float(value.loc[uid].values[cur - 1:cur + 1].sum()),
                         trend_alert=trend, naive_forecast_error=err,
                         units_last=float(u.values[cur])))
    f = pd.DataFrame(rows)
    p = products.copy()
    f = p.merge(f, on="unit_id", how="left")
    f = f[f.active_current.astype(bool)].copy()
    # V4 P/D
    f["pd_ratio"] = f.lt_production_weeks / f.lt_delivery_weeks
    # V5 material value: price rank within category ; V6 variants: colours rank within category
    q = params["high_quantile"]
    f["price_pct"] = f.groupby("category").price_fw25.rank(pct=True)
    f["material_high"] = f.price_pct > q
    f["variants_high"] = f.groupby("category").n_colours.transform(lambda x: x > x.quantile(q))
    return f.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Classification
# ---------------------------------------------------------------------------
def abc(values, cuts=(0.80, 0.95)):
    order = values.sort_values(ascending=False)
    share = order.cumsum() / order.sum()
    prev = share.shift(fill_value=0)                  # share before adding the item
    lab = np.where(prev < cuts[0], "A", np.where(prev < cuts[1], "B", "C"))
    return pd.Series(lab, index=order.index).reindex(values.index)


def classify(f, params=DEFAULTS, thresholds=None):
    """Classification S / V / I (N if history is insufficient).
    With cost_ratio = 1 a product is Variable when its CV estimate is above the threshold.
    With cost_ratio = r the DSS accounts for the uncertainty of the estimate and for the cost of
    the two errors: the product is treated as Variable when the probability that its true CV is
    above the threshold exceeds 1 / (1 + r) (r > 1: missing a variable product costs more)."""
    from scipy.stats import norm
    t1, t2 = thresholds or params["cv_thresholds"]
    r = params.get("cost_ratio", 1.0) or 1.0
    p_min = 1 / (1 + r)
    out = []
    for _, x in f.iterrows():
        if x.H < params["min_history"] or pd.isna(x.cv):
            out.append("N"); continue
        se = cv_se(x.cv, x.n_cv) if x.get("gaps", 1) == 0 else 0.0
        if r == 1.0 or se == 0:
            p1, p2 = float(x.cv >= t1), float(x.cv >= t2)
        else:
            p1, p2 = 1 - norm.cdf((t1 - x.cv) / se), 1 - norm.cdf((t2 - x.cv) / se)
        out.append("I" if p2 > p_min else ("V" if p1 > p_min else "S"))
    return pd.Series(out, index=f.index)


def review_flag(f, params=DEFAULTS, thresholds=None):
    """'S/V' or 'V/I' when the CV interval crosses a threshold: the product is close to the
    boundary and the class cannot be decided reliably from the data -> the manager decides.
    Only for products sold in every season of the window: seasons with no sales are direct
    evidence of intermittent (irregular) demand, and the interval approximation does not hold."""
    t1, t2 = thresholds or params["cv_thresholds"]
    ok = (f.H >= params["min_history"]) & f.cv.notna() & (f.gaps == 0)
    c1 = ok & (f.cv_low < t1) & (f.cv_high >= t1)
    c2 = ok & (f.cv_low < t2) & (f.cv_high >= t2)
    flag = pd.Series(np.where(c1 & c2, "S/V/I", np.where(c1, "S/V", np.where(c2, "V/I", ""))), index=f.index)
    return flag


def review_horizon(f, params=DEFAULTS, thresholds=None):
    """For products to review on a CV boundary: how many seasons of data (with the same CV)
    would make the class reliable. If more than the CV window, more data will not settle it
    and the class is a managerial decision."""
    t1, t2 = thresholds or params["cv_thresholds"]
    out = []
    for _, x in f.iterrows():
        b = str(x.to_review).split(" + ")[0]
        if "/" not in b:
            out.append(""); continue
        ths = [t for t, k in [(t1, "S/V"), (t2, "V/I")] if k in b or b == "S/V/I"]
        ms = [seasons_needed(x.cv, x.n_cv, th, params["review_confidence"], 30) for th in ths]
        m = None if None in ms else max(ms)
        if m is None:
            out.append("more than 30 seasons: the CV is on the threshold, manager decision")
        elif m <= params["cv_window"]:
            out.append(f"{m - int(x.n_cv)} more season(s)")
        else:
            out.append(f"about {m} seasons with the same CV, beyond the {params['cv_window']}-season window: manager decision")
    return pd.Series(out, index=f.index)


# ---------------------------------------------------------------------------
# 5. Strategy and order penetration point
# ---------------------------------------------------------------------------
def opp(row, params=DEFAULTS):
    if pd.isna(row.pd_ratio):
        return "n.a. (lead time missing)"
    if row.pd_ratio <= params["pd_threshold"]:
        return "MTO"
    return "ATO" if (row.material_high or row.variants_high) else "MTS"


def explain(row):
    """One-sentence explanation of the suggestion, shown in the product card."""
    if row.dss_class == "N":
        why = f"only {int(row.H)} season(s) of sales: variability cannot be estimated reliably"
    else:
        why = f"{int(row.H)} seasons on the market, variability of seasonal sales CV = {row.cv:.2f}"
    head = "Insufficient history" if row.dss_class == "N" else f"{CLASS_NAMES[row.dss_class]} demand"
    s = f"{head} ({why}) -> {row.strategy}"
    if not str(row.opp).startswith("n.a."):
        s += f"; P/D = {row.pd_ratio:.1f} -> {row.opp}"
        if row.opp == "ATO":
            s += " (" + " and ".join([x for x, c in [("costly materials", row.material_high), ("many variants", row.variants_high)] if c]) + ")"
    rv = str(getattr(row, "to_review", ""))
    if "/" in rv:
        b = rv.split(" + ")[0]
        where = "across both thresholds" if b == "S/V/I" else f"across the {b.replace('/', '-')} boundary"
        s += (f". TO REVIEW: with {int(row.n_cv)} seasons the CV could lie between {row.cv_low:.2f} and {row.cv_high:.2f}, "
              f"{where}: the class should be confirmed by the manager")
        if getattr(row, "review_horizon", ""):
            s += f" (reliable after: {row.review_horizon})"
    if getattr(row, "relaunch_of", ""):
        s += (f". POSSIBLE CODE CHANGE: it may continue {row.relaunch_of} (same category, line, material, supplier and "
              f"price; sold until {row.relaunch_until})")
    if row.trend_alert:
        s += ". ALERT: sales are declining" + (" - check whether the product is reaching the end of its life cycle"
                                               if getattr(row, "decline_check", False) else "")
    return s


# ---------------------------------------------------------------------------
# 6. KPI alignment
# ---------------------------------------------------------------------------
def kpi_alignment(f, kpi_measured):
    meas = {line: set(zip(g.process, g.kpi)) for line, g in kpi_measured.groupby("product_line")}
    out = []
    for _, r in f.iterrows():
        req = required_kpis(r.dss_class)
        have = meas.get(r.product_line, set())
        missing = sorted(req - have, key=lambda x: x[0])
        out.append(dict(alignment=len(req & have) / len(req), n_required=len(req), n_missing=len(missing),
                        missing_kpis="; ".join(k for _, k in missing),
                        missing_explained=" | ".join(f"{k}: {KPI_WHY.get(k, '')}" for _, k in missing),
                        n_not_required=len(have - req)))
    return pd.DataFrame(out, index=f.index)


# ---------------------------------------------------------------------------
# Possible code changes (relaunch check)
# ---------------------------------------------------------------------------
def relaunch_check(agg, products, f, params=DEFAULTS):
    """A product sold under a new code looks like a new product and loses its history.
    Using only master data and sales dates, flag recent continuative products that may continue an
    older code: same category, product line, material and supplier, price within a tolerance, and
    the old code stopped selling in the one or two seasons before the new one started.
    The histories are NOT merged automatically: the DSS shows what the class would be with the
    combined history, and the manager decides."""
    cur = SEASONS.index(params["current_season"])
    sold = agg[agg.units > 0].assign(si=lambda d: d.season.map(SEASONS.index))
    span = sold.groupby("unit_id").si.agg(["min", "max"])
    p = products.drop_duplicates("unit_id").set_index("unit_id")
    keys = ["category", "product_line", "material", "supplier"]
    out = pd.DataFrame(index=f.index, columns=["relaunch_of", "relaunch_until", "relaunch_class", "relaunch_H", "relaunch_cv"], dtype=object)
    new = f[f.product_line.isin(["Permanent", "Carry-over"]) & f.unit_id.isin(span.index)]
    new = new[new.unit_id.map(span["min"]) >= cur + 1 - params["relaunch_recent"]]
    olds = span[(span["max"] < cur)].join(p[keys + ["price_fw25"]], how="inner")
    pairs = {}
    for i, x in new.iterrows():
        first = span.loc[x.unit_id, "min"]
        c = olds[(olds["max"] >= first - 2) & (olds["max"] <= first - 1)]
        for k in keys:
            c = c[c[k] == p.loc[x.unit_id, k]]
        c = c[(c.price_fw25 - p.loc[x.unit_id, "price_fw25"]).abs() <= params["relaunch_price_tol"] * p.loc[x.unit_id, "price_fw25"]]
        if len(c):
            best = (c.price_fw25 - p.loc[x.unit_id, "price_fw25"]).abs().idxmin()
            pairs[best] = x.unit_id
            out.loc[i, "relaunch_of"] = best
            out.loc[i, "relaunch_until"] = SEASONS[int(c.loc[best, "max"])]
    if pairs:   # class the product would have with the combined history
        merged = agg.assign(unit_id=agg.unit_id.replace(pairs)).groupby(["unit_id", "season"], as_index=False)[["units", "value", "colours"]].sum()
        fm = features(merged[merged.unit_id.isin(list(pairs.values())) | ~merged.unit_id.isin(list(pairs))], products, params)
        fm = fm[fm.unit_id.isin(list(pairs.values()))].copy()
        fm["cls"] = classify(fm, params)
        fm = fm.set_index("unit_id")
        for i in out.index[out.relaunch_of.notna()]:
            u = f.loc[i, "unit_id"]
            out.loc[i, ["relaunch_class", "relaunch_H", "relaunch_cv"]] = [fm.loc[u, "cls"], fm.loc[u, "H"], fm.loc[u, "cv"]]
    out["relaunch_of"] = out.relaunch_of.fillna("")
    return out


# ---------------------------------------------------------------------------
# Priority list for the manager
# ---------------------------------------------------------------------------
def priority_list(res, top=None):
    """Products whose current management (by label) differs from the DSS suggestion, or whose
    class must be confirmed, ordered by value of the last 12 months (where to start)."""
    r = res.copy()
    r["label_strategy"] = r.product_line.map(LABEL_STRATEGY)
    mismatch = (r.dss_class != "N") & (r.strategy != r.label_strategy)
    review = r.to_review != ""
    missing = r.n_missing > 0
    decline = r.decline_check.astype(bool) if "decline_check" in r else False
    relaunch = (r.relaunch_of != "") if "relaunch_of" in r else False
    r = r[mismatch | review | missing | decline | relaunch].copy()
    def reason(x):
        out = []
        if x.dss_class != "N" and x.strategy != x.label_strategy:
            out.append(f"managed as {x.label_strategy} ({x.product_line}), DSS suggests {x.strategy}")
        if x.to_review:
            out.append(f"class to confirm ({x.to_review})")
        if x.n_missing:
            out.append(f"{x.n_missing} KPIs missing")
        if x.trend_alert:
            out.append("declining sales: end of life cycle?")
        if "relaunch_of" in x and x.relaunch_of:
            out.append(f"possible code change of {x.relaunch_of}")
        return "; ".join(out)
    r["reason"] = r.apply(reason, axis=1)
    r["value_share"] = r.value_12m / res.value_12m.sum()
    r = r.sort_values("value_12m", ascending=False)
    cols = ["unit_id", "category", "product_line", "abc", "dss_class", "strategy", "to_review", "decline_check", "value_12m", "value_share", "alignment", "reason"]
    cols += [c for c in ["relaunch_of"] if c in r]
    return (r[cols].head(top) if top else r[cols]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 7. Clustering check
# ---------------------------------------------------------------------------
def cluster_check(f, classes, k_range=range(2, 7), seed=2026):
    """k-means on standardised H, CV, log(value) for classified products (H >= min history).
    Returns silhouette by k, best k, ARI between k-means and rule classes, and
    a flag for products whose rule class differs from the majority class of their cluster."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    from sklearn.preprocessing import StandardScaler
    m = classes != "N"
    X = np.column_stack([f.loc[m, "H"], f.loc[m, "cv"], np.log1p(f.loc[m, "value_12m"])])
    X = StandardScaler().fit_transform(X)
    sil, labels = {}, {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)
        sil[k] = silhouette_score(X, km.labels_); labels[k] = km.labels_
    k_best = max(sil, key=sil.get)
    rule = classes[m].values
    lab3 = labels[3]
    maj = pd.Series(rule).groupby(lab3).agg(lambda x: x.value_counts().index[0])
    uncertain = pd.Series(False, index=f.index)
    uncertain[m] = rule != maj.reindex(lab3).values
    return dict(silhouette=sil, k_best=k_best,
                ari_best=adjusted_rand_score(rule, labels[k_best]), ari_k3=adjusted_rand_score(rule, lab3),
                uncertain=uncertain)


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------
def run(sales, products, kpi_measured, params=DEFAULTS):
    s, report = clean(sales, products, params)
    agg = aggregate(s)
    f = features(agg, products, params)
    f["abc"] = abc(f.value_12m.fillna(0), params["abc_cuts"])
    f["dss_class"] = classify(f, params)
    f["to_review"] = review_flag(f, params) if params.get("review_confidence") else ""
    f["review_horizon"] = review_horizon(f, params) if params.get("review_confidence") else ""
    f["trend_alert"] = f.trend_alert.eq(True)          # products without sales -> False (no object/NaN mix)
    f["decline_check"] = f.trend_alert & bool(params.get("decline_to_review"))
    for name, th in SCENARIOS.items():
        f[f"class_{name}"] = classify(f, params, th)
    f["strategy"] = f.dss_class.map(STRATEGY)
    f["opp"] = f.apply(opp, axis=1, params=params)
    if params.get("relaunch_check"):
        f = pd.concat([f, relaunch_check(agg, products, f, params)], axis=1)
    else:
        f["relaunch_of"] = ""
    f["explanation"] = f.apply(explain, axis=1)
    f = pd.concat([f, kpi_alignment(f, kpi_measured)], axis=1)
    report["possible_code_changes"] = int((f.relaunch_of != "").sum())
    return f, report
