"""
Maison A - Supply chain strategy DSS (dialogue component).

The dashboard only calls dss_core.py (model component) on the company data (data component):
what the manager sees is exactly what the analysis notebook computes.
"""
import inspect
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import dss_core as dss
import report_pdf

st.set_page_config(page_title="Maison A – Strategy DSS", page_icon="🧭", layout="wide")

# ---------------------------------------------------------------------------
# Apple-inspired look: system font, light grey canvas, white rounded cards,
# segmented controls instead of radio buttons, iOS system colours
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp, button, input, textarea, select {
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", Inter, "Segoe UI", sans-serif !important;
  -webkit-font-smoothing: antialiased;
}
.stApp { background: #FFFFFF; }
[data-testid="stAppDeployButton"], [data-testid="stDecoration"] { display: none; }
header[data-testid="stHeader"] { background: transparent; }
h1 { font-weight: 700 !important; letter-spacing: -0.03em; }
h2, h3, h4 { font-weight: 650 !important; letter-spacing: -0.02em; }
[data-testid="stCaptionContainer"], .stCaption { color: #6E6E73 !important; }
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"], [data-testid="stThumbValue"], [data-testid="stTickBar"] * {
  font-family: inherit !important;
}

/* sidebar */
[data-testid="stSidebar"] { background: #F5F5F7; border-right: 1px solid #E5E5EA; }

/* metric tiles, as on apple.com */
[data-testid="stMetric"] { background: #F5F5F7; border-radius: 18px; padding: 16px 20px; }
[data-testid="stMetricValue"] { font-weight: 650; letter-spacing: -0.02em; }
[data-testid="stExpander"] details, [data-testid="stExpander"] { border-radius: 14px !important; border-color: #E5E5EA !important; }
[data-testid="stAlert"] { border-radius: 14px; }
hr { border-color: #E5E5EA !important; }

/* buttons: pill shaped */
.stButton > button, .stDownloadButton > button { border-radius: 980px !important; padding: 0.35rem 1.1rem; font-weight: 500; }

/* radio buttons rendered as iOS segmented controls */
div[role="radiogroup"] {
  display: inline-flex !important; flex-wrap: wrap; gap: 2px; width: fit-content;
  background: #E3E3E8; padding: 3px; border-radius: 11px;
}
div[role="radiogroup"] > label {
  margin: 0 !important; padding: 6px 16px !important; border-radius: 9px; cursor: pointer; transition: background .15s;
}
div[role="radiogroup"] > label > div:first-child { display: none !important; }
div[role="radiogroup"] > label:has(input:checked) { background: #FFFFFF; box-shadow: 0 1px 3px rgba(0,0,0,.12), 0 1px 1px rgba(0,0,0,.04); }
div[role="radiogroup"] > label:has(input:checked) p { font-weight: 600; }
div[role="radiogroup"] > label p { font-size: 0.92rem; }
/* newer Streamlit markup */
div[role="radiogroup"] > div { margin: 0 !important; }
label[data-testid="stRadioOption"] { margin: 0 !important; padding: 6px 16px !important; border-radius: 9px; cursor: pointer; }
label[data-testid="stRadioOption"] > div > div:first-child { display: none !important; }
label[data-testid="stRadioOption"][data-selected="true"] { background: #FFFFFF; box-shadow: 0 1px 3px rgba(0,0,0,.12), 0 1px 1px rgba(0,0,0,.04); }
label[data-testid="stRadioOption"][data-selected="true"] p { font-weight: 600; }
label[data-testid="stRadioOption"] p { font-size: 0.92rem; }
</style>
""", unsafe_allow_html=True)

import plotly.io as pio
pio.templates["apple"] = go.layout.Template(layout=dict(
    font=dict(family="-apple-system, BlinkMacSystemFont, SF Pro Text, Inter, Segoe UI, sans-serif", color="#1D1D1F", size=13),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(gridcolor="#E5E5EA", zerolinecolor="#E5E5EA", linecolor="#E5E5EA"),
    yaxis=dict(gridcolor="#E5E5EA", zerolinecolor="#E5E5EA", linecolor="#E5E5EA"),
    title=dict(font=dict(size=15)), colorway=["#007AFF", "#FF9500", "#FF3B30", "#34C759", "#AF52DE", "#8E8E93"]))
pio.templates.default = "apple"

# ---------------------------------------------------------------------------
# Compatibility with older Streamlit versions (e.g. the one shipped with Anaconda)
# ---------------------------------------------------------------------------
_VER = tuple(int(x) for x in st.__version__.split(".")[:2])
W = {"width": "stretch"} if _VER >= (1, 46) else {"use_container_width": True}
_rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun")
CAN_SELECT = "on_select" in inspect.signature(st.dataframe).parameters


def box():
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


# ---------------------------------------------------------------------------
# Vocabulary shown to the manager
# ---------------------------------------------------------------------------
CLASS_COLORS = {"S": "#007AFF", "V": "#FF9500", "I": "#FF3B30", "N": "#8E8E93"}   # iOS system colours
CLASS_NAME = {"S": "Stable", "V": "Variable", "I": "Irregular", "N": "Too little history"}
CLASS_ICON = {"S": "🔵", "V": "🟠", "I": "🔴", "N": "⚪"}
STRAT_TXT = {"Lean": "Lean – efficiency and low cost",
             "Hybrid": "Hybrid – efficient base with a flexible response",
             "Agile": "Agile – responsiveness and flexibility",
             "Hybrid (prudent)": "Hybrid (prudent) – flexible until demand is known"}
OPP_TXT = {"MTS": "Make to stock – produce in advance, because production takes longer than customers wait",
           "MTO": "Make to order – produce on order, because production is faster than the expected delivery time",
           "ATO": "Assemble to order – prepare components in advance and finish on order (costly materials or many variants)"}
ATTITUDE = {"Neutral": 1.0, "Cautious": 2.0, "Very cautious": 3.0}
PAGES = ["Overview", "Where to start", "Product", "KPIs", "Settings & validation"]
# KPI configurations (Bindi et al., 2023): companies assign KPIs by product line,
# so the actionable output is "which products should be monitored with which KPI set"
KPI_SET_NAME = {"lean": "cost checklist (lean)", "hybrid": "flexibility & forecast checklist (hybrid)",
                "seasonal": "new-product checklist (seasonal / capsule)"}
MOVE_TXT = {("lean", "hybrid"): "have variable or irregular demand, but are monitored as stable products",
            ("lean", "seasonal"): "have too little sales history to be judged, but are already monitored as consolidated products",
            ("seasonal", "hybrid"): "have been sold for several seasons with uneven demand, but are still monitored as new products",
            ("seasonal", "lean"): "have been sold for several seasons with stable demand, but are still monitored as new products"}
def move_txt(key, n):
    s_ = MOVE_TXT.get(key, "should change KPI checklist")
    return s_ if n > 1 else s_.replace("have ", "has ", 1).replace("are ", "is ")


def kpi_set_today(line):
    return "lean" if line in ("Permanent", "Carry-over") else "seasonal"
def kpi_set_needed(cls):
    return {"S": "lean", "V": "hybrid", "I": "hybrid", "N": "seasonal"}[cls]
KPI_SETS_FULL = {"lean": dss.required_kpis("S"), "hybrid": dss.required_kpis("V"), "seasonal": dss.required_kpis("N")}
PROCESS_ORDER = ["New product development", "Plan", "Source", "Make", "Deliver", "Return"]
DT = {"model_code": str, "material_code": str, "colour_code": str, "article_code": str}
HERE = os.path.dirname(os.path.abspath(__file__))
DECISIONS_FILE = os.path.join(HERE, "manager_decisions.csv")
# Data of the dashboard: the development dataset of Maison A
DATA_DIR = os.path.join(HERE, "data") if os.path.isdir(os.path.join(HERE, "data")) else os.path.join(HERE, "maison_a")

GLOSSARY = """
**Demand classes**
- 🔵 **Stable** – sales are similar every season: they can be planned and produced efficiently.
- 🟠 **Variable** – sales change noticeably from season to season.
- 🔴 **Irregular** – sales are very uneven or stop and restart.
- ⚪ **Too little history** – fewer than 3 seasons of sales: too early to judge.

**Strategies**
- **Lean** – focus on efficiency and low cost.
- **Agile** – focus on speed and flexibility.
- **Hybrid** – efficient base plus a flexible part for what cannot be predicted.

**Order penetration point (when to produce)**
- **MTS** make to stock · **ATO** assemble to order · **MTO** make to order.
- **P/D** – production lead time divided by the delivery time customers accept. Above 1, waiting for the order is not possible.

**Other terms**
- **Variation (CV)** – how much seasonal sales move around their average. 0.30 = about ±30%.
- **ABC** – A = the products making 80% of revenue, B = the next 15%, C = the last 5%.
- **Class to confirm** – the product lies close to a boundary between two classes: the data cannot decide and the manager should.
- **KPI alignment** – share of the KPIs needed for the suggested strategy that are already measured.
"""

HOW_TO = """
1. **Overview** – read the key messages and see how the current product lines compare with actual demand.
2. **Where to start** – the products that need attention, most valuable first.
3. **Product** – click a product to see what the DSS suggests, why, and what to do.
4. **Decide** – confirm the suggestion or change the class. Your decisions are saved.
5. **KPIs** – check which KPIs are missing and for which products.

The DSS only *suggests*: every setting on this panel can be changed.
"""


# ---------------------------------------------------------------------------
# Data and model
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load(dataset):
    sfx = "" if dataset == "clean" else "_dirty"
    products = pd.read_csv(os.path.join(DATA_DIR, f"products{sfx}.csv"), dtype=DT)
    sales = pd.read_csv(os.path.join(DATA_DIR, f"sales{sfx}.csv"), dtype=DT)
    kpi = pd.read_csv(os.path.join(DATA_DIR, "kpi_measured.csv"))
    return products, sales, kpi


@st.cache_data(show_spinner=False)
def run_model(dataset, params_items):
    params = dict(params_items)
    products, sales, kpi = load(dataset)
    res, report = dss.run(sales, products, kpi, params)
    s = sales.copy()
    s["unit_id"] = s.model_code + "-" + s.material_code
    seasonal = s.groupby(["unit_id", "season"]).units.sum().unstack().reindex(columns=dss.SEASONS).fillna(0)
    return res, report, seasonal


@st.cache_data(show_spinner=False)
def classes_for(dataset, params_items, thresholds, cost_ratio):
    res, _, _ = run_model(dataset, params_items)
    return dss.classify(res, dict(dict(params_items), cost_ratio=cost_ratio), thresholds)


def load_decisions():
    if os.path.exists(DECISIONS_FILE):
        try:
            d = pd.read_csv(DECISIONS_FILE, dtype=str).fillna("")
            return {r.unit_id: {"decision": r.decision, "class": r.get("class", ""), "note": r.note} for _, r in d.iterrows()}
        except Exception:
            return {}
    return {}


def save_decisions(dec):
    try:
        pd.DataFrame([dict(unit_id=u, **v) for u, v in dec.items()]).to_csv(DECISIONS_FILE, index=False)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
SIDEBAR_DEFAULTS = {"dataset": "clean", "attitude": "Neutral", "scen": "base (0.5 / 1.0)", "custom_t": (0.5, 1.0),
                    "min_hist": dss.DEFAULTS["min_history"], "conf": dss.DEFAULTS["review_confidence"],
                    "abc": dss.DEFAULTS["abc_cuts"], "drop": dss.DEFAULTS["trend_drop"],
                    "pd_thr": dss.DEFAULTS["pd_threshold"], "deseas": dss.DEFAULTS["deseasonalise"]}
for k, v in SIDEBAR_DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("page", PAGES[0])
st.session_state.setdefault("uid", None)
st.session_state.setdefault("pc_tab", "Sales history")
st.session_state.setdefault("kpi_focus", None)
st.session_state.setdefault("kpi_highlight", None)
st.session_state.setdefault("ov_hist", True)
for _k, _v in {"pf_search": "", "pf_cat": [], "pf_line": [], "pf_class": [], "pf_attention": False, "pf_special": False}.items():
    st.session_state.setdefault(_k, _v)
if "decisions" not in st.session_state:
    st.session_state.decisions = load_decisions()


def reset_settings():
    for k, v in SIDEBAR_DEFAULTS.items():
        st.session_state[k] = v


PRODUCT_FILTERS = {"pf_search": "", "pf_cat": [], "pf_line": [], "pf_class": [], "pf_attention": False, "pf_special": False}


PC_TABS = ["Sales history", "KPIs", "Technical details"]


def open_product(uid, tab="Sales history"):
    for k_, v_ in PRODUCT_FILTERS.items():
        st.session_state[k_] = v_
    st.session_state.uid = uid
    st.session_state.pc_tab = tab
    st.session_state.kpi_focus = None
    st.session_state.page = PAGES[2]
    st.session_state._scroll_top = True


def open_kpi_group(a, b, uid):
    st.session_state.kpi_focus = (a, b)
    st.session_state.kpi_highlight = uid
    st.session_state.page = PAGES[3]
    st.session_state._scroll_top = True


def _on_pick():
    sel = st.session_state.get("plist_table")
    rows = sel.selection.rows if sel is not None and hasattr(sel, "selection") else []
    if rows:
        i = rows[0]
        open_product(st.session_state._plist_view[i], st.session_state._plist_tab[i])


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧭 Strategy DSS")
    st.caption("Maison A · simulated data · season FW25")
    with st.expander("📖 How to use this dashboard", expanded=False):
        st.markdown(HOW_TO)
    with st.expander("🔤 Glossary"):
        st.markdown(GLOSSARY)
    st.divider()

    st.radio("Sales data", ["clean", "dirty"], key="dataset", horizontal=True,
             format_func=lambda x: {"clean": "Clean", "dirty": "With errors"}[x],
             help="'With errors' contains missing rows, spikes, duplicates, returns, stock-outs, code changes and "
                  "missing lead times – to see how the DSS copes with real-world data.")
    st.select_slider("Attitude towards risk", list(ATTITUDE), key="attitude",
                     help="How serious is it to manage as *stable* a product whose demand is actually variable "
                          "(stock-outs, unsold stock, no control on forecasts)? 'Cautious' and 'Very cautious' treat "
                          "products close to the boundary as Variable more easily (cost ratio 2 and 3).")

    with st.expander("⚙️ Advanced settings"):
        st.selectbox("Variation thresholds", ["base (0.5 / 1.0)", "alternative (0.25 / 0.5)", "custom"], key="scen",
                     help="Variation (CV) below the first threshold = Stable, between the two = Variable, above = Irregular.")
        if st.session_state.scen == "custom":
            st.slider("Stable < t1 ≤ Variable < t2 ≤ Irregular", 0.05, 2.0, key="custom_t", step=0.05)
        st.number_input("Minimum seasons of history", 2, 6, key="min_hist",
                        help="Products with fewer seasons are 'Too little history'.")
        st.select_slider("Confidence of the 'class to confirm' check", [0.7, 0.8, 0.9, 0.95], key="conf",
                         help="Higher = wider uncertainty band = more products to confirm.")
        st.slider("ABC cut-offs (cumulative revenue)", 0.5, 0.99, key="abc", step=0.01)
        st.slider("Declining-sales alert: drop of last 2 seasons", 0.1, 0.6, key="drop", step=0.05,
                  help="Alert when the average of the last two seasons is this much below the previous ones.")
        st.number_input("P/D threshold (at or below: make to order)", 0.5, 3.0, key="pd_thr", step=0.1)
        st.checkbox("Remove SS/FW seasonality before measuring variation", key="deseas")
    st.button("↺ Reset settings", on_click=reset_settings)
    st.caption("Every setting is a suggestion of the DSS and can be changed by the manager.")

ss = st.session_state
if ss.scen.startswith("base"):
    t1, t2 = dss.SCENARIOS["base"]
elif ss.scen.startswith("alternative"):
    t1, t2 = dss.SCENARIOS["alternative"]
else:
    t1, t2 = ss.custom_t
cost_ratio = ATTITUDE[ss.attitude]
params = dict(dss.DEFAULTS, cv_thresholds=(t1, t2), cost_ratio=cost_ratio, min_history=int(ss.min_hist),
              review_confidence=ss.conf, abc_cuts=tuple(ss.abc), trend_drop=ss.drop, pd_threshold=ss.pd_thr,
              deseasonalise=ss.deseas)
params_items = tuple(sorted(params.items()))

with st.spinner("Analysing the portfolio…"):
    res, report, seasonal = run_model(ss.dataset, params_items)

# manager decisions applied on top of the DSS suggestion
dec = ss.decisions
res["manager_class"] = res.unit_id.map(lambda u: dec.get(u, {}).get("class") or None)
res["final_class"] = res.manager_class.fillna(res.dss_class)
res["final_strategy"] = res.final_class.map(dss.STRATEGY)
res["label_strategy"] = res.product_line.map(dss.LABEL_STRATEGY)
plist = dss.priority_list(res.assign(dss_class=res.final_class, strategy=res.final_strategy))
plist["decision"] = plist.unit_id.map(lambda u: dec.get(u, {}).get("decision", ""))


def signals(r):
    s = []
    if r.to_review: s.append("🔶 class to confirm")
    if r.decline_check: s.append("📉 declining")
    if "DSS suggests" in r.reason: s.append("🔁 strategy mismatch")
    if "KPIs missing" in r.reason: s.append("⚠️ KPIs missing")
    if "code change" in r.reason: s.append("🔗 code change?")
    return "  ".join(s)


plist["signals"] = plist.apply(signals, axis=1)


def _severity(r):
    if "DSS suggests" in r.reason:
        return 1, "Managed with a strategy that does not match demand"
    if r.to_review:
        return 2, "Class to confirm (close to a boundary)"
    if r.decline_check:
        return 2, "Declining sales – end of life cycle?"
    if r.get("relaunch_of", ""):
        return 2, "Possible code change – sales history may be split"
    return 3, "Only the KPI checklist to update"


_sev = plist.apply(_severity, axis=1)
plist["priority"] = [s_[0] for s_ in _sev]
plist["main_issue"] = [s_[1] for s_ in _sev]
plist["priority_label"] = plist.priority.map({1: "High", 2: "Medium", 3: "Low"})
plist = plist.sort_values(["priority", "value_12m"], ascending=[True, False]).reset_index(drop=True)
n_reviewed = int((plist.decision != "").sum())


def progress_bar():
    frac = n_reviewed / len(plist) if len(plist) else 1.0
    st.progress(frac, text=f"Review progress: {n_reviewed} of {len(plist)} products needing attention reviewed")

# ---------------------------------------------------------------------------
# Header and navigation
# ---------------------------------------------------------------------------
st.title("Supply chain strategy DSS")
st.caption(f"Maison A · {len(res)} active products · season {params['current_season']} · data: "
           f"{'clean' if ss.dataset == 'clean' else 'with errors'} · thresholds {t1:g} / {t2:g} · attitude: {ss.attitude.lower()}")
def _nav_changed():
    st.session_state._scroll_top = True
    st.session_state.kpi_focus = None


page = st.radio("Section", PAGES, key="page", horizontal=True, label_visibility="collapsed", on_change=_nav_changed)
if ss.get("_scroll_top"):
    ss._scroll_top = False
    import streamlit.components.v1 as components
    components.html("""<script>
      const d = window.parent.document;
      ['[data-testid="stMain"]', '[data-testid="stAppViewContainer"]', 'section.main', '.main'].forEach(s => {
        const e = d.querySelector(s); if (e) { e.scrollTo({top: 0}); }
      });
      window.parent.scrollTo({top: 0});
    </script>""", height=0)
st.divider()

def _key_messages():
    co = res[(res.product_line == "Carry-over") & (res.final_class != "N")]
    var = res[res.final_class.isin(["V", "I"]) & res.product_line.isin(["Carry-over", "Permanent"])]
    m = [f"{int((co.final_class != 'S').sum())} of {len(co)} carry-over products with enough history do not have stable "
         f"demand, but the label leads to managing them as stable.",
         f"{len(plist)} products need attention ({plist.value_share.sum():.0%} of revenue): "
         f"{int((plist.priority == 1).sum())} high, {int((plist.priority == 2).sum())} medium, {int((plist.priority == 3).sum())} low priority.",
         f"{int((res.to_review != '').sum())} products lie close to a class boundary and need the manager's confirmation.",
         f"{int(res.decline_check.sum())} products show declining sales."]
    if len(var):
        m.append(f"Continuative products with variable demand measure on average only {var.alignment.mean():.0%} of the KPIs they need.")
    return m


def _kpi_groups():
    r_ = res.assign(set_today=res.product_line.map(kpi_set_today), set_need=res.final_class.map(kpi_set_needed))
    out = []
    _order = [("lean", "hybrid"), ("lean", "seasonal"), ("seasonal", "hybrid"), ("seasonal", "lean")]
    _groups = sorted(r_[r_.set_today != r_.set_need].groupby(["set_today", "set_need"]),
                     key=lambda x: _order.index(x[0]) if x[0] in _order else 9)
    for (a_, b_), g in _groups:
        n_ok = int(g.unit_id.map(lambda u: dec.get(u, {}).get("decision", "") != "").sum())
        fe = g.naive_forecast_error.median()
        out.append({"title": f"{len(g)} product{'s' if len(g) > 1 else ''} {move_txt((a_, b_), len(g))}",
                    "from_to": f"Today: {KPI_SET_NAME[a_]}  ->  Needed: {KPI_SET_NAME[b_]}  "
                               f"(revenue concerned: EUR {g.value_12m.sum() / 1e6:,.1f} M)",
                    "status": f"{n_ok} decided by the manager, {len(g) - n_ok} not yet reviewed",
                    "preview": (f"Planning these products on past averages would be wrong by about {fe:.0%}."
                                if pd.notna(fe) and b_ == "hybrid" else None),
                    "add": [k_ for _, k_ in sorted(KPI_SETS_FULL[b_] - KPI_SETS_FULL[a_])],
                    "drop": [k_ for _, k_ in sorted(KPI_SETS_FULL[a_] - KPI_SETS_FULL[b_])]})
    return out


with st.sidebar:
    st.divider()
    st.markdown("**📄 Summary report**")
    st.caption("A PDF with key messages, priorities, KPI changes and your decisions – to share or print.")
    if st.button("Prepare PDF report"):
        with st.spinner("Preparing the report…"):
            ss.report_pdf = report_pdf.build(
                res, plist, dec,
                f"season {params['current_season']} · data {'clean' if ss.dataset == 'clean' else 'with errors'} · "
                f"thresholds {t1:g}/{t2:g} · attitude {ss.attitude.lower()}",
                _key_messages(), _kpi_groups(), CLASS_NAME)
    if ss.get("report_pdf"):
        st.download_button("⬇ Download PDF", ss.report_pdf, "DSS_summary_report.pdf", "application/pdf", type="primary")

legend = "  ".join(f"{CLASS_ICON[k]} {CLASS_NAME[k]}" for k in ["S", "V", "I", "N"])

# ---------------------------------------------------------------------------
# ① Overview
# ---------------------------------------------------------------------------
if page == PAGES[0]:
    co = res[(res.product_line == "Carry-over") & (res.final_class != "N")]
    co_ns = int((co.final_class != "S").sum())
    var = res[res.final_class.isin(["V", "I"]) & res.product_line.isin(["Carry-over", "Permanent"])]
    st.subheader("Key messages")
    with box():
        st.markdown(
            f"- **{co_ns} of {len(co)} carry-over products** with enough history do **not** have stable demand, "
            f"but the label leads to managing them as stable.\n"
            f"- **{len(plist)} products** need attention – together **{plist.value_share.sum():.0%} of revenue**. "
            f"Start from *Where to start*.\n"
            f"- **{int((res.to_review != '').sum())} products** lie close to a class boundary: the data cannot decide, "
            f"**your confirmation** is needed.\n"
            f"- **{int(res.decline_check.sum())} products** show **declining sales**: are they reaching the end of their life cycle?\n"
            + (f"- Continuative products with variable demand measure on average only **{var.alignment.mean():.0%}** "
               f"of the KPIs they need – forecast accuracy is the most frequent gap." if len(var) else ""))

    progress_bar()
    c = st.columns(4)
    for i, k in enumerate(["S", "V", "I", "N"]):
        c[i].metric(f"{CLASS_ICON[k]} {CLASS_NAME[k]}", int((res.final_class == k).sum()),
                    help={"S": "Regular sales – plan and produce efficiently (lean).",
                          "V": "Sales change from season to season – efficient base plus flexibility (hybrid).",
                          "I": "Very uneven sales – keep flexibility (agile).",
                          "N": "Fewer than 3 seasons of sales – manage prudently until demand is known."}[k])

    st.subheader("Product lines today vs actual demand")
    st.caption("Each row is a product line as the company labels it; each column is the demand class the DSS finds in the "
               "sales data. A label that matched demand would put all its products in one column.")
    _toggle = getattr(st, "toggle", st.checkbox)
    _toggle("Only products with enough sales history", key="ov_hist",
            help="Products with fewer than the minimum seasons of sales cannot be classified yet: hiding them makes the "
                 "comparison between labels and actual demand easier to read.")
    cols_ = ["S", "V", "I"] if ss.ov_hist else ["S", "V", "I", "N"]
    ct = pd.crosstab(res.product_line, res.final_class).reindex(index=["Permanent", "Carry-over", "Seasonal", "Capsule"],
                                                               columns=cols_, fill_value=0)
    if ss.ov_hist:
        ct = ct[ct.sum(axis=1) > 0]
        st.caption(f"{int((res.final_class == 'N').sum())} products with too little history are hidden "
                   f"(all {int((res.product_line == 'Capsule').sum())} capsules and most seasonal products are new this season).")
    fig = px.imshow(ct.rename(columns={k: f"{CLASS_ICON[k]} {CLASS_NAME[k]}" for k in ct.columns}), text_auto=True,
                    color_continuous_scale=["#F2F2F7", "#007AFF"], aspect="auto")
    fig.update_layout(height=300, coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=None,
                      yaxis_title=None, font=dict(size=14))
    st.plotly_chart(fig, **W)

    with st.expander("🔎 Explore: variation and value of every product"):
        st.caption("Each dot is a product. Further right = sales vary more between seasons; higher = more revenue. "
                   "Dashed lines are the thresholds between classes; diamonds are products whose class should be confirmed.")
        m = res.final_class != "N"
        d = res[m].assign(Class=res.final_class[m].map(lambda k: f"{CLASS_ICON[k]} {CLASS_NAME[k]}"),
                          Status=np.where(res.to_review[m] != "", "to confirm", "clear"))
        fig = px.scatter(d, x="cv", y="value_12m", color="Class", symbol="Status", log_y=True,
                         symbol_map={"clear": "circle", "to confirm": "diamond"},
                         color_discrete_map={f"{CLASS_ICON[k]} {CLASS_NAME[k]}": v for k, v in CLASS_COLORS.items()},
                         hover_data={"unit_id": True, "category": True, "product_line": True, "H": True},
                         labels={"cv": "Variation of seasonal sales (CV)", "value_12m": "Revenue last 12 months (€, log scale)",
                                 "H": "seasons", "product_line": "line"})
        for t in (t1, t2):
            fig.add_vline(x=t, line_dash="dash", line_color="grey")
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, **W)

# ---------------------------------------------------------------------------
# ② Where to start
# ---------------------------------------------------------------------------
elif page == PAGES[1]:
    st.subheader("Where to start")
    st.caption("Products whose management should be reviewed: **first** those managed with a strategy that does not match "
               "demand, **then** those whose class must be confirmed or whose sales are declining, **last** those where only "
               "the KPI checklist must be updated. Within each level, the most valuable first. "
               + ("**Click the box at the left of a row** to open its product card." if CAN_SELECT else "Pick a product below to open its card."))
    progress_bar()
    f0, f1, f2, f3 = st.columns(4)
    pri = f0.multiselect("Priority", ["High", "Medium", "Low"])
    sig = f1.multiselect("Signal", ["🔶 class to confirm", "📉 declining", "🔁 strategy mismatch", "⚠️ KPIs missing", "🔗 code change?"])
    cat = f2.multiselect("Category", sorted(res.category.unique()))
    rev = f3.multiselect("Review status", ["⏳ to review", "✔ reviewed"])
    keep = pd.Series(True, index=plist.index)
    if pri: keep &= plist.priority_label.isin(pri)
    for s_ in sig:
        keep &= plist.signals.str.contains(s_.split(" ", 1)[1], regex=False)
    if cat: keep &= plist.category.isin(cat)
    if rev and len(rev) == 1:
        keep &= (plist.decision != "") if rev[0].startswith("✔") else (plist.decision == "")
    view = plist[keep].reset_index(drop=True)
    st.markdown(f"**{len(view)} products** · {view.value_share.sum():.0%} of revenue")
    table = pd.DataFrame({
        "Priority": view.priority_label.map({"High": "● High", "Medium": "◐ Medium", "Low": "○ Low"}),
        "Product": view.unit_id, "Main issue": view.main_issue, "Category": view.category, "Line today": view.product_line,
        "ABC": view.abc, "DSS class": view.dss_class.map(lambda k: f"{CLASS_ICON[k]} {CLASS_NAME[k]}"),
        "Suggested strategy": view.strategy, "Signals": view.signals,
        "Revenue 12m (€ k)": (view.value_12m / 1000).round(0),
        "KPI alignment": view.alignment * 100,
        "Decision": view.decision.map({"confirmed": "✔ confirmed", "changed": "✎ changed"}).fillna("").replace("", "⏳ to review")})
    colcfg = {"Revenue 12m (€ k)": st.column_config.NumberColumn(format="%d"),
              "Main issue": st.column_config.TextColumn(width="large"),
              "Signals": st.column_config.TextColumn(width="medium"),
              "KPI alignment": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100)}
    tabs_for = ["KPIs" if p_ == 3 else "Sales history" for p_ in view.priority]
    if CAN_SELECT:
        ss._plist_view = view.unit_id.tolist()
        ss._plist_tab = tabs_for
        st.dataframe(table, hide_index=True, height=520, column_config=colcfg, on_select=_on_pick,
                     selection_mode="single-row", key="plist_table", **W)
    else:
        st.dataframe(table, hide_index=True, height=520, column_config=colcfg, **W)
        pick = st.selectbox("Open product", view.unit_id, format_func=lambda u: f"{u} · {view.set_index('unit_id').loc[u, 'main_issue']}")
        st.button("Open product card →", on_click=open_product, args=(pick,), type="primary")
    st.download_button("⬇ Download this list (CSV)", view.drop(columns=["signals"]).to_csv(index=False),
                       "priority_list.csv", "text/csv")

# ---------------------------------------------------------------------------
# ③ Product
# ---------------------------------------------------------------------------
elif page == PAGES[2]:
    st.subheader("Find a product")
    f = st.columns([2, 1.3, 1.3, 1.6, 1.2])
    f[0].text_input("Search by code", key="pf_search", placeholder="e.g. 2754 or 24F47")
    f[1].multiselect("Category", sorted(res.category.unique()), key="pf_cat")
    f[2].multiselect("Line today", ["Permanent", "Carry-over", "Seasonal", "Capsule"], key="pf_line")
    f[3].multiselect("DSS class", ["S", "V", "I", "N"], key="pf_class", format_func=lambda x: f"{CLASS_ICON[x]} {CLASS_NAME[x]}")
    f[4].checkbox("Needs attention", key="pf_attention", help="Only products in the 'Where to start' list")
    f[4].checkbox("Special editions", key="pf_special", help="Only special editions of a classic model")
    cand = res.copy()
    if ss.pf_search.strip():
        cand = cand[cand.unit_id.str.contains(ss.pf_search.strip(), case=False, regex=False)]
    if ss.pf_cat: cand = cand[cand.category.isin(ss.pf_cat)]
    if ss.pf_line: cand = cand[cand.product_line.isin(ss.pf_line)]
    if ss.pf_class: cand = cand[cand.final_class.isin(ss.pf_class)]
    if ss.pf_attention: cand = cand[cand.unit_id.isin(plist.unit_id)]
    if ss.pf_special: cand = cand[cand.is_special_edition.astype(str).str.lower() == "true"]
    attention = set(plist.unit_id)
    cand = cand.assign(_att=cand.unit_id.isin(attention)).sort_values(["_att", "value_12m"], ascending=[False, False])
    options = cand.unit_id.tolist()
    if not options:
        st.warning("No product matches these filters.")
        st.stop()
    if ss.uid not in options:
        ss.uid = options[0]
    lab = {r.unit_id: f"{r.unit_id} · {r.category} · {r.product_line}"
                      + (" (special edition)" if str(r.is_special_edition).lower() == "true" else "")
                      + f" · {CLASS_ICON[r.final_class]} {CLASS_NAME[r.final_class]}"
                      + ("  · needs attention" if r.unit_id in attention else "")
           for r in res.itertuples()}
    uid = st.selectbox(f"Product ({len(options)} found – you can also type in the box)", options, key="uid",
                       format_func=lambda u: lab[u])
    st.divider()
    r = res.set_index("unit_id").loc[uid]
    k = r.final_class
    strategy = dss.STRATEGY[k]

    st.markdown(f"#### {uid}  ·  {r.category}  ·  line *{r.product_line}*  ·  ABC {r.abc}")
    st.caption(f"{r.material} · {int(r.n_colours)} colour(s) · {int(r.n_sizes)} size(s) · price € {r.price_fw25:,.0f} · "
               f"revenue last 12 months € {r.value_12m:,.0f}")
    if str(r.get("is_special_edition", "")).lower() == "true" and str(r.get("parent_unit", "")) in set(res.unit_id):
        se = st.columns([4, 1.4])
        se[0].info(f"✨ Special edition of the classic model **{r.parent_unit}** – different material and supplier, so the DSS "
                   f"treats it as a separate product.")
        se[1].button("Open the classic model →", key=f"par_{uid}", on_click=open_product, args=(r.parent_unit,))
    sons = res[(res.get("parent_unit", pd.Series("", index=res.index)).astype(str) == uid)]
    if len(sons):
        st.caption("Special editions of this model: " + ", ".join(sons.unit_id))

    # --- verdict
    if k == "N":
        why = f"Only {int(r.H)} season(s) of sales: too early to judge how demand behaves."
    else:
        cvv = float(r.cv)
        level = ("very regular" if cvv < 0.20 else "fairly regular" if cvv < 0.35 else "moderately variable"
                 if cvv < 0.50 else "clearly variable" if cvv < 1.0 else "very uneven")
        why = (f"Over the last {int(r.n_cv)} seasons sales have been {level}: on average they move about "
               f"±{cvv:.0%} around their mean")
        why += f", and in {int(r.gaps)} season(s) there were no sales." if r.get("gaps", 0) else "."
    opp_txt = OPP_TXT.get(r.opp, "Not available – the lead time is missing in the master data.")

    actions = []
    if k != "N" and strategy != r.label_strategy:
        actions.append(f"**Review the strategy** – today managed as *{r.label_strategy}* ({r.product_line}), "
                       f"demand suggests *{strategy}*.")
    if r.to_review:
        actions.append("**Confirm the class** – the product lies on the boundary between "
                       f"{' and '.join(CLASS_NAME[x] for x in r.to_review.split('/'))}; more seasons of data would not settle it.")
    if r.decline_check:
        actions.append("**Check the life cycle** – sales are declining: is the product reaching the end of its life?")
    if r.n_missing:
        miss = str(r.missing_kpis).split("; ")
        actions.append(f"**Start measuring {len(miss)} KPI(s)** – " + ", ".join(miss[:4]) + ("…" if len(miss) > 4 else ""))
    if str(r.opp).startswith("n.a."):
        actions.append("**Complete the master data** – the production lead time is missing.")
    if r.get("relaunch_of", ""):
        rc_ = r.relaunch_class
        actions.append(f"**Check for a code change** – it may continue **{r.relaunch_of}** (same category, line, material, "
                       f"supplier and price; sold until {r.relaunch_until}). If it is the same product, with the combined "
                       f"history of {int(r.relaunch_H)} seasons its class would be **{CLASS_ICON[rc_]} {CLASS_NAME[rc_]}** "
                       f"(variation ±{float(r.relaunch_cv):.0%}): use *Change class* to apply it.")
    if k == "N":
        actions.append(f"**Re-assess later** – manage prudently and look again after "
                       f"{max(1, params['min_history'] - int(r.H))} more season(s).")
    if not actions:
        actions.append("**No action needed** – current management is consistent with demand.")

    today_set, need_set = kpi_set_today(r.product_line), kpi_set_needed(k)
    order_ = plist.unit_id.tolist()
    pending = [u for u in order_ if dec.get(u, {}).get("decision", "") == "" and u != uid]
    if uid in order_:
        after = [u for u in order_[order_.index(uid) + 1:] if u in pending]
        nxt = after[0] if after else (pending[0] if pending else None)
    else:
        nxt = pending[0] if pending else None
    nxt_tab = "KPIs" if nxt and int(plist.set_index("unit_id").loc[nxt, "priority"]) == 3 else "Sales history"

    left, right = st.columns([3, 2])
    with left:
        with box():
            st.markdown(f"<div style='border-left:6px solid {CLASS_COLORS[k]};padding:2px 0 2px 12px'>"
                        f"<div style='font-size:0.85rem;opacity:.7'>THE DSS SUGGESTS</div>"
                        f"<div style='font-size:1.25rem;font-weight:600'>{CLASS_ICON[k]} {CLASS_NAME[k]}{' demand' if k != 'N' else ''}"
                        f"{' (close to the boundary)' if r.to_review else ''} → {STRAT_TXT[strategy]}</div>"
                        f"<div style='margin-top:4px'>{opp_txt}</div></div>", unsafe_allow_html=True)
            if r.manager_class:
                st.info(f"Class set by the manager (DSS suggestion was {CLASS_NAME[r.dss_class]}).")
            st.markdown(f"**Why** – {why}")
            st.markdown("**What to do**\n" + "\n".join(f"- {a}" for a in actions))

    with right:
        if k != "N" and pd.notna(r.cv):
            xmax = max(1.5, float(r.cv_high) * 1.1, t2 * 1.3)
            fig = go.Figure()
            for x0, x1, c_, name in [(0, t1, "S", "Stable"), (t1, t2, "V", "Variable"), (t2, xmax, "I", "Irregular")]:
                fig.add_shape(type="rect", x0=x0, x1=x1, y0=0, y1=1, fillcolor=CLASS_COLORS[c_], opacity=0.25, line_width=0)
                fig.add_annotation(x=(x0 + x1) / 2, y=1.25, text=name, showarrow=False, font=dict(size=11))
            if r.to_review:
                fig.add_shape(type="rect", x0=r.cv_low, x1=r.cv_high, y0=0.2, y1=0.8, fillcolor="rgba(80,80,80,0.35)", line_width=0)
            fig.add_trace(go.Scatter(x=[r.cv], y=[0.5], mode="markers", marker=dict(size=16, color="#1f1e1d", symbol="diamond"),
                                     hovertemplate="variation %{x:.2f}<extra></extra>"))
            fig.update_layout(height=130, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                              title=dict(text="Where the product lies", font=dict(size=14)),
                              xaxis=dict(range=[0, xmax], title=None), yaxis=dict(visible=False, range=[0, 1.4]))
            st.plotly_chart(fig, **W)
            if r.to_review:
                st.caption("Grey band = range where the true variation may lie. It crosses a boundary: your call.")

        # --- decision, at the top
        prev = dec.get(uid, {})
        with box():
            st.markdown("**Your decision**")
            choice = st.radio("Decision", ["Confirm suggestion", "Change class"], horizontal=True,
                              index=1 if prev.get("decision") == "changed" else 0, key=f"d_{uid}", label_visibility="collapsed")
            dc = st.columns([1.2, 2])
            new_cls = dc[0].selectbox("New class", ["S", "V", "I", "N"], format_func=lambda x: f"{CLASS_ICON[x]} {CLASS_NAME[x]}",
                                      index=["S", "V", "I", "N"].index(prev.get("class") or r.dss_class), key=f"c_{uid}",
                                      disabled=choice.startswith("Confirm"))
            note = dc[1].text_input("Reason (optional)", prev.get("note", ""), key=f"n_{uid}")
            bc = st.columns([1, 1.6])
            if bc[0].button("💾 Save", type="primary", key=f"s_{uid}"):
                dec[uid] = dict(decision="confirmed" if choice.startswith("Confirm") else "changed",
                                **{"class": "" if choice.startswith("Confirm") else new_cls}, note=note)
                ok = save_decisions(dec)
                st.toast("Decision saved" + ("" if ok else " for this session only"))
                _rerun()
            if nxt:
                bc[1].button("Next product to review →", key=f"next_{uid}", on_click=open_product, args=(nxt, nxt_tab))
            if prev:
                st.caption(f"Saved: {prev['decision']}" + (f" → {CLASS_NAME.get(prev.get('class'), '')}" if prev.get("class") else "")
                           + (f" · {prev['note']}" if prev.get("note") else ""))
            st.caption(f"Reviewed {n_reviewed} of {len(plist)} products needing attention.")

    # --- details in tabs
    st.radio("Details", PC_TABS, key="pc_tab", horizontal=True, label_visibility="collapsed")
    if ss.pc_tab == "Sales history":
        hist = seasonal.loc[uid] if uid in seasonal.index else pd.Series(0, index=dss.SEASONS)
        fig = go.Figure(go.Bar(x=hist.index, y=hist.values, marker_color=CLASS_COLORS[k]))
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=35, b=10), title="Units sold per season", yaxis_title="units")
        st.plotly_chart(fig, **W)

    elif ss.pc_tab == "KPIs":
        kc = st.columns([3, 2])
        with kc[0]:
            st.markdown(f"**{r.alignment:.0%} of the KPIs this product needs are measured** "
                        f"({int(r.n_required - r.n_missing)} of {int(r.n_required)})")
            if today_set == need_set:
                st.success(f"Monitored with the right KPI checklist: *{KPI_SET_NAME[need_set]}*.")
            else:
                st.warning(f"Today monitored with the *{KPI_SET_NAME[today_set]}* (assigned by its line); "
                           f"it needs the **{KPI_SET_NAME[need_set]}**.")
            if r.n_missing:
                st.markdown("**Needed but not measured:**")
                for item in str(r.missing_explained).split(" | "):
                    name, _, why_k = item.partition(": ")
                    st.markdown(f"- **{name}** – {why_k}")
            if today_set != need_set:
                n_same = int(((res.product_line.map(kpi_set_today) == today_set) & (res.final_class.map(kpi_set_needed) == need_set)).sum())
                st.button(f"See the {n_same} products in the same situation →", on_click=open_kpi_group,
                          args=(today_set, need_set, uid))
        with kc[1]:
            if pd.notna(r.naive_forecast_error):
                fe = float(r.naive_forecast_error)
                msg = (f"**Preview from sales data** – planning each season on the average of the previous ones, "
                       f"the plan would have been wrong by about **{fe:.0%}** on this product")
                ref = res.loc[res.final_class == "S", "naive_forecast_error"].median()
                if pd.notna(ref) and k != "S":
                    msg += f" (stable products: about {ref:.0%})"
                msg += ". This is what forecast accuracy would show if it were measured."
                (st.info if fe < 0.35 else st.warning)(msg)
        with st.expander("All KPIs this product needs"):
            have = KPI_SETS_FULL[today_set]
            need = sorted(KPI_SETS_FULL[need_set], key=lambda x: (PROCESS_ORDER.index(x[0]), x[1]))
            chk = pd.DataFrame([{"Process": p_, "KPI": k_, "Status": "✅ measured" if (p_, k_) in have else "❌ missing"}
                                for p_, k_ in need])
            st.dataframe(chk, hide_index=True, **W)
            extra = sorted(have - KPI_SETS_FULL[need_set])
            if extra:
                st.caption("Measured today but not needed for this product: " + ", ".join(k_ for _, k_ in extra))

    else:
        vt = pd.DataFrame({"value": [
            str(int(r.H)) if pd.notna(r.H) else "–",
            f"{r.cv:.2f}" if pd.notna(r.cv) else "–",
            f"{r.cv_low:.2f} – {r.cv_high:.2f}" if pd.notna(r.cv) else "–",
            f"€ {r.value_12m:,.0f}" if pd.notna(r.value_12m) else "–",
            f"{r.pd_ratio:.1f}" if pd.notna(r.pd_ratio) else "missing",
            "high" if r.material_high else "normal", "many" if r.variants_high else "normal",
            f"{r.naive_forecast_error:.0%}" if pd.notna(r.naive_forecast_error) else "–"]},
            index=["V1 seasons on the market", "V2 variation (CV)", "CV uncertainty range", "V3 revenue last 12 months",
                   "V4 P/D", "V5 material value", "V6 variants", "Error if planned on past average"])
        st.table(vt)
        st.caption(r.explanation)

# ---------------------------------------------------------------------------
# ④ KPIs
# ---------------------------------------------------------------------------
elif page == PAGES[3]:
    st.subheader("Are products monitored with the right KPIs?")
    with box():
        st.markdown(
            "There are three KPI checklists (Bindi et al., 2023):\n"
            "- **cost checklist (lean)** – production, inventory and obsolescence costs: for products with stable demand;\n"
            "- **flexibility & forecast checklist (hybrid)** – forecast errors, time to market, available capacity: "
            "for products with variable or irregular demand;\n"
            "- **new-product checklist (seasonal / capsule)** – also prototype and sample costs: for new products.\n\n"
            "Today each product gets the checklist of its **product line**. The DSS gives it the checklist of its "
            "**actual demand**. Below are the products that should change checklist.")
        st.caption("This page reflects your decisions in *Product*: if you change a product's class there, it updates here. "
                   "Confirmed products are ready to apply; the others still need your review.")
    r_ = res.assign(set_today=res.product_line.map(kpi_set_today), set_need=res.final_class.map(kpi_set_needed),
                    status=res.unit_id.map(lambda u: "✔ confirmed" if dec.get(u, {}).get("decision") == "confirmed"
                                           else ("✎ class changed" if dec.get(u, {}).get("decision") == "changed" else "⏳ to review")))
    moves = r_[r_.set_today != r_.set_need]
    if moves.empty:
        st.success("Every product is monitored with the KPI checklist it needs.")
    else:
        groups = moves.groupby(["set_today", "set_need"])
        ORDER = [("lean", "hybrid"), ("lean", "seasonal"), ("seasonal", "hybrid"), ("seasonal", "lean")]
        focus = tuple(ss.kpi_focus) if ss.kpi_focus else None
        if focus:
            st.info(f"Showing first the group of product **{ss.kpi_highlight}** (marked with 👉).")
        for (a_, b_), g in sorted(groups, key=lambda x: (x[0] != focus, ORDER.index(x[0]) if x[0] in ORDER else 9)):
            add = sorted(KPI_SETS_FULL[b_] - KPI_SETS_FULL[a_], key=lambda x: (PROCESS_ORDER.index(x[0]), x[1]))
            drop = sorted(KPI_SETS_FULL[a_] - KPI_SETS_FULL[b_], key=lambda x: (PROCESS_ORDER.index(x[0]), x[1]))
            n_ok = int((g.status != "⏳ to review").sum()); n_todo = len(g) - n_ok
            lines = ", ".join(f"{c} {x.lower()}" for x, c in g.product_line.value_counts().items())
            with box():
                c1, c2 = st.columns([3, 1.2])
                c1.markdown(f"#### {len(g)} product{'s' if len(g) > 1 else ''} {move_txt((a_, b_), len(g))}")
                c1.markdown(f"Today: *{KPI_SET_NAME[a_]}* → Needed: **{KPI_SET_NAME[b_]}**")
                c1.caption(f"{lines} · ✔ {n_ok} decided by you – ready to apply · ⏳ {n_todo} not yet reviewed")
                fe_g = g.naive_forecast_error.median()
                if pd.notna(fe_g) and ("Forecast accuracy" in {k_ for _, k_ in KPI_SETS_FULL[b_] - KPI_SETS_FULL[a_]}):
                    fe_s = res.loc[res.final_class == "S", "naive_forecast_error"].median()
                    c1.markdown(f"📊 *Preview from sales data:* planning these products on past averages would be wrong by "
                                f"about **{fe_g:.0%}** (stable products: {fe_s:.0%}) – but today nobody measures it.")
                c2.metric("Revenue concerned", f"€ {g.value_12m.sum() / 1e6:,.1f} M")
                c2.caption(f"{g.value_12m.sum() / res.value_12m.sum():.0%} of total revenue")
                keep_ = sorted(KPI_SETS_FULL[a_] & KPI_SETS_FULL[b_], key=lambda x: (PROCESS_ORDER.index(x[0]), x[1]))
                cc = st.columns(3)
                for col_, title_, items_ in [(cc[0], "❌ Needed but not measured", add),
                                             (cc[1], "✅ Already measured correctly", keep_),
                                             (cc[2], "➖ Measured but not needed", drop)]:
                    with col_:
                        st.markdown(f"**{title_} ({len(items_)})**")
                        with st.expander("Show list"):
                            st.markdown("\n".join(f"- {k_} *({p_})*" for p_, k_ in items_) or "None.")

                with st.expander(f"Review and decide – {len(g)} products ({n_todo} not yet reviewed)",
                                 expanded=(a_, b_) == focus):
                    st.caption("Each product starts as *to review*. Choose **Confirm** if you agree with the DSS class, or pick "
                               "another class. If in doubt, open the product card below to see its sales history.")
                    opts = ["⏳ to review", "✔ confirm"] + [f"→ {CLASS_NAME[x]}" for x in ["S", "V", "I", "N"]]
                    def cur_opt(u):
                        d_ = dec.get(u, {})
                        if d_.get("decision") == "confirmed": return "✔ confirm"
                        if d_.get("decision") == "changed" and d_.get("class"): return f"→ {CLASS_NAME[d_['class']]}"
                        return "⏳ to review"
                    gg = g.assign(_hl=g.unit_id == ss.kpi_highlight).sort_values(["_hl", "value_12m"], ascending=[False, False])
                    ed_df = pd.DataFrame({
                        "Focus": np.where(gg._hl, "👉", ""),
                        "Product": gg.unit_id.values, "Line today": gg.product_line.values,
                        "DSS class": gg.dss_class.map(lambda x: f"{CLASS_ICON[x]} {CLASS_NAME[x]}").values,
                        "Sales variation": gg.cv.map(lambda v: f"±{v:.0%}" if pd.notna(v) else "–").values,
                        "Seasons sold": gg.H.astype("Int64").values,
                        "At a boundary": gg.to_review.map(lambda v: "🔶 yes" if v else "").values,
                        "Declining": gg.decline_check.map(lambda v: "📉 yes" if v else "").values,
                        "Revenue 12m (€ k)": (gg.value_12m / 1000).round(0).values,
                        "Your decision": [cur_opt(u) for u in gg.unit_id]})
                    edited = st.data_editor(
                        ed_df, hide_index=True, key=f"ed_{a_}_{b_}", disabled=[c for c in ed_df.columns if c != "Your decision"],
                        column_config={"Your decision": st.column_config.SelectboxColumn(options=opts, required=True, width="medium")},
                        **W)
                    bc = st.columns([1.2, 2, 1.3])
                    if bc[0].button("💾 Save decisions", type="primary", key=f"save_{a_}_{b_}"):
                        n_saved = 0
                        for u, choice in zip(edited["Product"], edited["Your decision"]):
                            if choice == cur_opt(u):
                                continue
                            if choice == "⏳ to review":
                                dec.pop(u, None)
                            elif choice == "✔ confirm":
                                dec[u] = {"decision": "confirmed", "class": "", "note": dec.get(u, {}).get("note", "")}
                            else:
                                new_k = {f"→ {CLASS_NAME[x]}": x for x in ["S", "V", "I", "N"]}[choice]
                                dec[u] = {"decision": "changed", "class": new_k, "note": dec.get(u, {}).get("note", "")}
                            n_saved += 1
                        save_decisions(dec)
                        st.toast(f"{n_saved} decision(s) saved")
                        _rerun()
                    pick = bc[1].selectbox("Open a product card", gg.unit_id.tolist(), key=f"pick_{a_}_{b_}",
                                           label_visibility="collapsed")
                    bc[2].button("Open product card →", key=f"open_{a_}_{b_}", on_click=open_product, args=(pick,))
        plan = []
        for _, x in moves.iterrows():
            for p_, k_ in sorted(KPI_SETS_FULL[x.set_need] - KPI_SETS_FULL[x.set_today]):
                plan.append({"product": x.unit_id, "category": x.category, "line": x.product_line,
                             "dss_class": CLASS_NAME[x.final_class], "your_decision": x.status,
                             "checklist_today": KPI_SET_NAME[x.set_today], "checklist_needed": KPI_SET_NAME[x.set_need],
                             "process": p_, "kpi_to_start": k_, "why": dss.KPI_WHY.get(k_, "")})
        st.download_button("⬇ Download the KPI changes (CSV)", pd.DataFrame(plan).to_csv(index=False),
                           "kpi_changes.csv", "text/csv")

    st.divider()
    st.subheader("Why: where the gap is")
    c = st.columns(2)
    with c[0]:
        st.markdown("**1 · How it looks today** – by product line label")
        by_line = res.groupby("product_line").alignment.mean().reindex(["Permanent", "Carry-over", "Seasonal", "Capsule"])
        fig = go.Figure(go.Bar(x=list(by_line.index), y=by_line.values * 100, marker_color="#8E8E93",
                               text=[f"{v:.0%}" for v in by_line.values], textposition="outside"))
        fig.update_layout(height=340, margin=dict(l=60, r=10, t=20, b=60), yaxis=dict(range=[0, 115], title="KPIs measured (%)"))
        fig.update_xaxes(tickangle=0, automargin=False); fig.update_yaxes(automargin=False)
        st.plotly_chart(fig, **W)
    with c[1]:
        st.markdown("**2 · Where the gap really is** – by actual demand class")
        by_cls = res.groupby(res.final_class).alignment.mean().reindex(["S", "V", "I", "N"]).dropna()
        fig = go.Figure(go.Bar(x=[CLASS_NAME[i].replace("Too little history", "Too little<br>history") for i in by_cls.index],
                               y=by_cls.values * 100,
                               marker_color=[CLASS_COLORS[i] for i in by_cls.index], text=[f"{v:.0%}" for v in by_cls.values],
                               textposition="outside"))
        fig.update_layout(height=340, margin=dict(l=60, r=10, t=20, b=60), yaxis=dict(range=[0, 115], title="KPIs measured (%)"))
        fig.update_xaxes(tickangle=0, automargin=False); fig.update_yaxes(automargin=False)
        st.plotly_chart(fig, **W)
    co_al = res.loc[res.product_line == "Carry-over", "alignment"].mean()
    v_al = res.loc[res.final_class == "V", "alignment"].mean()
    st.info(f"By label, the gap looks small: carry-over products measure {co_al:.0%} of the KPIs they need. "
            f"By actual demand, the gap is concentrated on products with **variable demand** ({v_al:.0%}) – "
            "most of them hidden inside the carry-over line and managed as if they were stable.")

# ---------------------------------------------------------------------------
# ⑤ Settings & validation
# ---------------------------------------------------------------------------
elif page == PAGES[4]:
    st.caption("This section is for analysts: it shows how the results depend on the settings and how the DSS was checked.")
    t_a, t_b, t_c = st.tabs(["What changes with the settings", "Clustering check", "Data quality"])

    with t_a:
        st.markdown("**Variation thresholds** – how many products fall in each class with each set of thresholds.")
        rows = []
        for name, th, cr in [("base", dss.SCENARIOS["base"], 1.0), ("alternative", dss.SCENARIOS["alternative"], 1.0),
                             ("current settings", (t1, t2), cost_ratio)]:
            cl = classes_for(ss.dataset, params_items, th, cr)
            rows.append({"scenario": name, "thresholds": f"{th[0]:g} / {th[1]:g}", "attitude (cost ratio)": cr,
                         **{CLASS_NAME[k]: int((cl == k).sum()) for k in ["S", "V", "I", "N"]},
                         "carry-over not stable": int(((cl != "S") & (cl != "N") & (res.product_line == "Carry-over")).sum())})
        st.dataframe(pd.DataFrame(rows), hide_index=True, **W)
        st.caption("The boundary between stable and variable is a managerial choice: the data do not show a natural gap.")

        st.markdown("**Attitude towards risk** – how many products change class when missing a variable product is "
                    "considered more costly than the opposite error.")
        base_cl = classes_for(ss.dataset, params_items, (t1, t2), 1.0)
        rows = []
        for name, cr in [("Neutral", 1.0), ("", 1.5), ("Cautious", 2.0), ("Very cautious", 3.0), ("", 5.0)]:
            cl = classes_for(ss.dataset, params_items, (t1, t2), cr)
            rows.append({"attitude": name, "cost ratio": cr, **{CLASS_NAME[k]: int((cl == k).sum()) for k in ["S", "V", "I"]},
                         "products changing class": int((cl != base_cl).sum())})
        st.dataframe(pd.DataFrame(rows), hide_index=True, **W)

    with t_b:
        st.markdown("An algorithm (k-means) groups the products **by itself**, without knowing the thresholds, using "
                    "seasons on the market, variation and revenue. If its groups resemble the DSS classes, the thresholds "
                    "follow a structure that exists in the data.")
        st.markdown("- **best k** – number of groups the algorithm finds most natural (silhouette).\n"
                    "- **ARI** – agreement between algorithm and rules: 1 = identical, 0 = no better than chance.\n"
                    "- **ARI k=3** – agreement when the algorithm is forced to make three groups, like the three classes.\n"
                    "- **uncertain products** – products on which rules and algorithm disagree.")
        if st.button("Run clustering check", type="primary"):
            out = []
            with st.spinner("Running k-means…"):
                for name, th in [("base", dss.SCENARIOS["base"]), ("alternative", dss.SCENARIOS["alternative"]), ("current", (t1, t2))]:
                    chk = dss.cluster_check(res, dss.classify(res, params, th))
                    out.append({"scenario": name, "best k": chk["k_best"], "ARI best k": round(chk["ari_best"], 2),
                                "ARI k=3": round(chk["ari_k3"], 2), "uncertain products": int(chk["uncertain"].sum())})
            st.dataframe(pd.DataFrame(out), hide_index=True, **W)
            st.caption("The scenario with the highest agreement is chosen as default (rule fixed before seeing the results).")

    with t_c:
        labels_rep = {"rows_in": "Sales rows received", "duplicates_removed": "Duplicates removed",
                      "return_rows_excluded": "Return rows excluded (not demand)", "spikes_smoothed": "Spikes smoothed",
                      "units_missing_lead_time": "Products with missing lead time", "rows_out": "Sales rows used"}
        st.table(pd.DataFrame({"count": [f"{report[k]:,}" for k in labels_rep]}, index=list(labels_rep.values())))
        st.caption("Stock-outs and product code changes are not repaired: they require data (stock levels, code mapping) "
                   "that a sales extract does not contain. Products with missing lead time get no order penetration point.")
        st.download_button("⬇ Download full DSS results (CSV)", res.to_csv(index=False), f"dss_results_{ss.dataset}.csv", "text/csv")
