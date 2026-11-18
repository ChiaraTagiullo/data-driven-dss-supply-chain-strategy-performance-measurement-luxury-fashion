# %% [markdown]
# # DSS evaluation – Maison A
#
# Artificial, formative evaluation of the DSS (Chapter 5), following the evaluation plan (D10) and the
# **pre-registration** of 13 November 2026 (`PREREGISTRATION.md`): hypotheses, success criteria and stress
# scenarios were fixed before the test data were opened.
#
# The same code runs first on the **development** data (dry run) and then **once** on the **test** data
# (`maison_a_test`, seed 2027) and on five **stress datasets** generated with new seeds, never inspected before.
#
# | # | Check | What it shows | What it does NOT show |
# |---|---|---|---|
# | 1 | Demonstration on typical cases | the DSS gives sensible, explainable suggestions | that they are right in general |
# | 2 | Verification against the generating profiles | the system works as designed | that the profiles resemble a real maison |
# | 3 | Comparison with label-based segmentation | what would change compared with today | the real economic benefit |
# | 4 | Clustering check | thresholds follow a structure present in the data | that thresholds are "right" |
# | 5 | Sensitivity analysis | how much results depend on parameters | which parameter is "correct" |
# | 6 | Dirty-data test | robustness to imperfect data | robustness to errors other than those simulated |
# | 7 | Requirements check (R1–R9) | the design answers its requirements | usefulness perceived by real managers |
# | 8 | Stress tests (S1–S5) | range of validity when data assumptions change | behaviour on real company data |
# | 9 | Pre-registered hypotheses (H1–H5) | results judged against criteria fixed in advance | – |

# %%
import hashlib, json, os, subprocess, sys, zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use(os.environ.get("MPLBACKEND", matplotlib.get_backend()))
import matplotlib.pyplot as plt
import dss_core as dss

MODE = os.environ.get("EVAL_MODE", "dev")          # "dev" = dry run on development data, "test" = final test (once)
FOLDER = "maison_a" if MODE == "dev" else "maison_a_test"
STRESS_SEEDS = {"dev": [9001, 9002, 9003, 9004, 9005], "test": [3001, 3002, 3003, 3004, 3005]}[MODE]
for z in ["maison_a.zip", "maison_a_test.zip"]:
    if os.path.exists(z) and not os.path.isdir(z[:-4]):
        zipfile.ZipFile(z).extractall(".")
OUT = f"results_{MODE}"
os.makedirs(OUT, exist_ok=True)
DT = {"model_code": str, "material_code": str, "colour_code": str, "article_code": str}
pd.set_option("display.width", 200, "display.max_columns", 30)


def load(folder, dirty=False):
    sfx = "_dirty" if dirty else ""
    return (pd.read_csv(f"{folder}/sales{sfx}.csv", dtype=DT), pd.read_csv(f"{folder}/products{sfx}.csv", dtype=DT),
            pd.read_csv(f"{folder}/kpi_measured.csv"))


def save(df, name):
    df.to_csv(f"{OUT}/{name}.csv", index=True)
    return df


def pct(v):
    return f"{v:.0%}" if pd.notna(v) else "–"

# %% [markdown]
# ## 0. Frozen version
# The model evaluated is version 3 with default parameters, frozen on 10 November 2026. On the test data the notebook
# refuses to run if `dss_core.py` or the parameters differ from the fingerprint in `VERSION.json`.

# %%
sha = hashlib.sha256(open("dss_core.py", "rb").read()).hexdigest()
params = dict(dss.DEFAULTS)
psha = hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()
ver = json.load(open("VERSION.json"))
same = ver["dss_core_sha256"] == sha and ver["params_sha256"] == psha
print(f"Frozen version {ver['version']} of {ver['frozen_on']}: model {'MATCHES' if same else 'DIFFERS'}")
assert same or MODE == "dev", "The model differs from the frozen version: the test must be run on the frozen version only."
print("dss_core.py sha256:", sha[:16], "…  params sha256:", psha[:16], "…  mode:", MODE, "· dataset:", FOLDER)

# %% [markdown]
# ## Run the DSS on clean data and load the hidden profiles
# `ground_truth.csv` is used **only here, for evaluation**; the DSS never reads it.

# %%
sales, products, kpi = load(FOLDER)
res, report = dss.run(sales, products, kpi, params)
gt = pd.read_csv(f"{FOLDER}/ground_truth.csv")
ev = res.merge(gt[["unit_id", "profile"]], on="unit_id", how="left")
CONT = ["Permanent", "Carry-over"]
NONSTABLE = ["oscillating", "declining"]
print(f"{len(res)} active products · profiles: {ev.profile.value_counts().to_dict()}")


def detection(e):
    """Continuative products with enough history: share of non-stable ones flagged, false alarms on stable ones."""
    c = e[e.product_line.isin(CONT) & (e.dss_class != "N")]
    st_, ns = c[c.profile == "stable"], c[c.profile.isin(NONSTABLE)]
    flag = lambda x: x.dss_class.isin(["V", "I"]) | (x.to_review.astype(str) != "") | x.decline_check.astype(bool)
    return {"stable products": len(st_), "non-stable products": len(ns),
            "non-stable classified V/I": ns.dss_class.isin(["V", "I"]).mean() if len(ns) else np.nan,
            "non-stable flagged": flag(ns).mean() if len(ns) else np.nan,
            "stable classified V/I": st_.dss_class.isin(["V", "I"]).mean() if len(st_) else np.nan,
            "stable flagged (false alarm)": flag(st_).mean() if len(st_) else np.nan}

# %% [markdown]
# ## 1. Demonstration on typical cases
# Cases are chosen **by rule** (the highest-revenue product of each type), not by hand.

# %%
def pick(mask):
    s_ = ev[mask].sort_values("value_12m", ascending=False)
    return s_.iloc[0] if len(s_) else None

cases = {
    "Carry-over with stable demand": pick((ev.product_line == "Carry-over") & (ev.dss_class == "S") & (ev.to_review == "")),
    "Carry-over with variable demand": pick((ev.product_line == "Carry-over") & (ev.dss_class == "V")),
    "Carry-over close to a boundary": pick((ev.product_line == "Carry-over") & (ev.to_review != "")),
    "Seasonal product re-proposed (irregular)": pick((ev.product_line == "Seasonal") & (ev.dss_class == "I")),
    "Recent carry-over (too little history)": pick((ev.product_line == "Carry-over") & (ev.dss_class == "N")),
    "Special edition of a classic model": pick(ev.is_special_edition.astype(str).str.lower() == "true"),
}
se = cases["Special edition of a classic model"]
if se is not None and se.parent_unit in set(ev.unit_id):
    cases["… its classic model"] = ev[ev.unit_id == se.parent_unit].iloc[0]

s_ = sales.copy(); s_["unit_id"] = s_.model_code + "-" + s_.material_code
seas = s_.groupby(["unit_id", "season"]).units.sum().unstack().reindex(columns=dss.SEASONS).fillna(0)
rows = []
fig, axes = plt.subplots(2, 4, figsize=(18, 7)); axes = axes.ravel()
for ax, (name, r) in zip(axes, [(n, r) for n, r in cases.items() if r is not None]):
    rows.append({"case": name, "product": r.unit_id, "line": r.product_line, "seasons": r.H,
                 "CV": round(r.cv, 2) if pd.notna(r.cv) else None, "DSS class": r.dss_class, "strategy": r.strategy,
                 "OPP": r.opp, "to confirm": r.to_review, "KPI alignment": pct(r.alignment), "explanation": r.explanation})
    ax.bar(range(len(dss.SEASONS)), seas.loc[r.unit_id] if r.unit_id in seas.index else 0, color="#007AFF")
    ax.set_xticks(range(len(dss.SEASONS))); ax.set_xticklabels(dss.SEASONS, rotation=60, fontsize=7)
    ax.set_title(f"{name}\n{r.unit_id} → {dss.CLASS_NAMES[r.dss_class]}", fontsize=9)
for ax in axes[len(rows):]:
    ax.axis("off")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_1_cases.png", dpi=150); plt.show()
demo = save(pd.DataFrame(rows).set_index("case"), "table_1_cases")
for c_, r in demo.iterrows():
    print(f"■ {c_} [{r['product']}]\n  {r.explanation}\n")

# %% [markdown]
# ## 2. Verification against the generating profiles

# %%
conf = save(pd.crosstab(ev.profile, ev.dss_class).reindex(columns=["S", "V", "I", "N"], fill_value=0), "table_2a_confusion_profile_vs_class")
print(conf, "\n")
cyc = {}
for vname, vp in [("Version 1", dss.V1), ("Version 2", dss.V2), ("Version 3 (final)", params)]:
    rv, _ = dss.run(sales, products, kpi, vp)
    cyc[vname] = detection(rv.merge(gt[["unit_id", "profile"]], on="unit_id"))
cyc["Label-based (today)"] = {**{k: cyc["Version 1"][k] for k in ["stable products", "non-stable products"]},
                              "non-stable classified V/I": 0.0, "non-stable flagged": 0.0,
                              "stable classified V/I": 0.0, "stable flagged (false alarm)": 0.0}
det = save(pd.DataFrame(cyc).T, "table_2b_detection_by_version")
print(det.T.to_string(float_format=lambda v: pct(v) if v <= 1 else f"{v:.0f}"))

c_ = ev[ev.product_line.isin(CONT) & (ev.dss_class != "N")].copy()
c_["flagged"] = c_.dss_class.isin(["V", "I"]) | (c_.to_review != "") | c_.decline_check
ns = c_[c_.profile.isin(NONSTABLE)]
sev = save(pd.DataFrame({
    "naive forecast error (median)": [c_[c_.profile == "stable"].naive_forecast_error.median(),
                                      ns[ns.flagged].naive_forecast_error.median(), ns[~ns.flagged].naive_forecast_error.median()],
    "products": [int((c_.profile == "stable").sum()), int(ns.flagged.sum()), int((~ns.flagged).sum())]},
    index=["stable", "non-stable flagged", "non-stable missed"]), "table_2c_missed_vs_flagged")
print("\n", sev.round(2))

# %% [markdown]
# ## 3. Comparison with label-based segmentation

# %%
co = ev[ev.product_line == "Carry-over"]
pl = dss.priority_list(res)
lab_rows = {
    "carry-over products": len(co),
    "carry-over with enough history": int((co.dss_class != "N").sum()),
    "carry-over not stable according to DSS": int(co.dss_class.isin(["V", "I"]).sum()),
    "carry-over too recent to be judged": int((co.dss_class == "N").sum()),
    "products needing attention (priority list)": len(pl),
    "revenue share of priority list": pl.value_share.sum(),
    "products to move to another KPI checklist": int((res.product_line.map(lambda l: "lean" if l in CONT else "seasonal")
                                                      != res.dss_class.map({"S": "lean", "V": "hybrid", "I": "hybrid", "N": "seasonal"})).sum()),
    "KPI alignment – carry-over (by label)": co.alignment.mean(),
    "KPI alignment – Stable class": res.loc[res.dss_class == "S", "alignment"].mean(),
    "KPI alignment – Variable class": res.loc[res.dss_class == "V", "alignment"].mean(),
    "KPI alignment – portfolio": res.alignment.mean(),
}
lab = save(pd.Series(lab_rows, name="value").to_frame(), "table_3a_label_vs_dss"); print(lab, "\n")
fe = save(res.groupby("dss_class").naive_forecast_error.median().rename("naive forecast error (median)").to_frame(), "table_3b_forecast_error_by_class")
print(fe.round(2))
ct = save(pd.crosstab(res.product_line, res.dss_class).reindex(index=["Permanent", "Carry-over", "Seasonal", "Capsule"],
                                                                columns=["S", "V", "I", "N"], fill_value=0), "table_3c_label_vs_class")
print("\n", ct)

# %% [markdown]
# ## 4. Clustering check (rule fixed in D4: the scenario with the highest ARI is the default)

# %%
crow = {}
for name, th in dss.SCENARIOS.items():
    chk = dss.cluster_check(res, dss.classify(res, params, th))
    crow[name] = {"best k": int(chk["k_best"]), "silhouette best k": round(chk["silhouette"][chk["k_best"]], 3),
                  "ARI best k": round(chk["ari_best"], 3), "ARI k=3": round(chk["ari_k3"], 3), "uncertain products": int(chk["uncertain"].sum())}
clus = save(pd.DataFrame(crow).T, "table_4_clustering"); print(clus)
chosen = clus["ARI best k"].astype(float).idxmax()
print("Scenario chosen by the rule:", chosen)

# %% [markdown]
# ## 5. Sensitivity analysis

# %%
srows = []
def sens(label, value, p):
    r_, _ = dss.run(sales, products, kpi, p)
    d_ = detection(r_.merge(gt[["unit_id", "profile"]], on="unit_id"))
    srows.append({"parameter": label, "value": str(value), **{k: int((r_.dss_class == k).sum()) for k in ["S", "V", "I", "N"]},
                  "to confirm": int((r_.to_review != "").sum()), **{k: d_[k] for k in
                  ["non-stable flagged", "stable flagged (false alarm)", "non-stable classified V/I", "stable classified V/I"]}})
for t1 in [0.3, 0.4, 0.5, 0.6, 0.7]:
    sens("CV thresholds (t1 / t2 = 2·t1)", f"{t1} / {2 * t1}", dict(params, cv_thresholds=(t1, 2 * t1)))
for cr in [1.0, 1.5, 2.0, 3.0, 5.0]:
    sens("attitude (cost ratio)", cr, dict(params, cost_ratio=cr))
for cf in [0.7, 0.8, 0.9, 0.95]:
    sens("confidence of 'to confirm'", cf, dict(params, review_confidence=cf))
for w in [4, 6, 8]:
    sens("CV window (seasons)", w, dict(params, cv_window=w))
for mh in [2, 3, 4]:
    sens("minimum history (seasons)", mh, dict(params, min_history=mh))
sens("seasonality correction", "off", dict(params, deseasonalise=False))
for dr in [0.2, 0.4]:
    sens("declining-sales drop", dr, dict(params, trend_drop=dr))
sen = save(pd.DataFrame(srows).set_index(["parameter", "value"]), "table_5_sensitivity")
print(sen.to_string(float_format=lambda v: pct(v) if v <= 1 else f"{v:.0f}"))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
th = sen.loc["CV thresholds (t1 / t2 = 2·t1)"]
ax[0].plot(th.index, th["non-stable classified V/I"] * 100, "o-", label="non-stable classified V/I", color="#FF9500")
ax[0].plot(th.index, th["stable classified V/I"] * 100, "o-", label="stable classified V/I", color="#007AFF")
ax[0].set_title("Thresholds: detected vs wrongly classified"); ax[0].set_ylabel("%"); ax[0].legend()
sub = sen.loc["attitude (cost ratio)"]
ax[1].plot(sub.index, sub["non-stable classified V/I"] * 100, "o-", color="#FF9500", label="non-stable classified V/I")
ax[1].plot(sub.index, sub["stable classified V/I"] * 100, "o-", color="#007AFF", label="stable classified V/I")
ax[1].set_title("Attitude towards risk (cost ratio)"); ax[1].legend()
plt.tight_layout(); plt.savefig(f"{OUT}/fig_5_sensitivity.png", dpi=150); plt.show()

# %% [markdown]
# ## 6. Dirty-data test

# %%
def dirty_test(folder, clean_res, clean_report):
    sd, pdirty, kp = load(folder, dirty=True)
    rd, rep_d = dss.run(sd, pdirty, kp, params)
    log = pd.read_csv(f"{folder}/dirty_log.csv")
    g_ = pd.read_csv(f"{folder}/ground_truth.csv")
    m = clean_res[["unit_id", "dss_class"]].merge(rd[["unit_id", "dss_class"]], on="unit_id", suffixes=("_clean", "_dirty"))
    cc = log[log.error_type == "code_change"].copy(); cc["new_unit"] = cc.detail.str.replace("new unit ", "", regex=False)
    fl = rd[rd.relaunch_of != ""][["unit_id", "relaunch_of", "relaunch_class"]].merge(cc[["code", "new_unit"]], left_on="unit_id", right_on="new_unit", how="left")
    fl["correct"] = fl.code == fl.relaunch_of
    clean_cls = clean_res.set_index("unit_id").dss_class
    d_ = detection(rd.merge(g_[["unit_id", "profile"]], on="unit_id", how="left"))
    return {"products compared (same code in both runs)": len(m),
            "same class clean vs dirty": (m.dss_class_clean == m.dss_class_dirty).mean(),
            "duplicates removed": rep_d["duplicates_removed"], "return rows excluded": rep_d["return_rows_excluded"],
            "spikes smoothed": rep_d["spikes_smoothed"], "spikes injected": int((log.error_type == "spike").sum()),
            "active products without OPP (missing lead time)": int(rd.opp.str.startswith("n.a.").sum()),
            "code changes injected": len(cc), "possible code changes flagged": len(fl), "… of which correct": int(fl.correct.sum()),
            "… combined-history class equal to clean class": int(sum(clean_cls.get(r.code) == r.relaunch_class for r in fl.itertuples() if r.correct)),
            "false code-change flags on clean data": clean_report["possible_code_changes"],
            "non-stable flagged (dirty)": d_["non-stable flagged"], "stable false alarm (dirty)": d_["stable flagged (false alarm)"]}

dq = dirty_test(FOLDER, res, report)
dirty = save(pd.Series(dq, name="value").to_frame(), "table_6_dirty"); print(dirty)

# %% [markdown]
# ## 7. Requirements check (R1–R9)

# %%
req = pd.DataFrame([
    ("R1", "Classify the product, not the label", f"{lab_rows['carry-over not stable according to DSS']} carry-over products classified differently from their label; special editions kept separate from their classic model"),
    ("R2", "Use data companies already have", "only product master data and a weekly sales extract; ground truth never read by the DSS"),
    ("R3", "Handle short histories explicitly", f"{int((res.dss_class == 'N').sum())} products in 'too little history'; CV interval and 'to confirm' flag for uncertain classes"),
    ("R4", "Consider the supply side", "P/D, material value and variants set the order penetration point (MTS/ATO/MTO)"),
    ("R5", "Transparent suggestions", "every product has an explanation (why / what to do) and a visual of its position against thresholds"),
    ("R6", "Link to KPIs and measure alignment", f"KPI alignment by product, class and line; {lab_rows['products to move to another KPI checklist']} products to move to another KPI checklist"),
    ("R7", "Support, do not replace the manager", "editable thresholds and attitude; confirm/change decisions saved and propagated"),
    ("R8", "Robust to imperfect data", f"{dq['same class clean vs dirty']:.1%} same class clean vs dirty; {dq['… of which correct']} of {dq['code changes injected']} code changes flagged"),
    ("R9", "Repeatable and scalable", "full portfolio analysed in seconds; same code for notebook and dashboard"),
], columns=["id", "requirement", "evidence"]).set_index("id")
save(req, "table_7_requirements"); print(req.to_string())

# %% [markdown]
# ## 8. Stress tests (S1–S5)
# Five datasets generated with the same generator, changing **one** assumption each, with seeds never used before.
# The DSS is the frozen version with default parameters; nothing is tuned.

# %%
scen = json.load(open("stress_scenarios.json"))
st_rows = {}
for (name, over), seed in zip(scen.items(), STRESS_SEEDS):
    folder = f"stress_{MODE}_{name.split()[0]}"
    if not os.path.isdir(folder):
        env = dict(os.environ, MAISON_SEED=str(seed), MAISON_STRESS=json.dumps({**over, "out": folder}), MPLBACKEND="Agg")
        subprocess.run([sys.executable, "-W", "ignore", "maison_a_generator.py"], env=env, check=True, capture_output=True)
    s2, p2, k2 = load(folder)
    r2, rep2 = dss.run(s2, p2, k2, params)
    g2 = pd.read_csv(f"{folder}/ground_truth.csv")
    d2 = detection(r2.merge(g2[["unit_id", "profile"]], on="unit_id", how="left"))
    dq2 = dirty_test(folder, r2, rep2)
    st_rows[name] = {"seed": seed, **d2, "too little history": int((r2.dss_class == "N").sum()),
                     "to confirm": int((r2.to_review != "").sum()), "same class clean vs dirty": dq2["same class clean vs dirty"],
                     "code changes flagged correctly": f"{dq2['… of which correct']}/{dq2['code changes injected']}"}
st_rows["Reference (main dataset)"] = {"seed": "–", **detection(ev), "too little history": int((res.dss_class == "N").sum()),
                                      "to confirm": int((res.to_review != "").sum()), "same class clean vs dirty": dq["same class clean vs dirty"],
                                      "code changes flagged correctly": f"{dq['… of which correct']}/{dq['code changes injected']}"}
stress = save(pd.DataFrame(st_rows).T, "table_8_stress")
print(stress.to_string(float_format=lambda v: pct(v) if v <= 1 else f"{v:.0f}"))

# %% [markdown]
# ## 9. Pre-registered hypotheses (H1–H5)
# Criteria fixed on 13 November 2026, before the test data were opened (`PREREGISTRATION.md`).

# %%
d3 = det.loc["Version 3 (final)"]
hyp = pd.DataFrame([
    ("H1", "Non-stable continuative products flagged", "≥ 50% (labels: 0%)", pct(d3["non-stable flagged"]), d3["non-stable flagged"] >= 0.50),
    ("H2a", "False alarms on stable products", "≤ 20%", pct(d3["stable flagged (false alarm)"]), d3["stable flagged (false alarm)"] <= 0.20),
    ("H2b", "Stable products wrongly classified", "≤ 5%", pct(d3["stable classified V/I"]), d3["stable classified V/I"] <= 0.05),
    ("H3", "Missed products have a lower forecast error than flagged ones", "median missed < median flagged",
     f"{pct(sev.loc['non-stable missed', 'naive forecast error (median)'])} vs {pct(sev.loc['non-stable flagged', 'naive forecast error (median)'])}",
     bool(sev.loc["non-stable missed", "naive forecast error (median)"] < sev.loc["non-stable flagged", "naive forecast error (median)"])),
    ("H4", "Same class with clean and dirty data", "≥ 95%", f"{dq['same class clean vs dirty']:.1%}", dq["same class clean vs dirty"] >= 0.95),
    ("H5", "KPI alignment lower for Variable than for Stable", "Variable < Stable",
     f"{pct(lab_rows['KPI alignment – Variable class'])} vs {pct(lab_rows['KPI alignment – Stable class'])}",
     bool(lab_rows["KPI alignment – Variable class"] < lab_rows["KPI alignment – Stable class"])),
], columns=["id", "hypothesis", "criterion", "result", "met"]).set_index("id")
hyp["met"] = hyp.met.map({True: "yes", False: "NO"})
save(hyp, "table_9_hypotheses"); print(hyp.to_string())

# %% [markdown]
# ## Summary

# %%
summary = f"""# Evaluation summary – {MODE} ({FOLDER})

Model: frozen version {ver['version']} ({ver['frozen_on']}), dss_core.py {sha[:16]}…, parameters {psha[:16]}…

## Pre-registered hypotheses
{hyp.to_markdown() if hasattr(hyp, 'to_markdown') else hyp.to_string()}

## Key results
- Non-stable continuative products flagged: {pct(d3['non-stable flagged'])} (labels: 0%); classified Variable/Irregular: {pct(d3['non-stable classified V/I'])}
- False alarms on stable products: {pct(d3['stable flagged (false alarm)'])}; stable wrongly classified: {pct(d3['stable classified V/I'])}
- Clustering: scenario chosen by the rule = {chosen} (ARI base {clus.loc['base', 'ARI best k']}, alternative {clus.loc['alternative', 'ARI best k']})
- Dirty data: {pct(dq['same class clean vs dirty'])} same class; code changes flagged {dq['… of which correct']}/{dq['code changes injected']}
- Carry-over not stable: {lab_rows['carry-over not stable according to DSS']} of {lab_rows['carry-over with enough history']} with enough history
"""
open(f"{OUT}/summary.md", "w").write(summary); print(summary)
