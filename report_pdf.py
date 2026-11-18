"""Summary report of the DSS as a PDF (matplotlib only, so it works wherever the dashboard runs)."""
import io
import textwrap
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

INK, GREY, LIGHT, BLUE = "#1D1D1F", "#6E6E73", "#F5F5F7", "#007AFF"
A4 = (8.27, 11.69)


def _page(pdf, title, subtitle=None):
    fig = plt.figure(figsize=A4)
    fig.text(0.07, 0.95, title, fontsize=17, fontweight="bold", color=INK, va="top")
    if subtitle:
        fig.text(0.07, 0.918, subtitle, fontsize=9, color=GREY, va="top")
    fig.add_artist(plt.Line2D([0.07, 0.93], [0.905, 0.905], color="#E5E5EA", lw=1))
    return fig


def _para(fig, y, text, size=10, width=95, bold=False, color=INK, gap=1.45):
    for line in textwrap.wrap(text, width) or [""]:
        fig.text(0.07, y, line, fontsize=size, color=color, fontweight="bold" if bold else "normal", va="top")
        y -= size * gap / 11.69 / 72 * 1.0 * 1.0 + 0.0  # line height in figure fraction
        y -= 0.0
    return y


def _lines(fig, y, items, size=9.5, width=100, bullet="•"):
    lh = size * 1.5 / 72 / 11.69
    for it in items:
        wrapped = textwrap.wrap(it, width)
        for i, line in enumerate(wrapped):
            fig.text(0.09 if i else 0.07, y, (bullet + " " if i == 0 else "  ") + line, fontsize=size, color=INK, va="top")
            y -= lh
        y -= lh * 0.25
    return y


def _table(fig, y_top, df, col_widths, height_per_row=0.022, fontsize=7.5):
    n = len(df)
    h = height_per_row * (n + 1)
    ax = fig.add_axes([0.07, y_top - h, 0.86, h]); ax.axis("off")
    t = ax.table(cellText=df.values.tolist(), colLabels=list(df.columns), colWidths=col_widths, loc="upper left",
                 cellLoc="left", colLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(fontsize); t.scale(1, 1.25)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#E5E5EA")
        if r == 0:
            cell.set_facecolor(LIGHT); cell.set_text_props(fontweight="bold", color=INK)
    return y_top - h - 0.02


def build(res, plist, decisions, settings, key_messages, kpi_groups, class_name):
    """Return the PDF report as bytes."""
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # ---- page 1: summary
        fig = _page(pdf, "Supply chain strategy DSS – summary report",
                    f"Maison A · {date.today():%d %B %Y} · {settings}")
        y = 0.88
        fig.text(0.07, y, "Key messages", fontsize=12, fontweight="bold", color=INK, va="top"); y -= 0.03
        y = _lines(fig, y, key_messages)
        y -= 0.01
        fig.text(0.07, y, "Products by demand class", fontsize=12, fontweight="bold", color=INK, va="top"); y -= 0.035
        counts = res.final_class.value_counts()
        for i, k in enumerate(["S", "V", "I", "N"]):
            x = 0.07 + i * 0.215
            ax = fig.add_axes([x, y - 0.06, 0.2, 0.06]); ax.axis("off")
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=LIGHT, transform=ax.transAxes))
            ax.text(0.08, 0.68, class_name[k], fontsize=9, color=GREY, transform=ax.transAxes)
            ax.text(0.08, 0.18, str(int(counts.get(k, 0))), fontsize=18, fontweight="bold", color=INK, transform=ax.transAxes)
        y -= 0.09
        fig.text(0.07, y, "Product lines today vs actual demand (number of products)", fontsize=12, fontweight="bold",
                 color=INK, va="top"); y -= 0.03
        import pandas as pd
        ct = pd.crosstab(res.product_line, res.final_class).reindex(
            index=["Permanent", "Carry-over", "Seasonal", "Capsule"], columns=["S", "V", "I", "N"], fill_value=0)
        ct = ct.rename(columns=class_name).reset_index().rename(columns={"product_line": "Line today"})
        y = _table(fig, y, ct.astype(str), [0.24, 0.19, 0.19, 0.19, 0.19], height_per_row=0.028, fontsize=9)
        reviewed = int((plist.decision != "").sum())
        fig.text(0.07, y, f"Review progress: {reviewed} of {len(plist)} products needing attention have been reviewed "
                          f"by the manager.", fontsize=9.5, color=INK, va="top")
        fig.text(0.07, 0.04, "The DSS suggests; the manager decides. Simulated data (fictitious company Maison A).",
                 fontsize=7.5, color=GREY)
        pdf.savefig(fig); plt.close(fig)

        # ---- page 2: where to start
        fig = _page(pdf, "Where to start", "Products needing attention, ordered by priority and revenue (top 30)")
        top = plist.head(30)
        df = pd.DataFrame({"Product": top.unit_id, "Line": top.product_line, "DSS class": top.dss_class.map(class_name),
                           "Priority": top.priority_label,
                           "Main issue": top.priority.map({1: "Strategy mismatch", 3: "KPI checklist only"}).fillna(
                               top.to_review.map(lambda v: "Class to confirm" if v else "Declining sales")),
                           "Rev. €k": (top.value_12m / 1000).round(0).astype(int).astype(str),
                           "Decision": top.decision.map({"confirmed": "confirmed", "changed": "changed"}).fillna("")
                           .replace("", "to review")})
        _table(fig, 0.88, df, [0.16, 0.12, 0.16, 0.10, 0.20, 0.10, 0.16], height_per_row=0.024)
        pdf.savefig(fig); plt.close(fig)

        # ---- page 3: KPI changes
        fig = _page(pdf, "KPI checklist changes", "Products that should be monitored with a different KPI checklist")
        y = 0.88
        if not kpi_groups:
            fig.text(0.07, y, "Every product is monitored with the KPI checklist it needs.", fontsize=10, va="top")
        for g in kpi_groups:
            if y < 0.18:
                pdf.savefig(fig); plt.close(fig)
                fig = _page(pdf, "KPI checklist changes (continued)"); y = 0.88
            for line in textwrap.wrap(g["title"], 78):
                fig.text(0.07, y, line, fontsize=11, fontweight="bold", color=INK, va="top"); y -= 0.02
            y -= 0.006
            y = _lines(fig, y, [g["from_to"], g["status"]] + ([g["preview"]] if g.get("preview") else []), size=9)
            if g["add"]:
                fig.text(0.07, y, f"Needed but not measured ({len(g['add'])}):", fontsize=9, fontweight="bold", va="top"); y -= 0.02
                y = _lines(fig, y, [", ".join(g["add"])], size=8.5, bullet="")
            if g["drop"]:
                fig.text(0.07, y, f"Measured but not needed ({len(g['drop'])}):", fontsize=9, fontweight="bold", va="top"); y -= 0.02
                y = _lines(fig, y, [", ".join(g["drop"])], size=8.5, bullet="")
            y -= 0.02
        pdf.savefig(fig); plt.close(fig)

        # ---- page 4: decisions
        fig = _page(pdf, "Manager decisions", f"{len(decisions)} decision(s) recorded")
        if decisions:
            rows = []
            for u, d in list(decisions.items())[:32]:
                r = res.set_index("unit_id").loc[u] if u in set(res.unit_id) else None
                rows.append({"Product": u, "Line": r.product_line if r is not None else "",
                             "DSS suggestion": class_name.get(r.dss_class, "") if r is not None else "",
                             "Decision": d.get("decision", ""), "Class set": class_name.get(d.get("class"), "") if d.get("class") else "",
                             "Reason": (d.get("note") or "")[:45]})
            _table(fig, 0.88, pd.DataFrame(rows), [0.15, 0.11, 0.16, 0.12, 0.14, 0.32], height_per_row=0.024)
            if len(decisions) > 32:
                fig.text(0.07, 0.06, f"… and {len(decisions) - 32} more (see manager_decisions.csv).", fontsize=8, color=GREY)
        else:
            fig.text(0.07, 0.88, "No decisions recorded yet.", fontsize=10, va="top")
        pdf.savefig(fig); plt.close(fig)
    return buf.getvalue()
