"""Airtel Telecom Analytics: premium fixed-screen Streamlit dashboard.

Views
  * Overview        - retention picture for the filtered subscriber base
  * Circle Analysis - India circle map (click a bubble to select) + selected-circle detail

Data comes only from the read-only SQLite table `customer_features`. Nothing is simulated:
the only supplied constant is the 23.75% national churn reference.
"""
import base64
import html
import math
import os
import re
import sqlite3
import sys
from contextlib import closing
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
sys.path.insert(0, str(APP_DIR))
from chatbot import render_chatbot  # noqa: E402

# ---------------------------------------------------------------- constants
NATIONAL_REF = 0.2375  # supplied reference value, not a target
RED, DEEP_RED, CHARCOAL = "#E60000", "#990000", "#17191F"
ROSE, BLUSH, GREEN = "#F47C7C", "#FBD5D8", "#12A150"
MUTED, INK, GRID = "#6B7280", "#1F2330", "rgba(23,25,31,.08)"
RED_SCALE = [[0, "#FCE4E6"], [.35, "#F58A8A"], [.7, "#E60000"], [1, "#6E0000"]]
REQUIRED = ["customer_id", "circle_name", "region", "connection_type", "tenure_years",
            "is_churned", "churn_date", "complaint_count", "unresolved_count"]
OVERVIEW, DETAIL = "Overview", "Circle Analysis"
FILTER_LABELS = {"connection_type": "Connection type", "region": "Region",
                 "tenure_band": "Tenure", "complaint_status": "Complaint status"}
GLOBAL_KEYS = {"connection_type": "connection", "region": "region", "tenure_band": "tenure"}

st.set_page_config(page_title="Airtel Telecom Analytics", page_icon="📶", layout="wide",
                   initial_sidebar_state="collapsed")


# ---------------------------------------------------------------- database
def find_database():
    """Find a database that actually contains the required customer_features schema."""
    override = os.getenv("AIRTEL_DB_PATH")
    candidates = [Path(override).expanduser()] if override else [
        ROOT / "data" / "raw" / "airtel_dashboard_slim.db",
        APP_DIR / "data" / "raw" / "airtel_dashboard_slim.db",
        ROOT / "data" / "raw" / "airtel.db",
        APP_DIR / "data" / "raw" / "airtel.db",
        ROOT / "data" / "airtel.db",
        ROOT / "airtel.db",
    ]
    if not override and ROOT.exists():
        for path in ROOT.rglob("airtel_dashboard_slim.db"):
            if path not in candidates:
                candidates.append(path)
        for path in ROOT.rglob("airtel.db"):
            if path not in candidates:
                candidates.append(path)

    checked, seen = [], set()
    for path in candidates:
        try:
            path = path.resolve()
        except OSError:
            continue
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            checked.append(f"{path} (not found)")
            continue
        try:
            uri = path.as_uri() + "?mode=ro"
            with closing(sqlite3.connect(uri, uri=True, timeout=3)) as con:
                tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' OR type='view'")}
                if "customer_features" not in tables:
                    checked.append(f"{path} (customer_features table missing; tables: {', '.join(sorted(tables)) or 'none'})")
                    continue
                present = {r[1] for r in con.execute("PRAGMA table_info(customer_features)")}
                missing = [c for c in REQUIRED if c not in present]
                if missing:
                    checked.append(f"{path} (missing columns: {', '.join(missing)})")
                    continue
            return path, checked
        except sqlite3.Error as e:
            checked.append(f"{path} (SQLite error: {e})")
    return None, checked


@st.cache_data(show_spinner="Reading the Airtel database…")
def load_data(db_path, modified):
    """Load only the needed columns, once per database version (`modified` is the cache key)."""
    uri = Path(db_path).as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as con:
        present = {r[1] for r in con.execute("PRAGMA table_info(customer_features)")}
        missing = [c for c in REQUIRED if c not in present]
        if missing:
            raise ValueError("customer_features is missing: " + ", ".join(missing))
        df = pd.read_sql_query(f"SELECT {', '.join(REQUIRED)} FROM customer_features", con)

    for col in ("circle_name", "region", "connection_type"):
        df[col] = df[col].fillna("Unknown").astype(str)
    flag = pd.to_numeric(df.is_churned, errors="coerce")
    if not flag.isin([0, 1]).all():
        raise ValueError("is_churned must contain only 0 or 1.")
    df["is_churned"] = flag.astype(int)
    for col in ("complaint_count", "unresolved_count"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["tenure_band"] = tenure_bands(df.tenure_years)
    dates = pd.to_datetime(df.churn_date, errors="coerce", format="mixed")
    df["churn_month"] = dates.dt.strftime("%Y-%m")
    df["complaint_status"] = "No complaints"
    df.loc[df.complaint_count > 0, "complaint_status"] = "None unresolved"
    df.loc[df.unresolved_count > 0, "complaint_status"] = "Has unresolved"
    return df


def tenure_bands(raw):
    """Keep a few whole-year values as they are; bin continuous values; keep text labels."""
    num = pd.to_numeric(raw, errors="coerce")
    if num.notna().sum() < raw.notna().sum():  # already text labels
        order = sorted(raw.dropna().astype(str).unique())
        return pd.Categorical(raw.astype("string"), categories=order, ordered=True)
    values = num.dropna()
    if values.nunique() <= 10 and (values % 1 == 0).all():
        order = [f"{int(v)} yr" for v in sorted(values.unique())]
        return pd.Categorical(num.map(lambda v: f"{int(v)} yr", na_action="ignore"),
                              categories=order, ordered=True)
    return pd.cut(num, [-float("inf"), 1, 2, 3, 5, float("inf")], right=False,
                  labels=["<1 yr", "1–2 yr", "2–3 yr", "3–5 yr", "5+ yr"])


# ---------------------------------------------------------------- calculations
def kpis(df):
    total = len(df)
    churned = int(df.is_churned.sum())
    return total, churned, (churned / total if total else 0.0)


def churn_by(df, column):
    out = df.groupby(column, observed=True).agg(
        subscribers=("is_churned", "size"), churned=("is_churned", "sum")).reset_index()
    out["rate"] = out.churned / out.subscribers
    return out


def month_options(df):
    """Every month between the first and last recorded churn date."""
    dated = df.loc[(df.is_churned == 1) & df.churn_month.notna(), "churn_month"]
    if dated.empty:
        return []
    return list(pd.period_range(dated.min(), dated.max(), freq="M").strftime("%Y-%m"))


def month_label(month):
    return pd.to_datetime(month, format="%Y-%m").strftime("%b %Y")


def monthly_churn(df, window=None):
    """Churned subscribers per recorded churn month; months with none are shown as zero."""
    empty = pd.DataFrame(columns=["month", "churned"])
    dated = df.loc[(df.is_churned == 1) & df.churn_month.notna(), "churn_month"]
    if dated.empty:
        return empty
    counts = dated.value_counts()
    months = list(pd.period_range(counts.index.min(), counts.index.max(), freq="M").strftime("%Y-%m"))
    if window:
        months = [m for m in months if window[0] <= m <= window[1]]
    if not months:
        return empty
    return counts.reindex(months, fill_value=0).rename_axis("month").reset_index(name="churned")


def circle_table(data):
    """Per-circle subscribers, churn and complaint share (all real, from the filtered rows)."""
    table = churn_by(data, "circle_name")
    regions = data.groupby("circle_name")["region"].apply(lambda s: sorted(s.unique())).rename("region_list")
    unresolved = (data.unresolved_count > 0).groupby(data.circle_name).mean().rename("unresolved_pct")
    table = table.merge(regions, left_on="circle_name", right_index=True)
    return table.merge(unresolved, left_on="circle_name", right_index=True)


def pct(value, places=1):
    return f"{value * 100:.{places}f}%"


# ---------------------------------------------------------------- map locations
# Approximate circle centre points. They only position markers; they are not exact boundaries.
CIRCLE_COORDS = {
    "andhra pradesh": (16.50, 80.64), "ap": (16.50, 80.64), "telangana": (17.40, 78.50),
    "assam": (26.20, 92.94), "bihar": (25.60, 85.10), "chhattisgarh": (21.30, 81.90),
    "chennai": (13.08, 80.27), "tamil nadu": (11.00, 78.00), "tamilnadu": (11.00, 78.00),
    "delhi": (28.61, 77.21), "delhi ncr": (28.61, 77.21), "goa": (15.40, 74.00),
    "gujarat": (22.30, 71.50), "haryana": (29.00, 76.00), "himachal pradesh": (31.10, 77.20),
    "jammu and kashmir": (33.50, 75.00), "jammu": (33.50, 75.00), "kashmir": (34.00, 74.80),
    "jharkhand": (23.60, 85.30), "karnataka": (14.80, 75.70), "kerala": (10.50, 76.30),
    "kochi": (9.93, 76.27), "kolkata": (22.57, 88.36), "madhya pradesh": (23.25, 77.40),
    "mp": (23.25, 77.40), "maharashtra": (19.00, 75.00), "mumbai": (19.08, 72.88),
    "north east": (26.00, 92.00), "northeast": (26.00, 92.00), "odisha": (20.30, 85.80),
    "orissa": (20.30, 85.80), "punjab": (31.00, 75.30), "rajasthan": (26.60, 74.00),
    "uttar pradesh": (26.85, 80.95), "up": (26.85, 80.95), "up east": (26.40, 82.50),
    "up west": (28.20, 78.20), "uttarakhand": (30.30, 79.00), "west bengal": (23.30, 87.80),
}
# Used when a circle is named after a broad region ("North", "South"...) or has no known name.
REGION_CENTRES = {
    "north": (29.00, 77.00), "south": (12.90, 78.00), "east": (22.60, 87.00),
    "west": (21.00, 73.00), "central": (23.30, 79.50), "north east": (26.00, 92.00),
    "northeast": (26.00, 92.00),
}


def norm(text):
    text = re.sub(r"[^a-z0-9 ]+", " ", str(text).lower().replace("&", " and "))
    return " ".join(text.split())


def locate_circle(name, regions):
    """Return ((lat, lon), basis) or None. Never invents a location for an unknown circle."""
    key = norm(name)
    if key in CIRCLE_COORDS:
        return CIRCLE_COORDS[key], "Circle centre (approximate)"
    if key in REGION_CENTRES:
        return REGION_CENTRES[key], "Region centre (approximate)"
    for known in sorted(CIRCLE_COORDS, key=len, reverse=True):
        if len(known) >= 4 and known in key:
            return CIRCLE_COORDS[known], "Circle centre (approximate)"
    for token in key.split():
        if token in CIRCLE_COORDS:
            return CIRCLE_COORDS[token], "Circle centre (approximate)"
    for region in regions:
        if norm(region) in REGION_CENTRES:
            return REGION_CENTRES[norm(region)], "Region centre (approximate)"
    return None


# ---------------------------------------------------------------- charts
def style(fig, **layout):
    base = dict(
        margin=dict(l=4, r=10, t=6, b=4), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", showlegend=False, autosize=True,
        font=dict(family="Inter, sans-serif", color="#374151", size=11),
        hoverlabel=dict(bgcolor=CHARCOAL, bordercolor=RED, font=dict(color="#FFFFFF", size=12)))
    base.update(layout)
    fig.update_layout(**base)
    axis = dict(gridcolor=GRID, zeroline=False, linecolor=GRID, automargin=True,
                tickfont=dict(color=MUTED, size=10), title_font=dict(size=10, color=MUTED),
                title_standoff=6)
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


COUNT_HOVER = ("Churn rate %{customdata[2]:.2%}<br>%{customdata[1]:,} churned of "
               "%{customdata[0]:,}<extra></extra>")


def counts_of(data):
    return data[["subscribers", "churned", "rate"]].to_numpy()


def reference_line(fig, vertical=False, label=True):
    line = dict(line_dash="dot", line_color=CHARCOAL, line_width=1.2)
    if label:
        line.update(annotation_text="23.75% ref", annotation_font=dict(size=9, color=MUTED),
                    annotation_position="top right" if not vertical else "top")
    if vertical:
        fig.add_vline(x=NATIONAL_REF, **line)
    else:
        fig.add_hline(y=NATIONAL_REF, **line)


def treemap_chart(data):
    """Composition + intensity: tile area is subscribers, tile colour is churn rate."""
    t = churn_by(data, "circle_name").sort_values("subscribers", ascending=False)
    low, high = float(t.rate.min()), float(t.rate.max())
    fig = go.Figure(go.Treemap(
        labels=t.circle_name.tolist(), parents=[""] * len(t), values=t.subscribers.tolist(),
        customdata=counts_of(t), branchvalues="total",
        marker=dict(colors=t.rate.tolist(), colorscale=RED_SCALE, cmin=low, cmax=max(high, low + .01),
                    line=dict(color="#FFFFFF", width=2),
                    colorbar=dict(title=dict(text="Churn", font=dict(size=9)), tickformat=".0%",
                                  thickness=8, len=.85, tickfont=dict(size=9))),
        texttemplate="<b>%{label}</b><br>%{value:,.0f}<br>%{customdata[2]:.1%}",
        textfont=dict(size=11), tiling=dict(pad=2), pathbar=dict(visible=False),
        hovertemplate="<b>%{label}</b><br>" + COUNT_HOVER))
    style(fig, margin=dict(l=2, r=2, t=4, b=2))
    return fig


def monthly_chart(month, colour=RED):
    """Trend over time: churned subscribers per month, with the peak labelled."""
    labels = pd.to_datetime(month.month, format="%Y-%m").dt.strftime("%b %y").tolist()
    fig = go.Figure(go.Scatter(
        x=labels, y=month.churned, mode="lines+markers", fill="tozeroy",
        fillcolor="rgba(230,0,0,.13)", line=dict(color=colour, width=2.6, shape="spline", smoothing=.5),
        marker=dict(size=6, color="#FFFFFF", line=dict(color=colour, width=2)),
        hovertemplate="<b>%{x}</b><br>%{y:,} churned<extra></extra>"))
    peak = int(month.churned.idxmax())
    fig.add_annotation(x=labels[peak], y=float(month.churned[peak]), text=f"Peak {int(month.churned[peak]):,}",
                       showarrow=True, arrowhead=0, arrowcolor=MUTED, ax=0, ay=-22,
                       font=dict(size=10, color=INK), bgcolor="rgba(255,255,255,.85)",
                       bordercolor=BLUSH, borderpad=3)
    style(fig, margin=dict(l=4, r=14, t=26, b=4))
    fig.update_xaxes(type="category", nticks=8, showgrid=False)
    fig.update_yaxes(rangemode="tozero")
    return fig


def funnel_chart(data):
    """Do unresolved complaints end in churn? Each stage is a subset of the one above."""
    unresolved = data.unresolved_count > 0
    raised = (data.complaint_count > 0) | unresolved
    stages = [("All subscribers", len(data)), ("Raised a complaint", int(raised.sum())),
              ("Unresolved complaint", int(unresolved.sum())),
              ("…and churned", int((unresolved & (data.is_churned == 1)).sum()))]
    fig = go.Figure(go.Funnel(
        y=[s for s, _ in stages], x=[v for _, v in stages], textposition="inside",
        textinfo="value+percent initial", textfont=dict(color="#FFFFFF", size=11),
        marker=dict(color=["#6E0000", "#A80000", RED, "#F25C5C"]),
        connector=dict(line=dict(color=BLUSH, width=1)),
        hovertemplate="<b>%{y}</b><br>%{x:,} subscribers<br>%{percentInitial:.1%} of all<extra></extra>"))
    note = ""
    if unresolved.any() and (~raised).any():
        note = (f"Churn rate: unresolved {pct(data.loc[unresolved, 'is_churned'].mean())}"
                f" · no complaints {pct(data.loc[~raised, 'is_churned'].mean())}")
    if note:
        fig.add_annotation(xref="paper", yref="paper", x=.5, y=-.04, text=note, showarrow=False,
                           font=dict(size=10, color=MUTED))
    style(fig, margin=dict(l=4, r=6, t=4, b=22 if note else 4))
    return fig


def connection_chart(data):
    """Deviation lollipop: each stem starts at the reference and ends at the observed rate."""
    t = churn_by(data, "connection_type")
    names = t.connection_type.tolist()
    fig = go.Figure()
    for name, rate in zip(names, t.rate):
        fig.add_shape(type="line", x0=NATIONAL_REF, x1=rate, y0=name, y1=name,
                      line=dict(color="rgba(153,0,0,.55)", width=4))
    fig.add_trace(go.Scatter(
        x=t.rate, y=names, mode="markers+text", customdata=counts_of(t),
        marker=dict(size=18, color=[RED if r > NATIONAL_REF else GREEN for r in t.rate],
                    line=dict(color="#FFFFFF", width=2.5)),
        text=[f"{r:.1%}" for r in t.rate], textposition="top center", cliponaxis=False,
        textfont=dict(size=11, color=INK), hovertemplate="<b>%{y}</b><br>" + COUNT_HOVER))
    reference_line(fig, vertical=True)
    values = t.rate.tolist() + [NATIONAL_REF]
    pad = max((max(values) - min(values)) * .35, .025)
    style(fig, margin=dict(l=4, r=16, t=18, b=4))
    fig.update_xaxes(tickformat=".0%", range=[max(0, min(values) - pad), max(values) + pad])
    fig.update_yaxes(type="category", range=[-.6, len(names) - .4], showgrid=False)
    return fig


def heatmap_chart(data):
    """Where does churn concentrate? Churn rate for each circle and tenure band."""
    biggest = churn_by(data, "circle_name").nlargest(12, "subscribers").circle_name
    sub = data[data.circle_name.isin(biggest)]
    cells = sub.groupby(["circle_name", "tenure_band"], observed=True).agg(
        n=("is_churned", "size"), c=("is_churned", "sum")).reset_index()
    cells["rate"] = cells.c / cells.n
    rate = cells.pivot(index="circle_name", columns="tenure_band", values="rate")
    size = cells.pivot(index="circle_name", columns="tenure_band", values="n")
    order = rate.mean(axis=1).sort_values().index  # highest churn ends up at the top
    rate, size = rate.loc[order], size.loc[order]
    z = rate.to_numpy(dtype=float)
    text = [[f"{v:.0%}" if pd.notna(v) else "" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=[str(c) for c in rate.columns], y=rate.index.tolist(), customdata=size.to_numpy(dtype=float),
        text=text, texttemplate="%{text}", textfont=dict(size=9), colorscale=RED_SCALE,
        xgap=2, ygap=2, hoverongaps=False,
        colorbar=dict(thickness=7, len=.9, tickformat=".0%", tickfont=dict(size=9)),
        hovertemplate="<b>%{y}</b> · %{x}<br>Churn rate %{z:.2%}<br>%{customdata:,.0f} subscribers<extra></extra>"))
    style(fig, margin=dict(l=4, r=4, t=4, b=4))
    fig.update_xaxes(type="category", showgrid=False)
    fig.update_yaxes(type="category", showgrid=False)
    return fig


def radar_chart(data):
    """Retention profile: churn rate across tenure bands against the reference ring."""
    t = churn_by(data, "tenure_band")
    if len(t) < 3:
        return "A radar needs at least three tenure bands."
    cats = t.tenure_band.astype(str).tolist()
    rates = t.rate.tolist()
    counts = counts_of(t).tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[NATIONAL_REF] * (len(cats) + 1), theta=cats + cats[:1], mode="lines",
        line=dict(color=CHARCOAL, width=1.3, dash="dot"), hovertemplate="Reference 23.75%<extra></extra>"))
    fig.add_trace(go.Scatterpolar(
        r=rates + rates[:1], theta=cats + cats[:1], mode="lines+markers", fill="toself",
        fillcolor="rgba(230,0,0,.16)", line=dict(color=RED, width=2.5),
        marker=dict(size=7, color=RED, line=dict(color="#FFFFFF", width=1.5)),
        customdata=counts + counts[:1], hovertemplate="<b>%{theta}</b><br>" + COUNT_HOVER))
    style(fig, margin=dict(l=30, r=30, t=14, b=14))
    fig.update_layout(polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(range=[0, max(rates + [NATIONAL_REF]) * 1.2], tickformat=".0%", angle=90,
                        gridcolor=GRID, linecolor=GRID, tickfont=dict(size=9, color=MUTED)),
        angularaxis=dict(gridcolor=GRID, linecolor=GRID, direction="clockwise",
                         tickfont=dict(size=10, color=INK))))
    return fig


def gauge_chart(data):
    """Selected circle's churn rate on a dial, with the reference as the threshold marker."""
    _, _, rate = kpis(data)
    ref = NATIONAL_REF * 100
    top = max(40, math.ceil(max(rate * 100, ref) * 1.3 / 5) * 5)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=rate * 100,
        number=dict(suffix="%", valueformat=".2f", font=dict(size=32, color=INK)),
        delta=dict(reference=ref, valueformat=".2f", suffix=" pts", font=dict(size=13),
                   increasing=dict(color=RED), decreasing=dict(color=GREEN)),
        title=dict(text="vs 23.75% reference", font=dict(size=10, color=MUTED)),
        gauge=dict(axis=dict(range=[0, top], ticksuffix="%", tickfont=dict(size=9, color=MUTED)),
                   bar=dict(color=RED, thickness=.35), bgcolor="#F3F4F6", borderwidth=0,
                   steps=[dict(range=[0, ref], color="#E3F4EA"), dict(range=[ref, top], color="#F9D3D6")],
                   threshold=dict(line=dict(color=CHARCOAL, width=3), thickness=.85, value=ref))))
    style(fig, margin=dict(l=22, r=22, t=26, b=6))
    return fig


def bubble_chart(data, selected=None):
    """Size vs rate: big and high-churn circles sit top-right; bubble area is churned subscribers."""
    t = churn_by(data, "circle_name")
    biggest = max(float(t.churned.max()), 1.0)
    labelled = set(t.nlargest(3, "rate").circle_name) | set(t.nlargest(2, "subscribers").circle_name)
    if selected:
        labelled.add(selected)
    fig = go.Figure(go.Scatter(
        x=t.subscribers, y=t.rate, mode="markers+text",
        customdata=t[["subscribers", "churned", "rate", "circle_name"]].to_numpy(),
        text=[c if c in labelled else "" for c in t.circle_name],
        textposition="top center", textfont=dict(size=10, color=INK), cliponaxis=False,
        marker=dict(size=[10 + 26 * math.sqrt(c / biggest) for c in t.churned], opacity=.9,
                    color=t.rate.tolist(), colorscale=RED_SCALE,
                    cmin=float(t.rate.min()), cmax=max(float(t.rate.max()), float(t.rate.min()) + .01),
                    line=dict(color=[CHARCOAL if c == selected else "#FFFFFF" for c in t.circle_name],
                              width=[3 if c == selected else 1 for c in t.circle_name])),
        hovertemplate="<b>%{customdata[3]}</b><br>" + COUNT_HOVER))
    reference_line(fig)
    style(fig, margin=dict(l=4, r=14, t=22, b=4))
    low, high = min(t.rate.min(), NATIONAL_REF), max(t.rate.max(), NATIONAL_REF)
    pad = max((high - low) * .2, .01)
    fig.update_yaxes(tickformat=".0%" if high - low > .06 else ".1%", title_text="Churn rate",
                     range=[max(0, low - pad), high + pad])
    fig.update_xaxes(title_text="Subscribers", tickformat="~s")
    return fig


def india_circle_map(data, selected=None):
    """India map: bubble size = subscribers, colour = churn rate. Returns (figure, plotted names, skipped)."""
    table = circle_table(data).sort_values("subscribers", ascending=False).reset_index(drop=True)
    rows, points = [], []
    for row in table.itertuples():
        found = locate_circle(row.circle_name, row.region_list)
        if found:
            rows.append((row, found[1]))
            points.append(found[0])
    skipped = len(table) - len(rows)
    if not rows:
        return None, [], skipped

    groups = {}
    for idx, (lat, lon) in enumerate(points):  # fan out markers that share a position
        groups.setdefault((round(lat, 1), round(lon, 1)), []).append(idx)
    for members in groups.values():
        if len(members) > 1:
            for k, idx in enumerate(members):
                angle = 2 * math.pi * k / len(members)
                points[idx] = (points[idx][0] + 1.4 * math.sin(angle), points[idx][1] + 1.4 * math.cos(angle))

    names = [r.circle_name for r, _ in rows]
    rates = [r.rate for r, _ in rows]
    biggest = max(max(r.subscribers for r, _ in rows), 1)
    sizes = [14 + 28 * math.sqrt(r.subscribers / biggest) for r, _ in rows]
    labelled = set(table.nlargest(5, "rate").circle_name) | set(table.nlargest(3, "subscribers").circle_name)
    if selected:
        labelled.add(selected)
    detail = [[r.subscribers, r.churned, r.rate, r.circle_name, ", ".join(r.region_list),
               r.unresolved_pct, basis, (r.rate - NATIONAL_REF) * 100] for r, basis in rows]

    fig = go.Figure()
    fig.add_trace(go.Choropleth(
        locations=["IND"], locationmode="ISO-3", z=[1], showscale=False, hoverinfo="skip",
        colorscale=[[0, "#FCE7E9"], [1, "#FCE7E9"]], marker=dict(line=dict(color="#E8A0A8", width=1))))
    fig.add_trace(go.Scattergeo(
        lat=[p[0] for p in points], lon=[p[1] for p in points], customdata=detail,
        text=[n if n in labelled else "" for n in names], mode="markers+text",
        textposition="top center", textfont=dict(size=10, color=CHARCOAL),
        marker=dict(size=sizes, color=rates, colorscale=RED_SCALE, cmin=min(rates),
                    cmax=max(max(rates), min(rates) + .01), opacity=.9,
                    line=dict(color="#FFFFFF", width=1.6),
                    colorbar=dict(title=dict(text="Churn", font=dict(size=9)), tickformat=".0%", thickness=8,
                                  len=.45, x=.02, xanchor="left", y=.28, tickfont=dict(size=9))),
        selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=.9)),
        hovertemplate=("<b>%{customdata[3]}</b><br>Subscribers: %{customdata[0]:,}<br>"
                       "Churned: %{customdata[1]:,}<br>Churn rate: %{customdata[2]:.2%}<br>"
                       "vs 23.75% reference: %{customdata[7]:+.2f} pts<br>"
                       "Customers with unresolved complaints: %{customdata[5]:.1%}<br>"
                       "Region: %{customdata[4]}<br><i>%{customdata[6]}</i><extra></extra>")))
    if selected in names:
        i = names.index(selected)
        fig.add_trace(go.Scattergeo(
            lat=[points[i][0]], lon=[points[i][1]], mode="markers", hoverinfo="skip",
            marker=dict(size=sizes[i] + 14, color="rgba(0,0,0,0)", symbol="circle",
                        line=dict(color=CHARCOAL, width=3))))
    fig.update_geos(
        scope="asia", resolution=50, projection_type="mercator", showland=True, landcolor="#EEF0F3",
        showcountries=True, countrycolor="#D5D9E0", countrywidth=.6, showcoastlines=True,
        coastlinecolor="#C9CED8", coastlinewidth=.6, showocean=True, oceancolor="#EAF3FA",
        showlakes=False, lonaxis_range=[67.5, 97.5], lataxis_range=[6, 37.5], showframe=False,
        bgcolor="rgba(0,0,0,0)")
    style(fig, margin=dict(l=0, r=0, t=0, b=0), clickmode="event+select")
    return fig, names, skipped


def picked_circle(event, names):
    """Circle name from a map click, if any."""
    try:
        points = event.selection.points
    except AttributeError:
        points = ((event or {}).get("selection") or {}).get("points", [])
    for p in points or []:
        if p.get("curve_number") not in (None, 1):
            continue
        detail = p.get("customdata")
        if detail and len(detail) > 3 and detail[3] in names:
            return detail[3]
        idx = p.get("point_index", p.get("point_number"))
        if isinstance(idx, int) and 0 <= idx < len(names):
            return names[idx]
    return None


# ---------------------------------------------------------------- icons
ICONS = {
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/>'
             '<path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "logout": '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/>'
              '<line x1="21" x2="9" y1="12" y2="12"/>',
    "shield": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 '
              '4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>',
    "ticket": '<path d="M2 9a3 3 0 0 1 0 6v2a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-2a3 3 0 0 1 0-6V7a2 2 0 0 0-2-2H4'
              'a2 2 0 0 0-2 2Z"/><path d="M13 5v2"/><path d="M13 17v2"/><path d="M13 11v2"/>',
    "pulse": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "pin": '<path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/>',
    "tiles": '<rect x="3" y="3" width="8" height="18" rx="1"/><rect x="13" y="3" width="8" height="8" rx="1"/>'
             '<rect x="13" y="13" width="8" height="8" rx="1"/>',
    "trend": '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    "filter": '<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>',
    "signal": '<path d="M5 12.55a11 11 0 0 1 14.08 0"/><path d="M1.42 9a16 16 0 0 1 21.16 0"/>'
              '<path d="M8.53 16.11a6 6 0 0 1 6.95 0"/><line x1="12" x2="12.01" y1="20" y2="20"/>',
    "grid": '<rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/>'
            '<rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "gauge": '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
    "bubbles": '<circle cx="7.5" cy="7.5" r="3.5"/><circle cx="16.5" cy="9" r="5"/><circle cx="9" cy="17" r="4"/>',
    "spark": '<path d="M12 3v3"/><path d="M12 18v3"/><path d="M3 12h3"/><path d="M18 12h3"/>'
             '<path d="m5.6 5.6 2.1 2.1"/><path d="m16.3 16.3 2.1 2.1"/><path d="m5.6 18.4 2.1-2.1"/>'
             '<path d="m16.3 7.7 2.1-2.1"/>',
}


def svg(name, size=16):
    return (f'<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" stroke="currentColor" '
            f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>')


TOWER = ('<svg class="tower" viewBox="0 0 48 48" width="34" height="34" fill="none" stroke="currentColor" '
         'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
         '<path class="w w2" d="M12 15a12 12 0 0 0 0 18"/><path class="w w2" d="M36 15a12 12 0 0 1 0 18"/>'
         '<path class="w w1" d="M18 20a6 6 0 0 0 0 8"/><path class="w w1" d="M30 20a6 6 0 0 1 0 8"/>'
         '<circle cx="24" cy="22" r="2" fill="currentColor"/>'
         '<path d="M19 42l5-17 5 17M21 36h6M22.5 31h3"/></svg>')


# ---------------------------------------------------------------- CSS and JavaScript
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root { --red:#E60000; --deep:#990000; --char:#17191F; --ink:#1F2330; --muted:#6B7280;
  --gap:clamp(6px,1vh,12px); --head:clamp(54px,8vh,72px); --kpi:clamp(76px,12vh,116px);
  --strip:clamp(26px,4vh,34px); --foot:16px;
  --body:calc(100vh - var(--head) - var(--foot) - 4 * var(--gap)); }
html, body, .stApp, .stApp button, .stApp input { font-family:'Inter',sans-serif; color-scheme:light; }
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  height:100vh; max-height:100vh; overflow:hidden !important; }
header[data-testid="stHeader"], footer, [data-testid="stSidebar"], [data-testid="stDecoration"] { display:none !important; }
[data-testid="stMainBlockContainer"] { max-width:100%; height:100vh; box-sizing:border-box;
  padding:var(--gap) clamp(10px,1.1vw,22px) 0; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] { gap:var(--gap); }
[data-stale="true"] { opacity:.6; transition:opacity .2s; }

/* Background: soft red light drifting slowly (transform only, so it stays cheap) */
.stApp { color:var(--ink); background:linear-gradient(160deg,#FFFFFF,#F3F4F7 55%,#EBEDF2); }
.stApp::after { content:""; position:fixed; inset:-25%; pointer-events:none; z-index:0;
  background:radial-gradient(circle at 18% 12%, rgba(230,0,0,.075), transparent 28%),
             radial-gradient(circle at 86% 88%, rgba(230,0,0,.06), transparent 30%);
  animation:sceneDrift 38s ease-in-out infinite alternate; }
@keyframes sceneDrift { from { transform:translate3d(-2%,-1%,0) scale(1); } to { transform:translate3d(3%,2%,0) scale(1.08); } }
[data-testid="stAppViewContainer"] { position:relative; z-index:1; }
/* Loading bar while Streamlit is re-running */
.stApp[data-teststate="running"]::before { content:""; position:fixed; top:0; left:0; width:100%; height:3px; z-index:9999;
  background:linear-gradient(90deg,transparent,#E60000,#FF7A7A,transparent) no-repeat; background-size:40% 100%;
  animation:loadbar 1s linear infinite; }
@keyframes loadbar { from { background-position:-40% 0; } to { background-position:140% 0; } }

/* Header: red gradient band, drifting glow, sweeping highlight, breathing logo */
.st-key-hdr { position:relative; overflow:hidden; height:var(--head); flex:none !important; box-sizing:border-box;
  padding:0 16px; justify-content:center; border-radius:14px; color:#fff;
  background:linear-gradient(110deg,#8F0000 0%,#C40000 36%,#E60000 68%,#B50000 100%); background-size:200% 100%;
  animation:hdrflow 16s ease-in-out infinite alternate; box-shadow:0 8px 24px rgba(153,0,0,.28); }
@keyframes hdrflow { to { background-position:100% 0; } }
.st-key-hdr::before { content:""; position:absolute; right:-6%; top:-70%; width:44%; height:240%; border-radius:50%;
  background:radial-gradient(circle at 30% 50%, rgba(255,255,255,.16), transparent 62%);
  animation:floatglow 9s ease-in-out infinite alternate; pointer-events:none; }
@keyframes floatglow { from { transform:translateX(0); } to { transform:translateX(-14%); } }
.st-key-hdr::after { content:""; position:absolute; bottom:0; left:-40%; width:40%; height:2px;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.95),transparent); animation:sweep 6s linear infinite; }
@keyframes sweep { from { left:-40%; } to { left:100%; } }
.st-key-hdr > *, .st-key-hdr [data-testid="stHorizontalBlock"] { position:relative; z-index:1; }
.brand { display:flex; align-items:center; gap:12px; min-width:0; }
.logo { background:#fff; border-radius:10px; padding:3px 9px; height:calc(var(--head) - 18px);
  display:flex; align-items:center; animation:breathe 4s ease-in-out infinite; }
.logo img { max-height:100%; max-width:112px; object-fit:contain; }
@keyframes breathe { 50% { box-shadow:0 0 20px rgba(255,255,255,.65); } }
.divider { width:1px; align-self:stretch; margin:6px 0; background:rgba(255,255,255,.35); }
.title { font-size:clamp(14px,1.3vw,22px); font-weight:800; white-space:nowrap; line-height:1.15; color:#fff; }
.status { font-size:clamp(9px,1.3vh,12px); color:rgba(255,255,255,.88); display:flex; align-items:center; gap:6px; white-space:nowrap; }
.dot { width:7px; height:7px; border-radius:50%; background:#6EE7A8; animation:pulse 2s infinite; }
@keyframes pulse { 0% { box-shadow:0 0 0 0 rgba(110,231,168,.7); } 70% { box-shadow:0 0 0 8px rgba(110,231,168,0); }
  100% { box-shadow:0 0 0 0 rgba(110,231,168,0); } }
.stamp { display:flex; align-items:center; justify-content:flex-end; gap:10px; color:#fff; font-size:10px; line-height:1.3; text-align:right; }
.stamp b { font-size:11px; }
.tower { flex:none; }
.tower .w { animation:wave 2.4s ease-in-out infinite; opacity:.25; }
.tower .w2 { animation-delay:.5s; }
@keyframes wave { 0%,100% { opacity:.2; } 50% { opacity:1; } }

/* Two-tab navigation with sliding white pill */
.st-key-view [role="radiogroup"] { position:relative; display:grid; grid-template-columns:1fr 1fr; gap:0;
  width:100%; padding:3px; border-radius:12px; background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.28); }
.st-key-view [role="radiogroup"]::before { content:""; position:absolute; top:3px; bottom:3px; left:3px;
  width:calc(50% - 3px); border-radius:9px; background:#fff; box-shadow:0 4px 14px rgba(0,0,0,.2);
  transition:transform .35s cubic-bezier(.4,0,.2,1); }
.st-key-view [role="radiogroup"]:has(> :nth-child(2) input:checked)::before { transform:translateX(100%); }
.st-key-view label { position:relative; z-index:1; justify-content:center; margin:0; padding:6px 4px; cursor:pointer; width:100%; }
.st-key-view label > div:not(:has(p)), .st-key-view label div:has(+ div p):not(:has(p)) { display:none !important; }
.st-key-view label p { font-size:12px; font-weight:700; white-space:nowrap; color:#fff; transition:color .3s; }
.st-key-view label:has(input:checked) p { color:var(--deep); }
.st-key-view [data-testid="stRadio"], .st-key-view [data-testid="stRadio"] > div { width:100%; }
.st-key-hdr button { min-height:34px; border:1px solid rgba(255,255,255,.7); background:#fff; color:var(--deep);
  border-radius:10px; transition:all .25s; }
.st-key-hdr button:hover { transform:translateY(-1px); box-shadow:0 6px 16px rgba(0,0,0,.25); border-color:#fff; }
.st-key-hdr button p { font-size:12px; font-weight:800; white-space:nowrap; color:var(--deep); }

[data-testid="stMainBlockContainer"] [data-testid="stMarkdownContainer"] { margin-bottom:0 !important; }
[data-testid="stLayoutWrapper"]:has(> .st-key-fx), .st-key-fx { position:absolute; width:0; height:0; overflow:hidden; }

/* Body: filter panel + content */
.st-key-body { display:grid !important; grid-template-columns:clamp(190px,15vw,262px) minmax(0,1fr);
  gap:var(--gap) !important; height:var(--body); min-height:0; flex:none !important; }
[data-testid="stLayoutWrapper"]:has(> .st-key-body) { flex:none !important; }
.st-key-body > * { min-height:0; min-width:0; height:100%; }
[data-testid="stLayoutWrapper"]:has(> .st-key-side), [data-testid="stLayoutWrapper"]:has(> .st-key-content) {
  height:100%; min-height:0; min-width:0; }
.st-key-content { height:100% !important; gap:var(--gap) !important; min-height:0; overflow:hidden; }
.st-key-content > [data-testid="stElementContainer"] { flex:none; }

/* Dark filter panel */
.st-key-side { position:relative; height:100% !important; box-sizing:border-box; overflow-y:auto; overflow-x:hidden;
  padding:14px 12px; gap:clamp(4px,.9vh,10px) !important; border-radius:16px; color:#fff;
  background:linear-gradient(180deg,#1C1F27,#17191F 62%,#2A0A0C); box-shadow:0 10px 28px rgba(23,25,31,.3);
  scrollbar-width:thin; scrollbar-color:rgba(255,255,255,.25) transparent; }
.st-key-side::before { content:""; position:absolute; left:-30%; right:-30%; bottom:-14%; height:36%; pointer-events:none;
  background:radial-gradient(ellipse at 50% 100%, rgba(230,0,0,.55), transparent 65%); animation:glowpulse 5s ease-in-out infinite alternate; }
@keyframes glowpulse { from { opacity:.45; } to { opacity:1; } }
.st-key-side > * { position:relative; z-index:1; }
.ph { display:flex; align-items:center; gap:8px; font-size:15px; font-weight:800; padding-bottom:8px;
  border-bottom:1px solid rgba(255,255,255,.14); }
.ph svg { color:#FF5A5A; }
.st-key-side [data-testid="stWidgetLabel"] { min-height:0; margin:0 0 2px; }
.st-key-side [data-testid="stWidgetLabel"] p { font-size:10px; font-weight:800; letter-spacing:1.1px; text-transform:uppercase; color:rgba(255,255,255,.78); }
.st-key-side [data-baseweb="select"] > div { min-height:34px; font-size:12px; background:#fff; border-color:transparent;
  border-radius:9px; transition:box-shadow .25s; }
.st-key-side [data-baseweb="select"] * { color:var(--char); }
.st-key-side [data-baseweb="select"] > div:hover, .st-key-side [data-baseweb="select"]:focus-within > div {
  box-shadow:0 0 0 2px rgba(230,0,0,.75), 0 0 14px rgba(230,0,0,.5); }
.st-key-side [data-testid="stSlider"] p, .st-key-side [data-testid="stSlider"] div[data-testid="stTickBar"] * { color:rgba(255,255,255,.85); font-size:10px; }
.st-key-side [data-baseweb="slider"] div[role="slider"] { background:var(--red); box-shadow:0 0 0 4px rgba(230,0,0,.3); }
.st-key-side button { min-height:34px; border-radius:10px; border:1px solid rgba(255,255,255,.55); background:transparent; transition:all .25s; }
.st-key-side button:hover { background:var(--red); border-color:var(--red); box-shadow:0 0 16px rgba(230,0,0,.55); }
.st-key-side button p { color:#fff; font-size:12px; font-weight:700; }
.sel { margin-top:2px; padding:10px 12px; border-radius:12px; background:rgba(255,255,255,.95); color:var(--char);
  animation:rise .6s cubic-bezier(.2,.7,.3,1) backwards; }
.sel .label { color:var(--muted); }
.sname { font-size:clamp(13px,1.9vh,17px); font-weight:800; color:var(--red); line-height:1.2; margin:2px 0 6px; overflow-wrap:anywhere; }
.srow { display:flex; align-items:center; justify-content:space-between; gap:8px; font-size:11px; color:#4B5563;
  border-top:1px solid #ECEEF2; padding-top:5px; margin-top:5px; }
.srow span { display:flex; align-items:center; gap:5px; } .srow svg { color:var(--red); }
.srow b { font-size:clamp(14px,2.3vh,19px); color:var(--char); font-variant-numeric:tabular-nums; }

/* KPI cards */
.kpis { display:grid; grid-template-columns:repeat(4,1fr); gap:var(--gap); height:var(--kpi); }
.glass, [class*="st-key-card_"] { position:relative; overflow:hidden; box-sizing:border-box; border-radius:14px;
  border:1px solid #E3E6EB; background:linear-gradient(155deg,#FFFFFF,#FAFAFB);
  box-shadow:0 3px 14px rgba(25,35,55,.07), inset 0 1px 0 rgba(255,255,255,.9);
  animation:rise .6s cubic-bezier(.2,.7,.3,1) backwards; transition:transform .25s, border-color .25s, box-shadow .25s; }
.glass:hover, [class*="st-key-card_"]:hover { transform:translateY(-2px); border-color:rgba(230,0,0,.55);
  box-shadow:0 10px 24px rgba(25,35,55,.12), 0 0 16px rgba(230,0,0,.09); }
@keyframes rise { from { opacity:0; transform:translateY(14px); } }
.kpi { padding:0 clamp(10px,1.1vw,18px); display:flex; align-items:center; gap:clamp(8px,.9vw,14px); }
.kpi:nth-child(2) { animation-delay:.08s; } .kpi:nth-child(3) { animation-delay:.16s; } .kpi:nth-child(4) { animation-delay:.24s; }
.kpi.hot { background:linear-gradient(155deg,#FFF3F4,#FDE4E6); border-color:rgba(230,0,0,.25); }
.kpi::before { content:""; position:absolute; inset:0 0 auto 0; height:3px;
  background:linear-gradient(90deg,transparent,var(--accent),transparent); background-size:200% 100%; animation:flow 3.5s linear infinite; }
@keyframes flow { from { background-position:200% 0; } to { background-position:-200% 0; } }
.kico { flex:none; width:clamp(30px,4.8vh,44px); height:clamp(30px,4.8vh,44px); border-radius:12px; display:grid; place-items:center;
  color:var(--red); background:linear-gradient(135deg,rgba(230,0,0,.14),rgba(230,0,0,.04)); }
.kico svg { width:56%; height:56%; }
.kbody { min-width:0; flex:1; }
.kmark { position:absolute; right:-6px; bottom:-8px; width:clamp(44px,8vh,76px); height:clamp(44px,8vh,76px); color:rgba(230,0,0,.07); }
.kmark svg { width:100%; height:100%; }
.label { font-size:clamp(9px,1.25vh,11px); font-weight:800; letter-spacing:1.3px; text-transform:uppercase; color:var(--muted); white-space:nowrap; }
.value { font-size:clamp(21px,4.2vh,38px); font-weight:800; line-height:1.12; font-variant-numeric:tabular-nums; color:var(--char); }
.note { font-size:clamp(9px,1.3vh,12px); color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.up { color:var(--red); font-weight:700; } .down { color:var(--green,#12A150); font-weight:700; }
/* cursor spotlight (from the IPL dashboard) */
.glass::after, [class*="st-key-card_"]::after { content:""; position:absolute; inset:0; pointer-events:none; opacity:0;
  background:radial-gradient(320px circle at var(--mx,50%) var(--my,50%),rgba(230,0,0,.10),transparent 62%); transition:opacity .3s; }
.glass:hover::after, [class*="st-key-card_"]:hover::after { opacity:1; }

/* Analytics grids */
.st-key-grid_ov, .st-key-grid_cd { display:grid !important; gap:var(--gap) !important; min-height:0; flex:none !important;
  height:calc(var(--body) - var(--kpi) - var(--strip) - 2 * var(--gap)); grid-template-columns:repeat(12,minmax(0,1fr));
  grid-template-rows:repeat(2,minmax(0,1fr)); }
[data-testid="stLayoutWrapper"]:has(> [class*="st-key-grid_"]) { flex:none !important; }
[data-testid="stLayoutWrapper"]:has(> [class*="st-key-card_"]) { height:100%; min-height:0; min-width:0; }
.st-key-grid_ov > :nth-child(1) { grid-column:span 5; } .st-key-grid_ov > :nth-child(2) { grid-column:span 4; }
.st-key-grid_ov > :nth-child(3) { grid-column:span 3; } .st-key-grid_ov > :nth-child(4) { grid-column:span 4; }
.st-key-grid_ov > :nth-child(5) { grid-column:span 5; } .st-key-grid_ov > :nth-child(6) { grid-column:span 3; }
.st-key-grid_cd > :nth-child(1) { grid-column:span 5; grid-row:span 2; } .st-key-grid_cd > :nth-child(2) { grid-column:span 3; }
.st-key-grid_cd > :nth-child(3) { grid-column:span 4; } .st-key-grid_cd > :nth-child(4) { grid-column:span 4; }
.st-key-grid_cd > :nth-child(5) { grid-column:span 3; }
[class*="st-key-card_"] { height:100% !important; min-height:0; padding:clamp(7px,1.1vh,12px) 12px 6px;
  gap:3px !important; flex-wrap:nowrap !important; }
.st-key-grid_ov > :nth-child(2) > div, .st-key-grid_cd > :nth-child(2) > div { animation-delay:.08s; }
.st-key-grid_ov > :nth-child(3) > div, .st-key-grid_cd > :nth-child(3) > div { animation-delay:.16s; }
.st-key-grid_ov > :nth-child(4) > div, .st-key-grid_cd > :nth-child(4) > div { animation-delay:.24s; }
.st-key-grid_ov > :nth-child(5) > div, .st-key-grid_cd > :nth-child(5) > div { animation-delay:.32s; }
.st-key-grid_ov > :nth-child(6) > div { animation-delay:.4s; }
[class*="st-key-card_"] > [data-testid="stElementContainer"]:has([data-testid="stPlotlyChart"]) {
  flex:1 1 0 !important; min-height:0; height:auto !important; animation:wipe .9s ease-out backwards; }
[class*="st-key-card_"] [data-testid="stFullScreenFrame"], [class*="st-key-card_"] [data-testid="stFullScreenFrame"] > div,
[class*="st-key-card_"] [data-testid="stPlotlyChart"] { height:100% !important; min-height:0; }
@keyframes wipe { from { clip-path:inset(0 100% 0 0); opacity:.2; } to { clip-path:inset(0 0 0 0); } }
.ch { display:flex; align-items:center; gap:9px; min-width:0; }
.ch.pad { margin-right:84px; }
.ci { flex:none; width:clamp(22px,3.4vh,30px); height:clamp(22px,3.4vh,30px); border-radius:9px; display:grid; place-items:center;
  color:#fff; background:linear-gradient(135deg,var(--red),var(--deep)); box-shadow:0 3px 10px rgba(230,0,0,.3); }
.ci svg { width:56%; height:56%; }
.cw { min-width:0; }
.ct { font-size:clamp(11px,1.65vh,15px); font-weight:800; line-height:1.25; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; color:var(--char); }
.cs { font-size:clamp(9px,1.2vh,11px); color:var(--muted); line-height:1.3; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
[data-testid="stLayoutWrapper"]:has(> [class*="st-key-flt_"]), [class*="st-key-flt_"] {
  position:absolute !important; top:clamp(6px,1vh,12px); right:10px; width:auto !important; height:auto !important; z-index:5; }
[data-testid="stLayoutWrapper"] > [class*="st-key-flt_"] { position:static !important; }
[class*="st-key-flt_"] button { min-height:24px; height:24px; padding:0 4px 0 10px; border-radius:8px;
  border:1px solid rgba(148,163,184,.4); background:#FFFFFF; transition:all .25s; }
[class*="st-key-flt_"] button:hover { border-color:var(--red); box-shadow:0 0 12px rgba(230,0,0,.4); }
[class*="st-key-flt_"] button[kind*="primary"], [class*="st-key-flt_"] button[data-testid*="rimary"] { background:var(--deep); border-color:var(--red); }
[class*="st-key-flt_"] button[kind*="primary"] p { color:#fff; }
[class*="st-key-flt_"] button p { font-size:10px; font-weight:700; white-space:nowrap; }
.empty { font-size:12px; color:var(--muted); text-align:center; padding:4vh 8px; }

/* Insight strip + footer */
.strip { display:flex; align-items:center; gap:10px; height:var(--strip); box-sizing:border-box; padding:0 14px; border-radius:11px;
  color:#fff; font-size:clamp(10px,1.45vh,12.5px); white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
  background:linear-gradient(100deg,#17191F,#262A33 70%,#3A0B0E); box-shadow:0 5px 16px rgba(23,25,31,.25);
  animation:rise .7s .35s cubic-bezier(.2,.7,.3,1) backwards; }
.strip svg { flex:none; color:#FF5A5A; animation:spin 12s linear infinite; }
@keyframes spin { to { transform:rotate(360deg); } }
.strip b { color:#fff; } .strip span { overflow:hidden; text-overflow:ellipsis; }
.foot { height:var(--foot); font-size:10px; line-height:var(--foot); color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.foot b { color:var(--red); }

/* Chatbot dialog */
[data-testid="stDialog"] > div { backdrop-filter:blur(4px); background:rgba(23,25,31,.45); }
div[role="dialog"] { border-radius:20px !important; overflow:hidden; background:#fff !important; color:var(--ink);
  border:1px solid rgba(230,0,0,.3); box-shadow:0 30px 80px rgba(23,25,31,.45), 0 0 0 1px rgba(255,255,255,.5);
  animation:dlg .4s cubic-bezier(.2,.8,.2,1); }
@keyframes dlg { from { opacity:0; transform:translateY(26px) scale(.96); } }
div[role="dialog"] h2 { font-size:12px !important; font-weight:800 !important; letter-spacing:1.4px; text-transform:uppercase; color:var(--red) !important; }
div[role="dialog"] button[aria-label="Close"] { border-radius:50%; background:#F3F4F7; border:1px solid #E3E6EB; transition:all .2s; }
div[role="dialog"] button[aria-label="Close"]:hover { background:var(--red); color:#fff; transform:rotate(90deg); }
.chat-hero { position:relative; overflow:hidden; display:flex; align-items:center; gap:14px; padding:14px 16px; margin-bottom:10px;
  border-radius:14px; color:#fff; background:linear-gradient(110deg,#8F0000,#E60000 70%,#B50000); }
.chat-hero::after { content:""; position:absolute; top:0; left:-40%; width:40%; height:100%;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.2),transparent); animation:shine 4.5s ease-in-out infinite; }
@keyframes shine { from { left:-40%; } to { left:110%; } }
.chat-orb { flex:none; width:40px; height:40px; border-radius:12px; display:grid; place-items:center; background:rgba(255,255,255,.2);
  animation:breathe 3s ease-in-out infinite; }
.chat-hero .t { font-size:16px; font-weight:800; } .chat-hero .s { font-size:11px; opacity:.9; }
.chips { display:flex; flex-wrap:wrap; gap:6px; margin-top:6px; }
.chip { font-size:10px; font-weight:700; padding:2px 9px; border-radius:99px; background:rgba(255,255,255,.2); border:1px solid rgba(255,255,255,.35); }
div[role="dialog"] [data-testid="stButton"] button { border-radius:99px; border:1px solid rgba(230,0,0,.35); background:#FFF5F6;
  color:var(--deep); font-weight:600; transition:all .2s; }
div[role="dialog"] [data-testid="stButton"] button:hover { background:var(--red); border-color:var(--red); color:#fff; transform:translateY(-1px);
  box-shadow:0 6px 16px rgba(230,0,0,.3); }
div[role="dialog"] [data-testid="stButton"] button:hover p { color:#fff; }
div[role="dialog"] [data-testid="stChatMessage"] { border-radius:14px; padding:10px 12px; margin-bottom:6px; background:#F7F8FA;
  border:1px solid #ECEEF2; border-left:3px solid var(--red); animation:rise .35s ease backwards; }
div[role="dialog"] [data-testid="stChatInput"], div[role="dialog"] [data-testid="stChatInput"] > div { border-radius:14px; }
div[role="dialog"] [data-testid="stChatInput"] > div { border:1px solid #D9DCE3; background:#fff; transition:box-shadow .25s, border-color .25s; }
div[role="dialog"] [data-testid="stChatInput"]:focus-within > div { border-color:var(--red); box-shadow:0 0 0 3px rgba(230,0,0,.15); }
div[role="dialog"] [data-testid="stChatInput"] button { background:var(--red); color:#fff; border-radius:10px; }
div[role="dialog"] [data-testid="stSpinner"] { color:var(--deep); font-weight:600; }
div[role="dialog"] [data-testid="stSpinner"] svg { stroke:var(--red); }
/* Styling for the controls chatbot.py really uses: text input + form button, answer table, SQL expander */
div[role="dialog"] [data-testid="stTextInput"] input { border-radius:12px; border:1px solid #D9DCE3; background:#fff;
  color:var(--ink); padding:10px 12px; transition:box-shadow .25s, border-color .25s; }
div[role="dialog"] [data-testid="stTextInput"] input:focus { border-color:var(--red); box-shadow:0 0 0 3px rgba(230,0,0,.15); }
div[role="dialog"] [data-testid="stTextInput"] > div > div { border-radius:12px; border:none; background:transparent; }
div[role="dialog"] [data-testid="stFormSubmitButton"] button { background:linear-gradient(135deg,var(--red),var(--deep));
  border:none; border-radius:12px; color:#fff; font-weight:700; transition:all .2s; }
div[role="dialog"] [data-testid="stFormSubmitButton"] button p { color:#fff; }
div[role="dialog"] [data-testid="stFormSubmitButton"] button:hover { transform:translateY(-1px); box-shadow:0 8px 18px rgba(230,0,0,.35); }
div[role="dialog"] [data-testid="stDataFrame"] { border:1px solid #ECEEF2; border-radius:12px; overflow:hidden; animation:rise .35s ease backwards; }
div[role="dialog"] [data-testid="stExpander"] { border:1px solid #ECEEF2; border-radius:12px; background:#FAFAFB; }
div[role="dialog"] [data-testid="stMarkdownContainer"] strong { color:var(--deep); }
[data-testid="stPopoverBody"] { border-radius:12px; }

/* Short desktop screens (e.g. 1366x768 inside a browser) */
@media (max-height:700px) { .cs, .status, .note { display:none; } .kpi .note { display:block; }
  .srow:last-child, .stamp { display:none; } .st-key-side [data-testid="stWidgetLabel"] p { font-size:9px; }
  .st-key-side [data-baseweb="select"] > div { min-height:30px; } .kmark { display:none; } }
@media (max-width:1250px) { .stamp div { display:none; } .kmark { display:none; } }
/* Phones, tiny windows: allow scrolling instead of cutting content off */
@media (max-width:1000px), (max-height:520px) {
  html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { overflow:auto !important; max-height:none; }
  [data-testid="stMainBlockContainer"] { height:auto; }
  .st-key-hdr { height:auto; overflow:visible; padding:8px; }
  .st-key-body { height:auto; grid-template-columns:1fr; } .st-key-content { height:auto !important; overflow:visible; }
  .kpis { grid-template-columns:1fr 1fr; height:auto; } .kpi { min-height:84px; }
  .st-key-grid_ov, .st-key-grid_cd { height:auto; grid-template-columns:1fr; grid-template-rows:none; grid-auto-rows:340px; }
  .st-key-grid_ov > *, .st-key-grid_cd > * { grid-column:auto !important; grid-row:auto !important; } }
@media (prefers-reduced-motion:reduce) { .stApp, .stApp *, .stApp *::before, .stApp *::after, .stApp::after,
  div[role="dialog"], div[role="dialog"] * { animation:none !important; transition:none !important; } }
</style>
"""

JS = """
<script>
(() => {
  if (window.__airtelFx) return;
  window.__airtelFx = true;
  document.addEventListener('mousemove', e => {
    const card = e.target.closest && e.target.closest('.glass, [class*="st-key-card_"]');
    if (!card) return;
    const r = card.getBoundingClientRect();
    card.style.setProperty('--mx', (e.clientX - r.left) + 'px');
    card.style.setProperty('--my', (e.clientY - r.top) + 'px');
  }, {passive: true});
  const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const countUp = el => {
    const target = el.dataset.count;
    if (el.dataset.done === target) return;
    el.dataset.done = target;
    const end = Number(target), places = Number(el.dataset.places), suffix = el.dataset.suffix || '';
    const show = v => el.textContent = v.toLocaleString('en-US',
      {minimumFractionDigits: places, maximumFractionDigits: places}) + suffix;
    if (still) return show(end);
    const start = performance.now();
    const step = now => {
      if (el.dataset.done !== target) return;
      const p = Math.min((now - start) / 900, 1);
      show(end * (1 - Math.pow(1 - p, 3)));
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  };
  const scan = () => document.querySelectorAll('[data-count]').forEach(countUp);
  new MutationObserver(scan).observe(document.body,
    {subtree: true, childList: true, attributes: true, attributeFilter: ['data-count']});
  scan();
})();
</script>
"""


# ---------------------------------------------------------------- layout pieces
def markup(content):
    st.markdown(" ".join(content.split()), unsafe_allow_html=True)


def counter(value, places=0, suffix=""):
    return (f'<span data-count="{value}" data-places="{places}" data-suffix="{suffix}">'
            f"{value:,.{places}f}{suffix}</span>")


def logo_html():
    for folder in (APP_DIR / "assets", ROOT / "assets"):
        path = folder / "airtel_logo.png"
        if path.is_file():
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            return f'<div class="logo"><img src="data:image/png;base64,{data}" alt="Airtel logo"></div>'
    return ""


def sticky(key, options):
    """Set a widget's value if it is missing or no longer valid."""
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]


def reset_filters(keys):
    for key in keys:
        st.session_state[key] = "All"


def reset_panel(window):
    for key in ("connection", "region", "tenure"):
        st.session_state[key] = "All"
    if window:
        st.session_state["window"] = window


def apply_filters(data, chosen):
    for column, value in chosen.items():
        data = data[data[column] == value]
    return data


def local_cols(ctx, *columns):
    """Chart-level filters are only offered for columns the side panel has not already fixed."""
    return [c for c in columns if ctx.get(GLOBAL_KEYS.get(c, ""), "All") == "All"]


def chart_filters(slot, data, columns):
    """Filter button for one chart. It narrows `data` for that chart only."""
    chosen = {}
    if not columns:
        return chosen
    keys = {column: f"f_{slot}_{column}" for column in columns}
    options = {}
    for column in columns:
        present = data[column].dropna()
        if column == "tenure_band":  # keep tenure in its natural order
            values = [v for v in data[column].cat.categories if v in set(present)]
        else:
            values = sorted(present.unique())
        options[column] = ["All"] + list(values)
        sticky(keys[column], options[column])
    active = sum(st.session_state[key] != "All" for key in keys.values())
    with st.container(key=f"flt_{slot}"):
        with st.popover(f"Filter · {active}" if active else "Filter",
                        type="primary" if active else "secondary"):
            for column in columns:
                value = st.selectbox(FILTER_LABELS[column], options[column], key=keys[column])
                if value != "All":
                    chosen[column] = value
            st.button("Reset", key=f"reset_{slot}", on_click=reset_filters,
                      args=(list(keys.values()),), disabled=not active, width="stretch")
    return chosen


def head_html(title, subtitle, icon, pad):
    cls = "ch pad" if pad else "ch"
    safe = html.escape(title)
    return " ".join((f'<div class="{cls}"><span class="ci">{svg(icon, 15)}</span><div class="cw">'
                     f'<div class="ct" title="{safe}">{safe}</div>'
                     f'<div class="cs">{html.escape(subtitle)}</div></div></div>').split())


def chart_card(slot, title, subtitle, icon, data, columns, build, signature):
    """One chart with its own filters. `build` turns the filtered rows into a figure (or a message)."""
    with st.container(key=f"card_{slot}"):
        heading = st.empty()
        chosen = chart_filters(slot, data, columns)
        subset = apply_filters(data, chosen)
        note = " · ".join(str(v) for v in chosen.values())
        name = f"{title} · {note}" if note else title
        heading.markdown(head_html(name, subtitle, icon, bool(columns)), unsafe_allow_html=True)
        fig = build(subset, chosen) if not subset.empty else None
        if fig is None or isinstance(fig, str):
            markup(f'<div class="empty">{html.escape(fig or "No data for this selection.")}</div>')
        else:
            st.plotly_chart(fig, width="stretch", height="stretch", theme=None,
                            key=f"{slot}|{signature}|{note}",
                            config={"displayModeBar": False, "responsive": True})


def map_card(data, circle, signature):
    """India map. Clicking a bubble selects that circle for the whole Circle Analysis view."""
    with st.container(key="card_cd_map"):
        heading = st.empty()
        fig, names, skipped = india_circle_map(data, circle) if not data.empty else (None, [], 0)
        subtitle = "Size: subscribers · Colour: churn rate · Click a bubble to select · Approximate circle centres"
        if skipped:
            subtitle += f" · {skipped} circle(s) not placed (no location match)"
        heading.markdown(head_html("Airtel circles across India", subtitle, "pin", False), unsafe_allow_html=True)
        if fig is None:
            markup('<div class="empty">No circle could be placed on the map for this selection.</div>')
            return
        event = st.plotly_chart(fig, width="stretch", height="stretch", theme=None,
                                key=f"cd_map|{signature}", on_select="rerun", selection_mode="points",
                                config={"displayModeBar": False, "responsive": True, "scrollZoom": True})
        picked = picked_circle(event, names)
        if picked and picked != circle and st.session_state.get("_map_seen") != (signature, picked):
            st.session_state["_map_seen"] = (signature, picked)
            st.session_state["_pending_circle"] = picked
            st.rerun()


def kpi_cards(data, base_note):
    total, churned, rate = kpis(data)
    gap = (rate - NATIONAL_REF) * 100
    arrow = "▲" if gap > 0 else "▼" if gap < 0 else "●"
    tone = "up" if gap > 0 else "down"
    rate_note = (f'<span class="{tone}">{arrow} {abs(gap):.2f} pts</span> vs 23.75% reference'
                 if total else "No subscribers in this selection")
    open_cases = int(data.unresolved_count.sum())
    open_customers = int((data.unresolved_count > 0).sum())
    cards = [
        ("Total subscribers", counter(total), base_note, "users", RED, ""),
        ("Churn rate", counter(round(rate * 100, 2), 2, "%"), rate_note, "pulse", RED, " hot"),
        ("Retained subscribers", counter(total - churned),
         f"{pct((total - churned) / total) if total else '0.0%'} retention · {churned:,} churned", "shield", "#12A150", ""),
        ("Open complaints", counter(open_cases),
         f"{open_customers:,} customers · {pct(open_customers / total) if total else '0.0%'} of base", "ticket", CHARCOAL, ""),
    ]
    markup('<div class="kpis">' + "".join(
        f'<div class="glass kpi{hot}" style="--accent:{a}"><div class="kico">{svg(i, 22)}</div>'
        f'<div class="kbody"><div class="label">{l}</div><div class="value">{v}</div>'
        f'<div class="note">{n}</div></div><div class="kmark">{svg(i, 76)}</div></div>'
        for l, v, n, i, a, hot in cards) + "</div>")


def strip_html(text):
    markup(f'<div class="strip">{svg("spark", 15)}<span><b>Overall insight:</b> {text}</span></div>')


def overview_insight(data):
    if data.empty:
        return "No subscribers match the current filters."
    parts = []
    circles = churn_by(data, "circle_name")
    if len(circles) > 1:
        hi, lo = circles.loc[circles.rate.idxmax()], circles.loc[circles.rate.idxmin()]
        parts.append(f"Highest churn: <b>{html.escape(hi.circle_name)}</b> ({pct(hi.rate)}); "
                     f"lowest: <b>{html.escape(lo.circle_name)}</b> ({pct(lo.rate)})")
    conns = churn_by(data, "connection_type")
    if len(conns) > 1:
        top = conns.loc[conns.rate.idxmax()]
        parts.append(f"<b>{html.escape(top.connection_type)}</b> churns most ({pct(top.rate)})")
    status = churn_by(data, "complaint_status").set_index("complaint_status").rate
    if {"Has unresolved", "No complaints"} <= set(status.index):
        parts.append(f"Unresolved-complaint customers churn at {pct(status['Has unresolved'])} vs "
                     f"{pct(status['No complaints'])} with no complaints (association, not cause)")
    return " · ".join(parts) or "Not enough variety in the current selection for a comparison."


def detail_insight(data, cdata, circle):
    if cdata.empty:
        return f"No subscribers for <b>{html.escape(circle)}</b> under the current filters."
    table = churn_by(data, "circle_name")
    _, churned, rate = kpis(cdata)
    rank = int((table.rate > rate).sum()) + 1
    gap = (rate - NATIONAL_REF) * 100
    parts = [f"<b>{html.escape(circle)}</b> ranks {rank} of {len(table)} circles by churn at {pct(rate, 2)} "
             f"({abs(gap):.2f} pts {'above' if gap > 0 else 'below' if gap < 0 else 'equal to'} the 23.75% reference)"]
    conns = churn_by(cdata, "connection_type")
    if len(conns) > 1:
        top = conns.loc[conns.rate.idxmax()]
        parts.append(f"{html.escape(top.connection_type)} is its highest-churn connection type ({pct(top.rate)})")
    parts.append(f"{churned:,} subscribers churned")
    return " · ".join(parts)


# ---------------------------------------------------------------- header and panel
def render_header(df, logo, stamp):
    for key in list(st.session_state):  # keep selections while the other view is open
        if key == "circle" or str(key).startswith("f_"):
            st.session_state[key] = st.session_state[key]
    with st.container(key="hdr"):
        brand, nav, when, ask = st.columns([3.4, 2.1, 1.9, 1.1], vertical_alignment="center")
        with brand:
            note = f"Read-only SQLite · {len(df):,} records" + ("" if logo else " · logo file not found")
            markup(f'<div class="brand">{logo}<div class="divider"></div><div><div class="title">Airtel Telecom Analytics</div>'
                   f'<div class="status"><span class="dot"></span>Retention &amp; Circle Analysis · {note}</div></div></div>')
        with nav:
            view = st.radio("View", [OVERVIEW, DETAIL], key="view", horizontal=True,
                            label_visibility="collapsed", width="stretch")
        with when:
            markup(f'<div class="stamp">{TOWER}<div>Data refreshed<br><b>{html.escape(stamp)}</b></div></div>')
        with ask:
            asked = st.button("✦ Ask data", width="stretch", help="AI Data Analyst")
    return view, asked


def render_panel(df, view):
    """Dark filter panel. Returns the filter context and the globally filtered rows."""
    conns = ["All"] + sorted(df.connection_type.unique())
    regions = ["All"] + sorted(df.region.unique())
    present = set(df.tenure_band.dropna())
    tenures = ["All"] + [str(c) for c in df.tenure_band.cat.categories if c in present]
    months = month_options(df)
    default_window = (months[0], months[-1]) if len(months) >= 2 else None
    circle = None
    with st.container(key="side"):
        markup(f'<div class="ph">{svg("filter", 18)}<span>Filters</span></div>')
        if view == DETAIL:
            circles = sorted(df.circle_name.unique())
            pending = st.session_state.pop("_pending_circle", None)
            if pending in circles:
                st.session_state["circle"] = pending
            sticky("circle", circles)
            circle = st.selectbox("Circle", circles, key="circle")
        sticky("connection", conns)
        sticky("region", regions)
        sticky("tenure", tenures)
        connection = st.selectbox("Connection type", conns, key="connection")
        region = st.selectbox("Region", regions, key="region")
        tenure = st.selectbox("Tenure", tenures, key="tenure")
        window = None
        if default_window:
            current = st.session_state.get("window")
            if not (isinstance(current, (tuple, list)) and len(current) == 2
                    and current[0] in months and current[1] in months):
                st.session_state["window"] = default_window
            window = tuple(st.select_slider("Churn period", options=months, key="window",
                                            format_func=month_label,
                                            help="Applies to the month-by-month charts only."))
        st.button("↺ Reset filters", key="reset_all", on_click=reset_panel, args=(default_window,), width="stretch")

        data = df
        if connection != "All":
            data = data[data.connection_type == connection]
        if region != "All":
            data = data[data.region == region]
        if tenure != "All":
            data = data[data.tenure_band == tenure]
        scope_parts = [p for p in (circle, connection if connection != "All" else None,
                                   region if region != "All" else None, tenure if tenure != "All" else None) if p]
        scope = " · ".join(scope_parts) or "All circles"
        shown = data[data.circle_name == circle] if circle else data
        total, churned, _ = kpis(shown)
        markup(f'<div class="sel"><div class="label">Current selection</div>'
               f'<div class="sname">{html.escape(scope)}</div>'
               f'<div class="srow"><span>{svg("users", 14)} Subscribers</span><b>{total:,}</b></div>'
               f'<div class="srow"><span>{svg("logout", 14)} Churned</span><b>{churned:,}</b></div></div>')
    ctx = dict(connection=connection, region=region, tenure=tenure, window=window, circle=circle, scope=scope,
               sig=f"{view}|{connection}|{region}|{tenure}|{window}|{circle}")
    return ctx, data


@st.dialog("AI Data Analyst", width="large")
def chat_dialog(connection, region, circle):
    chips = "".join(f'<span class="chip">{html.escape(label)}: {html.escape(value)}</span>'
                    for label, value in (("Connection", connection), ("Region", region), ("Circle", circle)))
    markup(f'<div class="chat-hero"><div class="chat-orb">{svg("spark", 22)}</div><div>'
           f'<div class="t">Ask the Airtel data</div>'
           f'<div class="s">Answers are computed from the SQLite database, scoped to your current view.</div>'
           f'<div class="chips">{chips}</div></div></div>')
    render_chatbot(st.session_state.db_path, connection, region, circle)


# ---------------------------------------------------------------- views
def render_overview(data, df, ctx):
    total_all = len(df)
    share = len(data) / total_all if total_all else 0
    kpi_cards(data, f"{share:.1%} of all records")
    window = ctx["window"]
    sig = ctx["sig"]

    def month_fig(rows, _):
        month = monthly_churn(rows, window)
        return monthly_chart(month) if not month.empty else "No dated churn in this period."

    undated = int(((data.is_churned == 1) & data.churn_month.isna()).sum())
    with st.container(key="grid_ov"):
        chart_card("treemap", "Subscriber base × churn by circle", "Area: subscribers · Colour: churn rate", "tiles",
                   data, local_cols(ctx, "connection_type") + ["complaint_status"],
                   lambda d, _: treemap_chart(d), sig)
        chart_card("month", "Monthly churned subscribers",
                   "Counts by churn date" + (f" · {undated:,} undated excluded" if undated else ""), "trend",
                   data, local_cols(ctx, "connection_type") + ["complaint_status"], month_fig, sig)
        chart_card("funnel", "Complaint-to-churn funnel", "Do unresolved issues end in churn?", "filter",
                   data, [], lambda d, _: funnel_chart(d), sig)
        chart_card("conn", "Churn by connection type", "Stem: gap from 23.75% reference", "signal",
                   data, ["complaint_status"] + local_cols(ctx, "tenure_band"),
                   lambda d, _: connection_chart(d), sig)
        chart_card("heat", "Churn heatmap: circle × tenure", "Largest 12 circles · Colour: churn rate", "grid",
                   data, ["complaint_status"] + local_cols(ctx, "connection_type"),
                   lambda d, _: heatmap_chart(d), sig)
        chart_card("radar", "Tenure retention profile", "Radar: churn rate by tenure band", "target",
                   data, ["complaint_status"] + local_cols(ctx, "connection_type"),
                   lambda d, _: radar_chart(d), sig)
    strip_html(overview_insight(data))
    return f"Scope: {ctx['scope']}"


def render_detail(data, df, ctx):
    circle = ctx["circle"]
    cdata = data[data.circle_name == circle]
    share = len(cdata) / len(data) if len(data) else 0
    kpi_cards(cdata, f"{share:.1%} of the filtered base")
    window = ctx["window"]
    sig = ctx["sig"]

    def month_fig(rows, _):
        month = monthly_churn(rows, window)
        return monthly_chart(month) if not month.empty else "No dated churn in this period."

    with st.container(key="grid_cd"):
        map_card(data, circle, sig)
        chart_card("cd_gauge", "Churn gauge", circle, "gauge", cdata, [],
                   lambda d, _: gauge_chart(d), sig)
        chart_card("cd_bubble", "Circle comparison: size vs churn", "Bubble area: churned subscribers", "bubbles",
                   data, ["complaint_status"] + local_cols(ctx, "connection_type"),
                   lambda d, _: bubble_chart(d, circle), sig)
        chart_card("cd_month", "Monthly churned subscribers", "Counts by churn date", "trend",
                   cdata, local_cols(ctx, "connection_type") + ["complaint_status"], month_fig, sig)
        chart_card("cd_conn", "Connection type analysis", "Stem: gap from 23.75% reference", "signal",
                   cdata, ["complaint_status"] + local_cols(ctx, "tenure_band"),
                   lambda d, _: connection_chart(d), sig)
    strip_html(detail_insight(data, cdata, circle))
    return f"Circle: {circle}"


# ---------------------------------------------------------------- main
def main():
    st.html(CSS)
    db_path, searched = find_database()
    if db_path is None:
        st.error("Airtel database not found. No file was created. Looked in:\n\n"
                 + "\n".join(f"- `{p}`" for p in searched)
                 + "\n\nPlace airtel_dashboard_slim.db in data/raw/ or set AIRTEL_DB_PATH.")
        st.stop()
    try:
        df = load_data(str(db_path), db_path.stat().st_mtime_ns)
    except (sqlite3.Error, pd.errors.DatabaseError, ValueError) as error:
        st.error(f"Could not read `{db_path}`: {error}")
        st.stop()
    if df.empty:
        st.error("customer_features has no rows.")
        st.stop()
    st.session_state.db_path = str(db_path)
    stamp = datetime.fromtimestamp(db_path.stat().st_mtime).strftime("%d %b %Y, %H:%M")

    view, asked = render_header(df, logo_html(), stamp)
    with st.container(key="body"):
        ctx, data = render_panel(df, view)
        with st.container(key="content"):
            context = render_overview(data, df, ctx) if view == OVERVIEW else render_detail(data, df, ctx)

    repeats = len(df) - df.customer_id.nunique()
    warning = f" · <b>Warning: {repeats:,} repeated customer_id rows</b>" if repeats else ""
    markup(f'<div class="foot"><b>23.75%</b> is the supplied national churn reference, not a target · '
           f"{html.escape(context)} · Chart filters affect only their own chart · Snapshot churn status{warning}</div>")
    with st.container(key="fx"):
        st.html(JS, unsafe_allow_javascript=True)
    if asked:
        chat_dialog(ctx["connection"], ctx["region"], ctx["circle"] or "All")


main()
