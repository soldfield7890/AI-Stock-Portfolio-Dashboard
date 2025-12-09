# app.py

import os
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.loaders import load_csv
from src.scoring import score_portfolio
from src.fundamentals import get_fundamentals_for_symbols
from src.ai_research import run_mini_trading_desk, _get_client


# ---------------------------
#  STREAMLIT BASE CONFIG
# ---------------------------

st.set_page_config(
    page_title="Oldfield AI Stock Dashboard",
    page_icon="💹",
    layout="wide",
)

# ---------------------------
#  GLOBAL STYLING (GLASS / MODERN)
# ---------------------------

st.markdown(
    """
<style>
/* Overall background */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top left, #e0f2fe 0, #f9fafb 40%, #e5e7eb 100%);
    color: #0f172a;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
}

/* Sidebar (if used) */
[data-testid="stSidebar"] {
    background: #020617;
}

/* Make default text darker / readable */
html, body, [class*="css"] {
    color: #020617 !important;
}

/* Top title block */
.hero-title {
    font-size: 32px;
    font-weight: 700;
    letter-spacing: 0.03em;
    color: #020617;
}

.hero-subtitle {
    font-size: 14px;
    color: #4b5563;
    margin-top: 2px;
}

/* Soft glass card */
.soft-card {
    background: rgba(255,255,255,0.92);
    border-radius: 18px;
    padding: 18px 22px;
    box-shadow:
        0 18px 35px rgba(15,23,42,0.08),
        0 0 0 0.5px rgba(148,163,184,0.35);
    backdrop-filter: blur(20px);
}

/* Metric tiles */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
}

.metric-tile {
    background: rgba(248,250,252,0.98);
    border-radius: 16px;
    padding: 14px 16px 12px 16px;
    border: 1px solid rgba(148,163,184,0.4);
}

.metric-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6b7280;
}

.metric-value-main {
    font-size: 24px;
    font-weight: 650;
    margin-top: 2px;
    color: #020617;
}

.metric-subtext {
    font-size: 11px;
    color: #9ca3af;
}

.metric-positive {
    color: #16a34a;
    font-weight: 600;
}

.metric-negative {
    color: #dc2626;
    font-weight: 600;
}

/* Section headers */
.section-title {
    font-size: 18px;
    font-weight: 650;
    margin-bottom: 4px;
    color: #020617;
}

.section-subtitle {
    font-size: 12px;
    color: #6b7280;
}

/* Tabs */
[data-testid="stTabs"] button {
    border-radius: 999px !important;
    padding: 6px 18px;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    background: linear-gradient(90deg,#2563eb,#38bdf8);
    color: #f9fafb !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border-radius: 16px !important;
    overflow: hidden !important;
    box-shadow: 0 12px 30px rgba(15,23,42,0.08);
}

/* File uploader */
[data-testid="stFileUploader"] {
    border-radius: 16px;
}

[data-testid="stFileUploader"] section {
    background: #020617;
    border-radius: 14px;
    color: #f9fafb;
}

/* Buttons */
.stButton>button {
    border-radius: 999px;
    padding: 6px 20px;
    border: none;
    background: linear-gradient(135deg,#1d4ed8,#38bdf8);
    color: white;
    font-weight: 500;
    box-shadow: 0 10px 25px rgba(37,99,235,0.35);
}

.stButton>button:hover {
    background: linear-gradient(135deg,#1e3a8a,#0ea5e9);
}

/* AI Desk card text */
.ai-pill {
    display: inline-flex;
    align-items: center;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 11px;
    background: rgba(15,23,42,0.04);
    color: #4b5563;
}

/* Watchlist badge */
.badge-watchlist-yes {
    background: rgba(22,163,74,0.12);
    color: #15803d;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 11px;
    font-weight: 600;
}

.badge-watchlist-no {
    background: rgba(248,250,252,0.9);
    color: #6b7280;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 11px;
}

/* Health text colors */
.health-good { color: #16a34a; }
.health-ok { color: #f59e0b; }
.health-bad { color: #dc2626; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------
#  HELPER FUNCTIONS
# ---------------------------


def format_currency(v: float) -> str:
    if pd.isna(v):
        return "—"
    sign = "-" if v < 0 else ""
    v_abs = abs(v)
    if v_abs >= 1_000_000_000:
        return f"{sign}${v_abs/1_000_000_000:,.1f}B"
    if v_abs >= 1_000_000:
        return f"{sign}${v_abs/1_000_000:,.1f}M"
    return f"{sign}${v_abs:,.0f}"


def format_pct(v: float) -> str:
    if pd.isna(v):
        return "—"
    return f"{v*100:,.1f}%"


def ensure_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------------------
#  OVERVIEW TAB
# ---------------------------


def build_health_gauge(avg_score: float) -> go.Figure:
    # Portfolio health: 0–100, segments <40 red, 40–60 yellow, >60 green.
    fig = go.Figure()

    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=avg_score,
            number={"suffix": " / 100", "font": {"size": 20}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1d4ed8", "thickness": 0.35},
                "bgcolor": "rgba(15,23,42,0.03)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(248,113,113,0.55)"},
                    {"range": [40, 60], "color": "rgba(234,179,8,0.55)"},
                    {"range": [60, 100], "color": "rgba(34,197,94,0.55)"},
                ],
            },
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Portfolio Health", "font": {"size": 14}},
        )
    )

    fig.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_overview(scored_df: Optional[pd.DataFrame]):
    if scored_df is None or scored_df.empty:
        st.info("Upload a positions CSV on the left to see your portfolio cockpit.")
        return

    # Make sure numeric fields are numeric
    scored_df = ensure_numeric(
        scored_df,
        [
            "CurrentValue",
            "UnrealizedPL",
            "TodayPLDollar",
            "TodayPLPct",
            "Score",
        ],
    )

    total_value = scored_df["CurrentValue"].sum() if "CurrentValue" in scored_df else np.nan
    total_unreal = scored_df["UnrealizedPL"].sum() if "UnrealizedPL" in scored_df else np.nan
    positions = len(scored_df)

    avg_score = scored_df["Score"].mean() if "Score" in scored_df else np.nan
    if "Score" in scored_df and "CurrentValue" in scored_df and total_value > 0:
        value_weighted_score = np.average(
            scored_df["Score"].fillna(avg_score),
            weights=scored_df["CurrentValue"].fillna(0),
        )
    else:
        value_weighted_score = np.nan

    # Largest position
    largest_symbol = "—"
    largest_weight = np.nan
    if "CurrentValue" in scored_df and "Symbol" in scored_df and total_value > 0:
        idx = scored_df["CurrentValue"].idxmax()
        row = scored_df.loc[idx]
        largest_symbol = str(row.get("Symbol", "—"))
        largest_weight = row["CurrentValue"] / total_value

    # Score buckets
    speculative = middle = core = 0
    if "Score" in scored_df:
        speculative = (scored_df["Score"] < 40).sum()
        middle = ((scored_df["Score"] >= 40) & (scored_df["Score"] < 60)).sum()
        core = (scored_df["Score"] >= 60).sum()

    # Decision mix
    decisions = scored_df.get("Decision", pd.Series(dtype=str)).fillna("Unknown")
    decision_counts = decisions.value_counts()
    strong_buy_n = decision_counts.get("Strong Buy", 0)
    buy_n = decision_counts.get("Buy", 0)
    hold_n = decision_counts.get("Hold", 0)
    trim_n = decision_counts.get("Trim", 0)
    exit_n = decision_counts.get("Exit / Avoid", 0)

    # Winners / losers for "Today's Focus"
    winners = []
    losers = []
    if "UnrealizedPL" in scored_df and "Symbol" in scored_df:
        tmp = scored_df[["Symbol", "UnrealizedPL"]].dropna()
        winners = tmp.nlargest(3, "UnrealizedPL").to_dict("records")
        losers = tmp.nsmallest(3, "UnrealizedPL").to_dict("records")

    # ---------------- KPIs + HEALTH GAUGE ----------------
    top_col_left, top_col_right = st.columns([1.8, 1])

    with top_col_left:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)

        st.markdown(
            '<div class="section-title">Portfolio Snapshot</div>'
            '<div class="section-subtitle">High-level read on value, risk, and concentration.</div>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="metric-grid">', unsafe_allow_html=True)

            # Tile 1 – Total value
            st.markdown(
                f"""
                <div class="metric-tile">
                  <div class="metric-label">Total portfolio value</div>
                  <div class="metric-value-main">{format_currency(total_value)}</div>
                  <div class="metric-subtext">Sum of all positions</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Tile 2 – Unrealized P/L
            unreal_class = "metric-positive" if total_unreal >= 0 else "metric-negative"
            st.markdown(
                f"""
                <div class="metric-tile">
                  <div class="metric-label">Total unrealized P/L</div>
                  <div class="metric-value-main {unreal_class}">{format_currency(total_unreal)}</div>
                  <div class="metric-subtext">Across all holdings</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Tile 3 – Positions
            st.markdown(
                f"""
                <div class="metric-tile">
                  <div class="metric-label">Positions</div>
                  <div class="metric-value-main">{positions}</div>
                  <div class="metric-subtext">Unique tickers</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="metric-grid">', unsafe_allow_html=True)

            # Tile 4 – Average score
            st.markdown(
                f"""
                <div class="metric-tile">
                  <div class="metric-label">Average score</div>
                  <div class="metric-value-main">{avg_score:0.1f}</div>
                  <div class="metric-subtext">Simple average of scoring model</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Tile 5 – Value-weighted score
            st.markdown(
                f"""
                <div class="metric-tile">
                  <div class="metric-label">Value-weighted score</div>
                  <div class="metric-value-main">{value_weighted_score:0.1f}</div>
                  <div class="metric-subtext">Score weighted by position size</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Tile 6 – Largest position
            lw_text = f"{largest_symbol} • {largest_weight*100:0.1f}%" if not pd.isna(largest_weight) else largest_symbol
            st.markdown(
                f"""
                <div class="metric-tile">
                  <div class="metric-label">Largest position</div>
                  <div class="metric-value-main">{largest_symbol}</div>
                  <div class="metric-subtext">{lw_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with top_col_right:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        health_class = "health-good" if avg_score >= 60 else "health-ok" if avg_score >= 40 else "health-bad"
        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
              <div class="section-title" style="margin-bottom:0;">Health Gauge</div>
              <div class="section-subtitle {health_class}">
                {"Healthy core" if avg_score >= 60 else "Mixed risk" if avg_score >= 40 else "Speculative tilt"}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig_gauge = build_health_gauge(float(avg_score if not pd.isna(avg_score) else 0.0))
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")  # spacing

    # ---------------- SCORE MIX + ACTION SUMMARY STRIP ----------------
    strip_col1, strip_col2 = st.columns([1.2, 1.3])

    with strip_col1:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">Score Mix</div>'
            '<div class="section-subtitle">How your positions stack up by conviction bucket.</div>',
            unsafe_allow_html=True,
        )

        score_mix_df = pd.DataFrame(
            {
                "Bucket": ["Speculative (<40)", "Middle (40–59)", "Core (60+)"],
                "Count": [speculative, middle, core],
            }
        )
        fig_mix = px.bar(
            score_mix_df,
            x="Bucket",
            y="Count",
            text="Count",
        )
        fig_mix.update_traces(
            marker_color="#60a5fa",
            marker_line_color="#1d4ed8",
            marker_line_width=1,
            textposition="outside",
        )
        fig_mix.update_layout(
            height=260,
            margin=dict(l=0, r=0, t=10, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0)",
            xaxis_title=None,
            yaxis_title=None,
        )
        st.plotly_chart(fig_mix, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with strip_col2:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">Action Summary</div>'
            '<div class="section-subtitle">Headcount by decision – what your model is telling you.</div>',
            unsafe_allow_html=True,
        )

        c_a1, c_a2, c_a3, c_a4, c_a5 = st.columns(5)
        tiles = [
            ("Strong Buy", strong_buy_n, "#16a34a"),
            ("Buy", buy_n, "#22c55e"),
            ("Hold", hold_n, "#64748b"),
            ("Trim", trim_n, "#f97316"),
            ("Exit / Avoid", exit_n, "#dc2626"),
        ]
        for col, (label, count, color) in zip(
            [c_a1, c_a2, c_a3, c_a4, c_a5], tiles
        ):
            with col:
                st.markdown(
                    f"""
                    <div class="metric-tile" style="padding:10px 12px;">
                      <div class="metric-label" style="color:{color};">{label}</div>
                      <div class="metric-value-main" style="font-size:20px;">{count}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")  # spacing

    # ---------------- UNREALIZED P/L + TODAY'S FOCUS ----------------
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Unrealized P/L &amp; Today’s Focus</div>'
        '<div class="section-subtitle">Where your money is working for you – and where risk is building.</div>',
        unsafe_allow_html=True,
    )

    left_bar, right_focus = st.columns([2.1, 1])

    with left_bar:
        if "UnrealizedPL" in scored_df and "Symbol" in scored_df:
            bar_df = scored_df.copy()
            bar_df["UnrealizedPL"] = pd.to_numeric(bar_df["UnrealizedPL"], errors="coerce")
            bar_df = bar_df.sort_values("UnrealizedPL", ascending=False)
            fig_bar = px.bar(
                bar_df,
                x="Symbol",
                y="UnrealizedPL",
                labels={"UnrealizedPL": "Unrealized P/L ($)"},
            )
            colors = ["#22c55e" if v >= 0 else "#f97316" for v in bar_df["UnrealizedPL"].fillna(0)]
            fig_bar.update_traces(
                marker_color=colors,
                marker_line_color="#0f172a",
                marker_line_width=0.4,
            )
            fig_bar.update_layout(
                height=320,
                margin=dict(l=0, r=0, t=15, b=40),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0)",
                xaxis_title=None,
                yaxis_title=None,
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Unrealized P/L data not available in this file.")

    with right_focus:
        st.markdown("#### Today’s Focus")
        st.markdown("**Top Winners**", unsafe_allow_html=True)
        if winners:
            for w in winners:
                st.markdown(
                    f"- **{w['Symbol']}** up {format_currency(w['UnrealizedPL'])} unrealized",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("- None yet.")

        st.markdown("**Top Losers**")
        if losers:
            for l in losers:
                st.markdown(
                    f"- **{l['Symbol']}** down {format_currency(l['UnrealizedPL'])} unrealized",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("- None yet.")

        # Quick moves based on decisions
        st.markdown("**Immediate Moves Radar**")
        radar_lines = []

        if "Decision" in scored_df and "Symbol" in scored_df and "CurrentValue" in scored_df:
            tmp = scored_df.copy()
            tmp["WeightPct"] = tmp["CurrentValue"] / total_value * 100 if total_value > 0 else np.nan

            trim_df = tmp[tmp["Decision"] == "Trim"].sort_values("WeightPct", ascending=False).head(3)
            for _, r in trim_df.iterrows():
                radar_lines.append(
                    f"Trim **{r['Symbol']}** (≈{r['WeightPct']:0.1f}% of portfolio)."
                )

            exit_df = tmp[tmp["Decision"].str.contains("Exit", na=False)].sort_values(
                "WeightPct", ascending=False
            ).head(3)
            for _, r in exit_df.iterrows():
                radar_lines.append(
                    f"Re-evaluate / exit **{r['Symbol']}** (≈{r['WeightPct']:0.1f}%)."
                )

            buy_df = tmp[tmp["Decision"].str.contains("Strong Buy", na=False)].sort_values(
                "WeightPct", ascending=False
            ).head(3)
            for _, r in buy_df.iterrows():
                radar_lines.append(
                    f"Consider adding on dips to **{r['Symbol']}** (score {r['Score']:0.0f})."
                )

        if radar_lines:
            for line in radar_lines:
                st.markdown(f"- {line}")
        else:
            st.markdown("- No clear actions surfaced by the model yet.")

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------
#  POSITIONS TAB
# ---------------------------


def render_positions(scored_df: Optional[pd.DataFrame]):
    if scored_df is None or scored_df.empty:
        st.info("Upload and score a portfolio to see positions.")
        return

    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Positions &amp; Metrics</div>'
        '<div class="section-subtitle">Sortable table of all holdings with scores and P/L.</div>',
        unsafe_allow_html=True,
    )

    filter_decision = st.selectbox(
        "Filter by decision",
        options=["All"] + sorted(scored_df.get("Decision", pd.Series(dtype=str)).dropna().unique().tolist()),
    )

    sort_col = st.selectbox(
        "Sort by column",
        options=[c for c in scored_df.columns if scored_df[c].dtype != "object"]
        + [c for c in scored_df.columns if scored_df[c].dtype == "object"],
        index=0 if "Score" not in scored_df.columns else list(scored_df.columns).index("Score"),
    )

    sort_asc = st.checkbox("Sort ascending?", value=False)

    df_view = scored_df.copy()
    if filter_decision != "All" and "Decision" in df_view.columns:
        df_view = df_view[df_view["Decision"] == filter_decision]

    df_view = df_view.sort_values(sort_col, ascending=sort_asc, na_position="last")

    st.dataframe(df_view, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------
#  FUNDAMENTALS TAB
# ---------------------------


def render_fundamentals(scored_df: Optional[pd.DataFrame]):
    if scored_df is None or scored_df.empty:
        st.info("Upload and score a portfolio to see fundamentals.")
        return

    symbols = scored_df["Symbol"].dropna().unique().tolist()
    with st.spinner("Pulling fundamentals from Yahoo Finance..."):
        fundamentals_df = get_fundamentals_for_symbols(symbols)

    if fundamentals_df is None or fundamentals_df.empty:
        st.warning("No fundamentals could be retrieved.")
        return

    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Fundamentals Snapshot</div>'
        '<div class="section-subtitle">Valuation, profitability, risk, and size by ticker.</div>',
        unsafe_allow_html=True,
    )

    numeric_cols = [c for c in fundamentals_df.columns if fundamentals_df[c].dtype != "object"]
    sort_col = st.selectbox("Sort by", options=numeric_cols if numeric_cols else fundamentals_df.columns)
    sort_asc = st.checkbox("Sort ascending?", value=False)

    fundamentals_df = fundamentals_df.sort_values(sort_col, ascending=sort_asc, na_position="last")
    st.dataframe(fundamentals_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------
#  SIGNALS TAB
# ---------------------------


def render_signals(scored_df: Optional[pd.DataFrame]):
    if scored_df is None or scored_df.empty:
        st.info("Upload and score a portfolio to see signals.")
        return

    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Signals &amp; Action List</div>'
        '<div class="section-subtitle">Grouped list of Strong Buys, Trims, and Exits.</div>',
        unsafe_allow_html=True,
    )

    df = scored_df.copy()
    if "PortfolioWeightPct" not in df.columns and "CurrentValue" in df.columns:
        tv = df["CurrentValue"].sum()
        if tv > 0:
            df["PortfolioWeightPct"] = df["CurrentValue"] / tv * 100

    strong_buy = df[df["Decision"] == "Strong Buy"]
    buy = df[df["Decision"] == "Buy"]
    trim = df[df["Decision"] == "Trim"]
    exit_avoid = df[df["Decision"].str.contains("Exit", na=False)]

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Buy / Add Radar")
        st.markdown("**Strong Buy**")
        if strong_buy.empty:
            st.markdown("_None flagged right now._")
        else:
            st.dataframe(
                strong_buy[
                    ["Symbol", "Score", "PortfolioWeightPct", "UnrealizedPL", "PE_TTM"]
                    if "PE_TTM" in strong_buy.columns
                    else ["Symbol", "Score", "PortfolioWeightPct", "UnrealizedPL"]
                ],
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("**Buy**")
        if buy.empty:
            st.markdown("_None flagged right now._")
        else:
            st.dataframe(
                buy[
                    ["Symbol", "Score", "PortfolioWeightPct", "UnrealizedPL", "PE_TTM"]
                    if "PE_TTM" in buy.columns
                    else ["Symbol", "Score", "PortfolioWeightPct", "UnrealizedPL"]
                ],
                use_container_width=True,
                hide_index=True,
            )

    with col_right:
        st.markdown("### De-Risk Radar")
        st.markdown("**Trim**")
        if trim.empty:
            st.markdown("_None flagged right now._")
        else:
            st.dataframe(
                trim[
                    ["Symbol", "Score", "PortfolioWeightPct", "UnrealizedPL", "PE_TTM"]
                    if "PE_TTM" in trim.columns
                    else ["Symbol", "Score", "PortfolioWeightPct", "UnrealizedPL"]
                ],
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("**Exit / Avoid**")
        if exit_avoid.empty:
            st.markdown("_None flagged right now._")
        else:
            st.dataframe(
                exit_avoid[
                    ["Symbol", "Score", "PortfolioWeightPct", "UnrealizedPL", "PE_TTM"]
                    if "PE_TTM" in exit_avoid.columns
                    else ["Symbol", "Score", "PortfolioWeightPct", "UnrealizedPL"]
                ],
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------
#  AI DESK – QUICK GLANCE
# ---------------------------


def render_ai_desk(scored_df: Optional[pd.DataFrame]):
    if scored_df is None or scored_df.empty:
        st.info("Upload and score a portfolio to use the AI Desk.")
        return

    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">AI Desk – Quick Glance</div>'
        '<div class="section-subtitle">Select a ticker to get an AI-generated snapshot based on fundamentals, trends, and risk.</div>',
        unsafe_allow_html=True,
    )

    symbols = scored_df["Symbol"].dropna().unique().tolist()
    selected = st.selectbox("Select ticker for AI check", options=sorted(symbols))

    depth = st.radio("Depth", options=["lite", "full"], horizontal=True, index=0)
    run = st.button("Run Mini Trading Desk")

    client = _get_client()
    if client is None:
        api_status = "offline"
    else:
        api_status = "online"

    if not run:
        status_text = (
            "AI Desk online – ready to analyze."
            if api_status == "online"
            else "AI Desk offline – add `OPENAI_API_KEY` in secrets to enable."
        )
        st.markdown(f'<span class="ai-pill">{status_text}</span>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Locate row for selected ticker
    row = scored_df[scored_df["Symbol"] == selected].iloc[0]

    # If no API key, use stub output (handled inside run_mini_trading_desk per our earlier patch)
    with st.spinner("Running Mini Trading Desk view..."):
        view: Dict[str, Any] = run_mini_trading_desk(
            ticker=selected,
            row=row,
            mode=depth,
        )

    # --- Layout of AI card
    header_row1, header_row2 = st.columns([2.2, 1])

    title = f"{view.get('ticker', selected)} – {view.get('final_decision', 'View')}"
    with header_row1:
        st.markdown(f"### {title}")
        st.markdown(
            f"Conviction: **{view.get('conviction_score', '—')}** · "
            f"Horizon: **{view.get('time_horizon', 'n/a')}** · "
            f"Bucket: **{view.get('bucket_view', '—')}**"
        )

    with header_row2:
        prim = view.get("primary_action", "Review position")
        st.markdown(
            f'<div style="text-align:right;"><span class="ai-pill">Primary action: {prim}</span></div>',
            unsafe_allow_html=True,
        )

    # Three columns: fundamentals, trend/technicals, sentiment/story
    c1, c2, c3 = st.columns(3)

    def _render_bullets(lines, title: str, col_handle):
        with col_handle:
            st.markdown(f"**{title}**")
            if not lines:
                st.markdown("_No notes._")
                return
            if isinstance(lines, str):
                lines = [lines]
            for line in lines:
                # Strip any raw HTML tags the model might add
                safe = str(line).replace("<ul>", "").replace("</ul>", "").replace("<li>", "• ").replace("</li>", "")
                st.markdown(f"- {safe}")

    _render_bullets(view.get("fundamental_view", []), "Fundamentals", c1)
    _render_bullets(view.get("technical_view", []) or view.get("trend_view", []), "Trend & Technics", c2)
    _render_bullets(view.get("sentiment_view", []), "Sentiment & Story", c3)

    # Key risks + next actions
    st.markdown("---")
    col_risks, col_actions, col_meta = st.columns([1.6, 1.6, 1])

    _render_bullets(view.get("risk_factors", []), "Key Risks", col_risks)

    with col_actions:
        st.markdown("**Next Actions**")
        next_actions = view.get("next_actions", {})
        if isinstance(next_actions, dict) and next_actions:
            for label, detail in next_actions.items():
                st.markdown(f"- **{label}:** {detail}")
        else:
            st.markdown("_No structured next actions returned._")

    with col_meta:
        st.markdown("**Meta**")
        st.markdown(
            "AI view generated by your Mini Trading Desk engine (multi-role prompt).",
        )
        wl = view.get("watchlist", {})
        if isinstance(wl, dict) and wl.get("add_to_watchlist"):
            st.markdown('<span class="badge-watchlist-yes">Watchlist: Add / Keep</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-watchlist-no">Watchlist: No change</span>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------
#  MAIN APP
# ---------------------------


def main():
    # Header row with logo and version chip
    top_l, top_mid, top_r = st.columns([3, 4, 1])

    with top_l:
        st.markdown('<div class="hero-title">Oldfield AI Stock Dashboard</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="hero-subtitle">Glass-style cockpit for your AI-scored portfolio — positions, fundamentals, and actions in one view.</div>',
            unsafe_allow_html=True,
        )

    with top_r:
        # Logo in the top-right (use your clover logo file in the repo root as "logo.png")
        logo_path = Path("logo.png")
        if logo_path.exists():
            st.image(str(logo_path), width=70)
        st.markdown(
            '<div style="text-align:right;"><span class="ai-pill">v1.2 · Experimental</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")  # spacing

    # File upload
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    upload_col, helper_col = st.columns([2.4, 3])

    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload portfolio CSV",
            type=["csv"],
            help="Use the same column layout you’ve been using for positions.",
        )

    with helper_col:
        st.markdown(
            "Keep a single positions CSV in your drive, overwrite it after trades, then reload this app to re-score the whole portfolio in one click.",
        )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("")  # spacing

    scored_df: Optional[pd.DataFrame] = None
    if uploaded_file is not None:
        try:
            raw_df = load_csv(uploaded_file)
            scored_df = score_portfolio(raw_df)
        except Exception as e:
            st.error(f"Error loading or scoring portfolio: {e}")
            scored_df = None

    # Tabs
    tab_overview, tab_positions, tab_fundamentals, tab_signals, tab_ai = st.tabs(
        ["Overview", "Positions", "Fundamentals", "Signals", "AI Desk"]
    )

    with tab_overview:
        render_overview(scored_df)

    with tab_positions:
        render_positions(scored_df)

    with tab_fundamentals:
        render_fundamentals(scored_df)

    with tab_signals:
        render_signals(scored_df)

    with tab_ai:
        render_ai_desk(scored_df)


if __name__ == "__main__":
    main()
