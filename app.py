# app.py

import os
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# Optional imports from your src/ package (with fallbacks)
# ---------------------------------------------------------
try:
    from src.scoring import score_portfolio  # your custom scoring model
except Exception:
    def score_portfolio(df: pd.DataFrame) -> pd.DataFrame:
        """Fallback: no scoring, just return df."""
        return df

try:
    from src.loaders import load_csv  # your custom loader
except Exception:
    def load_csv(file) -> pd.DataFrame:
        return pd.read_csv(file)

try:
    from src.ai_research import run_mini_trading_desk  # mini trading desk engine
except Exception:
    def run_mini_trading_desk(ticker: str, row: pd.Series, mode: str = "lite") -> Dict[str, Any]:
        """Fallback AI desk if src.ai_research is missing."""
        return {
            "ticker": ticker,
            "final_decision": "NO_ENGINE",
            "conviction_score": None,
            "time_horizon": None,
            "bucket_view": None,
            "fundamental_view": ["AI engine not available."],
            "technical_view": [],
            "sentiment_view": [],
            "risk_factors": [],
            "primary_action": "Watch Only",
            "next_actions": {},
            "watchlist": {
                "add_to_watchlist": False,
                "watchlist_bucket": "None",
                "notes": "",
            },
            "_raw_text": "",
        }


# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Oldfield AI Stock Dashboard",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------
# Global styling (glass / liquid look)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global background */
    .stApp {
        background: radial-gradient(circle at top left, #e0f2fe 0, #eff6ff 32%, #f9fafb 70%, #e5e7eb 100%);
        color: #0F172A;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1400px;
    }

    .glass-card {
        background: rgba(255,255,255,0.92);
        border-radius: 24px;
        box-shadow: 0 18px 45px rgba(15,23,42,0.15);
        border: 1px solid rgba(148,163,184,0.30);
        backdrop-filter: blur(18px);
    }

    /* File uploader tweak */
    .uploadedFile { color: #0F172A !important; }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-size: 0.9rem;
        padding-top: 8px;
        padding-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------
def fmt_currency(v: Optional[float], decimals: int = 0) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    try:
        return f"${v:,.{decimals}f}"
    except Exception:
        return str(v)


def fmt_pct(v: Optional[float], decimals: int = 1) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    try:
        return f"{v:.{decimals}f}%"
    except Exception:
        return str(v)


def detect_symbol_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["Symbol", "Ticker", "symbol", "ticker", "SYM"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def get_value_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["CurrentValue_clean", "CurrentValue", "Current Value", "MarketValue", "Market Value"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def compute_overview_analytics(df: pd.DataFrame) -> Dict[str, Any]:
    symbol_col = detect_symbol_column(df)
    value_col = get_value_column(df)

    metrics: Dict[str, Any] = {
        "total_value": None,
        "total_pl": None,
        "avg_score": None,
        "weighted_score": None,
        "score_buckets": {},
        "top_winners": pd.DataFrame(),
        "top_losers": pd.DataFrame(),
    }

    if value_col and value_col in df.columns:
        try:
            metrics["total_value"] = float(df[value_col].fillna(0).sum())
        except Exception:
            metrics["total_value"] = None

    if "UnrealizedPL" in df.columns:
        try:
            metrics["total_pl"] = float(df["UnrealizedPL"].fillna(0).sum())
        except Exception:
            metrics["total_pl"] = None

    if "Score" in df.columns:
        try:
            metrics["avg_score"] = float(df["Score"].dropna().mean())
        except Exception:
            metrics["avg_score"] = None

        # Weighted score by value
        if value_col and value_col in df.columns:
            try:
                tmp = df[["Score", value_col]].dropna()
                w = tmp[value_col].astype(float)
                s = tmp["Score"].astype(float)
                if w.sum() > 0:
                    metrics["weighted_score"] = float((s * w).sum() / w.sum())
            except Exception:
                metrics["weighted_score"] = None

        # Score buckets
        buckets = {"Speculative (<40)": 0, "Middle (40–59)": 0, "Core (60+)": 0}
        for val in df["Score"].dropna():
            try:
                v = float(val)
                if v < 40:
                    buckets["Speculative (<40)"] += 1
                elif v < 60:
                    buckets["Middle (40–59)"] += 1
                else:
                    buckets["Core (60+)"] += 1
            except Exception:
                continue
        metrics["score_buckets"] = buckets

    # Winners / Losers
    if symbol_col and "UnrealizedPL" in df.columns:
        tmp = df[[symbol_col, "UnrealizedPL"]].copy()
        tmp = tmp.dropna(subset=["UnrealizedPL"])
        if not tmp.empty:
            metrics["top_winners"] = tmp.sort_values("UnrealizedPL", ascending=False).head(3)
            metrics["top_losers"] = tmp.sort_values("UnrealizedPL", ascending=True).head(3)

    return metrics


# ---------------------------------------------------------
# AI Desk Card Renderer (new, polished)
# ---------------------------------------------------------
def render_ai_glance_card(ai_view: Dict[str, Any]):
    """Pretty, color-coded AI Trading Desk card."""

    if not ai_view:
        st.info("Run the Mini Trading Desk to see AI output.")
        return

    # ---- Unpack data safely ----
    ticker = ai_view.get("ticker", "—")
    decision_raw = ai_view.get("final_decision") or "No decision"
    decision_clean = decision_raw.replace("_", " ").title()
    conviction = ai_view.get("conviction_score")
    horizon = ai_view.get("time_horizon") or "n/a"
    bucket = ai_view.get("bucket_view") or "Unknown"
    primary_action = ai_view.get("primary_action") or "—"

    fundamentals = ai_view.get("fundamental_view") or []
    technicals = ai_view.get("technical_view") or []
    sentiment = ai_view.get("sentiment_view") or []
    risks = ai_view.get("risk_factors") or []

    wl = ai_view.get("watchlist") or {}
    wl_flag = bool(wl.get("add_to_watchlist", False))
    wl_bucket = wl.get("watchlist_bucket", "None")
    wl_notes = wl.get("notes", "")

    next_actions = ai_view.get("next_actions") or {}
    add_on_dip = next_actions.get("add_on_dip_level")
    trim_above = next_actions.get("trim_above_level")
    hard_exit = next_actions.get("hard_exit_level")
    sizing_note = next_actions.get("position_sizing_note")

    # ---- Decision chip color map ----
    decision_palette = {
        "Strong Buy": ("#16a34a", "rgba(22,163,74,0.14)"),
        "Buy": ("#10b981", "rgba(16,185,129,0.14)"),
        "Hold": ("#6366f1", "rgba(99,102,241,0.14)"),
        "Trim": ("#facc15", "rgba(250,204,21,0.20)"),
        "Exit": ("#ef4444", "rgba(239,68,68,0.20)"),
        "Exit / Avoid": ("#ef4444", "rgba(239,68,68,0.20)"),
        "No_Api_Key": ("#6b7280", "rgba(148,163,184,0.30)"),
        "No Decision": ("#6b7280", "rgba(148,163,184,0.30)"),
    }

    if decision_raw.upper() == "NO_API_KEY":
        palette_key = "No_Api_Key"
    else:
        palette_key = decision_clean

    decision_color, decision_bg = decision_palette.get(
        palette_key, ("#0f172a", "rgba(15,23,42,0.06)")
    )

    # ---- Conviction badge ----
    conviction_label = "—"
    conviction_badge_html = ""

    if conviction is not None:
        conviction_label = f"{conviction:.0f}/100"
        if conviction >= 75:
            cv_color, cv_bg = "#16a34a", "rgba(22,163,74,0.14)"
        elif conviction >= 50:
            cv_color, cv_bg = "#eab308", "rgba(234,179,8,0.18)"
        else:
            cv_color, cv_bg = "#ef4444", "rgba(239,68,68,0.20)"

        conviction_badge_html = f"""
            <span style="
                border-radius:999px;
                padding:3px 10px;
                font-size:0.75rem;
                background:{cv_bg};
                color:{cv_color};
                margin-left:8px;
            ">
                Conviction {conviction_label}
            </span>
        """

    # ---- Card wrapper ----
    st.markdown(
        '<div class="glass-card" style="padding:18px 22px; margin-top:10px;">',
        unsafe_allow_html=True,
    )

    # =======================
    #   TOP: Header row
    # =======================
    top_l, top_r = st.columns([3, 1])

    with top_l:
        st.markdown(
            f"""
            <div style="display:flex; flex-direction:column; gap:4px;">
              <div style="font-size:0.9rem; color:#6B7280;">AI Trading Desk View</div>
              <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
                <div style="font-size:1.1rem; font-weight:600; color:#0F172A;">
                  {ticker} – {decision_clean}
                </div>
                <span style="
                    border-radius:999px;
                    padding:4px 10px;
                    font-size:0.75rem;
                    background:{decision_bg};
                    color:{decision_color};
                    ">
                  {decision_clean}
                </span>
                {conviction_badge_html}
              </div>
              <div style="font-size:0.85rem; color:#6B7280;">
                Conviction: {conviction_label}
                · Horizon: {horizon}
                · Bucket: {bucket}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with top_r:
        st.markdown(
            f"""
            <div style="display:flex; justify-content:flex-end;">
              <span style="
                  border-radius:999px;
                  padding:4px 12px;
                  font-size:0.75rem;
                  background:rgba(15,23,42,0.04);
                  color:#4B5563;">
                Primary action:
                <span style="font-weight:600;">{primary_action}</span>
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =======================
    #   MID: 3 analysis columns
    # =======================
    def render_bullets(title: str, items: list[str]):
        st.markdown(f"**{title}**")
        if items:
            for txt in items:
                st.markdown(f"- {txt}")
        else:
            st.caption("No insight yet.")

    mid1, mid2, mid3 = st.columns(3)
    with mid1:
        render_bullets("Fundamentals", fundamentals)
    with mid2:
        render_bullets("Trend & Technicals", technicals)
    with mid3:
        render_bullets("Sentiment & Story", sentiment)

    st.markdown(
        "<hr style='border:none; border-top:1px solid rgba(148,163,184,0.35); "
        "margin:12px 0 16px;'>",
        unsafe_allow_html=True,
    )

    # =======================
    #   BOTTOM: Risks + Watchlist
    # =======================
    bottom_l, bottom_r = st.columns([2, 1])

    with bottom_l:
        st.markdown("**Key Risks**")
        if risks:
            for r in risks:
                st.markdown(f"- {r}")
        else:
            st.caption("No specific risks highlighted yet.")

        st.markdown("**Next Actions**")
        st.markdown(f"- Add on dip: {add_on_dip or '—'}")
        st.markdown(f"- Trim above: {trim_above or '—'}")
        st.markdown(f"- Hard exit: {hard_exit or '—'}")
        if sizing_note:
            st.markdown(f"- {sizing_note}")

    with bottom_r:
        st.markdown("**Watchlist**")
        st.markdown(f"- Add to watchlist: **{'Yes' if wl_flag else 'No'}**")
        st.markdown(f"- Bucket: **{wl_bucket}**")
        if wl_notes:
            st.markdown(f"- Notes: {wl_notes}")
        st.caption("AI view generated by your Mini Trading Desk engine.")

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# Overview tab
# ---------------------------------------------------------
def render_overview(scored_df: pd.DataFrame):
    metrics = compute_overview_analytics(scored_df)
    symbol_col = detect_symbol_column(scored_df)

    # ---------- TOP: Glass summary band (KPIs + Health gauge) ----------
    st.markdown(
        '<div class="glass-card" style="padding:18px 22px; margin-top:4px; margin-bottom:18px;">',
        unsafe_allow_html=True,
    )

    top_left, top_right = st.columns([2, 1.1])

    # Left side – KPI tiles
    with top_left:
        st.markdown(
            """
            <div style="display:flex; flex-direction:column; gap:10px;">
              <div style="font-size:0.9rem; color:#6B7280;">Portfolio Snapshot</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        k1, k2, k3, k4 = st.columns(4)

        k1.metric(
            "Total Portfolio Value",
            fmt_currency(metrics["total_value"], 0) if metrics["total_value"] is not None else "—",
        )
        k2.metric(
            "Total Unrealized P/L",
            fmt_currency(metrics["total_pl"], 0) if metrics["total_pl"] is not None else "—",
        )
        k3.metric("Positions", len(scored_df))
        if metrics["avg_score"] is not None:
            k4.metric("Avg Score", f"{metrics['avg_score']:.1f}")
        else:
            k4.metric("Avg Score", "—")

        # Second KPI row: weighted score + mix
        c1, c2, c3, c4 = st.columns(4)
        if metrics.get("weighted_score") is not None:
            c1.metric("Value-Weighted Score", f"{metrics['weighted_score']:.1f}")
        else:
            c1.metric("Value-Weighted Score", "—")

        buckets = metrics.get("score_buckets", {})
        c2.metric("Speculative (<40)", buckets.get("Speculative (<40)", 0))
        c3.metric("Middle (40–59)", buckets.get("Middle (40–59)", 0))
        c4.metric("Core (60+)", buckets.get("Core (60+)", 0))

    # Right side – Health gauge
    with top_right:
        base_score = metrics.get("weighted_score") or metrics.get("avg_score")
        if base_score is not None:
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=base_score,
                    number={"suffix": " / 100", "font": {"size": 22, "color": "#0F172A"}},
                    title={"text": "Portfolio Health", "font": {"size": 14, "color": "#4B5563"}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#2563EB"},
                        "bgcolor": "rgba(241,245,249,0.9)",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 40], "color": "rgba(248,113,113,0.45)"},
                            {"range": [40, 60], "color": "rgba(234,179,8,0.45)"},
                            {"range": [60, 100], "color": "rgba(34,197,94,0.45)"},
                        ],
                    },
                )
            )
            fig_gauge.update_layout(
                margin=dict(l=10, r=10, t=40, b=0),
                paper_bgcolor="rgba(255,255,255,0.0)",
            )
            st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("Portfolio health gauge will appear once scores are available.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- MIDDLE: Score mix + Action summary tiles ----------
    st.markdown(
        '<div class="glass-card" style="padding:18px 22px; margin-bottom:18px;">',
        unsafe_allow_html=True,
    )

    m1, m2 = st.columns(2)

    # Score mix as mini bar chart
    with m1:
        st.markdown("#### Score Mix")
        buckets = metrics.get("score_buckets", {})
        if buckets:
            mix_df = pd.DataFrame(
                {
                    "Bucket": list(buckets.keys()),
                    "Count": list(buckets.values()),
                }
            )
            fig_mix = px.bar(
                mix_df,
                x="Bucket",
                y="Count",
                text="Count",
                title="",
            )
            fig_mix.update_traces(textposition="outside")
            fig_mix.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="rgba(255,255,255,0.96)",
                paper_bgcolor="rgba(255,255,255,0.0)",
                font=dict(size=11, color="#0F172A"),
            )
            st.plotly_chart(fig_mix, use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("Scores not available yet.")

    # Action summary (Strong buy / buy / hold / trim / exit)
    with m2:
        st.markdown("#### Action Summary")
        if "Decision" in scored_df.columns:
            decision_counts = scored_df["Decision"].value_counts().to_dict()
            r1, r2, r3, r4, r5 = st.columns(5)
            r1.metric("Strong Buy", decision_counts.get("Strong Buy", 0))
            r2.metric("Buy", decision_counts.get("Buy", 0))
            r3.metric("Hold", decision_counts.get("Hold", 0))
            r4.metric("Trim", decision_counts.get("Trim", 0))
            r5.metric("Exit / Avoid", decision_counts.get("Exit / Avoid", 0))
        else:
            st.caption("No decision column in data – scoring step may need adjustment.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- LOWER: Key charts row ----------
    st.markdown(
        '<div class="glass-card" style="padding:18px 22px; margin-bottom:18px;">',
        unsafe_allow_html=True,
    )
    st.markdown("#### Key Charts")

    col_a, col_b = st.columns(2)

    # Score distribution
    with col_a:
        if "Score" in scored_df.columns:
            score_series = scored_df["Score"].dropna()
            if not score_series.empty:
                fig_score = px.histogram(score_series, nbins=15, title="Score Distribution")
                fig_score.update_traces(marker=dict(line=dict(width=0)))
                fig_score.update_layout(
                    margin=dict(l=0, r=0, t=30, b=0),
                    plot_bgcolor="rgba(255,255,255,0.96)",
                    paper_bgcolor="rgba(255,255,255,0.0)",
                    font=dict(size=11, color="#0F172A"),
                )
                st.plotly_chart(fig_score, use_container_width=True, config={"displayModeBar": False})
            else:
                st.caption("No scores available.")
        else:
            st.caption("No scores available.")

    # Allocation by decision
    with col_b:
        if "PortfolioWeightPct" in scored_df.columns and "Decision" in scored_df.columns:
            alloc = (
                scored_df.dropna(subset=["PortfolioWeightPct"])
                .groupby("Decision")["PortfolioWeightPct"]
                .sum()
                .reset_index()
            )
            if not alloc.empty:
                fig_alloc = px.pie(
                    alloc,
                    values="PortfolioWeightPct",
                    names="Decision",
                    title="Allocation by Decision",
                )
                fig_alloc.update_layout(
                    margin=dict(l=0, r=0, t=30, b=0),
                    paper_bgcolor="rgba(255,255,255,0.0)",
                    font=dict(size=11, color="#0F172A"),
                )
                st.plotly_chart(fig_alloc, use_container_width=True, config={"displayModeBar": False})
            else:
                st.caption("No allocation data.")
        else:
            st.caption("No allocation data.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- BOTTOM: Unrealized P/L focus & Top movers ----------
    st.markdown(
        '<div class="glass-card" style="padding:18px 22px; margin-bottom:8px;">',
        unsafe_allow_html=True,
    )

    st.markdown("#### Unrealized P/L & Today’s Focus")

    u1, u2 = st.columns([2, 1])

    # Unrealized P/L by ticker
    with u1:
        if symbol_col and "UnrealizedPL" in scored_df.columns:
            pl_df = scored_df[[symbol_col, "UnrealizedPL"]].copy().dropna(subset=["UnrealizedPL"])
            if not pl_df.empty:
                fig_pl = px.bar(pl_df, x=symbol_col, y="UnrealizedPL", title="Unrealized P/L by Ticker")
                fig_pl.update_layout(
                    margin=dict(l=0, r=0, t=30, b=0),
                    plot_bgcolor="rgba(255,255,255,0.96)",
                    paper_bgcolor="rgba(255,255,255,0.0)",
                    xaxis_title="",
                    yaxis_title="Unrealized P/L",
                    font=dict(size=11, color="#0F172A"),
                )
                st.plotly_chart(fig_pl, use_container_width=True, config={"displayModeBar": False})
            else:
                st.caption("No unrealized P/L data.")
        else:
            st.caption("No unrealized P/L data.")

    # Today’s Focus – top winners/losers bullets
    with u2:
        winners = metrics["top_winners"]
        losers = metrics["top_losers"]
        if not winners.empty:
            st.markdown("**Top Winners**")
            for _, r in winners.iterrows():
                st.markdown(f"- **{r[symbol_col]}** up {fmt_currency(r['UnrealizedPL'],0)} unrealized")
        else:
            st.caption("No winners yet.")

        st.markdown("---")

        if not losers.empty:
            st.markdown("**Top Losers**")
            for _, r in losers.iterrows():
                st.markdown(f"- **{r[symbol_col]}** down {fmt_currency(r['UnrealizedPL'],0)} unrealized")
        else:
            st.caption("No losers yet.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- AI Desk quick glance ----------
    st.markdown("### AI Desk – Quick Glance")

    if symbol_col is None:
        st.info("No ticker/symbol column found for AI desk.")
        return

    tickers = sorted(scored_df[symbol_col].dropna().unique())
    col_left, col_right = st.columns([1, 2])

    with col_left:
        selected_ticker = st.selectbox(
            "Select ticker for AI check",
            options=tickers,
            index=0,
            key="ai_glance_ticker",
        )
        mode = st.radio(
            "Depth",
            options=["lite", "full"],
            index=0,
            horizontal=True,
            key="ai_glance_mode",
        )
        run_btn = st.button("Run Mini Trading Desk", key="ai_glance_run")

    with col_right:
        if run_btn and selected_ticker:
            row = scored_df[scored_df[symbol_col] == selected_ticker].iloc[0]
            with st.spinner("Thinking like a 4-role trading desk..."):
                ai_view = run_mini_trading_desk(
                    selected_ticker,
                    row,
                    mode=mode,
                )
            render_ai_glance_card(ai_view)


# ---------------------------------------------------------
# Positions tab
# ---------------------------------------------------------
def render_positions(scored_df: pd.DataFrame):
    st.markdown(
        '<div class="glass-card" style="padding:18px 22px; margin-top:8px;">',
        unsafe_allow_html=True,
    )
    st.markdown("### Positions & Metrics")

    df = scored_df.copy()
    decision_col = "Decision" if "Decision" in df.columns else None

    # Filters
    if decision_col:
        options = ["All"] + sorted(df[decision_col].dropna().unique().tolist())
        filter_decision = st.selectbox("Filter by decision", options=options, index=0)
        if filter_decision != "All":
            df = df[df[decision_col] == filter_decision]

    sort_col = st.selectbox("Sort by column", options=df.columns.tolist(), index=df.columns.get_loc("Score") if "Score" in df.columns else 0)
    sort_ascending = st.checkbox("Sort ascending?", value=False)

    if sort_col:
        df = df.sort_values(sort_col, ascending=sort_ascending)

    # Simple formatting
    format_dict = {}
    value_col = get_value_column(df)
    if value_col and value_col in df.columns:
        format_dict[value_col] = lambda x: fmt_currency(x, 0)
    for col in ["UnrealizedPL", "TodayGainLossDollar", "TotalGainLossDollar"]:
        if col in df.columns:
            format_dict[col] = lambda x: fmt_currency(x, 0)
    for col in ["UnrealizedPLPct", "TodayGainLossPercent"]:
        if col in df.columns:
            format_dict[col] = lambda x: fmt_pct(x, 1)

    # Highlight Score / Decision
    def score_color(val):
        try:
            v = float(val)
        except Exception:
            return ""
        if v >= 70:
            return "background-color: rgba(22,163,74,0.18);"
        elif v >= 60:
            return "background-color: rgba(234,179,8,0.18);"
        elif v < 40:
            return "background-color: rgba(248,113,113,0.18);"
        return ""

    def decision_color(val):
        if not isinstance(val, str):
            return ""
        mapping = {
            "Strong Buy": "background-color: rgba(22,163,74,0.18);",
            "Buy": "background-color: rgba(34,197,94,0.14);",
            "Hold": "background-color: rgba(129,140,248,0.14);",
            "Trim": "background-color: rgba(250,204,21,0.18);",
            "Exit / Avoid": "background-color: rgba(248,113,113,0.18);",
            "Exit": "background-color: rgba(248,113,113,0.18);",
        }
        return mapping.get(val, "")

    styler = df.style.format(format_dict)
    if "Score" in df.columns:
        styler = styler.applymap(score_color, subset=["Score"])
    if decision_col:
        styler = styler.applymap(decision_color, subset=[decision_col])

    st.dataframe(styler, use_container_width=True, height=500)
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# Fundamentals tab
# ---------------------------------------------------------
def render_fundamentals(scored_df: pd.DataFrame):
    st.markdown(
        '<div class="glass-card" style="padding:18px 22px; margin-top:8px;">',
        unsafe_allow_html=True,
    )
    st.markdown("### Fundamentals Snapshot")

    df = scored_df.copy()
    fundamental_cols = [
        col
        for col in [
            "Symbol",
            "Ticker",
            "PE_TTM",
            "ForwardPE",
            "DividendYield",
            "ProfitMargin",
            "MarketCap",
            "Beta",
            "Score",
            "Decision",
        ]
        if col in df.columns
    ]
    if fundamental_cols:
        df = df[fundamental_cols]

    sort_col = st.selectbox("Sort by", options=fundamental_cols if fundamental_cols else df.columns.tolist())
    sort_ascending = st.checkbox("Sort ascending?", value=False, key="fund_sort_asc")
    df = df.sort_values(sort_col, ascending=sort_ascending)

    # Formatting
    format_dict = {}
    if "DividendYield" in df.columns:
        format_dict["DividendYield"] = lambda x: fmt_pct(x, 2)
    if "ProfitMargin" in df.columns:
        format_dict["ProfitMargin"] = lambda x: fmt_pct(x, 1)
    if "MarketCap" in df.columns:
        format_dict["MarketCap"] = lambda x: fmt_currency(x, 0)

    styler = df.style.format(format_dict)

    def score_color(val):
        try:
            v = float(val)
        except Exception:
            return ""
        if v >= 70:
            return "background-color: rgba(22,163,74,0.18);"
        elif v >= 60:
            return "background-color: rgba(234,179,8,0.18);"
        elif v < 40:
            return "background-color: rgba(248,113,113,0.18);"
        return ""

    def decision_color(val):
        if not isinstance(val, str):
            return ""
        mapping = {
            "Strong Buy": "background-color: rgba(22,163,74,0.18);",
            "Buy": "background-color: rgba(34,197,94,0.14);",
            "Hold": "background-color: rgba(129,140,248,0.14);",
            "Trim": "background-color: rgba(250,204,21,0.18);",
            "Exit / Avoid": "background-color: rgba(248,113,113,0.18);",
            "Exit": "background-color: rgba(248,113,113,0.18);",
        }
        return mapping.get(val, "")

    if "Score" in df.columns:
        styler = styler.applymap(score_color, subset=["Score"])
    if "Decision" in df.columns:
        styler = styler.applymap(decision_color, subset=["Decision"])

    st.dataframe(styler, use_container_width=True, height=500)
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# Signals tab
# ---------------------------------------------------------
def render_signals(scored_df: pd.DataFrame):
    st.markdown(
        '<div class="glass-card" style="padding:18px 22px; margin-top:8px;">',
        unsafe_allow_html=True,
    )
    st.markdown("### Signals & Action List")

    if "Decision" not in scored_df.columns:
        st.info("No Decision column found – run scoring first.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    symbol_col = detect_symbol_column(scored_df) or "Symbol"
    base_cols = [
        c
        for c in [
            symbol_col,
            "Score",
            "PortfolioWeightPct",
            "UnrealizedPLPct",
            "PE_TTM",
            "ProfitMargin",
        ]
        if c in scored_df.columns
    ]

    def section(title: str, decision: str):
        st.markdown(f"#### {title}")
        subset = scored_df[scored_df["Decision"] == decision]
        if subset.empty:
            st.caption("None at the moment.")
        else:
            st.dataframe(
                subset[base_cols].sort_values("Score", ascending=False),
                use_container_width=True,
                height=min(280, 60 + 28 * len(subset)),
            )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Buy / Add Radar")
        section("Strong Buy", "Strong Buy")
        section("Buy", "Buy")

    with col2:
        st.markdown("#### De-Risk Radar")
        section("Trim", "Trim")
        section("Exit / Avoid", "Exit / Avoid")

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# Header / Logo
# ---------------------------------------------------------
def render_header():
    left, right = st.columns([4, 1])

    with left:
        st.markdown(
            """
            <div class="glass-card" style="padding:18px 22px; margin-bottom:16px;">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                  <div style="font-size:1.6rem; font-weight:650; color:#0F172A;">
                    Oldfield AI Stock Dashboard
                  </div>
                  <div style="font-size:0.9rem; color:#6B7280; margin-top:4px;">
                    Liquid-glass cockpit for your AI-scored portfolio — positions, fundamentals, and actions in one view.
                  </div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        # Adjust path to your logo file as needed (repo-relative)
        logo_path = "assets/oldfield_logo.png"
        if os.path.exists(logo_path):
            st.image(logo_path, width=80)
        st.markdown(
            """
            <div style="display:flex; justify-content:flex-end; margin-top:8px;">
              <span style="
                  border-radius:999px;
                  padding:4px 10px;
                  font-size:0.75rem;
                  background:rgba(37,99,235,0.08);
                  color:#1D4ED8;
                  border:1px solid rgba(37,99,235,0.25);">
                v1.1 · Experimental
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# Main app
# ---------------------------------------------------------
def main():
    render_header()

    st.markdown(
        '<div class="glass-card" style="padding:16px 22px; margin-bottom:16px;">',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([2, 3])
    with c1:
        uploaded_file = st.file_uploader(
            "Upload portfolio CSV",
            type=["csv"],
            help="Keep a single positions CSV in your drive, overwrite it, and reload to re-score quickly.",
        )
    with c2:
        st.caption(
            "Keep a single positions CSV in your drive, overwrite it, and reload this app to re-score the whole portfolio in one click."
        )

    st.markdown("</div>", unsafe_allow_html=True)

    if not uploaded_file:
        st.info("Upload a portfolio CSV to get started.")
        return

    # Load + score
    try:
        raw_df = load_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return

    try:
        scored_df = score_portfolio(raw_df)
    except Exception as e:
        st.error(f"Error scoring portfolio: {e}")
        st.write("Raw data preview:")
        st.dataframe(raw_df.head(), use_container_width=True)
        return

    # Tabs
    tab_overview, tab_positions, tab_fund, tab_signals = st.tabs(
        ["Overview", "Positions", "Fundamentals", "Signals"]
    )

    with tab_overview:
        render_overview(scored_df)

    with tab_positions:
        render_positions(scored_df)

    with tab_fund:
        render_fundamentals(scored_df)

    with tab_signals:
        render_signals(scored_df)


if __name__ == "__main__":
    main()
