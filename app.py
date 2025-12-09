import os
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --- Optional imports from your project modules ---
try:
    from src.scoring import score_portfolio  # type: ignore
except Exception:  # pragma: no cover
    score_portfolio = None  # type: ignore

try:
    from src.loaders import import_load_csv  # type: ignore
except Exception:  # pragma: no cover
    import_load_csv = None  # type: ignore

try:
    from src.ai_research import run_mini_trading_desk  # type: ignore
except Exception:  # pragma: no cover
    run_mini_trading_desk = None  # type: ignore


# ---------------------------------------------------------------------
#  Page config & global styling
# ---------------------------------------------------------------------


st.set_page_config(
    page_title="Oldfield AI Stock Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_global_css() -> None:
    """Glass / modern styling with better contrast."""
    st.markdown(
        """
        <style>
        html, body, [class*="stApp"] {
            background: radial-gradient(circle at top left, #e4edff 0, #f5f7fb 40%, #f9fbff 100%);
            color: #0f172a;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text",
                         "Segoe UI", sans-serif;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }

        /* Glass cards */
        .of-glass-card {
            background: rgba(255,255,255,0.96);
            border-radius: 22px;
            padding: 18px 22px;
            box-shadow:
                0 18px 45px rgba(15,23,42,0.18),
                0 0 0 1px rgba(148,163,184,0.16);
            backdrop-filter: blur(18px);
        }

        .of-section-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            color: #6b7280;
            margin-bottom: 4px;
        }

        .of-kpi-label {
            font-size: 0.9rem;
            color: #6b7280;
            margin-bottom: 2px;
        }

        .of-kpi-value {
            font-size: 1.6rem;
            font-weight: 600;
            color: #020617;
        }

        .of-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 500;
            border: 1px solid rgba(148,163,184,0.4);
            background: rgba(248,250,252,0.9);
        }

        /* Tabs bar glass effect */
        div[data-baseweb="tab-list"] {
            background: rgba(255,255,255,0.95);
            border-radius: 999px;
            padding: 4px 6px;
            box-shadow:
                0 10px 25px rgba(15,23,42,0.18),
                0 0 0 1px rgba(148,163,184,0.25);
        }
        button[role="tab"] {
            border-radius: 999px !important;
        }

        /* File uploader contrast */
        .stFileUploader label, .stFileUploader div {
            color: #020617 !important;
        }

        /* Dataframe text size */
        .stDataFrame table {
            font-size: 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_global_css()


# ---------------------------------------------------------------------
#  Utilities
# ---------------------------------------------------------------------


def friendly_currency(x: float) -> str:
    if pd.isna(x):
        return "—"
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1_000_000_000:
        return f"{sign}${x/1_000_000_000:,.1f}B"
    if x >= 1_000_000:
        return f"{sign}${x/1_000_000:,.1f}M"
    if x >= 1_000:
        return f"{sign}${x/1_000:,.1f}K"
    return f"{sign}${x:,.0f}"


def bucket_counts(df: pd.DataFrame) -> tuple[int, int, int]:
    """Speculative (<40), Middle (40–59), Core (60+)."""
    if "Score" not in df.columns:
        return 0, 0, 0
    s = df["Score"]
    return int((s < 40).sum()), int(((s >= 40) & (s < 60)).sum()), int((s >= 60).sum())


def decision_counts(df: pd.DataFrame) -> Dict[str, int]:
    if "Decision" not in df.columns:
        return {}
    decisions = ["Strong Buy", "Buy", "Hold", "Trim", "Exit / Avoid"]
    return {d: int((df["Decision"] == d).sum()) for d in decisions}


def top_unrealized(df: pd.DataFrame, n: int = 3):
    if "UnrealizedPL" not in df.columns:
        return df.head(0), df.head(0)
    df_sorted = df.sort_values("UnrealizedPL", ascending=False)
    winners = df_sorted.head(n)
    losers = df_sorted.tail(n).sort_values("UnrealizedPL")
    return winners, losers


def portfolio_summary(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    out["total_value"] = float(df["CurrentValue"].sum()) if "CurrentValue" in df.columns else float("nan")
    out["total_unrealized"] = float(df["UnrealizedPL"].sum()) if "UnrealizedPL" in df.columns else float("nan")
    out["avg_score"] = float(df["Score"].mean()) if "Score" in df.columns else float(50.0)
    out["positions"] = int(len(df))

    if "PortfolioWeightPct" in df.columns and "Symbol" in df.columns and not df.empty:
        idx = df["PortfolioWeightPct"].idxmax()
        row = df.loc[idx]
        out["largest_symbol"] = str(row.get("Symbol", "—"))
        out["largest_weight"] = float(row.get("PortfolioWeightPct", np.nan))
    else:
        out["largest_symbol"] = "—"
        out["largest_weight"] = float("nan")

    out["health"] = float(df["Score"].mean()) if "Score" in df.columns and not df.empty else 50.0

    spec, mid, core = bucket_counts(df)
    out["spec"], out["mid"], out["core"] = spec, mid, core

    return out


# ---------------------------------------------------------------------
#  Overview-specific helpers
# ---------------------------------------------------------------------


def render_health_gauge(health: float):
    fig = go.Figure()

    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=health,
            number={"suffix": " / 100", "font": {"size": 22}},
            gauge={
                "shape": "angular",
                "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": "rgba(0,0,0,0)"},
                "bar": {"color": "#1d4ed8", "thickness": 0.22},
                "bgcolor": "rgba(255,255,255,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "#fee2e2"},
                    {"range": [40, 60], "color": "#fef3c7"},
                    {"range": [60, 100], "color": "#dcfce7"},
                ],
            },
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        height=260,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_score_mix(df: pd.DataFrame):
    spec, mid, core = bucket_counts(df)
    data = pd.DataFrame(
        {
            "Bucket": ["Speculative (<40)", "Middle (40–59)", "Core (60+)"],
            "Count": [spec, mid, core],
        }
    )
    fig = px.bar(
        data,
        x="Bucket",
        y="Count",
        text="Count",
    )
    fig.update_traces(
        textposition="outside",
        marker=dict(
            color=["#fecaca", "#bfdbfe", "#bbf7d0"],
            line=dict(color="#1f2937", width=0.5),
        ),
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(showgrid=False),
        xaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_action_summary(df: pd.DataFrame):
    counts = decision_counts(df)
    cols = st.columns(5)
    labels = ["Strong Buy", "Buy", "Hold", "Trim", "Exit / Avoid"]
    colors = ["#22c55e", "#4ade80", "#e5e7eb", "#fbbf24", "#fca5a5"]
    for col, label, color in zip(cols, labels, colors):
        with col:
            st.markdown('<div class="of-kpi-label">{}</div>'.format(label), unsafe_allow_html=True)
            val = counts.get(label, 0)
            st.markdown(
                f'<div class="of-kpi-value" style="color:{color};">{val}</div>',
                unsafe_allow_html=True,
            )


def render_unrealized_focus(df: pd.DataFrame):
    if "Symbol" not in df.columns or "UnrealizedPL" not in df.columns:
        st.info("Unrealized P/L columns not available in this dataset.")
        return

    chart_df = df[["Symbol", "UnrealizedPL"]].copy()
    chart_df["UnrealizedPL"] = chart_df["UnrealizedPL"].astype(float)

    fig = px.bar(
        chart_df.sort_values("UnrealizedPL", ascending=False),
        x="Symbol",
        y="UnrealizedPL",
    )
    fig.update_traces(marker=dict(color="#93c5fd", line=dict(color="#1f2937", width=0.3)))
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Unrealized P/L ($)",
        xaxis_title="",
    )

    left, right = st.columns([2.3, 1])
    with left:
        st.subheader("Unrealized P/L by Ticker")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        winners, losers = top_unrealized(df)
        st.subheader("Today’s Focus")

        def bullet_list(title: str, subset: pd.DataFrame, positive: bool):
            st.markdown(f"**{title}**")
            if subset.empty:
                st.write("None.")
                return
            for _, row in subset.iterrows():
                ticker = row.get("Symbol", "—")
                pl = row.get("UnrealizedPL", np.nan)
                color = "#16a34a" if positive else "#b91c1c"
                st.markdown(
                    f"- **{ticker}** "
                    f"<span style='color:{color};'>{friendly_currency(pl)} unrealized</span>",
                    unsafe_allow_html=True,
                )

        bullet_list("Top Winners", winners, positive=True)
        bullet_list("Top Losers", losers, positive=False)


# ---------------------------------------------------------------------
#  Tab renderers
# ---------------------------------------------------------------------


def render_overview(df: pd.DataFrame):
    summary = portfolio_summary(df)

    # HEADER TILE
    st.markdown(
        """
        <div class="of-glass-card" style="margin-bottom:18px;">
          <div class="of-section-label">Dashboard</div>
          <div style="display:flex; align-items:center; justify-content:space-between;">
            <div>
              <h1 style="margin:0; font-size:2rem;">Overview</h1>
              <p style="margin:2px 0 0; color:#6b7280; font-size:0.95rem;">
                Command center for value, risk, and next actions across your portfolio.
              </p>
            </div>
            <div class="of-pill">
              Mode: <span style="font-weight:600;">Live review</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ROW 1 – KPIs + HEALTH GAUGE
    col_kpi, col_health = st.columns([2, 1.3])

    with col_kpi:
        st.markdown('<div class="of-glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="of-section-label">Portfolio snapshot</div>', unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown('<div class="of-kpi-label">Total Value</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="of-kpi-value">{friendly_currency(summary["total_value"])}</div>',
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown('<div class="of-kpi-label">Total Unrealized P/L</div>', unsafe_allow_html=True)
            color = "#16a34a" if summary["total_unrealized"] >= 0 else "#b91c1c"
            st.markdown(
                f'<div class="of-kpi-value" style="color:{color};">'
                f'{friendly_currency(summary["total_unrealized"])}</div>',
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown('<div class="of-kpi-label">Average Score</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="of-kpi-value">{summary["avg_score"]:.1f}</div>',
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown('<div class="of-kpi-label">Positions</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="of-kpi-value">{summary["positions"]}</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            "<hr style='margin:12px 0; border:none; border-top:1px solid #e5e7eb;' />",
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown('<div class="of-kpi-label">Speculative (&lt;40)</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="of-kpi-value">{summary["spec"]}</div>', unsafe_allow_html=True)
        with m2:
            st.markdown('<div class="of-kpi-label">Middle (40–59)</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="of-kpi-value">{summary["mid"]}</div>', unsafe_allow_html=True)
        with m3:
            st.markdown('<div class="of-kpi-label">Core (60+)</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="of-kpi-value">{summary["core"]}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col_health:
        st.markdown('<div class="of-glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="of-section-label">Health</div>', unsafe_allow_html=True)
        render_health_gauge(summary["health"])
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    # ROW 2 – SCORE MIX + ACTION QUEUE
    left, right = st.columns([1.4, 1.6])
    with left:
        st.markdown('<div class="of-glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="of-section-label">Score mix</div>', unsafe_allow_html=True)
        render_score_mix(df)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="of-glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="of-section-label">Action queue</div>', unsafe_allow_html=True)
        render_action_summary(df)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    # ROW 3 – UNREALIZED P/L FOCUS
    st.markdown('<div class="of-glass-card">', unsafe_allow_html=True)
    render_unrealized_focus(df)
    st.markdown('</div>', unsafe_allow_html=True)


def render_positions(df: pd.DataFrame):
    st.markdown(
        """
        <div class="of-glass-card" style="margin-bottom:18px;">
          <div class="of-section-label">Positions</div>
          <h1 style="margin:0; font-size:1.6rem;">Positions & Metrics</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df.empty:
        st.info("No data loaded yet.")
        return

    decision_filter = st.selectbox(
        "Filter by decision",
        options=["All"] + sorted(df.get("Decision", pd.Series()).dropna().unique().tolist()),
    )

    sort_col = st.selectbox(
        "Sort by column",
        options=[c for c in df.columns if df[c].dtype != "object"] or df.columns.tolist(),
        index=0,
    )

    sort_ascending = st.checkbox("Sort ascending?", value=False)

    filtered = df.copy()
    if decision_filter != "All" and "Decision" in df.columns:
        filtered = filtered[filtered["Decision"] == decision_filter]

    filtered = filtered.sort_values(sort_col, ascending=sort_ascending)

    st.dataframe(filtered.reset_index(drop=True), use_container_width=True, height=600)


def render_fundamentals(df: pd.DataFrame):
    st.markdown(
        """
        <div class="of-glass-card" style="margin-bottom:18px;">
          <div class="of-section-label">Fundamentals</div>
          <h1 style="margin:0; font-size:1.6rem;">Fundamentals Snapshot</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df.empty:
        st.info("No data loaded yet.")
        return

    metric_cols = [c for c in ["PE_TTM", "ForwardPE", "DividendYield", "ProfitMargin", "MarketCap", "Beta", "Score"] if c in df.columns]
    if not metric_cols:
        st.dataframe(df, use_container_width=True)
        return

    sort_metric = st.selectbox("Sort by", options=metric_cols, index=metric_cols.index("Score") if "Score" in metric_cols else 0)
    sort_ascending = st.checkbox("Sort ascending?", value=False, key="fund_sort")

    view = df.sort_values(sort_metric, ascending=sort_ascending).copy()

    # Format key metrics
    if "DividendYield" in view.columns:
        view["DividendYield"] = view["DividendYield"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "—")
    if "ProfitMargin" in view.columns:
        view["ProfitMargin"] = view["ProfitMargin"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
    if "MarketCap" in view.columns:
        view["MarketCap"] = view["MarketCap"].apply(lambda x: friendly_currency(x) if pd.notna(x) else "—")

    st.dataframe(view.reset_index(drop=True), use_container_width=True, height=650)


def render_signals(df: pd.DataFrame):
    st.markdown(
        """
        <div class="of-glass-card" style="margin-bottom:18px;">
          <div class="of-section-label">Signals</div>
          <h1 style="margin:0; font-size:1.6rem;">Signals & Action List</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df.empty or "Decision" not in df.columns:
        st.info("No signal data available.")
        return

    strong_buy = df[df["Decision"] == "Strong Buy"]
    buy = df[df["Decision"] == "Buy"]
    trim = df[df["Decision"] == "Trim"]
    exit_avoid = df[df["Decision"] == "Exit / Avoid"]

    # BUY RADAR
    st.markdown('<div class="of-glass-card" style="margin-bottom:14px;">', unsafe_allow_html=True)
    st.markdown("### Buy / Add Radar", unsafe_allow_html=True)
    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.markdown("**Strong Buy**")
        st.dataframe(
            strong_buy[["Symbol", "Score", "PortfolioWeightPct", "UnrealizedPLPct", "PE_TTM", "ProfitMargin"]].reset_index(drop=True)
            if not strong_buy.empty else strong_buy,
            use_container_width=True,
            height=220,
        )
    with c2:
        st.markdown("**Buy**")
        st.dataframe(
            buy[["Symbol", "Score", "PortfolioWeightPct", "UnrealizedPLPct", "PE_TTM", "ProfitMargin"]].reset_index(drop=True)
            if not buy.empty else buy,
            use_container_width=True,
            height=220,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # DE-RISK RADAR
    st.markdown('<div class="of-glass-card">', unsafe_allow_html=True)
    st.markdown("### De-Risk Radar", unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Trim**")
        st.dataframe(
            trim[["Symbol", "Score", "PortfolioWeightPct", "UnrealizedPLPct"]].reset_index(drop=True)
            if not trim.empty else trim,
            use_container_width=True,
            height=250,
        )
    with c4:
        st.markdown("**Exit / Avoid**")
        st.dataframe(
            exit_avoid[["Symbol", "Score", "PortfolioWeightPct", "UnrealizedPLPct"]].reset_index(drop=True)
            if not exit_avoid.empty else exit_avoid,
            use_container_width=True,
            height=250,
        )
    st.markdown('</div>', unsafe_allow_html=True)


def render_ai_desk(df: pd.DataFrame):
    st.markdown(
        """
        <div class="of-glass-card" style="margin-bottom:18px;">
          <div class="of-section-label">AI Desk</div>
          <h1 style="margin:0; font-size:1.6rem;">AI Desk – Quick Glance</h1>
          <p style="margin:4px 0 0; color:#6b7280; font-size:0.9rem;">
            Lightweight replication of the TradingAgents "trading desk" idea –
            pick a ticker and let the AI summarize fundamentals, trend, risk, and next actions.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df.empty or "Symbol" not in df.columns:
        st.info("Load a portfolio first to use the AI Desk.")
        return

    if run_mini_trading_desk is None:
        st.warning("AI engine (run_mini_trading_desk) is not available in this environment.")
        return

    symbols = sorted(df["Symbol"].dropna().unique().tolist())
    left, right = st.columns([1, 2])

    with left:
        selected_ticker = st.selectbox("Select ticker for AI check", options=symbols)
        depth = st.radio("Depth", options=["lite", "full"], index=0, horizontal=True)
        run = st.button("Run mini trading desk", use_container_width=True)

    if not run:
        return

    row = df[df["Symbol"] == selected_ticker].iloc[0].to_dict()

    # Call your AI engine
    try:
        ai_view = run_mini_trading_desk(
            selected_ticker,
            row,
            mode=depth,
        )
    except RuntimeError as e:
        st.error(str(e))
        return
    except Exception as e:  # pragma: no cover
        st.error(f"AI Desk error: {e}")
        return

    with right:
        st.markdown('<div class="of-glass-card">', unsafe_allow_html=True)

        header = st.columns([2, 1])
        with header[0]:
            title = ai_view.get("ticker", selected_ticker)
            final_decision = ai_view.get("final_decision", "—")
            st.markdown(f"**AI Trading Desk View – {title}**")
            st.markdown(
                f"<span class='of-kpi-label'>Decision:</span> "
                f"<span style='font-weight:600;'>{final_decision}</span>",
                unsafe_allow_html=True,
            )
        with header[1]:
            primary = ai_view.get("primary_action", "")
            if primary:
                st.markdown(
                    f"<div class='of-pill' style='justify-content:flex-end;'>"
                    f"Primary action: <span style='font-weight:600;'>{primary}</span></div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<hr />", unsafe_allow_html=True)

        meta_cols = st.columns(3)
        with meta_cols[0]:
            st.markdown("**Conviction**")
            conv = ai_view.get("conviction_score")
            st.write(conv if conv is not None else "—")
        with meta_cols[1]:
            st.markdown("**Horizon**")
            st.write(ai_view.get("time_horizon", "—"))
        with meta_cols[2]:
            st.markdown("**Bucket**")
            st.write(ai_view.get("bucket_view", "—"))

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

        # Fundamentals / Technicals / Sentiment
        sec1, sec2, sec3 = st.columns(3)
        with sec1:
            st.markdown("**Fundamentals**")
            for line in ai_view.get("fundamental_view", []):
                st.markdown(f"- {line}")
        with sec2:
            st.markdown("**Trend & Technicals**")
            for line in ai_view.get("technical_view", []):
                st.markdown(f"- {line}")
        with sec3:
            st.markdown("**Sentiment & Story**")
            for line in ai_view.get("sentiment_view", []):
                st.markdown(f"- {line}")

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

        # Key risks + next actions
        st.markdown("**Key Risks**")
        risks = ai_view.get("risk_factors", [])
        if risks:
            for r in risks:
                st.markdown(f"- {r}")
        else:
            st.write("None highlighted.")

        next_actions = ai_view.get("next_actions", {})
        if next_actions:
            st.markdown("<br/>**Next actions**", unsafe_allow_html=True)
            for label, text in next_actions.items():
                st.markdown(f"- **{label}:** {text}")

        st.markdown(
            "<p style='margin-top:10px; font-size:0.75rem; color:#9ca3af;'>"
            "AI view generated by your mini trading desk engine.</p>",
            unsafe_allow_html=True,
        )

        st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------
#  Data loading
# ---------------------------------------------------------------------


def load_positions_from_upload(upload) -> pd.DataFrame:
    if upload is None:
        return pd.DataFrame()

    try:
        if import_load_csv is not None:
            return import_load_csv(upload)  # type: ignore
    except Exception:
        pass

    try:
        return pd.read_csv(upload)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return pd.DataFrame()


def score_positions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    try:
        if score_portfolio is not None:
            return score_portfolio(df)  # type: ignore
    except Exception as e:
        st.warning(f"Score engine failed, showing raw data only. ({e})")
    return df


# ---------------------------------------------------------------------
#  Main layout
# ---------------------------------------------------------------------


def main():
    # Top header with optional logo slot
    header_left, header_right = st.columns([4, 1])
    with header_left:
        st.markdown(
            """
            <div style="margin-bottom:6px;">
              <h1 style="margin:0; font-size:2.1rem;">Oldfield AI Stock Dashboard</h1>
              <p style="margin:2px 0 0; color:#6b7280; font-size:0.9rem;">
                Glass-style cockpit for tracking positions, fundamentals, signals, and AI-driven views.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with header_right:
        logo_path = os.getenv("OF_LOGO_PATH", "logo.png")
        if os.path.exists(logo_path):
            st.image(logo_path, width=74)

    upload = st.file_uploader("Upload portfolio CSV", type=["csv"], label_visibility="collapsed")
    df_raw = load_positions_from_upload(upload)
    df_scored = score_positions(df_raw)

    if df_scored.empty:
        st.info("Upload a positions CSV to get started.")
        return

    tabs = st.tabs(["Overview", "Positions", "Fundamentals", "Signals", "AI Desk"])

    with tabs[0]:
        render_overview(df_scored)

    with tabs[1]:
        render_positions(df_scored)

    with tabs[2]:
        render_fundamentals(df_scored)

    with tabs[3]:
        render_signals(df_scored)

    with tabs[4]:
        render_ai_desk(df_scored)


if __name__ == "__main__":
    main()
