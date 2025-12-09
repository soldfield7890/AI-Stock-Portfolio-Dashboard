import os
import json
from datetime import date
from typing import Any, Dict, Optional

import pandas as pd

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ---------------------------
# LLM client helper
# ---------------------------

def _get_client():
    """
    Returns an OpenAI client if OPENAI_API_KEY is set and the library is installed.
    Returns None if not configured, so callers can degrade gracefully instead of crashing.
    """
    if OpenAI is None:
        return None
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    return OpenAI(api_key=key)


# ---------------------------
# Context builder
# ---------------------------

def build_context_from_scored_row(ticker: str, row: pd.Series) -> Dict[str, Any]:
    """
    Build a compact, structured context from your scored_df row.
    This is what we feed into the Mini Trading Desk prompt.
    """

    def g(col: str, default=None):
        try:
            return row.get(col, default)
        except Exception:
            return default

    # Try to infer basic price from value / quantity
    qty = g("Quantity") or g("Shares") or g("Qty")
    current_value = g("CurrentValue_clean") or g("CurrentValue") or g("Current Value")
    avg_cost = g("AverageCost") or g("AvgCost") or g("CostBasisPerShare")

    current_price = None
    if current_value is not None and qty not in (None, 0):
        try:
            current_price = float(current_value) / float(qty)
        except Exception:
            current_price = None

    context: Dict[str, Any] = {
        "ticker": ticker.upper(),
        "as_of_date": date.today().isoformat(),

        # Position / portfolio
        "quantity": qty,
        "current_value": current_value,
        "average_cost": avg_cost,
        "current_price": current_price,
        "portfolio_weight_pct": g("PortfolioWeightPct"),
        "unrealized_pl": g("UnrealizedPL"),
        "unrealized_pl_pct": g("UnrealizedPLPct"),

        # Your scoring meta
        "score": g("Score"),
        "model_decision": g("Decision"),
        "bucket": None,  # will derive below

        # Fundamentals
        "pe_ttm": g("PE_TTM"),
        "forward_pe": g("ForwardPE"),
        "dividend_yield": g("DividendYield"),
        "profit_margin": g("ProfitMargin"),
        "market_cap": g("MarketCap"),
        "beta": g("Beta"),

        # Optional description / sector if present
        "description": g("Description"),
        "sector": g("Sector"),
        "industry": g("Industry"),
    }

    # Derive a simple bucket: Forever Core / Growth Engine / Speculative Sidecar
    score = context["score"]
    weight = context["portfolio_weight_pct"] or 0

    if score is not None:
        try:
            s = float(score)
        except Exception:
            s = None
        if s is not None:
            if s >= 70 and weight >= 4:
                bucket = "Forever Core"
            elif s >= 60:
                bucket = "Growth Engine"
            elif s >= 50:
                bucket = "Growth / Speculative Blend"
            else:
                bucket = "Speculative Sidecar"
        else:
            bucket = None
    else:
        bucket = None

    context["bucket"] = bucket

    return context


# ---------------------------
# Prompt builder
# ---------------------------

def build_research_prompt(context: Dict[str, Any], mode: str = "lite") -> str:
    """
    Build a single prompt that simulates four roles:
      - Fundamental analyst
      - Technical / trend analyst
      - Sentiment / narrative analyst
      - Portfolio manager / decision maker

    The model must reply in strict JSON.
    """

    # Macro / strategy notes (your personal thesis)
    macro_thesis = (
        "The user is an aggressive but rational long-term investor who believes we are in an AI/tech/USA-led "
        "growth cycle over the next several years. The portfolio is built around three buckets: "
        "Forever Core, Growth Engine, and Speculative Sidecar. The goal is to grow wealth aggressively while "
        "gradually increasing the share of Forever Core positions over time."
    )

    # Mode guidance
    if mode == "full":
        detail_instruction = (
            "Provide rich but concise analysis for each role (3-5 bullets each) while still keeping the output "
            "compact enough to show in a dashboard. "
        )
    else:
        detail_instruction = (
            "Provide very concise analysis (2-3 bullets per role) optimized for a quick-glance dashboard tile. "
        )

    # Extract context safely
    t = context.get("ticker")
    as_of = context.get("as_of_date")

    # Clean up some values
    def safe(v):
        return "" if v is None else v

    prompt = f"""
You are an elite AI trading desk compressed into one model. Act as four distinct roles:
1) Fundamental Analyst
2) Technical / Trend Analyst
3) Sentiment / Narrative Analyst
4) Portfolio Manager / Decision Maker

Ticker: {t}
As-of date: {as_of}

Portfolio & Position Context:
- Quantity: {safe(context.get("quantity"))}
- Current price (approx): {safe(context.get("current_price"))}
- Current value: {safe(context.get("current_value"))}
- Average cost per share: {safe(context.get("average_cost"))}
- Unrealized P/L ($): {safe(context.get("unrealized_pl"))}
- Unrealized P/L (%): {safe(context.get("unrealized_pl_pct"))}
- Portfolio weight (%): {safe(context.get("portfolio_weight_pct"))}
- Model score (0-100): {safe(context.get("score"))}
- Model decision (internal scoring): {safe(context.get("model_decision"))}
- Bucket: {safe(context.get("bucket"))}

Fundamental Snapshot:
- P/E TTM: {safe(context.get("pe_ttm"))}
- Forward P/E: {safe(context.get("forward_pe"))}
- Dividend yield: {safe(context.get("dividend_yield"))}
- Profit margin: {safe(context.get("profit_margin"))}
- Market cap: {safe(context.get("market_cap"))}
- Beta: {safe(context.get("beta"))}
- Sector: {safe(context.get("sector"))}
- Industry: {safe(context.get("industry"))}
- Description: {safe(context.get("description"))}

Macro & Strategy Context:
{macro_thesis}

{detail_instruction}

For each of the 4 roles, think through the stock from that angle using the data above and your general market knowledge.
Then, as the Portfolio Manager, synthesize everything into a single decision and action plan.

### VERY IMPORTANT OUTPUT FORMAT

Reply in **valid JSON ONLY** with this exact top-level structure and field names:

{{
  "ticker": "{t}",
  "as_of_date": "{as_of}",
  "final_decision": "Strong Buy | Buy | Hold | Trim | Exit",
  "conviction_score": 0-100,
  "time_horizon": "Short-term trade (0-6m) | Medium-term (6-24m) | Long-term (2y+)",

  "bucket_view": "Forever Core | Growth Engine | Growth / Speculative Blend | Speculative Sidecar | Unknown",

  "fundamental_view": [
    "bullet 1",
    "bullet 2",
    "... (2-5 items total)"
  ],
  "technical_view": [
    "bullet 1",
    "bullet 2"
  ],
  "sentiment_view": [
    "bullet 1",
    "bullet 2"
  ],
  "risk_factors": [
    "key risk 1",
    "key risk 2"
  ],

  "primary_action": "Add | Hold | Trim | Exit | Watch Only",
  "next_actions": {{
    "add_on_dip_level": "price level or rule, or null",
    "trim_above_level": "price level or rule, or null",
    "hard_exit_level": "price level or rule, or null",
    "position_sizing_note": "brief guidance on sizing relative to current weight"
  }},

  "watchlist": {{
    "add_to_watchlist": true or false,
    "watchlist_bucket": "High Priority | Monitor | Speculative Lottery | None",
    "notes": "1-2 sentence watchlist rationale"
  }}
}}

Rules:
- Output MUST be valid JSON. Do not include any explanation outside JSON.
- Use double quotes for all JSON keys and string values.
- Keep bullets short and information-dense, geared for a dashboard.
"""
    return prompt


# ---------------------------
# Parser
# ---------------------------

def parse_research_output(raw_text: str) -> Dict[str, Any]:
    """
    Parse the model's JSON output safely.
    If parsing fails, fall back to a minimal structure.
    """
    raw_text = raw_text.strip()

    # Try to find JSON substring if model wrapped it accidentally
    first_brace = raw_text.find("{")
    last_brace = raw_text.rfind("}")
    if first_brace != -1 and last_brace != -1:
        candidate = raw_text[first_brace:last_brace + 1]
    else:
        candidate = raw_text

    try:
        data = json.loads(candidate)
    except Exception:
        # Fallback - wrap raw text
        return {
            "ticker": None,
            "as_of_date": None,
            "final_decision": "NO_DECISION",
            "conviction_score": None,
            "time_horizon": None,
            "bucket_view": None,
            "fundamental_view": [candidate],
            "technical_view": [],
            "sentiment_view": [],
            "risk_factors": [],
            "primary_action": "Watch Only",
            "next_actions": {
                "add_on_dip_level": None,
                "trim_above_level": None,
                "hard_exit_level": None,
                "position_sizing_note": None,
            },
            "watchlist": {
                "add_to_watchlist": False,
                "watchlist_bucket": "None",
                "notes": "Raw unstructured output (JSON parse failed).",
            },
            "_raw_text": raw_text,
        }

    # Ensure all expected keys exist
    def ensure(key, default):
        if key not in data or data[key] is None:
            data[key] = default

    ensure("fundamental_view", [])
    ensure("technical_view", [])
    ensure("sentiment_view", [])
    ensure("risk_factors", [])
    ensure("next_actions", {})
    ensure("watchlist", {})

    na = data["next_actions"]
    if not isinstance(na, dict):
        na = {}
    for k, default in [
        ("add_on_dip_level", None),
        ("trim_above_level", None),
        ("hard_exit_level", None),
        ("position_sizing_note", None),
    ]:
        if k not in na:
            na[k] = default
    data["next_actions"] = na

    wl = data["watchlist"]
    if not isinstance(wl, dict):
        wl = {}
    for k, default in [
        ("add_to_watchlist", False),
        ("watchlist_bucket", "None"),
        ("notes", ""),
    ]:
        if k not in wl:
            wl[k] = default
    data["watchlist"] = wl

    # Attach raw text for debugging
    data["_raw_text"] = raw_text

    return data


# ---------------------------
# Main entrypoint
# ---------------------------

def run_mini_trading_desk(
    ticker: str,
    scored_row: pd.Series,
    mode: str = "lite",
    model_name: str = "gpt-4.1-mini",
) -> Dict[str, Any]:
    """
    Core function your Streamlit app will call.

    - ticker: symbol to analyze
    - scored_row: a single row from scored_df (pandas.Series)
    - mode: "lite" or "full" (lite = cheaper/faster, full = more verbose)
    - model_name: override model if needed

    Returns a dict with:
      - ticker, as_of_date
      - final_decision, conviction_score, time_horizon, bucket_view
      - fundamental_view[], technical_view[], sentiment_view[], risk_factors[]
      - primary_action, next_actions{...}
      - watchlist{...}
      - _raw_text (for debugging)
    """
    # OPTION B behavior: if no API key/client, return a stub instead of crashing
    client = _get_client()
    if client is None:
        context = build_context_from_scored_row(ticker, scored_row)
        return {
            "ticker": context.get("ticker", ticker),
            "as_of_date": context.get("as_of_date"),
            "final_decision": "NO_API_KEY",
            "conviction_score": None,
            "time_horizon": None,
            "bucket_view": context.get("bucket"),
            "fundamental_view": [
                "API key not configured – AI Desk is currently offline."
            ],
            "technical_view": [
                "Add OPENAI_API_KEY to enable AI analysis in the dashboard."
            ],
            "sentiment_view": [],
            "risk_factors": [],
            "primary_action": "Configure API key",
            "next_actions": {},
            "watchlist": {
                "add_to_watchlist": False,
                "watchlist_bucket": "None",
                "notes": "AI analysis unavailable until OPENAI_API_KEY is set.",
            },
            "_raw_text": "",
        }

    # Normal AI flow if key is configured
    context = build_context_from_scored_row(ticker, scored_row)
    prompt = build_research_prompt(context, mode=mode)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an elite multi-role trading desk for an experienced, "
                    "macro-aware equity investor. You must follow the JSON schema exactly."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.25 if mode == "lite" else 0.35,
    )

    raw_text = response.choices[0].message.content
    parsed = parse_research_output(raw_text)

    # If ticker/as_of missing, backfill from context
    if not parsed.get("ticker"):
        parsed["ticker"] = context.get("ticker")
    if not parsed.get("as_of_date"):
        parsed["as_of_date"] = context.get("as_of_date")

    return parsed
