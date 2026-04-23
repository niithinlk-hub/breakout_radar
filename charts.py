"""Plotly chart builders — all dark-themed."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from styles import PLOTLY_TEMPLATE, plotly_layout

COLORS = {
    "green":    "#00FF88",
    "red":      "#FF4444",
    "blue":     "#4488FF",
    "amber":    "#FFB800",
    "purple":   "#AA44FF",
    "muted":    "#8B949E",
    "bg":       "#0E1117",
    "bg_card":  "#1C2230",
    "border":   "#30363D",
}


def _apply_template(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_TEMPLATE["layout"])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Candlestick deep-dive chart
# ─────────────────────────────────────────────────────────────────────────────

def build_candlestick(
    df: pd.DataFrame,
    ticker: str,
    patterns: Optional[Dict] = None,
    levels: Optional[Dict] = None,
    show_volume: bool = True,
) -> go.Figure:
    """Full interactive candlestick with EMAs, BB, VWAP, volume, patterns."""
    rows = 3 if show_volume else 2
    row_heights = [0.55, 0.25, 0.20] if show_volume else [0.70, 0.30]
    specs = [[{"type": "candlestick"}]] + [[{"type": "scatter"}]] * (rows - 1)

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.02,
        specs=specs,
    )

    # ── Candlesticks ──
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price",
        increasing=dict(fillcolor=COLORS["green"], line=dict(color=COLORS["green"], width=1)),
        decreasing=dict(fillcolor=COLORS["red"], line=dict(color=COLORS["red"], width=1)),
        showlegend=False,
    ), row=1, col=1)

    # ── EMAs ──
    ema_colors = {20: COLORS["blue"], 50: COLORS["amber"], 100: "#AA44FF", 200: COLORS["red"]}
    for w, color in ema_colors.items():
        col_name = f"ema{w}"
        if col_name in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col_name], name=f"EMA{w}",
                line=dict(color=color, width=1.2),
                opacity=0.85,
            ), row=1, col=1)

    # ── Bollinger Bands ──
    if "bb_upper" in df.columns and "bb_lower" in df.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([pd.Series(df.index), pd.Series(df.index[::-1])]),
            y=pd.concat([df["bb_upper"], df["bb_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(68,136,255,0.07)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Bollinger Bands",
            showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_upper"], name="BB Upper",
            line=dict(color=COLORS["blue"], width=0.8, dash="dot"),
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_lower"], name="BB Lower",
            line=dict(color=COLORS["blue"], width=0.8, dash="dot"),
            showlegend=False,
        ), row=1, col=1)

    # ── Key levels (horizontal lines) ──
    if levels:
        level_styles = {
            "entry":   (COLORS["green"], "solid", "Entry"),
            "stop":    (COLORS["red"], "dash", "Stop"),
            "target1": (COLORS["amber"], "dot", "T1"),
            "target2": (COLORS["amber"], "dot", "T2"),
            "target3": (COLORS["amber"], "dot", "T3"),
        }
        for key, (color, dash, label) in level_styles.items():
            val = levels.get(key)
            if val and val > 0:
                fig.add_hline(y=val, line=dict(color=color, width=1, dash=dash),
                              annotation_text=f" {label}: ₹{val:.0f}",
                              annotation_font=dict(color=color, size=10),
                              row=1, col=1)

    # ── MACD panel ──
    if "macd_hist" in df.columns:
        colors_hist = [COLORS["green"] if v >= 0 else COLORS["red"] for v in df["macd_hist"].fillna(0)]
        fig.add_trace(go.Bar(
            x=df.index, y=df["macd_hist"], name="MACD Hist",
            marker_color=colors_hist, opacity=0.8,
        ), row=2, col=1)
        if "macd" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD",
                                     line=dict(color=COLORS["blue"], width=1)), row=2, col=1)
        if "macd_signal" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="Signal",
                                     line=dict(color=COLORS["amber"], width=1)), row=2, col=1)

    # ── Volume ──
    if show_volume and rows == 3:
        vol_colors = [COLORS["green"] if df["Close"].iloc[i] >= df["Open"].iloc[i] else COLORS["red"]
                      for i in range(len(df))]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"], name="Volume",
            marker_color=vol_colors, opacity=0.7,
        ), row=3, col=1)
        # Volume MA
        if "vol_avg20" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["vol_avg20"], name="Vol MA20",
                line=dict(color=COLORS["muted"], width=1, dash="dot"),
            ), row=3, col=1)

    fig.update_layout(**plotly_layout(
        title=dict(text=f"<b>{ticker}</b>", font=dict(size=16, color=COLORS["blue"])),
        xaxis_rangeslider_visible=False,
        height=680,
    ))
    fig.update_yaxes(tickformat=",.0f", tickprefix="₹")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# BPS Gauge
# ─────────────────────────────────────────────────────────────────────────────

def build_bps_gauge(score: float) -> go.Figure:
    """Semicircular gauge for BPS score."""
    if score >= 75:
        bar_color = COLORS["green"]
    elif score >= 55:
        bar_color = COLORS["amber"]
    else:
        bar_color = COLORS["red"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        number=dict(font=dict(size=36, color=bar_color), suffix=""),
        gauge=dict(
            axis=dict(range=[0, 105], tickwidth=1, tickcolor=COLORS["muted"],
                      tickfont=dict(color=COLORS["muted"])),
            bar=dict(color=bar_color, thickness=0.25),
            bgcolor=COLORS["bg_card"],
            borderwidth=1,
            bordercolor=COLORS["border"],
            steps=[
                dict(range=[0, 55], color="#1a0a0a"),
                dict(range=[55, 75], color="#1a1500"),
                dict(range=[75, 105], color="#001a0d"),
            ],
            threshold=dict(line=dict(color="white", width=2), thickness=0.75, value=75),
        ),
        title=dict(text="Breakout Probability Score", font=dict(size=13, color=COLORS["muted"])),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))
    fig.update_layout(**plotly_layout(height=250, margin=dict(l=20, r=20, t=40, b=10)))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Score breakdown bar chart
# ─────────────────────────────────────────────────────────────────────────────

def build_score_breakdown(factor_scores: Dict[str, float]) -> go.Figure:
    """Horizontal bar chart showing each factor's 0-10 score."""
    labels = {
        "trend":    "Trend (Price vs 20/50/200 SMA)",
        "momentum": "Momentum (RSI + MACD)",
        "volume":   "Volume (vs 20d Avg)",
        "setup":    "Setup (52W Proximity + BB Squeeze)",
        "weekly":   "Weekly Trend Confirmation",
    }
    weights = {
        "trend": 30, "momentum": 25, "volume": 20, "setup": 25, "weekly": 5,
    }

    names, scores, contributions, colors_list = [], [], [], []
    for k, v in factor_scores.items():
        names.append(f"{labels.get(k, k)} ({weights.get(k, 0)}%)")
        scores.append(v)
        contributions.append(round(v * weights.get(k, 0) / 100, 1))
        if v >= 7:
            colors_list.append(COLORS["green"])
        elif v >= 5:
            colors_list.append(COLORS["amber"])
        else:
            colors_list.append(COLORS["red"])

    fig = go.Figure(go.Bar(
        y=names,
        x=scores,
        orientation="h",
        marker_color=colors_list,
        text=[f"{s:.1f}/10 (contrib: {c})" for s, c in zip(scores, contributions)],
        textposition="outside",
        textfont=dict(color=COLORS["muted"], size=10),
    ))
    fig.update_layout(**plotly_layout(
        xaxis=dict(range=[0, 12], title=dict(text="Score (0-10)")),
        height=300,
        margin=dict(l=200, r=80, t=20, b=20),
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Market Heatmap (Treemap)
# ─────────────────────────────────────────────────────────────────────────────

def build_treemap(metrics_df: pd.DataFrame) -> go.Figure:
    """Treemap sized by market cap (or equal if no mc), colored by BPS."""
    df = metrics_df.copy()
    df = df[df["bps"].notna() & (df["name"].notna())]

    # Color mapping: BPS 0-100 → red-amber-green
    def bps_to_color(bps: float) -> str:
        if bps >= 70:
            return COLORS["green"]
        elif bps >= 50:
            return COLORS["amber"]
        return COLORS["red"]

    labels = df["name"].tolist()
    parents = df["sector"].tolist()
    values = [1] * len(df)  # equal size (mc data unreliable from yf)
    bps_values = df["bps"].tolist()
    tickers = df["ticker"].tolist()
    cmps = df["cmp"].tolist()

    # Build sector nodes
    sectors = df["sector"].unique().tolist()
    all_labels = sectors + labels
    all_parents = [""] * len(sectors) + parents
    all_values = [0] * len(sectors) + values
    all_bps = [df[df["sector"] == s]["bps"].mean() for s in sectors] + bps_values
    all_text = [f"Sector avg BPS: {b:.0f}" for b in [df[df["sector"] == s]["bps"].mean() for s in sectors]] + \
               [f"₹{c:.0f} | BPS {b:.0f}" for c, b in zip(cmps, bps_values)]

    color_vals = all_bps

    fig = go.Figure(go.Treemap(
        labels=all_labels,
        parents=all_parents,
        values=all_values,
        text=all_text,
        hovertemplate="<b>%{label}</b><br>%{text}<extra></extra>",
        marker=dict(
            colors=color_vals,
            colorscale=[[0, COLORS["red"]], [0.5, COLORS["amber"]], [1, COLORS["green"]]],
            cmin=0,
            cmax=100,
            showscale=True,
            colorbar=dict(
                title=dict(text="BPS"),
                tickvals=[0, 50, 70, 100],
                tickfont=dict(color=COLORS["muted"]),
            ),
        ),
        pathbar=dict(visible=True),
        tiling=dict(packing="squarify"),
    ))
    fig.update_layout(**plotly_layout(
        height=600,
        margin=dict(l=10, r=10, t=30, b=10),
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sector heatmap
# ─────────────────────────────────────────────────────────────────────────────

def build_sector_heatmap(metrics_df: pd.DataFrame) -> go.Figure:
    """Bar chart of average BPS per sector, sorted descending."""
    sector_avg = (
        metrics_df.groupby("sector")["bps"]
        .mean()
        .reset_index()
        .sort_values("bps", ascending=True)
    )

    bar_colors = []
    for bps in sector_avg["bps"]:
        if bps >= 70:
            bar_colors.append(COLORS["green"])
        elif bps >= 50:
            bar_colors.append(COLORS["amber"])
        else:
            bar_colors.append(COLORS["red"])

    fig = go.Figure(go.Bar(
        y=sector_avg["sector"],
        x=sector_avg["bps"],
        orientation="h",
        marker_color=bar_colors,
        text=[f"{b:.0f}" for b in sector_avg["bps"]],
        textposition="outside",
        textfont=dict(color=COLORS["muted"], size=11),
    ))
    fig.update_layout(**plotly_layout(
        xaxis=dict(range=[0, 105], title=dict(text="Average BPS")),
        height=max(400, len(sector_avg) * 28),
        margin=dict(l=180, r=60, t=20, b=30),
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Backtest equity curve
# ─────────────────────────────────────────────────────────────────────────────

def build_equity_curve(
    portfolio_curve: pd.Series,
    nifty_curve: pd.Series,
    title: str = "Backtest Equity Curve",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_curve.index, y=portfolio_curve,
        name="BPS>70 Portfolio", line=dict(color=COLORS["green"], width=2),
        fill="tozeroy", fillcolor="rgba(0,255,136,0.07)",
    ))
    fig.add_trace(go.Scatter(
        x=nifty_curve.index, y=nifty_curve,
        name="Nifty 50 Benchmark", line=dict(color=COLORS["blue"], width=2, dash="dot"),
    ))
    fig.add_hline(y=100, line=dict(color=COLORS["border"], width=1, dash="dash"))
    fig.update_layout(**plotly_layout(
        title=dict(text=title, font=dict(size=14, color=COLORS["muted"])),
        yaxis_title="Indexed Return (100 = start)",
        height=380,
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sector coverage pie
# ─────────────────────────────────────────────────────────────────────────────

def build_sector_pie(metrics_df: pd.DataFrame) -> go.Figure:
    counts = metrics_df["sector"].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.45,
        marker=dict(
            colors=[f"hsl({int(i * 360 / len(counts))}, 65%, 45%)" for i in range(len(counts))],
            line=dict(color=COLORS["bg"], width=2),
        ),
        textfont=dict(size=10, color=COLORS["text_primary"] if "text_primary" in COLORS else "white"),
    ))
    fig.update_layout(**plotly_layout(
        height=320,
        showlegend=True,
        legend=dict(font=dict(size=9)),
        margin=dict(l=10, r=10, t=10, b=10),
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# BPS history line for watchlist
# ─────────────────────────────────────────────────────────────────────────────

def build_bps_sparkline(bps_history: List[float], ticker: str) -> go.Figure:
    x = list(range(len(bps_history)))
    color = COLORS["green"] if bps_history[-1] >= 70 else COLORS["amber"] if bps_history[-1] >= 50 else COLORS["red"]
    fig = go.Figure(go.Scatter(
        x=x, y=bps_history, mode="lines+markers",
        line=dict(color=color, width=2),
        marker=dict(size=4, color=color),
        name=ticker,
    ))
    fig.update_layout(**plotly_layout(
        height=120,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(range=[0, 100]),
    ))
    return fig
