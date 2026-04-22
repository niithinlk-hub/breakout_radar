"""Dark terminal-style CSS for the Breakout Radar app."""

DARK_CSS = """
<style>
/* ── Base ─────────────────────────────────────────────────────────────────── */
:root {
    --bg-primary:   #0E1117;
    --bg-secondary: #161B22;
    --bg-card:      #1C2230;
    --accent-green: #00FF88;
    --accent-red:   #FF4444;
    --accent-blue:  #4488FF;
    --accent-amber: #FFB800;
    --text-primary: #E6EDF3;
    --text-muted:   #8B949E;
    --border:       #30363D;
}

html, body, [class*="css"]  {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace, sans-serif;
}

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Metric cards ─────────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, var(--bg-card) 0%, #1a2535 100%);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover { border-color: var(--accent-blue); }
[data-testid="metric-container"] label { color: var(--text-muted) !important; font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: var(--text-primary) !important; font-size: 1.6rem; font-weight: 700; }

/* ── Buttons ──────────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #1a3a5c 0%, #0d2138 100%) !important;
    color: var(--accent-blue) !important;
    border: 1px solid var(--accent-blue) !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: var(--accent-blue) !important;
    color: var(--bg-primary) !important;
    box-shadow: 0 0 12px rgba(68,136,255,0.4) !important;
}

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background-color: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-muted) !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.6rem 1.2rem;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.stTabs [aria-selected="true"] {
    color: var(--accent-blue) !important;
    border-bottom-color: var(--accent-blue) !important;
    background-color: transparent !important;
}

/* ── Dataframe ────────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 8px; overflow: hidden; }

/* ── Progress bar ─────────────────────────────────────────────────────────── */
.stProgress > div > div { background-color: var(--accent-green) !important; }

/* ── Selectbox / multiselect ─────────────────────────────────────────────── */
[data-baseweb="select"] { background-color: var(--bg-card) !important; border-color: var(--border) !important; }
[data-baseweb="select"] * { color: var(--text-primary) !important; }
[data-baseweb="popover"] { background-color: var(--bg-card) !important; }

/* ── Expander ─────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background-color: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
}

/* ── Score badge ──────────────────────────────────────────────────────────── */
.bps-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-weight: 700;
    font-size: 0.85rem;
}
.bps-green  { background: rgba(0,255,136,0.15); color: #00FF88; border: 1px solid #00FF88; }
.bps-yellow { background: rgba(255,184,0,0.15);  color: #FFB800; border: 1px solid #FFB800; }
.bps-red    { background: rgba(255,68,68,0.15);   color: #FF4444; border: 1px solid #FF4444; }

/* ── Section headings ─────────────────────────────────────────────────────── */
.section-header {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 6px;
    margin-bottom: 12px;
}

/* ── Terminal banner ──────────────────────────────────────────────────────── */
.terminal-banner {
    background: linear-gradient(90deg, #0E1117 0%, #1a2535 50%, #0E1117 100%);
    border: 1px solid var(--accent-blue);
    border-radius: 8px;
    padding: 12px 20px;
    margin-bottom: 16px;
    font-size: 0.78rem;
    color: var(--accent-blue);
    letter-spacing: 0.06em;
}

/* ── Slider ───────────────────────────────────────────────────────────────── */
[data-testid="stSlider"] [data-baseweb="slider"] [data-testid*="thumb"] {
    background-color: var(--accent-blue) !important;
}

/* ── Scrollbar ────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-secondary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }
</style>
"""


def inject_css() -> None:
    import streamlit as st
    st.markdown(DARK_CSS, unsafe_allow_html=True)


PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="#E6EDF3", family="JetBrains Mono, monospace"),
        xaxis=dict(gridcolor="#30363D", linecolor="#30363D", zerolinecolor="#30363D"),
        yaxis=dict(gridcolor="#30363D", linecolor="#30363D", zerolinecolor="#30363D"),
        legend=dict(bgcolor="#161B22", bordercolor="#30363D", borderwidth=1),
        margin=dict(l=50, r=20, t=40, b=40),
    )
)
