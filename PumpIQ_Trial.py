"""
Equipment Reliability Dashboard
================================
Weibull analysis for process/equipment model combinations.
Pulls failure + currently-running data from cloud CMMS on load.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from lifelines import WeibullFitter
from datetime import datetime

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Equipment Reliability",
    page_icon="🔧",
    layout="wide",
)

# ─────────────────────────────────────────────
# DATA LOADING  ← replace with your CMMS connection
# ─────────────────────────────────────────────
def load_data():
    df = pd.read_csv("pump_reliability_data.csv")
    df["event_observed"] = (df["status"] == "Failed").astype(int)
    df = df.rename(columns={"pump_runtime_hrs": "duration",
                             "pump_make": "equipment_make",
                             "pump_model": "equipment_model"})
    return df, datetime.now()


# ─────────────────────────────────────────────
# WEIBULL FITTING
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def fit_all_combos(df_json: str) -> pd.DataFrame:
    """Fit a Weibull model for every process/equipment_model combination."""
    df = pd.read_json(df_json, orient="split")
    results = []

    for (process, model), group in df.groupby(["process", "equipment_model"]):
        n_failures = group["event_observed"].sum()
        n_total = len(group)

        if n_failures < 3:
            results.append({
                "process": process, "equipment_model": model,
                "beta": None, "eta": None, "mttf": None,
                "b10": None, "b50": None, "current_risk_pct": None,
                "n_failures": int(n_failures), "n_total": int(n_total),
                "fit_status": "⚠️ Insufficient data",
            })
            continue

        try:
            wf = WeibullFitter()
            wf.fit(group["duration"], event_observed=group["event_observed"])

            beta = wf.rho_
            eta = wf.lambda_

            from scipy.special import gamma as gamma_fn
            mttf_true = eta * gamma_fn(1 + 1 / beta)

            b10 = float(wf.percentile(0.90))
            b50 = float(wf.percentile(0.50))

            running = group[group["event_observed"] == 0]
            if len(running) > 0:
                avg_age = running["duration"].mean()
                risk_pct = float((1 - wf.survival_function_at_times([avg_age]).values[0]) * 100)
            else:
                risk_pct = None

            if beta < 1:
                failure_mode = "⚠️ Infant Mortality"
            elif beta < 1.5:
                failure_mode = "🎲 Random"
            else:
                failure_mode = "⏳ Wear-out"

            results.append({
                "process": process, "equipment_model": model,
                "beta": round(beta, 3), "eta": round(eta, 1),
                "mttf": round(mttf_true, 1), "b10": round(b10, 1), "b50": round(b50, 1),
                "current_risk_pct": round(risk_pct, 1) if risk_pct is not None else None,
                "n_failures": int(n_failures), "n_total": int(n_total),
                "failure_mode": failure_mode, "fit_status": "✅ OK",
                "_wf_lambda": eta, "_wf_rho": beta,
            })

        except Exception as e:
            results.append({
                "process": process, "equipment_model": model,
                "beta": None, "eta": None, "mttf": None,
                "b10": None, "b50": None, "current_risk_pct": None,
                "n_failures": int(n_failures), "n_total": int(n_total),
                "fit_status": f"❌ Error: {e}",
            })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# PLOTTING HELPERS
# ─────────────────────────────────────────────
def plot_survival_and_hazard(beta, eta, b10, b50, mttf, combo_label):
    t = np.linspace(0.01, eta * 3, 500)
    survival = np.exp(-((t / eta) ** beta))
    hazard = (beta / eta) * ((t / eta) ** (beta - 1))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t, y=survival * 100,
        name="Survival %", line=dict(color="#00C9A7", width=2.5),
        hovertemplate="t=%{x:.0f}<br>Survival=%{y:.1f}%<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=t, y=hazard,
        name="Hazard Rate", line=dict(color="#FF6B6B", width=2, dash="dot"),
        yaxis="y2",
        hovertemplate="t=%{x:.0f}<br>Hazard=%{y:.4f}<extra></extra>",
    ))

    for val, label, color in [(b10, "B10", "#FFD93D"), (b50, "B50", "#6BCB77"), (mttf, "MTTF", "#4D96FF")]:
        if val is not None:
            fig.add_vline(x=val, line_dash="dash", line_color=color, opacity=0.6,
                          annotation_text=label, annotation_position="top right")

    fig.update_layout(
        title=f"Reliability Analysis — {combo_label}",
        xaxis_title="Time in Service",
        yaxis_title="Survival Probability (%)",
        yaxis2=dict(title="Hazard Rate", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=420,
        margin=dict(t=60, b=40, l=60, r=60),
        plot_bgcolor="#0f1117",
        paper_bgcolor="#0f1117",
        font=dict(color="#fafafa"),
        xaxis=dict(gridcolor="#2a2a3e"),
        yaxis=dict(gridcolor="#2a2a3e", range=[0, 105]),
    )
    return fig


def plot_weibull_probability(df: pd.DataFrame, beta: float, eta: float, combo_label: str) -> go.Figure:
    """
    Weibull probability plot on linearised axes:
      x-axis: ln(t)  →  displayed as t with log scale
      y-axis: ln(ln(1 / (1 - F(t))))  →  displayed as unreliability %

    Failure points use Bernard's median-rank approximation:
        F(i) = (i - 0.3) / (n + 0.4)
    where n is the total number of observations (failures + suspensions)
    and i is the adjusted rank accounting for suspensions (mean order number).

    The fitted Weibull line is a straight line on these axes by definition.
    """
    failures   = df[df["event_observed"] == 1]["duration"].sort_values().values
    n_failures = len(failures)
    n_total    = len(df)

    if n_failures < 2:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient failure data for probability plot",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(color="#fafafa", size=14))
        fig.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
                          font=dict(color="#fafafa"))
        return fig

    # ── Median rank (Bernard's approximation) ──────────────────────────────
    # Simple version: rank failures directly; suspensions handled implicitly
    # by using n_total in the denominator.
    ranks = np.arange(1, n_failures + 1)
    # Adjusted median rank using total sample size (accounts for censoring simply)
    F_i = (ranks - 0.3) / (n_total + 0.4)
    F_i = np.clip(F_i, 1e-6, 1 - 1e-6)

    # ── Linearised Weibull transforms ──────────────────────────────────────
    # y = ln(-ln(1 - F))  — the Weibull double-log transform
    y_points = np.log(-np.log(1 - F_i))
    x_points = np.log(failures)   # ln(t)

    # ── Fitted line over a wide time range ─────────────────────────────────
    t_line = np.logspace(np.log10(max(failures.min() * 0.1, 0.1)),
                         np.log10(failures.max() * 5), 300)
    F_line = 1 - np.exp(-((t_line / eta) ** beta))
    F_line = np.clip(F_line, 1e-6, 1 - 1e-6)
    y_line = np.log(-np.log(1 - F_line))
    x_line = np.log(t_line)

    # ── Y-axis: convert double-log back to unreliability % for display ──────
    # We'll display tick labels as unreliability percentages
    unreliability_ticks = [1, 5, 10, 20, 30, 50, 63.2, 80, 90, 95, 99]
    y_tick_vals  = [np.log(-np.log(1 - p / 100)) for p in unreliability_ticks]
    y_tick_texts = [f"{p}%" for p in unreliability_ticks]

    # ── X-axis: nice time ticks ─────────────────────────────────────────────
    t_min = max(failures.min() * 0.5, 1)
    t_max = failures.max() * 3
    exp_min = int(np.floor(np.log10(t_min)))
    exp_max = int(np.ceil(np.log10(t_max)))
    x_tick_vals_raw = []
    for e in range(exp_min, exp_max + 1):
        for m in [1, 2, 5]:
            v = m * 10 ** e
            if t_min * 0.8 <= v <= t_max * 1.2:
                x_tick_vals_raw.append(v)
    x_tick_vals  = [np.log(v) for v in x_tick_vals_raw]
    x_tick_texts = [f"{int(v):,}" if v >= 1 else f"{v:.2f}" for v in x_tick_vals_raw]

    # ── Hover text: show original t and F% ─────────────────────────────────
    hover_texts = [
        f"t = {t:,.0f}<br>Unreliability = {f*100:.1f}%"
        for t, f in zip(failures, F_i)
    ]

    fig = go.Figure()

    # Fitted line
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode="lines",
        name=f"Weibull fit (β={beta:.2f}, η={eta:,.0f})",
        line=dict(color="#4D96FF", width=2.5),
        hovertemplate="t=%{customdata:,.0f}<br>Unreliability=%{text}<extra></extra>",
        customdata=t_line,
        text=[f"{(1-np.exp(-((t/eta)**beta)))*100:.1f}%" for t in t_line],
    ))

    # Observed failure points
    fig.add_trace(go.Scatter(
        x=x_points, y=y_points,
        mode="markers",
        name="Observed failures",
        marker=dict(color="#FF6B6B", size=9, symbol="circle",
                    line=dict(color="#fafafa", width=1)),
        text=hover_texts,
        hovertemplate="%{text}<extra></extra>",
    ))

    # η reference line (63.2% unreliability — always passes through eta on Weibull paper)
    y_632 = np.log(-np.log(1 - 0.632))
    fig.add_hline(y=y_632, line_dash="dot", line_color="#FFD93D", opacity=0.5,
                  annotation_text="63.2% (η)", annotation_position="right")
    fig.add_vline(x=np.log(eta), line_dash="dot", line_color="#FFD93D", opacity=0.5,
                  annotation_text=f"η={eta:,.0f}", annotation_position="top left")

    fig.update_layout(
        title=f"Weibull Probability Plot — {combo_label}",
        xaxis=dict(
            title="Time in Service",
            tickvals=x_tick_vals,
            ticktext=x_tick_texts,
            gridcolor="#2a2a3e",
            zeroline=False,
        ),
        yaxis=dict(
            title="Unreliability F(t)",
            tickvals=y_tick_vals,
            ticktext=y_tick_texts,
            gridcolor="#2a2a3e",
            zeroline=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=480,
        margin=dict(t=70, b=50, l=80, r=80),
        plot_bgcolor="#0f1117",
        paper_bgcolor="#0f1117",
        font=dict(color="#fafafa"),
        hovermode="closest",
    )
    return fig


# ─────────────────────────────────────────────
# STYLING HELPERS
# ─────────────────────────────────────────────
def style_risk(val):
    if val is None or pd.isna(val):
        return ""
    if val >= 70:
        return "background-color: #5c1a1a; color: #ff6b6b; font-weight: bold"
    elif val >= 40:
        return "background-color: #4a3a00; color: #FFD93D"
    else:
        return "background-color: #0d3320; color: #6BCB77"


# ─────────────────────────────────────────────
# REPLACEMENT COSTS
# ─────────────────────────────────────────────
REPLACEMENT_COST = {
    "B500":  6000,
    "B1500": 12000,
    "B2500": 14000,
    "E1000": 10000,
    "E2000": 15000,
    "E3000": 18000,
}

# BM penalty constants (on top of the shared rebuild cost)
BM_DOWNTIME_HRS  = 8
BM_DOWNTIME_RATE = 500     # $/hr
WAFER_VALUE      = 5_000   # $ per wafer


# ─────────────────────────────────────────────
# 30-DAY COST PROJECTION
# ─────────────────────────────────────────────
def compute_30day_cost_projection(df: pd.DataFrame, results: pd.DataFrame, horizon_hrs: float) -> pd.DataFrame:
    """
    For each running pump, compute the conditional probability of failing
    within the next horizon_hrs hours given it has survived to its current age.

    Conditional failure prob: P(fail in [t, t+h] | survived to t)
        = 1 - S(t + h) / S(t)
        = 1 - exp( -(t+h/η)^β + (t/η)^β )

    Expected cost per pump = conditional_prob × replacement_cost
    Sum across all running pumps per process/model combo.
    """
    running = df[df["event_observed"] == 0].copy()
    valid_fits = results[results["fit_status"] == "✅ OK"][
        ["process", "equipment_model", "_wf_rho", "_wf_lambda"]
    ]

    rows = []
    for _, fit in valid_fits.iterrows():
        process   = fit["process"]
        model     = fit["equipment_model"]
        beta      = fit["_wf_rho"]
        eta       = fit["_wf_lambda"]
        cost      = REPLACEMENT_COST.get(model, 0)

        pumps = running[
            (running["process"] == process) &
            (running["equipment_model"] == model)
        ]
        n_running = len(pumps)
        if n_running == 0:
            continue

        t = pumps["duration"].values

        # Weibull survival: S(t) = exp(-(t/η)^β)
        S_t     = np.exp(-((t       / eta) ** beta))
        S_t_h   = np.exp(-(((t + horizon_hrs) / eta) ** beta))

        # Conditional probability of failing in next horizon_hrs
        cond_prob = np.where(S_t > 0, 1 - S_t_h / S_t, 1.0)

        expected_failures = float(cond_prob.sum())
        expected_cost     = expected_failures * cost

        rows.append({
            "process":            process,
            "equipment_model":    model,
            "replacement_cost":   cost,
            "n_running":          n_running,
            "expected_failures":  round(expected_failures, 2),
            "expected_cost":      round(expected_cost, 2),
        })

    return pd.DataFrame(rows).sort_values("expected_cost", ascending=False)


def plot_cost_projection(cost_df: pd.DataFrame, horizon_days: int = 30) -> go.Figure:
    """Horizontal bar chart of expected rebuild cost by process/model."""
    labels = cost_df["process"] + " / " + cost_df["equipment_model"]
    costs  = cost_df["expected_cost"]

    colors = [
        "#ff6b6b" if c >= 50000 else "#FFD93D" if c >= 20000 else "#00C9A7"
        for c in costs
    ]

    fig = go.Figure(go.Bar(
        x=costs,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"${c:,.0f}" for c in costs],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Expected Cost: $%{x:,.0f}<extra></extra>",
    ))

    fig.update_layout(
        title=f"Expected Rebuild Cost — Next {horizon_days} Days",
        xaxis_title="Expected Cost ($)",
        xaxis_tickprefix="$",
        xaxis_tickformat=",",
        yaxis=dict(autorange="reversed"),
        height=max(300, len(cost_df) * 35 + 80),
        margin=dict(t=50, b=40, l=180, r=100),
        plot_bgcolor="#0f1117",
        paper_bgcolor="#0f1117",
        font=dict(color="#fafafa"),
        xaxis_gridcolor="#2a2a3e",
    )
    return fig


# ─────────────────────────────────────────────
# PM COST-RISK OPTIMIZATION
# ─────────────────────────────────────────────
def _cycle_len_grid(T_grid: np.ndarray, beta: float, eta: float) -> np.ndarray:
    """
    Expected cycle length E[min(X, T)] = integral_0^T R(t) dt for each T in T_grid.
    Uses a single quad call for [0, T_grid[0]] then cumulative trapezoid from there.
    """
    from scipy.integrate import quad
    R_vals = np.exp(-((T_grid / eta) ** beta))
    offset, _ = quad(lambda t: np.exp(-((t / eta) ** beta)), 0.0, float(T_grid[0]))
    dx   = np.diff(T_grid)
    trap = 0.5 * (R_vals[:-1] + R_vals[1:]) * dx
    return np.concatenate([[offset], offset + np.cumsum(trap)])


@st.cache_data
def compute_pm_cost_curve(
    beta: float, eta: float, c_pm: float, c_bm: float, n_pts: int = 600
) -> tuple:
    """
    Age-replacement cost-rate model.

    C(T) = [c_pm * R(T) + c_bm * F(T)] / integral_0^T R(t) dt

    Returns (T_grid, cost_rates, T_star, C_star, T_lo, T_hi, c_bm_only_rate).
    T_lo/T_hi bound the acceptable window where C(T) <= 1.10 * C_star.
    """
    from scipy.optimize import minimize_scalar
    from scipy.integrate import quad
    from scipy.special import gamma as gamma_fn

    T_grid = np.linspace(eta * 0.02, eta * 4.0, n_pts)
    R_grid = np.exp(-((T_grid / eta) ** beta))
    F_grid = 1.0 - R_grid
    cycle_lens = _cycle_len_grid(T_grid, beta, eta)

    cost_rates = np.where(
        cycle_lens > 1e-9,
        (c_pm * R_grid + c_bm * F_grid) / cycle_lens,
        np.inf,
    )

    # BM-only baseline: no PM ever, avg cycle = MTTF
    mttf = eta * gamma_fn(1.0 + 1.0 / beta)
    c_bm_only_rate = c_bm / mttf

    # Precise optimum via bounded scalar minimization
    def _rate(T):
        R = float(np.exp(-((T / eta) ** beta)))
        cyc, _ = quad(lambda t: np.exp(-((t / eta) ** beta)), 0.0, T)
        return (c_pm * R + c_bm * (1.0 - R)) / max(cyc, 1e-9)

    opt    = minimize_scalar(_rate, bounds=(eta * 0.02, eta * 4.0), method="bounded")
    T_star = float(opt.x)
    C_star = float(opt.fun)

    # Acceptable window: contiguous region around T* where cost rate <= 1.10 * C_star
    threshold = C_star * 1.10
    mask      = cost_rates <= threshold
    if mask.any():
        idx_star = int(np.argmin(np.abs(T_grid - T_star)))
        idx_lo   = idx_star
        while idx_lo > 0 and mask[idx_lo - 1]:
            idx_lo -= 1
        idx_hi = idx_star
        while idx_hi < len(T_grid) - 1 and mask[idx_hi + 1]:
            idx_hi += 1
        T_lo, T_hi = float(T_grid[idx_lo]), float(T_grid[idx_hi])
    else:
        T_lo = T_hi = T_star

    return T_grid, cost_rates, T_star, C_star, T_lo, T_hi, c_bm_only_rate


def plot_pm_cost_curve(
    T_grid, cost_rates, T_star, C_star, T_lo, T_hi, c_bm_only_rate, combo_label
):
    finite_mask = np.isfinite(cost_rates)
    y_min = cost_rates[finite_mask].min() if finite_mask.any() else 0
    y_max = min(c_bm_only_rate * 2.5, cost_rates[finite_mask].max()) if finite_mask.any() else c_bm_only_rate * 2

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=T_grid[finite_mask], y=cost_rates[finite_mask],
        name="PM Cost Rate ($/hr)",
        line=dict(color="#4D96FF", width=2.5),
        hovertemplate="Interval=%{x:,.0f} hrs<br>Cost Rate=$%{y:.4f}/hr<extra></extra>",
    ))

    fig.add_hline(
        y=c_bm_only_rate, line_dash="dash", line_color="#FF6B6B", opacity=0.85,
        annotation_text="BM-only (run-to-failure)", annotation_position="right",
    )

    fig.add_vrect(
        x0=T_lo, x1=T_hi, fillcolor="#6BCB77", opacity=0.13,
        layer="below", line_width=0,
    )
    mid_window = (T_lo + T_hi) / 2
    fig.add_annotation(
        x=mid_window, y=y_min * 0.97,
        text="Optimal Window (≤+10%)", font=dict(color="#6BCB77", size=11),
        showarrow=False, yref="y",
    )

    fig.add_vline(
        x=T_star, line_dash="dash", line_color="#FFD93D", opacity=0.9,
        annotation_text=f"T* = {T_star:,.0f} hrs", annotation_position="top right",
    )
    fig.add_trace(go.Scatter(
        x=[T_star], y=[C_star],
        mode="markers",
        marker=dict(color="#FFD93D", size=13, symbol="star"),
        name=f"T* = {T_star:,.0f} hrs  (min cost)",
        hovertemplate=f"Optimal T* = {T_star:,.0f} hrs<br>Min Rate = ${C_star:.4f}/hr<extra></extra>",
    ))

    fig.update_layout(
        title=f"PM Interval Cost-Rate Optimization — {combo_label}",
        xaxis_title="PM Interval (hrs)",
        yaxis_title="Expected Cost Rate ($/hr)",
        yaxis=dict(gridcolor="#2a2a3e", range=[y_min * 0.9, y_max * 1.05]),
        height=440,
        margin=dict(t=60, b=40, l=90, r=80),
        plot_bgcolor="#0f1117",
        paper_bgcolor="#0f1117",
        font=dict(color="#fafafa"),
        xaxis=dict(gridcolor="#2a2a3e"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    st.title("🔧 Equipment Reliability Dashboard")

    # ── Load data ──
    with st.spinner("Pulling data from CMMS..."):
        df, pulled_at = load_data()

    st.caption(
        f"Data loaded: {pulled_at.strftime('%Y-%m-%d %H:%M:%S')}  |  "
        f"{len(df):,} records  |  "
        f"{df['event_observed'].sum():,} failures  |  "
        f"{(~df['event_observed'].astype(bool)).sum():,} censored (still running)"
    )

    # ── Fit models ──
    with st.spinner("Fitting Weibull models..."):
        results = fit_all_combos(df.to_json(orient="split"))

    st.divider()

    # ════════════════════════════════════════
    # TABS
    # ════════════════════════════════════════
    tab_table, tab_plots, tab_cost, tab_pm = st.tabs([
        "📊 Fleet Summary",
        "📈 Reliability Plots",
        "💰 Cost Projection",
        "🔧 PM Optimization",
    ])

    # ────────────────────────────────────────
    # TAB 1: FLEET SUMMARY TABLE
    # ────────────────────────────────────────
    with tab_table:
        st.subheader("Fleet Summary — All Process / Model Combinations")

        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            filter_process = st.multiselect(
                "Filter by Process",
                options=sorted(results["process"].unique()),
                key="tbl_process",
            )
        with col2:
            filter_mode = st.multiselect(
                "Filter by Failure Mode",
                options=["⚠️ Infant Mortality", "🎲 Random", "⏳ Wear-out"],
                key="tbl_mode",
            )
        with col3:
            min_failures = st.slider("Min failures to show", 0, 20, 3, key="tbl_min_fail")

        display = results.copy()
        if filter_process:
            display = display[display["process"].isin(filter_process)]
        if filter_mode:
            display = display[display["failure_mode"].isin(filter_mode)]
        display = display[display["n_failures"] >= min_failures]

        table_cols = ["process", "equipment_model", "failure_mode", "beta", "eta",
                      "mttf", "b10", "b50", "current_risk_pct", "n_failures", "n_total", "fit_status"]
        display_table = display[table_cols].rename(columns={
            "process": "Process", "equipment_model": "Model",
            "failure_mode": "Mode", "beta": "β (shape)", "eta": "η (scale)",
            "mttf": "MTTF", "b10": "B10 Life", "b50": "B50 Life",
            "current_risk_pct": "Fleet Risk %", "n_failures": "Failures",
            "n_total": "Total Units", "fit_status": "Status",
        }).copy()

        styled = display_table.style.applymap(style_risk, subset=["Fleet Risk %"])
        st.dataframe(
            styled,
            use_container_width=True,
            height=420,
            column_config={
                "β (shape)":    st.column_config.NumberColumn(format="%.2f"),
                "η (scale)":    st.column_config.NumberColumn(format="%.0f"),
                "MTTF":         st.column_config.NumberColumn(format="%.0f"),
                "B10 Life":     st.column_config.NumberColumn(format="%.0f"),
                "B50 Life":     st.column_config.NumberColumn(format="%.0f"),
                "Fleet Risk %": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        st.caption(f"Showing {len(display)} of {len(results)} combinations")

    # ────────────────────────────────────────
    # TAB 2: RELIABILITY PLOTS
    # ────────────────────────────────────────
    with tab_plots:
        st.subheader("Detailed Reliability Plots")

        valid = results[results["fit_status"] == "✅ OK"]

        col_a, col_b = st.columns(2)
        with col_a:
            sel_process = st.selectbox(
                "Select Process", sorted(valid["process"].unique()), key="plt_process"
            )
        with col_b:
            models_for_process = sorted(
                valid[valid["process"] == sel_process]["equipment_model"].unique()
            )
            sel_model = st.selectbox(
                "Select Equipment Model", models_for_process, key="plt_model"
            )

        row = valid[
            (valid["process"] == sel_process) & (valid["equipment_model"] == sel_model)
        ]

        if row.empty:
            st.warning("No valid Weibull fit for this combination.")
        else:
            row = row.iloc[0]
            combo_label = f"{sel_process} / {sel_model}"

            # ── Key metrics ──
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("β (Shape)", f"{row['beta']:.2f}",
                      help="<1 infant mortality | 1 random | >1 wear-out")
            m2.metric("η (Scale)", f"{row['eta']:,.0f} hrs")
            m3.metric("MTTF", f"{row['mttf']:,.0f} hrs")
            m4.metric("B10 Life", f"{row['b10']:,.0f} hrs",
                      help="10% of units failed by this time")
            m5.metric("Fleet Risk", f"{row['current_risk_pct']:.1f}%",
                      help="Avg probability a currently-running unit has failed by its current age")

            st.caption(
                f"Failure mode: {row.get('failure_mode', '')}  |  "
                f"Based on {row['n_failures']} failures, {row['n_total']} total units"
            )

            # ── Two plots side by side ──
            left_col, right_col = st.columns(2)

            with left_col:
                st.markdown("##### Survival & Hazard")
                fig_sh = plot_survival_and_hazard(
                    beta=row["_wf_rho"], eta=row["_wf_lambda"],
                    b10=row["b10"], b50=row["b50"], mttf=row["mttf"],
                    combo_label=combo_label,
                )
                st.plotly_chart(fig_sh, use_container_width=True)

            with right_col:
                st.markdown("##### Weibull Probability Plot")
                # Filter raw data for the selected combo
                combo_df = df[
                    (df["process"] == sel_process) &
                    (df["equipment_model"] == sel_model)
                ]
                fig_wp = plot_weibull_probability(
                    df=combo_df,
                    beta=row["_wf_rho"],
                    eta=row["_wf_lambda"],
                    combo_label=combo_label,
                )
                st.plotly_chart(fig_wp, use_container_width=True)

            with st.expander("ℹ️ How to interpret these plots"):
                st.markdown("""
**Survival & Hazard (left)**

| β value | Failure pattern | Maintenance implication |
|---|---|---|
| β < 1 | Infant mortality — failures decrease over time | Burn-in / screening recommended |
| β ≈ 1 | Random failures — constant hazard rate | Corrective maintenance; PM won't help much |
| β > 1 | Wear-out — failures increase over time | Preventive maintenance intervals make sense |
| β > 3 | Rapid wear-out | Tight, consistent PM schedule needed |

**Weibull Probability Plot (right)**

Each red dot represents a **single failure event**, ranked by time-to-failure.
Unreliability F(t) is estimated using Bernard's median rank: `(i − 0.3) / (n + 0.4)`,
where *n* is the total sample size (failures + suspensions) to account for censoring.

- Points lying **on or near the blue line** indicate the Weibull distribution fits the data well.
- **Systematic curvature** may suggest a mixed failure population or a three-parameter Weibull.
- The **dashed yellow lines** mark η (the characteristic life), where exactly 63.2% of the
  population is expected to have failed.
                """)

    # ────────────────────────────────────────
    # TAB 3: COST PROJECTION
    # ────────────────────────────────────────
    with tab_cost:
        st.subheader("Projected Rebuild Cost — Next 30 Days")

        c1, c2 = st.columns([3, 2])
        with c1:
            horizon_days = st.slider(
                "Projection horizon (days)", min_value=1, max_value=90, value=30,
                key="cost_horizon_days",
            )
        with c2:
            hours_per_day = st.number_input(
                "Equipment hours per day", min_value=1, max_value=24, value=24,
                help="How many hours per day does your equipment run?",
                key="cost_hrs_day",
            )
        horizon_hrs = horizon_days * hours_per_day

        st.caption(
            f"Projection horizon: **{horizon_hrs:,} hours** "
            f"({horizon_days} days × {hours_per_day} hrs/day)"
        )

        cost_df = compute_30day_cost_projection(df, results, horizon_hrs)

        if cost_df.empty:
            st.warning(
                "No cost projection available — check that Weibull fits completed successfully."
            )
        else:
            # ── Top-line metrics ──
            total_cost     = cost_df["expected_cost"].sum()
            total_failures = cost_df["expected_failures"].sum()
            total_running  = cost_df["n_running"].sum()
            worst_combo    = cost_df.iloc[0]

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Total Projected Cost",    f"${total_cost:,.0f}")
            mc2.metric(f"Expected Failures ({horizon_days}d)", f"{total_failures:.1f}")
            mc3.metric("Running Pumps at Risk",   f"{total_running:,}")
            mc4.metric(
                "Highest Risk Combo",
                f"{worst_combo['process']} / {worst_combo['equipment_model']}",
                f"${worst_combo['expected_cost']:,.0f}",
            )

        st.plotly_chart(plot_cost_projection(cost_df, horizon_days), use_container_width=True)
        with st.expander("📋 Full cost breakdown by combo"):
                cost_display = cost_df.rename(columns={
                    "process":           "Process",
                    "equipment_model":   "Model",
                    "replacement_cost":  "Unit Cost ($)",
                    "n_running":         "Running Pumps",
                    "expected_failures": "Exp. Failures",
                    "expected_cost":     "Exp. Cost ($)",
                })
                st.dataframe(
                    cost_display,
                    use_container_width=True,
                    column_config={
                        "Unit Cost ($)":  st.column_config.NumberColumn(format="$%.0f"),
                        "Exp. Failures":  st.column_config.NumberColumn(format="%.2f"),
                        "Exp. Cost ($)":  st.column_config.NumberColumn(format="$%.0f"),
                    },
                )

        with st.expander("ℹ️ How is this calculated?"):
                st.markdown(rf"""
For each **currently-running** pump, we know its age (current runtime hours).
Using the fitted Weibull curve for its process/model combo, we calculate the
**conditional probability** of it failing in the next **{horizon_days} days**, given it has
already survived to its current age:

$$P(\text{{fail in next }} h \text{{ hrs}} \mid \text{{survived to }} t) = 1 - \frac{{S(t+h)}}{{S(t)}}$$

Where $S(t) = e^{{-(t/\eta)^{{\beta}}}}$ is the Weibull survival function.

Each pump's failure probability is multiplied by its **replacement cost**,
then summed across all running pumps in each combo to give the **expected cost**.
This is not a worst-case estimate — it is a statistically expected value.
                """)

    # ────────────────────────────────────────
    # TAB 4: PM OPTIMIZATION
    # ────────────────────────────────────────
    with tab_pm:
        st.subheader("Optimal PM Interval — Cost-Risk Optimization")

        valid_pm = results[results["fit_status"] == "✅ OK"]

        if valid_pm.empty:
            st.warning("No valid Weibull fits available.")
        else:
            col_pa, col_pb = st.columns(2)
            with col_pa:
                pm_process = st.selectbox(
                    "Select Process", sorted(valid_pm["process"].unique()), key="pm_process"
                )
            with col_pb:
                pm_models = sorted(
                    valid_pm[valid_pm["process"] == pm_process]["equipment_model"].unique()
                )
                pm_model = st.selectbox(
                    "Select Equipment Model", pm_models, key="pm_model"
                )

            pm_row = valid_pm[
                (valid_pm["process"] == pm_process) &
                (valid_pm["equipment_model"] == pm_model)
            ]

            if pm_row.empty:
                st.warning("No valid Weibull fit for this combination.")
            else:
                pm_row = pm_row.iloc[0]
                beta        = float(pm_row["_wf_rho"])
                eta         = float(pm_row["_wf_lambda"])
                c_pm        = float(REPLACEMENT_COST.get(pm_model, 0))
                combo_label = f"{pm_process} / {pm_model}"

                if c_pm == 0:
                    st.warning(
                        f"No rebuild cost defined for model **{pm_model}**. "
                        "Add it to `REPLACEMENT_COST` to enable optimization."
                    )
                else:
                    st.divider()
                    st.markdown("#### BM Risk Parameters")
                    st.caption(
                        f"PM part cost = **${c_pm:,.0f}** (same rebuild as BM). "
                        "BM adds unplanned downtime and potential wafer loss on top."
                    )

                    bm_col1, bm_col2, bm_col3 = st.columns(3)
                    with bm_col1:
                        st.metric(
                            "BM Downtime Cost",
                            f"${BM_DOWNTIME_HRS * BM_DOWNTIME_RATE:,}",
                            help=f"{BM_DOWNTIME_HRS} hrs × ${BM_DOWNTIME_RATE:,}/hr (fixed)",
                        )
                    with bm_col2:
                        wafer_risk_pct = st.number_input(
                            "Wafer scrap risk per BM event (%)",
                            min_value=0.0, max_value=100.0, value=20.0, step=5.0,
                            key="pm_wafer_risk",
                        )
                    with bm_col3:
                        wafer_qty = st.number_input(
                            "Wafers at risk per BM event",
                            min_value=0, max_value=125, value=4, step=1,
                            key="pm_wafer_qty",
                        )

                    wafer_risk_cost = (wafer_risk_pct / 100.0) * wafer_qty * WAFER_VALUE
                    bm_extra        = BM_DOWNTIME_HRS * BM_DOWNTIME_RATE + wafer_risk_cost
                    c_bm            = c_pm + bm_extra

                    st.info(
                        f"**BM total cost** = ${c_pm:,.0f} (parts) "
                        f"+ ${BM_DOWNTIME_HRS * BM_DOWNTIME_RATE:,} (downtime) "
                        f"+ ${wafer_risk_cost:,.0f} (wafer risk at {wafer_risk_pct:.0f}% × {wafer_qty} wafers × $5k) "
                        f"= **${c_bm:,.0f}**  |  "
                        f"PM avoids **${bm_extra:,.0f}** in extra BM costs per avoided failure"
                    )

                    st.divider()

                    if beta <= 1.0:
                        st.warning(
                            f"**β = {beta:.2f} — age-based PM optimization is not applicable.** "
                            "Failure rate is constant (β ≈ 1) or decreasing (β < 1); replacing "
                            "equipment before failure does not reduce failure probability. "
                            "Focus on root-cause analysis or incoming inspection instead."
                        )
                    else:
                        with st.spinner("Computing optimal PM interval..."):
                            T_grid, cost_rates, T_star, C_star, T_lo, T_hi, c_bm_only_rate = \
                                compute_pm_cost_curve(beta, eta, c_pm, c_bm)

                        n_running = int((
                            (df["event_observed"] == 0) &
                            (df["process"] == pm_process) &
                            (df["equipment_model"] == pm_model)
                        ).sum())

                        savings_per_unit  = max((c_bm_only_rate - C_star) * 8760, 0)
                        savings_fleet     = savings_per_unit * n_running

                        pm1, pm2, pm3, pm4, pm5 = st.columns(5)
                        pm1.metric(
                            "Optimal PM Interval", f"{T_star:,.0f} hrs",
                            help="Minimises expected cost per operating hour",
                        )
                        pm2.metric(
                            "Acceptable Window (≤+10%)",
                            f"{T_lo:,.0f} – {T_hi:,.0f} hrs",
                            help="Any interval in this range stays within 10% of the theoretical minimum cost",
                        )
                        pm3.metric(
                            "Annual Savings / Unit",
                            f"${savings_per_unit:,.0f}",
                            help="Per-pump savings over 8,760 hrs at optimal PM interval vs run-to-failure",
                        )
                        pm4.metric(
                            f"Fleet Savings / Year ({n_running} running)",
                            f"${savings_fleet:,.0f}",
                            help=f"Per-unit savings × {n_running} currently-running pumps in this combo",
                        )
                        pm5.metric(
                            "Min Cost Rate",
                            f"${C_star:.3f}/hr",
                            delta=f"${C_star - c_bm_only_rate:.3f}/hr vs BM-only",
                            delta_color="inverse",
                            help="Expected cost per operating hour at optimal PM interval",
                        )

                        st.plotly_chart(
                            plot_pm_cost_curve(
                                T_grid, cost_rates, T_star, C_star, T_lo, T_hi,
                                c_bm_only_rate, combo_label,
                            ),
                            use_container_width=True,
                        )

                        with st.expander("ℹ️ How is the optimal interval calculated?"):
                            st.markdown(rf"""
The model minimises the **long-run expected cost per operating hour** for an
age-based replacement policy — replace at scheduled interval T (PM) *or* at failure (BM), whichever comes first.

$$C(T) = \frac{{C_{{PM}} \cdot R(T) \;+\; C_{{BM}} \cdot F(T)}}{{\displaystyle\int_0^T R(t)\,dt}}$$

| Symbol | Meaning | This combo |
|--------|---------|------------|
| $C_{{PM}}$ | Planned replacement cost (rebuild only) | ${c_pm:,.0f} |
| $C_{{BM}}$ | Unplanned failure cost (rebuild + downtime + wafer risk) | ${c_bm:,.0f} |
| $R(T)$ | Weibull survival — probability unit reaches age T | — |
| $F(T)=1-R(T)$ | Failure probability before age T | — |
| $\int_0^T R(t)\,dt$ | Expected cycle length (mean time to replacement) | — |

The **green shaded window** is the range of PM intervals where cost stays within **10% of the minimum** — use this for practical scheduling flexibility.

The **red dashed line** is the cost rate if no PM is performed (run-to-failure). Whenever the blue curve sits below the red line, proactive PM is cost-justified.

> BM downtime assumed at **{BM_DOWNTIME_HRS} hrs × ${BM_DOWNTIME_RATE:,}/hr = ${BM_DOWNTIME_HRS * BM_DOWNTIME_RATE:,}**.
> Wafer value assumed at **${WAFER_VALUE:,}/wafer**. Adjust inputs above to reflect your actual risk exposure.
                            """)

    st.divider()
    st.caption(
        "Weibull fits use Maximum Likelihood Estimation via lifelines. "
        "Censored (still-running) units are correctly handled. "
        "Combos with < 3 failures are excluded from fitting."
    )


if __name__ == "__main__":
    main()