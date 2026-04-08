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
    tab_table, tab_plots, tab_cost = st.tabs([
        "📊 Fleet Summary",
        "📈 Reliability Plots",
        "💰 Cost Projection",
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

    st.divider()
    st.caption(
        "Weibull fits use Maximum Likelihood Estimation via lifelines. "
        "Censored (still-running) units are correctly handled. "
        "Combos with < 3 failures are excluded from fitting."
    )


if __name__ == "__main__":
    main()