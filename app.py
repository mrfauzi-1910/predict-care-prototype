"""
Predict & Care — Mitra Churn Prediction & Recommendation Dashboard.

Streamlit prototype for Astro Hackathon 2026.
Owner of dashboard: Tim Mitra Management.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from model import train_model, score_mitra, confidence_score, FEATURE_DISPLAY
from recommendations import recommend_for_mitra, combo_recommendation


# =================== PAGE CONFIG ===================
st.set_page_config(
    page_title="Predict & Care · Astro Mitra Management",
    page_icon="🟣",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Theme colors
VIOLET = "#5B21B6"
INK = "#1E1B4B"
AMBER = "#F59E0B"
GREEN = "#10B981"
RED = "#EF4444"
LAVENDER = "#F5F3FF"
MUTED = "#64748B"

TIER_COLOR = {"Green": GREEN, "Yellow": AMBER, "Red": RED}


# Custom CSS for cleaner look
st.markdown(f"""
<style>
    .main {{ padding-top: 1rem; }}
    .stMetric {{
        background: {LAVENDER};
        padding: 16px;
        border-radius: 8px;
        border-left: 4px solid {VIOLET};
    }}
    h1, h2, h3 {{ color: {INK}; }}
    .tier-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        font-size: 0.85em;
    }}
    .driver-pill {{
        display: inline-block;
        background: {LAVENDER};
        color: {INK};
        padding: 4px 10px;
        border-radius: 6px;
        margin: 2px 4px 2px 0;
        font-size: 0.85em;
        font-weight: 500;
    }}
    .action-card {{
        background: white;
        border: 1px solid #E2E8F0;
        border-left: 4px solid {VIOLET};
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }}
    .stButton button {{
        background: {VIOLET};
        color: white;
        border: 0;
        padding: 8px 16px;
        border-radius: 6px;
    }}
</style>
""", unsafe_allow_html=True)


# =================== DATA LOADING ===================
@st.cache_data
def load_data():
    csv_path = Path(__file__).parent / "data" / "mitra.csv"
    if not csv_path.exists():
        st.error(f"Data file not found at {csv_path}. Run `python data_generator.py` first.")
        st.stop()
    return pd.read_csv(csv_path)


@st.cache_resource
def train_and_score(df):
    model, scaler, feature_names, metrics = train_model(df)
    scored = score_mitra(df, model, scaler, feature_names)
    scored["model_confidence"] = confidence_score(scored["churn_prob_pred"].values)
    return model, scaler, feature_names, metrics, scored


df_raw = load_data()
model, scaler, feature_names, metrics, df = train_and_score(df_raw)


# =================== HEADER ===================
col_t1, col_t2 = st.columns([3, 1])
with col_t1:
    st.markdown(
        f"<h1 style='margin-bottom:0;color:{INK};'>Predict & Care</h1>"
        f"<p style='color:{MUTED};margin-top:0;'>Mitra Churn Prediction & Recommendation — owned by Tim Mitra Management</p>",
        unsafe_allow_html=True,
    )
with col_t2:
    st.markdown(
        f"<div style='text-align:right;color:{MUTED};font-size:0.9em;'>"
        f"Astro Hackathon 2026<br>Theme: Mitra Happiness</div>",
        unsafe_allow_html=True,
    )


# =================== SIDEBAR FILTERS ===================
with st.sidebar:
    st.markdown(f"### Filters")
    role_filter = st.multiselect(
        "Role", options=sorted(df["role"].unique()),
        default=sorted(df["role"].unique()),
    )
    hub_filter = st.multiselect(
        "Hub", options=sorted(df["hub"].unique()),
        default=sorted(df["hub"].unique()),
    )
    tier_filter = st.multiselect(
        "Risk Tier", options=["Red", "Yellow", "Green"],
        default=["Red", "Yellow", "Green"],
    )
    tenure_range = st.slider(
        "Tenure (days)",
        int(df["tenure_days"].min()), int(df["tenure_days"].max()),
        (int(df["tenure_days"].min()), int(df["tenure_days"].max())),
    )
    st.divider()
    st.markdown(f"<small style='color:{MUTED};'>Model: Logistic Regression<br>"
                f"Test AUC: <b>{metrics['auc']:.3f}</b><br>"
                f"Train n={metrics['n_train']} · Test n={metrics['n_test']}</small>",
                unsafe_allow_html=True)

# Apply filters
mask = (
    df["role"].isin(role_filter)
    & df["hub"].isin(hub_filter)
    & df["risk_tier"].isin(tier_filter)
    & df["tenure_days"].between(tenure_range[0], tenure_range[1])
)
fdf = df[mask].copy()


# =================== TABS ===================
tab_overview, tab_atrisk, tab_detail, tab_cohort, tab_model = st.tabs([
    "📊 Overview",
    "🚨 At-Risk List",
    "🔍 Mitra Detail",
    "📈 Cohort Survival",
    "🧠 Model Performance",
])


# =================== TAB: OVERVIEW ===================
with tab_overview:
    st.markdown("### Today's snapshot — Mitra Management view")
    st.caption(f"Filtered: {len(fdf)} of {len(df)} mitra")

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    n_red = (fdf["risk_tier"] == "Red").sum()
    n_yellow = (fdf["risk_tier"] == "Yellow").sum()
    n_green = (fdf["risk_tier"] == "Green").sum()
    avg_risk = fdf["churn_prob_pred"].mean() if len(fdf) else 0
    n_new = (fdf["tenure_days"] < 30).sum()

    c1.metric("Total Mitra", f"{len(fdf):,}")
    c2.metric("🔴 Red (urgent)", f"{n_red}", f"{n_red/max(len(fdf),1):.0%}")
    c3.metric("🟡 Yellow (watch)", f"{n_yellow}", f"{n_yellow/max(len(fdf),1):.0%}")
    c4.metric("🟢 Green", f"{n_green}", f"{n_green/max(len(fdf),1):.0%}")
    c5.metric("New joiners (<30d)", f"{n_new}")

    st.divider()

    # 2-column charts
    cl, cr = st.columns(2)

    with cl:
        st.markdown("##### Risk distribution")
        tier_counts = fdf["risk_tier"].value_counts().reindex(["Green", "Yellow", "Red"]).fillna(0)
        fig = go.Figure()
        for tier in ["Green", "Yellow", "Red"]:
            fig.add_trace(go.Bar(
                x=[tier], y=[tier_counts.get(tier, 0)],
                marker_color=TIER_COLOR[tier], name=tier,
                text=[int(tier_counts.get(tier, 0))], textposition="outside",
            ))
        fig.update_layout(
            showlegend=False, height=320,
            margin=dict(l=10, r=10, t=20, b=10),
            yaxis_title="Mitra count", xaxis_title="",
            plot_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        st.markdown("##### Risk by hub")
        hub_tier = fdf.groupby(["hub", "risk_tier"]).size().reset_index(name="count")
        fig2 = px.bar(
            hub_tier, x="hub", y="count", color="risk_tier",
            color_discrete_map=TIER_COLOR,
            category_orders={"risk_tier": ["Green", "Yellow", "Red"]},
        )
        fig2.update_layout(
            height=320, margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="", yaxis_title="Mitra count",
            plot_bgcolor="white", legend_title_text="",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # Risk by role + tenure segment
    cl2, cr2 = st.columns(2)

    with cl2:
        st.markdown("##### Risk by role")
        role_tier = fdf.groupby(["role", "risk_tier"]).size().reset_index(name="count")
        fig3 = px.bar(
            role_tier, x="role", y="count", color="risk_tier",
            color_discrete_map=TIER_COLOR, barmode="stack",
            category_orders={"risk_tier": ["Green", "Yellow", "Red"]},
        )
        fig3.update_layout(
            height=300, margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="", yaxis_title="Mitra count",
            plot_bgcolor="white", legend_title_text="",
        )
        st.plotly_chart(fig3, use_container_width=True)

    with cr2:
        st.markdown("##### Risk vs tenure")
        tdf = fdf.copy()
        tdf["tenure_segment"] = pd.cut(
            tdf["tenure_days"],
            bins=[0, 30, 90, 180, 365, 9999],
            labels=["0-30d", "30-90d", "90-180d", "180-365d", "1y+"],
        )
        seg_risk = tdf.groupby("tenure_segment", observed=True)["churn_prob_pred"].mean().reset_index()
        fig4 = px.bar(
            seg_risk, x="tenure_segment", y="churn_prob_pred",
            color="churn_prob_pred", color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
        )
        fig4.update_layout(
            height=300, margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="Tenure segment", yaxis_title="Avg churn prob",
            plot_bgcolor="white", coloraxis_showscale=False,
        )
        fig4.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig4, use_container_width=True)


# =================== TAB: AT-RISK LIST ===================
with tab_atrisk:
    st.markdown("### Mitra at-risk — sorted by churn probability")
    st.caption("Klik mitra mana saja di tab '🔍 Mitra Detail' untuk drill-down dan lihat recommended actions.")

    display = fdf.sort_values("churn_prob_pred", ascending=False).copy()
    display["Top Driver"] = display["top_drivers"].apply(
        lambda d: d[0]["display_name"] if d else "—"
    )
    display["Risk %"] = (display["churn_prob_pred"] * 100).round(1)
    display["Confidence %"] = (display["model_confidence"] * 100).round(0).astype(int)

    table = display[[
        "mitra_id", "name", "role", "hub", "tenure_days",
        "Risk %", "risk_tier", "Confidence %", "Top Driver",
        "captain_assigned",
    ]].rename(columns={
        "mitra_id": "ID",
        "name": "Nama",
        "role": "Role",
        "hub": "Hub",
        "tenure_days": "Tenure (d)",
        "risk_tier": "Tier",
        "captain_assigned": "Captain?",
    })

    st.dataframe(
        table, use_container_width=True, height=520, hide_index=True,
        column_config={
            "Risk %": st.column_config.ProgressColumn(
                "Risk %", format="%.1f%%", min_value=0, max_value=100,
            ),
            "Confidence %": st.column_config.NumberColumn(
                "Conf %", format="%d%%",
            ),
            "Captain?": st.column_config.NumberColumn(
                "Captain?", help="1 = sudah ada Captain assigned",
            ),
        },
    )


# =================== TAB: MITRA DETAIL ===================
with tab_detail:
    st.markdown("### Drill-down per mitra + recommended actions")

    # Default: pick highest-risk mitra
    mitra_options = fdf.sort_values("churn_prob_pred", ascending=False)
    if len(mitra_options) == 0:
        st.warning("Tidak ada mitra yang match dengan filter saat ini.")
        st.stop()

    label_options = [
        f"{r['mitra_id']} — {r['name']} ({r['role']}, {r['hub']}, "
        f"{r['risk_tier']}, {r['churn_prob_pred']*100:.0f}%)"
        for _, r in mitra_options.iterrows()
    ]
    pick_idx = st.selectbox(
        "Pilih mitra:",
        range(len(label_options)),
        format_func=lambda i: label_options[i],
    )
    mitra = mitra_options.iloc[pick_idx]

    # ============= Mitra summary panel =============
    st.markdown("---")
    cs1, cs2, cs3 = st.columns([2, 1, 1])

    with cs1:
        st.markdown(
            f"<h3 style='margin-bottom:4px;color:{INK};'>"
            f"{mitra['name']} <span style='font-weight:400;color:{MUTED};font-size:0.7em;'>{mitra['mitra_id']}</span></h3>"
            f"<p style='color:{MUTED};margin:0;'>{mitra['role']} · {mitra['hub']} · "
            f"Tenure {int(mitra['tenure_days'])} hari · "
            f"Source: {mitra['source_channel']} · "
            f"Distance: {mitra['distance_to_hub_km']:.1f} km</p>",
            unsafe_allow_html=True,
        )

    with cs2:
        tier = mitra["risk_tier"]
        risk_pct = mitra["churn_prob_pred"] * 100
        tier_color = TIER_COLOR[tier]
        st.markdown(
            f"<div style='background:{tier_color};color:white;padding:10px 14px;"
            f"border-radius:8px;text-align:center;'>"
            f"<div style='font-size:0.85em;opacity:0.9;'>Risk Tier</div>"
            f"<div style='font-size:1.5em;font-weight:700;'>{tier}</div>"
            f"<div style='font-size:1.1em;'>{risk_pct:.0f}% prob</div></div>",
            unsafe_allow_html=True,
        )

    with cs3:
        conf = mitra["model_confidence"] * 100
        st.markdown(
            f"<div style='background:{LAVENDER};color:{INK};padding:10px 14px;"
            f"border-radius:8px;text-align:center;'>"
            f"<div style='font-size:0.85em;opacity:0.7;'>Model Confidence</div>"
            f"<div style='font-size:1.5em;font-weight:700;'>{conf:.0f}%</div>"
            f"<div style='font-size:0.85em;color:{MUTED};'>"
            f"{'High' if conf > 75 else 'Medium' if conf > 55 else 'Low'}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("##### Top 3 Churn Drivers (per mitra)")
    drivers = mitra["top_drivers"]
    if not drivers:
        st.info("Mitra ini tidak punya driver bermasalah — Green tier.")
    else:
        cols = st.columns(len(drivers))
        for i, d in enumerate(drivers):
            with cols[i]:
                value_str = f"{d['value']:.2f}" if d["value"] is not None else "—"
                st.markdown(
                    f"<div style='background:{LAVENDER};padding:14px;border-radius:8px;"
                    f"border-left:3px solid {VIOLET};'>"
                    f"<div style='color:{MUTED};font-size:0.85em;'>#{i+1} Driver</div>"
                    f"<div style='font-weight:600;color:{INK};font-size:1.05em;'>{d['display_name']}</div>"
                    f"<div style='color:{VIOLET};font-size:0.95em;margin-top:4px;'>"
                    f"Value: <b>{value_str}</b></div>"
                    f"<div style='color:{MUTED};font-size:0.8em;margin-top:4px;'>"
                    f"Contribution: {d['contribution']:+.3f}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ============= Recommendations =============
    st.markdown("---")
    st.markdown("### 🎯 Recommended Actions (ranked by ROI × confidence)")

    if drivers:
        recs = recommend_for_mitra(
            mitra_row=mitra.to_dict(),
            top_drivers=drivers,
            risk_prob=float(mitra["churn_prob_pred"]),
            model_confidence=float(mitra["model_confidence"]),
            top_k=4,
        )

        if not recs:
            st.info("Tidak ada action yang match dengan driver mitra ini.")
        else:
            for i, r in enumerate(recs):
                with st.container():
                    cR1, cR2, cR3, cR4 = st.columns([3.5, 1, 1, 1])
                    with cR1:
                        st.markdown(
                            f"<div class='action-card'>"
                            f"<div style='color:{MUTED};font-size:0.8em;'>"
                            f"#{i+1} · {r['category']} · Owner: {r['owner']}</div>"
                            f"<div style='font-weight:600;color:{INK};font-size:1.0em;margin:4px 0;'>"
                            f"{r['label']}</div>"
                            f"<div style='color:{MUTED};font-size:0.8em;'>"
                            f"Targets: {', '.join([FEATURE_DISPLAY.get(f, f) for f in r['matched_drivers']])} · "
                            f"Evidence: n = {r['evidence_n']}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    with cR2:
                        st.metric("Confidence", f"{r['confidence']*100:.0f}%")
                    with cR3:
                        st.metric("Δ Retention", f"+{r['expected_delta_pts']:.0f} pts")
                    with cR4:
                        cost_short = f"Rp {r['cost_rp']/1_000_000:.1f}M" if r['cost_rp'] >= 1_000_000 else \
                                     f"Rp {r['cost_rp']/1000:.0f}k" if r['cost_rp'] > 0 else "Free"
                        st.metric("Cost", cost_short, f"ROI {r['roi']:.1f}x")

            # ============= Combo recommendation =============
            combo = combo_recommendation(recs)
            if combo:
                st.markdown(
                    f"<div style='background:{INK};color:white;padding:16px 20px;"
                    f"border-radius:10px;margin-top:14px;'>"
                    f"<div style='font-size:0.85em;opacity:0.7;'>RECOMMENDED COMBO</div>"
                    f"<div style='font-size:1.1em;font-weight:600;margin:6px 0;'>"
                    f"{combo['labels'][0]}<br>+ {combo['labels'][1]}</div>"
                    f"<div style='font-size:0.95em;'>"
                    f"Combined retention lift: <b style='color:{AMBER};'>+{combo['expected_delta_pts']:.0f} pts</b> · "
                    f"Total cost: <b>Rp {combo['cost_rp']/1_000_000:.2f} jt</b> · "
                    f"Confidence: <b>{combo['confidence']*100:.0f}%</b></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.markdown(" ")
            cb1, cb2, cb3 = st.columns(3)
            with cb1:
                st.button("✅ Approve combo", use_container_width=True)
            with cb2:
                st.button("✏️ Custom action", use_container_width=True)
            with cb3:
                st.button("⏸️ Defer 24h", use_container_width=True)


# =================== TAB: COHORT SURVIVAL ===================
with tab_cohort:
    st.markdown("### Cohort survival curves (illustrative)")
    st.caption(
        "Track survival rate per join-cohort. Red line: actual cohort. "
        "Grey line: historical baseline. Drop below baseline = trigger investigation."
    )

    # Synthetic survival data — cohort by week-of-join (simplified for prototype)
    # Real impl: query from event log
    weeks = list(range(1, 13))
    np.random.seed(7)
    cohorts = ["Cohort W12 (Mar)", "Cohort W14 (Mar-Apr)", "Cohort W16 (Apr)"]
    survival_data = []
    for cohort in cohorts:
        base = np.random.uniform(0.78, 0.86)
        decay = np.random.uniform(0.012, 0.020)
        anomaly_week = np.random.choice([4, 5, 6]) if "W16" in cohort else None
        for w in weeks:
            rate = base * np.exp(-decay * w)
            if anomaly_week and w >= anomaly_week:
                rate *= 0.92  # the anomaly we want to catch
            survival_data.append({"cohort": cohort, "week": w, "survival_rate": rate})

    sdf = pd.DataFrame(survival_data)
    # Baseline = average across all
    baseline = sdf.groupby("week")["survival_rate"].mean().reset_index()
    baseline["cohort"] = "Baseline (historical avg)"

    fig = go.Figure()
    colors = [VIOLET, GREEN, RED]
    for i, cohort in enumerate(cohorts):
        cdata = sdf[sdf["cohort"] == cohort]
        fig.add_trace(go.Scatter(
            x=cdata["week"], y=cdata["survival_rate"],
            mode="lines+markers", name=cohort,
            line=dict(color=colors[i], width=2.5),
        ))
    fig.add_trace(go.Scatter(
        x=baseline["week"], y=baseline["survival_rate"],
        mode="lines", name="Baseline (historical)",
        line=dict(color=MUTED, width=2, dash="dash"),
    ))
    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Weeks since joining", yaxis_title="Survival rate",
        plot_bgcolor="white", legend=dict(orientation="h", y=-0.15),
        hovermode="x unified",
    )
    fig.update_yaxes(tickformat=".0%", range=[0.5, 1.0])
    st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "⚠️ Cohort W16 (Apr) shows degraded survival starting Week 4 — "
        "trigger investigasi: Captain coverage menurun? Onboarding berubah? "
        "Hub manager rotation?"
    )


# =================== TAB: MODEL PERFORMANCE ===================
with tab_model:
    st.markdown("### Model performance & explainability")

    cm1, cm2, cm3 = st.columns(3)
    cm1.metric("Test AUC", f"{metrics['auc']:.3f}")
    cm2.metric("Train n", f"{metrics['n_train']}")
    cm3.metric("Test n", f"{metrics['n_test']}")

    st.markdown("##### Top features driving churn (global)")
    fi = metrics["feature_importance"].head(10).copy()
    fi["abs"] = fi["abs_coef"]
    fig_fi = px.bar(
        fi.iloc[::-1],
        x="abs", y="display_name",
        orientation="h",
        color="coef",
        color_continuous_scale=["#10B981", "#F5F3FF", "#EF4444"],
        color_continuous_midpoint=0,
        labels={"abs": "Importance (|coefficient|)", "display_name": ""},
    )
    fig_fi.update_layout(
        height=420, margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor="white", coloraxis_colorbar=dict(title="Direction"),
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("##### Confusion matrix")
    cm = metrics["confusion_matrix"]
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Negative", "Actual Positive"],
        columns=["Predicted Negative", "Predicted Positive"],
    )
    fig_cm = px.imshow(
        cm_df, text_auto=True, aspect="auto",
        color_continuous_scale=["white", VIOLET],
    )
    fig_cm.update_layout(
        height=300, margin=dict(l=10, r=10, t=10, b=10),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.caption(
        f"Note: synthetic data, model performance metrics indikatif saja. "
        f"Production model dengan data real diharapkan AUC 0.75-0.85."
    )


# =================== FOOTER ===================
st.markdown("---")
st.markdown(
    f"<div style='text-align:center;color:{MUTED};font-size:0.85em;padding:10px;'>"
    f"<b>Predict & Care</b> · Tim Mitra Management · Astro Hackathon 2026 · "
    f"<i>Mitra yang merasa dilihat dan ditemani tidak akan pergi diam-diam.</i>"
    f"</div>",
    unsafe_allow_html=True,
)
