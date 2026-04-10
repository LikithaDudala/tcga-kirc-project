"""
TCGA KIRC Survival Analysis — Streamlit Dashboard
==================================================
Single-page interactive dashboard for exploring the TCGA Kidney Renal
Clear Cell Carcinoma (KIRC) survival analysis pipeline results.

Sections:
  1. Dataset Overview       — cohort statistics & demographics
  2. Kaplan-Meier Curves    — overall and by cancer stage
  3. Model Performance      — C-index comparison & Brier score
  4. Gene Importance        — LASSO Cox & RSF top biomarkers
  5. Risk Prediction Demo   — clinical input → predicted risk score

Run with:  streamlit run app.py
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

warnings.filterwarnings("ignore")

# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="TCGA KIRC Survival Analysis",
    page_icon="https://img.icons8.com/color/96/dna-helix.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global Style ────────────────────────────────────────────────────────────
COLORS = {
    "primary":   "#2c3e50",
    "accent":    "#3498db",
    "dead":      "#e74c3c",
    "alive":     "#27ae60",
    "orange":    "#e67e22",
    "purple":    "#9b59b6",
    "teal":      "#1abc9c",
    "stage_I":   "#27ae60",
    "stage_II":  "#3498db",
    "stage_III": "#e67e22",
    "stage_IV":  "#e74c3c",
}

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, Arial, sans-serif", size=13),
    title_font=dict(size=16, color=COLORS["primary"]),
    margin=dict(l=40, r=30, t=50, b=40),
    paper_bgcolor="white",
    plot_bgcolor="white",
)

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
DATA_DIR   = ROOT / "data"
RESULTS    = ROOT / "outputs" / "results"
MODELS_DIR = ROOT / "outputs" / "models"


# ─── Data Loaders ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_results():
    """Load all pre-computed JSON / CSV results from the pipeline."""
    cohort      = json.loads((RESULTS / "cohort_summary.json").read_text())
    model_comp  = pd.read_csv(RESULTS / "model_comparison.csv")
    gene_imp    = json.loads((RESULTS / "gene_importance.json").read_text())
    lasso_coef  = pd.read_csv(RESULTS / "lasso_coefficients.csv", index_col=0)
    lasso_coef.columns = ["coefficient"]
    rsf_imp     = pd.read_csv(RESULTS / "rsf_feature_importance.csv", index_col=0)
    rsf_imp.columns = ["importance"]
    feat_info   = json.loads((RESULTS / "feature_info.json").read_text())
    model_res   = json.loads((RESULTS / "model_results.json").read_text())
    return cohort, model_comp, gene_imp, lasso_coef, rsf_imp, feat_info, model_res


@st.cache_data(show_spinner=False)
def load_clinical():
    """Load and pre-process clinical data for KM curves and demographics."""
    clinical = pd.read_csv(DATA_DIR / "clinical.tsv", sep="\t")
    clinical = clinical.replace("'--", np.nan)
    clinical = clinical.drop_duplicates(subset="cases.submitter_id", keep="first")

    # Survival target
    clinical["days_to_death"] = pd.to_numeric(
        clinical["demographic.days_to_death"], errors="coerce"
    )
    clinical["days_to_fup"] = pd.to_numeric(
        clinical["diagnoses.days_to_last_follow_up"], errors="coerce"
    )
    clinical["event"] = (clinical["demographic.vital_status"] == "Dead").astype(int)
    clinical["time"] = np.where(
        clinical["event"] == 1, clinical["days_to_death"], clinical["days_to_fup"]
    )

    # Follow-up supplement from follow_up.tsv
    try:
        fu = pd.read_csv(DATA_DIR / "follow_up.tsv", sep="\t").replace("'--", np.nan)
        fu["days_val"] = pd.to_numeric(
            fu.get("follow_ups.days_to_follow_up", pd.Series(dtype=float)), errors="coerce"
        )
        fu_max = fu.groupby("cases.submitter_id")["days_val"].max().reset_index()
        fu_max.columns = ["cases.submitter_id", "fu_max"]
        clinical = clinical.merge(fu_max, on="cases.submitter_id", how="left")
        mask_fup = clinical["time"].isna() & (clinical["event"] == 0)
        clinical.loc[mask_fup, "time"] = clinical.loc[mask_fup, "fu_max"]
    except Exception:
        pass

    clinical = clinical[clinical["time"].notna() & (clinical["time"] > 0)].copy()

    # Stage
    def map_stage(val):
        if pd.isna(val):
            return np.nan
        v = str(val).upper().replace(" ", "")
        if "IV" in v:
            return "Stage IV"
        if "III" in v:
            return "Stage III"
        if "II" in v:
            return "Stage II"
        if "I" in v:
            return "Stage I"
        return np.nan

    if "diagnoses.ajcc_pathologic_stage" in clinical.columns:
        clinical["stage"] = clinical["diagnoses.ajcc_pathologic_stage"].map(map_stage)

    # Age & gender
    if "demographic.age_at_index" in clinical.columns:
        clinical["age"] = pd.to_numeric(clinical["demographic.age_at_index"], errors="coerce")
    if "demographic.gender" in clinical.columns:
        clinical["gender"] = clinical["demographic.gender"].str.lower()

    return clinical


@st.cache_resource(show_spinner=False)
def load_models():
    """Load pickled Cox PH model and scaler for risk prediction."""
    try:
        cph    = joblib.load(MODELS_DIR / "cox_ph.pkl")
        scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        return cph, scaler
    except Exception:
        return None, None


# ─── KM Plotting Helper ───────────────────────────────────────────────────────

def km_to_plotly(kmf: KaplanMeierFitter, name: str, color: str,
                 fig: go.Figure, show_ci: bool = True):
    """Add a KM survival curve (and optional CI band) to a Plotly figure."""
    sf = kmf.survival_function_
    t  = sf.index.values
    s  = sf.iloc[:, 0].values

    fig.add_trace(go.Scatter(
        x=t, y=s, mode="lines", name=name,
        line=dict(color=color, width=2.5),
    ))

    if show_ci and kmf.confidence_interval_ is not None:
        ci = kmf.confidence_interval_
        lower = ci.iloc[:, 0].values
        upper = ci.iloc[:, 1].values
        t_ci  = ci.index.values
        # Convert hex (#rrggbb) → rgba for Plotly CI band
        if color.startswith("#") and len(color) == 7:
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            fill_color = f"rgba({r},{g},{b},0.12)"
        elif color.startswith("rgb"):
            fill_color = color.replace(")", ",0.12)").replace("rgb(", "rgba(")
        else:
            fill_color = "rgba(100,100,100,0.12)"
        fig.add_trace(go.Scatter(
            x=np.concatenate([t_ci, t_ci[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor=fill_color,
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        ))


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://www.cancer.gov/sites/g/files/xnygph186/files/styles/cgov_article/public/cgov_image"
        "/media_image/2022-04/kidneys-and-ureters-anatomy-1200px.jpg",
        width='stretch',
    ) if False else None  # skip external image fetch in offline env

    st.markdown("## 🧬 TCGA KIRC")
    st.markdown("**Kidney Renal Clear Cell Carcinoma**  \nSurvival Analysis Dashboard")
    st.divider()

    st.markdown("### Navigation")
    section = st.radio(
        "Jump to section",
        ["📊 Dataset Overview", "📈 Kaplan-Meier Curves",
         "🏆 Model Performance", "🔬 Gene Importance", "⚕️ Risk Prediction Demo"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("**Dataset:** TCGA-KIRC (GDC Portal)  \n"
               "**Pipeline:** Cox PH · LASSO Cox · RSF · DeepSurv  \n"
               "**Metric:** Concordance Index (C-index)")


# ─── Load Data ───────────────────────────────────────────────────────────────

with st.spinner("Loading results…"):
    cohort_summary, model_comp, gene_imp, lasso_coef, rsf_imp, feat_info, model_res = load_results()

with st.spinner("Loading clinical data…"):
    clinical = load_clinical()

cph_model, scaler = load_models()


# ─── Main Header ─────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='color:#2c3e50;margin-bottom:0'>TCGA KIRC — Survival Analysis Dashboard</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#7f8c8d;margin-top:4px'>Kidney Renal Clear Cell Carcinoma · "
    "Multi-model survival analysis · TCGA cohort</p>",
    unsafe_allow_html=True,
)
st.divider()


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# SECTION 1 — DATASET OVERVIEW
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
if "Overview" in section:
    st.header("📊 Dataset Overview")

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Patients",  cohort_summary["total_patients"])
    c2.metric("Deaths (events)", cohort_summary["events_dead"])
    c3.metric("Censored (alive)", cohort_summary["censored_alive"])
    c4.metric("Median Follow-up",
              f"{cohort_summary['median_time_days']:.0f} days")
    c5.metric("Genes Selected", f"{cohort_summary['num_genes']:,}")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    # ── Vital status donut
    with col_l:
        status_counts = clinical["event"].map({1: "Dead", 0: "Alive"}).value_counts()
        fig_vs = go.Figure(go.Pie(
            labels=status_counts.index,
            values=status_counts.values,
            hole=0.50,
            marker=dict(colors=[COLORS["dead"], COLORS["alive"]],
                        line=dict(color="white", width=2)),
            textinfo="label+percent",
            textfont_size=13,
        ))
        fig_vs.update_layout(
            title="Vital Status Distribution",
            showlegend=True,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_vs, width='stretch')

    # ── Age distribution
    with col_r:
        if "age" in clinical.columns:
            age_vals = clinical["age"].dropna()
            fig_age = go.Figure()
            fig_age.add_trace(go.Histogram(
                x=age_vals, nbinsx=28,
                marker_color=COLORS["accent"],
                marker_line=dict(color="white", width=0.8),
                opacity=0.85, name="Patients",
            ))
            fig_age.add_vline(
                x=float(age_vals.median()), line_dash="dash",
                line_color=COLORS["dead"], line_width=2,
                annotation_text=f"Median: {age_vals.median():.0f} yrs",
                annotation_position="top right",
            )
            fig_age.update_layout(
                title="Age at Diagnosis Distribution",
                xaxis_title="Age (years)",
                yaxis_title="Count",
                showlegend=False,
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_age, width='stretch')

    col_a, col_b = st.columns(2)

    # ── Stage distribution
    with col_a:
        if "stage" in clinical.columns:
            stage_order = ["Stage I", "Stage II", "Stage III", "Stage IV"]
            stage_counts = (
                clinical["stage"].value_counts()
                .reindex(stage_order).dropna()
            )
            fig_stage = go.Figure(go.Bar(
                x=stage_counts.index,
                y=stage_counts.values,
                marker_color=[COLORS["stage_I"], COLORS["stage_II"],
                              COLORS["stage_III"], COLORS["stage_IV"]],
                marker_line=dict(color="white", width=1),
                text=stage_counts.values.astype(int),
                textposition="outside",
            ))
            fig_stage.update_layout(
                title="AJCC Pathologic Stage Distribution",
                xaxis_title="Stage",
                yaxis_title="Number of Patients",
                showlegend=False,
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_stage, width='stretch')

    # ── Gender distribution
    with col_b:
        if "gender" in clinical.columns:
            gdr = clinical["gender"].value_counts()
            fig_gdr = go.Figure(go.Bar(
                x=gdr.index.str.capitalize(),
                y=gdr.values,
                marker_color=[COLORS["accent"], COLORS["orange"]],
                marker_line=dict(color="white", width=1),
                text=gdr.values,
                textposition="outside",
            ))
            fig_gdr.update_layout(
                title="Gender Distribution",
                xaxis_title="Gender",
                yaxis_title="Count",
                showlegend=False,
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_gdr, width='stretch')

    # ── Survival time distribution
    st.markdown("#### Follow-up / Survival Time Distribution")
    time_dead    = clinical.loc[clinical["event"] == 1, "time"]
    time_censored = clinical.loc[clinical["event"] == 0, "time"]
    fig_time = go.Figure()
    fig_time.add_trace(go.Histogram(
        x=time_dead, name="Deceased", nbinsx=40,
        marker_color=COLORS["dead"], opacity=0.7,
    ))
    fig_time.add_trace(go.Histogram(
        x=time_censored, name="Censored", nbinsx=40,
        marker_color=COLORS["alive"], opacity=0.7,
    ))
    fig_time.update_layout(
        barmode="overlay",
        title="Survival / Follow-up Time (days)",
        xaxis_title="Days",
        yaxis_title="Count",
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_time, width='stretch')


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# SECTION 2 — KAPLAN-MEIER CURVES
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
elif "Kaplan" in section:
    st.header("📈 Kaplan-Meier Survival Curves")

    tab1, tab2, tab3 = st.tabs(["Overall Survival", "By Cancer Stage", "Stage Comparison Table"])

    # ── Overall KM
    with tab1:
        kmf = KaplanMeierFitter()
        kmf.fit(clinical["time"], event_observed=clinical["event"], label="Overall Survival")

        fig_km = go.Figure()
        km_to_plotly(kmf, "Overall Survival", COLORS["primary"], fig_km, show_ci=True)

        med = kmf.median_survival_time_
        if not np.isinf(med):
            fig_km.add_hline(y=0.5, line_dash="dot", line_color="gray", line_width=1)
            fig_km.add_vline(x=float(med), line_dash="dot", line_color="gray", line_width=1,
                             annotation_text=f"Median: {med:.0f} days",
                             annotation_position="top right",
                             annotation_font=dict(color=COLORS["primary"]))

        fig_km.update_layout(
            title="Overall Kaplan-Meier Survival Curve — TCGA KIRC",
            xaxis_title="Time (days)",
            yaxis_title="Survival Probability",
            yaxis=dict(range=[0, 1.05]),
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_km, width='stretch')

        col_l, col_r = st.columns(2)
        col_l.metric("Median Survival", f"{med:.0f} days" if not np.isinf(med) else "Not reached")
        col_r.metric("12-month survival est.",
                     f"{float(kmf.predict(365)):.1%}")

    # ── KM by stage
    with tab2:
        if "stage" in clinical.columns:
            stage_order  = ["Stage I", "Stage II", "Stage III", "Stage IV"]
            stage_colors = [COLORS["stage_I"], COLORS["stage_II"],
                            COLORS["stage_III"], COLORS["stage_IV"]]

            fig_stage_km = go.Figure()
            lr_results = {}

            for stage, color in zip(stage_order, stage_colors):
                mask = clinical["stage"] == stage
                if mask.sum() < 5:
                    continue
                kmf_s = KaplanMeierFitter()
                kmf_s.fit(
                    clinical.loc[mask, "time"],
                    event_observed=clinical.loc[mask, "event"],
                    label=stage,
                )
                km_to_plotly(kmf_s, stage, color, fig_stage_km, show_ci=True)
                lr_results[stage] = {
                    "n": mask.sum(),
                    "median": kmf_s.median_survival_time_,
                    "events": int(clinical.loc[mask, "event"].sum()),
                }

            fig_stage_km.update_layout(
                title="Kaplan-Meier Curves by AJCC Pathologic Stage",
                xaxis_title="Time (days)",
                yaxis_title="Survival Probability",
                yaxis=dict(range=[0, 1.05]),
                legend=dict(x=0.75, y=0.95),
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_stage_km, width='stretch')

            # Log-rank test Stage I vs IV
            mask_i  = clinical["stage"] == "Stage I"
            mask_iv = clinical["stage"] == "Stage IV"
            if mask_i.sum() > 5 and mask_iv.sum() > 5:
                lr = logrank_test(
                    clinical.loc[mask_i, "time"], clinical.loc[mask_iv, "time"],
                    clinical.loc[mask_i, "event"], clinical.loc[mask_iv, "event"],
                )
                st.info(f"**Log-rank test (Stage I vs IV):** p = {lr.p_value:.2e} "
                        f"({'significant' if lr.p_value < 0.05 else 'not significant'})")
        else:
            st.warning("Stage information not available in the clinical data.")

    # ── Comparison table
    with tab3:
        if "stage" in clinical.columns:
            summary_rows = []
            for stage in ["Stage I", "Stage II", "Stage III", "Stage IV"]:
                mask = clinical["stage"] == stage
                if mask.sum() < 5:
                    continue
                kmf_t = KaplanMeierFitter()
                kmf_t.fit(clinical.loc[mask, "time"], event_observed=clinical.loc[mask, "event"])
                med = kmf_t.median_survival_time_
                summary_rows.append({
                    "Stage": stage,
                    "N": int(mask.sum()),
                    "Events (Deaths)": int(clinical.loc[mask, "event"].sum()),
                    "Censored": int((clinical.loc[mask, "event"] == 0).sum()),
                    "Median Survival (days)": f"{med:.0f}" if not np.isinf(med) else "Not reached",
                    "1-year Survival": f"{float(kmf_t.predict(365)):.1%}",
                    "3-year Survival": f"{float(kmf_t.predict(1095)):.1%}",
                    "5-year Survival": f"{float(kmf_t.predict(1825)):.1%}",
                })
            st.dataframe(pd.DataFrame(summary_rows), width='stretch', hide_index=True)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# SECTION 3 — MODEL PERFORMANCE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
elif "Model" in section:
    st.header("🏆 Model Performance")

    # Best model callout
    best_row = model_comp.loc[model_comp["C-index"].idxmax()]
    st.success(
        f"**Best model: {best_row['Model']}** — C-index = **{best_row['C-index']:.4f}**  \n"
        f"Integrated Brier Score (RSF): **{cohort_summary['ibs_rsf']:.4f}**  "
        f"*(lower is better, 0.25 = random)*"
    )

    col1, col2 = st.columns([2, 1])

    # ── C-index bar chart
    with col1:
        bar_colors = [
            COLORS["accent"] if row["Model"] != best_row["Model"] else COLORS["teal"]
            for _, row in model_comp.iterrows()
        ]
        fig_ci = go.Figure(go.Bar(
            x=model_comp["C-index"],
            y=model_comp["Model"],
            orientation="h",
            marker_color=bar_colors,
            marker_line=dict(color="white", width=1),
            text=[f"{v:.4f}" for v in model_comp["C-index"]],
            textposition="outside",
        ))
        fig_ci.add_vline(x=0.5, line_dash="dot", line_color="gray",
                         annotation_text="Random (0.5)", annotation_position="bottom right")
        fig_ci.update_layout(
            title="Model Comparison — Concordance Index (Test Set)",
            xaxis=dict(title="C-index", range=[0.45, 0.88]),
            yaxis=dict(title=""),
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_ci, width='stretch')

    # ── Metrics table
    with col2:
        st.markdown("#### Detailed Metrics")
        metrics_df = model_comp.copy()
        metrics_df = metrics_df.sort_values("C-index", ascending=False)
        metrics_df["Rank"] = range(1, len(metrics_df) + 1)
        metrics_df["C-index"] = metrics_df["C-index"].map("{:.4f}".format)
        st.dataframe(
            metrics_df[["Rank", "Model", "C-index"]],
            width='stretch',
            hide_index=True,
        )

        st.markdown("#### Interpretation")
        st.markdown("""
| C-index | Interpretation |
|---------|---------------|
| 1.0     | Perfect        |
| 0.8–1.0 | Excellent      |
| 0.7–0.8 | Good           |
| 0.6–0.7 | Moderate       |
| 0.5     | Random         |
""")

    st.markdown("---")
    st.markdown("#### Model Descriptions")
    exp1, exp2, exp3, exp4 = st.columns(4)
    with exp1:
        with st.expander("Cox PH (clinical)"):
            st.markdown(
                "Baseline **Cox Proportional Hazards** model trained on 3 clinical features "
                "(age, gender, stage). Assumes proportional hazards. Interpretable hazard ratios."
            )
    with exp2:
        with st.expander("LASSO Cox â­"):
            st.markdown(
                "**L1-penalized Cox regression** on clinical + 2000 top-variance genes. "
                "Automatically selects sparse set of predictive features. Best overall C-index."
            )
    with exp3:
        with st.expander("Random Survival Forest"):
            st.markdown(
                "**Non-parametric ensemble** (300 trees) on LASSO-selected features. "
                "Captures non-linear interactions. Also provides permutation-based feature importance."
            )
    with exp4:
        with st.expander("DeepSurv"):
            st.markdown(
                "**Deep learning** MLP (128→64→1) trained with Cox partial likelihood loss "
                "(PyTorch). 100 epochs. Captures complex patterns in high-dimensional expression data."
            )


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# SECTION 4 — GENE IMPORTANCE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
elif "Gene" in section:
    st.header("🔬 Gene Importance")

    top_n = st.slider("Number of top features to display", 10, 40, 20, step=5)

    tab_lasso, tab_rsf, tab_overlap, tab_table = st.tabs(
        ["LASSO Cox Coefficients", "RSF Permutation Importance",
         "Overlap Analysis", "Full Table"]
    )

    # ── LASSO top features
    with tab_lasso:
        top_lasso = pd.concat([
            lasso_coef.nlargest(top_n, "coefficient"),
            lasso_coef.nsmallest(top_n, "coefficient"),
        ]).drop_duplicates()
        top_lasso = top_lasso.sort_values("coefficient")

        bar_colors_l = [COLORS["dead"] if v > 0 else COLORS["alive"]
                        for v in top_lasso["coefficient"]]
        fig_l = go.Figure(go.Bar(
            x=top_lasso["coefficient"],
            y=top_lasso.index,
            orientation="h",
            marker_color=bar_colors_l,
            marker_line=dict(color="white", width=0.5),
        ))
        fig_l.add_vline(x=0, line_color="gray", line_width=0.8)
        fig_l.update_layout(
            title=f"Top LASSO Cox Features (by |coefficient|)",
            xaxis_title="LASSO Cox Coefficient",
            yaxis=dict(title="", tickfont=dict(size=11)),
            height=max(400, top_n * 22),
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_l, width='stretch')
        st.caption(
            "🔴 **Red bars** → positive coefficient → higher risk of death  \n"
            "🟢 **Green bars** → negative coefficient → protective factor"
        )

    # ── RSF top features
    with tab_rsf:
        rsf_top_n = rsf_imp.nlargest(top_n, "importance")
        fig_rsf = go.Figure(go.Bar(
            x=rsf_top_n["importance"],
            y=rsf_top_n.index,
            orientation="h",
            marker_color=COLORS["purple"],
            marker_line=dict(color="white", width=0.5),
        ))
        fig_rsf.update_layout(
            title=f"Top {top_n} Features — RSF Permutation Importance",
            xaxis_title="Permutation Importance",
            yaxis=dict(title="", tickfont=dict(size=11), autorange="reversed"),
            height=max(400, top_n * 22),
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_rsf, width='stretch')
        st.caption("Permutation importance: how much the C-index drops when this feature is randomly shuffled.")

    # ── Overlap analysis
    with tab_overlap:
        lasso_genes = set(gene_imp["lasso_top_genes"].keys())
        rsf_genes   = set(gene_imp["rsf_top_genes"].keys())
        overlap     = lasso_genes & rsf_genes
        only_lasso  = lasso_genes - rsf_genes
        only_rsf    = rsf_genes - lasso_genes

        c1, c2, c3 = st.columns(3)
        c1.metric("LASSO top genes", len(lasso_genes))
        c2.metric("RSF top genes",   len(rsf_genes))
        c3.metric("Shared genes",    len(overlap))

        fig_venn_bar = go.Figure(go.Bar(
            x=["LASSO only", "Shared", "RSF only"],
            y=[len(only_lasso), len(overlap), len(only_rsf)],
            marker_color=[COLORS["accent"], COLORS["teal"], COLORS["purple"]],
            text=[len(only_lasso), len(overlap), len(only_rsf)],
            textposition="outside",
        ))
        fig_venn_bar.update_layout(
            title="Gene Overlap between LASSO Cox and Random Survival Forest",
            yaxis_title="Number of Genes",
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_venn_bar, width='stretch')

        if overlap:
            st.markdown("**Genes identified by both models:**")
            overlap_df = pd.DataFrame({
                "Gene":             sorted(overlap),
                "LASSO coefficient": [gene_imp["lasso_top_genes"].get(g, 0) for g in sorted(overlap)],
                "RSF importance":   [gene_imp["rsf_top_genes"].get(g, 0) for g in sorted(overlap)],
            })
            st.dataframe(overlap_df, width='stretch', hide_index=True)

    # ── Full searchable table
    with tab_table:
        combined = lasso_coef.copy()
        combined["rsf_importance"] = rsf_imp["importance"]
        combined = combined.fillna(0).reset_index()
        combined.columns = ["Feature", "LASSO Coefficient", "RSF Importance"]
        combined = combined.sort_values("LASSO Coefficient", key=abs, ascending=False)

        search = st.text_input("Search feature name", "")
        if search:
            combined = combined[
                combined["Feature"].str.contains(search, case=False, na=False)
            ]
        st.dataframe(combined, width='stretch', hide_index=True)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# SECTION 5 — RISK PREDICTION DEMO
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
elif "Risk" in section:
    st.header("⚕️ Risk Prediction Demo")
    st.markdown(
        "Enter a patient's clinical profile to obtain a predicted survival risk score "
        "using the **Cox Proportional Hazards** model (trained on clinical features)."
    )

    if cph_model is None or scaler is None:
        st.warning(
            "Pre-trained models not found. Make sure `outputs/models/cox_ph.pkl` and "
            "`outputs/models/scaler.pkl` exist (run `src/pipeline.py` first)."
        )
    else:
        col_inp, col_out = st.columns([1, 2])

        with col_inp:
            st.markdown("#### Patient Profile")
            age_val   = st.slider("Age at diagnosis (years)", 20, 90, 60)
            gender    = st.radio("Gender", ["Male", "Female"])
            stage_sel = st.select_slider(
                "AJCC Pathologic Stage",
                options=["Stage I", "Stage II", "Stage III", "Stage IV"],
                value="Stage II",
            )

            is_male   = 1 if gender == "Male" else 0
            stage_num = {"Stage I": 1, "Stage II": 2, "Stage III": 3, "Stage IV": 4}[stage_sel]

            # Scale input using trained scaler (first 3 features: age, is_male, stage_num)
            input_raw = np.array([[age_val, is_male, stage_num]], dtype=float)
            clin_idx  = slice(0, 3)    # age, is_male, stage_num are first 3 columns in scaler
            input_scaled = (input_raw - scaler.mean_[clin_idx]) / scaler.scale_[clin_idx]
            input_df = pd.DataFrame(input_scaled, columns=["age", "is_male", "stage_num"])

            # Predict
            partial_hazard = float(cph_model.predict_partial_hazard(input_df).iloc[0])
            log_hazard     = float(np.log(partial_hazard + 1e-9))

            # Compute reference range from training set percentiles to normalize
            # Use log(partial hazard) z-score relative to training data
            st.markdown("---")
            st.markdown("#### Predicted Risk Score")

            # Color-code by hazard magnitude (relative to reference patient: age=60, male, Stage II)
            ref_raw    = np.array([[60, 1, 2]], dtype=float)
            ref_scaled = (ref_raw - scaler.mean_[clin_idx]) / scaler.scale_[clin_idx]
            ref_df     = pd.DataFrame(ref_scaled, columns=["age", "is_male", "stage_num"])
            ref_hazard = float(cph_model.predict_partial_hazard(ref_df).iloc[0])

            ratio = partial_hazard / ref_hazard
            if ratio < 0.5:
                colour = "🟢"; label = "Low Risk"
            elif ratio < 1.5:
                colour = "🟡"; label = "Moderate Risk"
            else:
                colour = "🔴"; label = "High Risk"

            st.metric("Partial Hazard", f"{partial_hazard:.4f}")
            st.markdown(
                f"<div style='font-size:28px;text-align:center;padding:12px;"
                f"border-radius:8px;background:#f8f9fa'>"
                f"{colour} <b>{label}</b></div>",
                unsafe_allow_html=True,
            )
            st.caption(f"Relative to median patient (hazard ratio vs. reference = {ratio:.2f}×)")

        with col_out:
            st.markdown("#### Predicted Survival Curve")
            pred_sf = cph_model.predict_survival_function(input_df)
            times   = pred_sf.index.values
            surv    = pred_sf.iloc[:, 0].values

            fig_pred = go.Figure()

            # Population baseline KM in background
            kmf_bg = KaplanMeierFitter()
            kmf_bg.fit(clinical["time"], event_observed=clinical["event"])
            bg_sf  = kmf_bg.survival_function_
            fig_pred.add_trace(go.Scatter(
                x=bg_sf.index.values,
                y=bg_sf.iloc[:, 0].values,
                mode="lines", name="Population Average",
                line=dict(color="lightgray", width=2, dash="dash"),
            ))

            # Predicted patient curve
            fig_pred.add_trace(go.Scatter(
                x=times, y=surv, mode="lines",
                name=f"This patient ({stage_sel})",
                line=dict(
                    color=COLORS["dead"] if label == "High Risk"
                          else COLORS["alive"] if label == "Low Risk"
                          else COLORS["orange"],
                    width=3,
                ),
            ))

            # Annotate 1-year and 5-year survival
            for t_mark, t_label in [(365, "1-yr"), (1825, "5-yr")]:
                idx = np.searchsorted(times, t_mark)
                if idx < len(surv):
                    sv = surv[idx]
                    fig_pred.add_annotation(
                        x=t_mark, y=sv,
                        text=f"{t_label}: {sv:.1%}",
                        showarrow=True, arrowhead=2,
                        arrowcolor=COLORS["primary"],
                        font=dict(size=11, color=COLORS["primary"]),
                        bgcolor="white", bordercolor=COLORS["primary"],
                        ax=50, ay=-30,
                    )

            fig_pred.add_hline(y=0.5, line_dash="dot", line_color="gray", line_width=1,
                               annotation_text="50% survival", annotation_position="right")
            fig_pred.update_layout(
                title="Predicted Survival Function",
                xaxis_title="Time (days)",
                yaxis_title="Survival Probability",
                yaxis=dict(range=[0, 1.05]),
                legend=dict(x=0.65, y=0.95),
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_pred, width='stretch')

            # Survival probability table
            landmarks = [90, 180, 365, 730, 1095, 1460, 1825]
            rows = []
            for t_mark in landmarks:
                idx = np.searchsorted(times, t_mark)
                if idx < len(surv):
                    rows.append({"Time": f"{t_mark} days ({t_mark//365:.0f} yr)" if t_mark >= 365
                                         else f"{t_mark} days",
                                 "Predicted Survival": f"{surv[idx]:.1%}"})
            if rows:
                st.markdown("#### Landmark Survival Probabilities")
                st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)


# ─── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "**Data source:** The Cancer Genome Atlas (TCGA-KIRC), GDC Data Portal &nbsp;|&nbsp; "
    "**Pipeline:** lifelines · scikit-survival · PyTorch &nbsp;|&nbsp; "
    "**Dashboard:** Streamlit + Plotly"
)

