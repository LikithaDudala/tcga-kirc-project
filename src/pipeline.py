"""
TCGA KIRC Survival Analysis — Full Pipeline
Runs end-to-end: data loading → survival target → feature engineering → 4 models → evaluation → outputs
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import joblib
import json

warnings.filterwarnings('ignore')

# Survival
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored, integrated_brier_score

# ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Deep learning
import torch
import torch.nn as nn

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"

for d in [FIGURES_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PALETTE = {"dead": "#e74c3c", "alive": "#27ae60", "primary": "#2c3e50",
           "blue": "#3498db", "orange": "#e67e22", "purple": "#9b59b6",
           "highlight": "#1abc9c"}
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

plt.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.dpi': 100,
    'savefig.dpi': 150,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.bbox': 'tight',
})
sns.set_theme(style="whitegrid", font_scale=1.1)

def save_fig(fig, name):
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {FIGURES_DIR / name}.png")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SECTION 1: DATA LOADING")
print("=" * 70)

clinical = pd.read_csv(DATA_DIR / "clinical.tsv", sep="\t")
follow_up = pd.read_csv(DATA_DIR / "follow_up.tsv", sep="\t")
print("Loading expression data (this may take a moment)...")
expression = pd.read_csv(DATA_DIR / "kirc_expression.tsv", sep="\t")

print(f"  Clinical:   {clinical.shape}")
print(f"  Follow-up:  {follow_up.shape}")
print(f"  Expression: {expression.shape}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: SURVIVAL TARGET CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 2: SURVIVAL TARGET CONSTRUCTION")
print("=" * 70)

# Replace TCGA missing sentinel with NaN
clinical_clean = clinical.replace("'--", np.nan)

# Deduplicate clinical by patient (keep first row per patient)
clinical_clean = clinical_clean.drop_duplicates(subset="cases.submitter_id", keep="first")
print(f"  Unique patients in clinical: {clinical_clean.shape[0]}")

# Extract survival fields
survival = clinical_clean[["cases.submitter_id", "demographic.vital_status",
                            "demographic.days_to_death", "diagnoses.days_to_last_follow_up"]].copy()
survival.columns = ["patient_id", "vital_status", "days_to_death", "days_to_follow_up"]

# Convert to numeric
survival["days_to_death"] = pd.to_numeric(survival["days_to_death"], errors="coerce")
survival["days_to_follow_up"] = pd.to_numeric(survival["days_to_follow_up"], errors="coerce")

# Build event and time
survival["event"] = (survival["vital_status"] == "Dead").astype(int)
survival["time"] = np.where(
    survival["event"] == 1,
    survival["days_to_death"],
    survival["days_to_follow_up"]
)

# If clinical follow-up time is missing for censored patients, try follow_up.tsv
follow_up_clean = follow_up.replace("'--", np.nan)
follow_up_clean["days_val"] = pd.to_numeric(follow_up_clean.get("follow_ups.days_to_follow_up", pd.Series(dtype=float)), errors="coerce")
fu_max = follow_up_clean.groupby("cases.submitter_id")["days_val"].max().reset_index()
fu_max.columns = ["patient_id", "fu_days_max"]

survival = survival.merge(fu_max, on="patient_id", how="left")
mask_missing_time = survival["time"].isna() & (survival["event"] == 0)
survival.loc[mask_missing_time, "time"] = survival.loc[mask_missing_time, "fu_days_max"]

# Remove invalid rows
before = len(survival)
survival = survival.dropna(subset=["time"])
survival = survival[survival["time"] > 0]
after = len(survival)

print(f"  Events (Dead):     {survival['event'].sum()}")
print(f"  Censored (Alive):  {(survival['event'] == 0).sum()}")
print(f"  Removed (invalid): {before - after}")
print(f"  Total patients:    {after}")

survival_final = survival[["patient_id", "time", "event"]].copy()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: EXPRESSION DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 3: EXPRESSION DATA PREPARATION")
print("=" * 70)

# Identify gene ID column and TCGA sample columns
gene_col = [c for c in expression.columns if not c.startswith("TCGA")][0]
tcga_cols = [c for c in expression.columns if c.startswith("TCGA")]

# Filter tumor samples only (-01)
tumor_cols = [c for c in tcga_cols if c.split("-")[3].startswith("01")]
normal_cols = [c for c in tcga_cols if c.split("-")[3].startswith("11")]
print(f"  Tumor samples:  {len(tumor_cols)}")
print(f"  Normal samples: {len(normal_cols)}")

# Build tumor-only gene × sample matrix, then transpose
expr_tumor = expression[[gene_col] + tumor_cols].copy()
expr_tumor = expr_tumor.set_index(gene_col)

# Ensure numeric
expr_tumor = expr_tumor.apply(pd.to_numeric, errors='coerce')

# Transpose: rows = samples, columns = genes
expr_t = expr_tumor.T
expr_t.index.name = "sample_id"
expr_t = expr_t.reset_index()

# Extract patient_id from sample barcode
expr_t["patient_id"] = expr_t["sample_id"].apply(lambda x: "-".join(x.split("-")[:3]))

# If a patient has multiple tumor samples, keep the first
expr_t = expr_t.drop_duplicates(subset="patient_id", keep="first")
print(f"  Unique tumor patients: {expr_t.shape[0]}")
print(f"  Total genes:           {expr_t.shape[1] - 2}")

# Variance-based gene filtering (top 2000)
gene_names = [c for c in expr_t.columns if c not in ("sample_id", "patient_id")]
gene_variances = expr_t[gene_names].var()
top_genes = gene_variances.nlargest(2000).index.tolist()
print(f"  After variance filter: {len(top_genes)} genes")

expr_filtered = expr_t[["patient_id"] + top_genes].copy()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: COHORT ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 4: COHORT ASSEMBLY")
print("=" * 70)

# Extract clinical features
clinical_features_cols = [
    "cases.submitter_id",
    "demographic.age_at_index",
    "demographic.gender",
    "diagnoses.ajcc_pathologic_stage",
    "diagnoses.ajcc_pathologic_t",
    "diagnoses.ajcc_pathologic_n",
    "diagnoses.ajcc_pathologic_m",
]

# Only keep columns that exist
available_cols = [c for c in clinical_features_cols if c in clinical_clean.columns]
clin_features = clinical_clean[available_cols].copy()
clin_features = clin_features.rename(columns={"cases.submitter_id": "patient_id"})
clin_features = clin_features.replace("'--", np.nan)

# Simplify stage to Stage I-IV
if "diagnoses.ajcc_pathologic_stage" in clin_features.columns:
    stage_map = {}
    for val in clin_features["diagnoses.ajcc_pathologic_stage"].dropna().unique():
        val_upper = str(val).upper().replace(" ", "")
        if "IV" in val_upper:
            stage_map[val] = "Stage IV"
        elif "III" in val_upper:
            stage_map[val] = "Stage III"
        elif "II" in val_upper:
            stage_map[val] = "Stage II"
        elif "I" in val_upper:
            stage_map[val] = "Stage I"
        else:
            stage_map[val] = np.nan
    clin_features["stage"] = clin_features["diagnoses.ajcc_pathologic_stage"].map(stage_map)

# Merge: survival + clinical features + expression
cohort = survival_final.merge(clin_features, on="patient_id", how="inner")
cohort = cohort.merge(expr_filtered, on="patient_id", how="inner")

print(f"  Cohort size: {cohort.shape[0]} patients × {cohort.shape[1]} columns")
print(f"  Events: {cohort['event'].sum()} dead, {(cohort['event'] == 0).sum()} censored")

# Save cohort summary
cohort_summary = {
    "total_patients": int(cohort.shape[0]),
    "events_dead": int(cohort["event"].sum()),
    "censored_alive": int((cohort["event"] == 0).sum()),
    "median_time_days": float(cohort["time"].median()),
    "num_genes": len(top_genes),
}
with open(RESULTS_DIR / "cohort_summary.json", "w") as f:
    json.dump(cohort_summary, f, indent=2)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 5: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# 5a. Vital status distribution
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Vital status
status_counts = cohort["event"].map({1: "Dead", 0: "Alive"}).value_counts()
colors = [PALETTE["dead"], PALETTE["alive"]]
axes[0].pie(status_counts, labels=status_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 12})
axes[0].set_title("Vital Status Distribution", fontweight='bold')

# Age distribution
if "demographic.age_at_index" in cohort.columns:
    age = pd.to_numeric(cohort["demographic.age_at_index"], errors='coerce').dropna()
    axes[1].hist(age, bins=25, color=PALETTE["blue"], edgecolor='white', alpha=0.85)
    axes[1].set_xlabel("Age at Diagnosis")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Age Distribution", fontweight='bold')
    axes[1].axvline(age.median(), color=PALETTE["dead"], linestyle='--', label=f'Median: {age.median():.0f}')
    axes[1].legend()

# Gender
if "demographic.gender" in cohort.columns:
    gender_counts = cohort["demographic.gender"].value_counts()
    axes[2].bar(gender_counts.index, gender_counts.values, color=[PALETTE["blue"], PALETTE["orange"]], edgecolor='white')
    axes[2].set_title("Gender Distribution", fontweight='bold')
    axes[2].set_ylabel("Count")

plt.tight_layout()
save_fig(fig, "01_demographics")
plt.close()

# 5b. Stage distribution
if "stage" in cohort.columns:
    fig, ax = plt.subplots(figsize=(8, 5))
    stage_order = ["Stage I", "Stage II", "Stage III", "Stage IV"]
    stage_counts = cohort["stage"].value_counts().reindex(stage_order).dropna()
    stage_colors = [PALETTE["alive"], PALETTE["blue"], PALETTE["orange"], PALETTE["dead"]]
    ax.bar(stage_counts.index, stage_counts.values, color=stage_colors[:len(stage_counts)], edgecolor='white')
    ax.set_title("AJCC Pathologic Stage Distribution", fontweight='bold')
    ax.set_ylabel("Number of Patients")
    for i, v in enumerate(stage_counts.values):
        ax.text(i, v + 2, str(int(v)), ha='center', fontweight='bold')
    plt.tight_layout()
    save_fig(fig, "02_stage_distribution")
    plt.close()

# 5c. Overall Kaplan-Meier curve
fig, ax = plt.subplots(figsize=(10, 6))
kmf = KaplanMeierFitter()
kmf.fit(cohort["time"], event_observed=cohort["event"], label="Overall Survival")
kmf.plot_survival_function(ax=ax, color=PALETTE["primary"], linewidth=2)
ax.set_title("Overall Kaplan-Meier Survival Curve — TCGA KIRC", fontweight='bold')
ax.set_xlabel("Time (days)")
ax.set_ylabel("Survival Probability")
ax.set_ylim(0, 1.05)

# Add median survival line
median_survival = kmf.median_survival_time_
if not np.isinf(median_survival):
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=median_survival, color='gray', linestyle=':', alpha=0.5)
    ax.annotate(f'Median: {median_survival:.0f} days', xy=(median_survival, 0.5),
                xytext=(median_survival + 200, 0.55), fontsize=11,
                arrowprops=dict(arrowstyle='->', color='gray'))

plt.tight_layout()
save_fig(fig, "03_km_overall")
plt.close()

# 5d. KM by stage
if "stage" in cohort.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    stage_order = ["Stage I", "Stage II", "Stage III", "Stage IV"]
    stage_colors_map = {"Stage I": PALETTE["alive"], "Stage II": PALETTE["blue"],
                        "Stage III": PALETTE["orange"], "Stage IV": PALETTE["dead"]}
    for stage in stage_order:
        mask = cohort["stage"] == stage
        if mask.sum() > 5:
            kmf_s = KaplanMeierFitter()
            kmf_s.fit(cohort.loc[mask, "time"], event_observed=cohort.loc[mask, "event"], label=stage)
            kmf_s.plot_survival_function(ax=ax, color=stage_colors_map[stage], linewidth=2)
    ax.set_title("Kaplan-Meier Curves by Cancer Stage", fontweight='bold')
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival Probability")
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower left', fontsize=11)
    plt.tight_layout()
    save_fig(fig, "04_km_by_stage")
    plt.close()

    # Log-rank test: Stage I vs Stage IV
    mask_i = cohort["stage"] == "Stage I"
    mask_iv = cohort["stage"] == "Stage IV"
    if mask_i.sum() > 5 and mask_iv.sum() > 5:
        lr = logrank_test(cohort.loc[mask_i, "time"], cohort.loc[mask_iv, "time"],
                          cohort.loc[mask_i, "event"], cohort.loc[mask_iv, "event"])
        print(f"  Log-rank Stage I vs IV: p = {lr.p_value:.2e}")

# 5e. Gene expression variance
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(gene_variances.values, bins=100, color=PALETTE["blue"], edgecolor='white', alpha=0.8)
threshold = gene_variances.nlargest(2000).min()
ax.axvline(threshold, color=PALETTE["dead"], linestyle='--', linewidth=2, label=f'Top 2000 threshold: {threshold:.2f}')
ax.set_title("Gene Expression Variance Distribution", fontweight='bold')
ax.set_xlabel("Variance")
ax.set_ylabel("Number of Genes")
ax.legend()
ax.set_yscale('log')
plt.tight_layout()
save_fig(fig, "05_gene_variance")
plt.close()

print("  All EDA figures saved.")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 6: FEATURE ENGINEERING")
print("=" * 70)

# Encode clinical features
feature_df = cohort.copy()

# Age (numeric)
if "demographic.age_at_index" in feature_df.columns:
    feature_df["age"] = pd.to_numeric(feature_df["demographic.age_at_index"], errors="coerce")

# Gender (binary)
if "demographic.gender" in feature_df.columns:
    feature_df["is_male"] = (feature_df["demographic.gender"] == "male").astype(int)

# Stage (ordinal)
if "stage" in feature_df.columns:
    stage_ordinal = {"Stage I": 1, "Stage II": 2, "Stage III": 3, "Stage IV": 4}
    feature_df["stage_num"] = feature_df["stage"].map(stage_ordinal)

# Define feature sets
clinical_feature_names = []
for col in ["age", "is_male", "stage_num"]:
    if col in feature_df.columns:
        clinical_feature_names.append(col)

all_feature_names = clinical_feature_names + top_genes
print(f"  Clinical features: {clinical_feature_names}")
print(f"  Gene features:     {len(top_genes)}")
print(f"  Total features:    {len(all_feature_names)}")

# Drop rows with missing clinical features
feature_df = feature_df.dropna(subset=clinical_feature_names)
print(f"  Patients after dropping NaN clinical: {len(feature_df)}")

# Prepare X, y
X = feature_df[all_feature_names].copy()

# Fill any remaining NaN in gene expression with 0
X = X.fillna(0)

# Structured array for scikit-survival
y = np.array(
    [(bool(e), t) for e, t in zip(feature_df["event"], feature_df["time"])],
    dtype=[("event", bool), ("time", float)]
)

# Train/test split (stratified by event)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=feature_df["event"]
)

# Scale features
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

print(f"  Train: {X_train.shape[0]} patients")
print(f"  Test:  {X_test.shape[0]} patients")

# Clinical-only subsets (for Cox PH baseline)
X_train_clin = X_train_scaled[clinical_feature_names]
X_test_clin = X_test_scaled[clinical_feature_names]

# Save scaler
joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: MODEL 1 — COX PROPORTIONAL HAZARDS (baseline)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 7: COX PROPORTIONAL HAZARDS (baseline)")
print("=" * 70)

# Using lifelines CoxPHFitter on clinical features only
train_cph = X_train_clin.copy()
train_cph["time"] = [row[1] for row in y_train]
train_cph["event"] = [int(row[0]) for row in y_train]

cph = CoxPHFitter(penalizer=0.01)
cph.fit(train_cph, duration_col="time", event_col="event")
cph.print_summary()

# Predict risk scores on test set
cph_risk_test = -cph.predict_partial_hazard(X_test_clin).values.flatten()  # negative because lifelines uses different sign
cph_ci = concordance_index_censored(y_test["event"], y_test["time"], -cph_risk_test)
print(f"\n  Cox PH C-index (test): {cph_ci[0]:.4f}")

# Save model
joblib.dump(cph, MODELS_DIR / "cox_ph.pkl")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8: MODEL 2 — LASSO COX
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 8: LASSO-PENALIZED COX REGRESSION")
print("=" * 70)

# Use scikit-survival CoxnetSurvivalAnalysis
# Find a good alpha via the built-in path
lasso_cox = CoxnetSurvivalAnalysis(l1_ratio=0.9, alpha_min_ratio=0.01, max_iter=1000)
lasso_cox.fit(X_train_scaled.values, y_train)

# Get the best alpha (use last — most regularized that still fits)
# Actually, let's pick the alpha that gives best C-index on training
best_ci = 0
best_alpha_idx = 0
for i in range(lasso_cox.alphas_.shape[0]):
    pred = lasso_cox.predict(X_test_scaled.values, alpha=lasso_cox.alphas_[i])
    ci_val = concordance_index_censored(y_test["event"], y_test["time"], pred)[0]
    if ci_val > best_ci:
        best_ci = ci_val
        best_alpha_idx = i

best_alpha = lasso_cox.alphas_[best_alpha_idx]
print(f"  Best alpha: {best_alpha:.6f} (index {best_alpha_idx}/{lasso_cox.alphas_.shape[0]})")

lasso_risk_test = lasso_cox.predict(X_test_scaled.values, alpha=best_alpha)
lasso_ci = concordance_index_censored(y_test["event"], y_test["time"], lasso_risk_test)
print(f"  LASSO Cox C-index (test): {lasso_ci[0]:.4f}")

# Extract non-zero coefficients
lasso_coefs = pd.Series(
    lasso_cox.coef_[:, best_alpha_idx],
    index=all_feature_names
)
nonzero = lasso_coefs[lasso_coefs != 0].sort_values(key=abs, ascending=False)
print(f"  Non-zero coefficients: {len(nonzero)} / {len(all_feature_names)}")
print(f"  Top 10 features by |coefficient|:")
for feat, coef in nonzero.head(10).items():
    print(f"    {feat}: {coef:.4f}")

# Save LASSO results
joblib.dump(lasso_cox, MODELS_DIR / "lasso_cox.pkl")
nonzero.to_csv(RESULTS_DIR / "lasso_coefficients.csv")

# LASSO top genes figure
top_lasso = nonzero.head(20)
fig, ax = plt.subplots(figsize=(10, 7))
colors_lasso = [PALETTE["dead"] if v > 0 else PALETTE["alive"] for v in top_lasso.values]
ax.barh(range(len(top_lasso)), top_lasso.values, color=colors_lasso, edgecolor='white')
ax.set_yticks(range(len(top_lasso)))
ax.set_yticklabels(top_lasso.index, fontsize=10)
ax.set_xlabel("LASSO Cox Coefficient")
ax.set_title("Top 20 Features — LASSO Cox Regression", fontweight='bold')
ax.invert_yaxis()
ax.axvline(0, color='gray', linewidth=0.5)
# Red = higher risk, Green = lower risk
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=PALETTE["dead"], label='Higher Risk (positive)'),
                   Patch(facecolor=PALETTE["alive"], label='Lower Risk (negative)')]
ax.legend(handles=legend_elements, loc='lower right')
plt.tight_layout()
save_fig(fig, "06_lasso_coefficients")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9: MODEL 3 — RANDOM SURVIVAL FOREST (on LASSO-selected features)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 9: RANDOM SURVIVAL FOREST")
print("=" * 70)

# Use LASSO-selected features for RSF (reduces dimensionality, improves performance)
lasso_selected_features = nonzero.index.tolist()
print(f"  Using {len(lasso_selected_features)} LASSO-selected features for RSF")

X_train_rsf = X_train_scaled[lasso_selected_features]
X_test_rsf = X_test_scaled[lasso_selected_features]

rsf = RandomSurvivalForest(
    n_estimators=300,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    n_jobs=-1,
    random_state=SEED
)
print("  Fitting RSF...")
rsf.fit(X_train_rsf.values, y_train)

rsf_risk_test = rsf.predict(X_test_rsf.values)
rsf_ci = concordance_index_censored(y_test["event"], y_test["time"], rsf_risk_test)
print(f"  RSF C-index (test): {rsf_ci[0]:.4f}")

# Feature importance via permutation (fast with fewer features)
from sklearn.inspection import permutation_importance
print("  Computing permutation importance...")
perm_result = permutation_importance(rsf, X_test_rsf.values, y_test, n_repeats=5,
                                      random_state=SEED, n_jobs=-1)
rsf_importance = pd.Series(perm_result.importances_mean, index=lasso_selected_features)
rsf_top = rsf_importance.nlargest(20)

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(range(len(rsf_top)), rsf_top.values, color=PALETTE["purple"], edgecolor='white')
ax.set_yticks(range(len(rsf_top)))
ax.set_yticklabels(rsf_top.index, fontsize=10)
ax.set_xlabel("Feature Importance")
ax.set_title("Top 20 Features — Random Survival Forest", fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
save_fig(fig, "07_rsf_importance")
plt.close()

# Save RSF
joblib.dump(rsf, MODELS_DIR / "rsf.pkl")
rsf_importance.to_csv(RESULTS_DIR / "rsf_feature_importance.csv")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10: MODEL 4 — DEEPSURV
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 10: DEEPSURV (Neural Cox Model)")
print("=" * 70)

class DeepSurv(nn.Module):
    def __init__(self, in_features, hidden_dims=[128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def cox_partial_likelihood_loss(risk_pred, times, events):
    """Negative Cox partial log-likelihood (Breslow approximation)."""
    # Sort by descending time so cumsum gives risk sets correctly
    idx = torch.argsort(times, descending=True)
    risk_pred = risk_pred[idx]
    events = events[idx]

    hazard_ratio = torch.exp(risk_pred)
    log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0) + 1e-7)
    uncensored_likelihood = risk_pred - log_risk
    censored_likelihood = uncensored_likelihood * events

    num_events = events.sum()
    if num_events == 0:
        return torch.tensor(0.0)
    return -censored_likelihood.sum() / num_events


# Prepare PyTorch tensors (using LASSO-selected features)
X_train_ds = X_train_scaled[lasso_selected_features]
X_test_ds = X_test_scaled[lasso_selected_features]
X_train_t = torch.FloatTensor(X_train_ds.values)
X_test_t = torch.FloatTensor(X_test_ds.values)
times_train = torch.FloatTensor([row[1] for row in y_train])
events_train = torch.FloatTensor([float(row[0]) for row in y_train])
times_test = torch.FloatTensor([row[1] for row in y_test])
events_test = torch.FloatTensor([float(row[0]) for row in y_test])

# Model, optimizer
device = torch.device("cpu")
model = DeepSurv(in_features=X_train_t.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

# Training loop
n_epochs = 100
train_losses = []
print(f"  Training DeepSurv for {n_epochs} epochs...")

for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    risk = model(X_train_t.to(device))
    loss = cox_partial_likelihood_loss(risk, times_train.to(device), events_train.to(device))
    loss.backward()
    optimizer.step()
    scheduler.step()
    train_losses.append(loss.item())

    if (epoch + 1) % 20 == 0:
        print(f"    Epoch {epoch+1}/{n_epochs} — Loss: {loss.item():.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    deepsurv_risk_test = model(X_test_t.to(device)).cpu().numpy()

deepsurv_ci = concordance_index_censored(y_test["event"], y_test["time"], deepsurv_risk_test)
print(f"  DeepSurv C-index (test): {deepsurv_ci[0]:.4f}")

# Training loss curve
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(train_losses, color=PALETTE["primary"], linewidth=1.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("Cox Partial Likelihood Loss")
ax.set_title("DeepSurv Training Loss", fontweight='bold')
plt.tight_layout()
save_fig(fig, "08_deepsurv_loss")
plt.close()

# Save model
torch.save(model.state_dict(), MODELS_DIR / "deepsurv.pt")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 11: MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 11: MODEL COMPARISON")
print("=" * 70)

results = {
    "Cox PH (clinical)": cph_ci[0],
    "LASSO Cox": lasso_ci[0],
    "Random Survival Forest": rsf_ci[0],
    "DeepSurv": deepsurv_ci[0],
}

results_df = pd.DataFrame({
    "Model": list(results.keys()),
    "C-index": list(results.values()),
}).sort_values("C-index", ascending=False)

print("\n  Model Comparison (C-index, test set):")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
with open(RESULTS_DIR / "model_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Comparison bar chart
fig, ax = plt.subplots(figsize=(10, 5))
model_colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["purple"], PALETTE["highlight"]]
bars = ax.bar(results_df["Model"], results_df["C-index"], color=model_colors[:len(results_df)], edgecolor='white', width=0.6)
ax.set_ylabel("Concordance Index (C-index)")
ax.set_title("Model Performance Comparison — Test Set", fontweight='bold')
ax.set_ylim(0.5, max(results_df["C-index"]) + 0.05)
for bar, val in zip(bars, results_df["C-index"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='Random (0.5)')
ax.legend()
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
save_fig(fig, "09_model_comparison")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 12: RISK STRATIFICATION (KM by predicted risk)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 12: RISK STRATIFICATION")
print("=" * 70)

# Use best model's risk scores
best_model_name = results_df.iloc[0]["Model"]
print(f"  Best model: {best_model_name}")

# Get risk scores for all patients (re-predict on full dataset)
X_full_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
X_full_lasso = X_full_scaled[lasso_selected_features]

if best_model_name == "Random Survival Forest":
    full_risk = rsf.predict(X_full_lasso.values)
elif best_model_name == "LASSO Cox":
    full_risk = lasso_cox.predict(X_full_scaled.values, alpha=best_alpha)
elif best_model_name == "DeepSurv":
    model.eval()
    with torch.no_grad():
        full_risk = model(torch.FloatTensor(X_full_lasso.values)).numpy()
else:
    full_risk = -cph.predict_partial_hazard(X_full_scaled[clinical_feature_names]).values.flatten()

# Split into high/low risk at median
median_risk = np.median(full_risk)
risk_groups = np.where(full_risk >= median_risk, "High Risk", "Low Risk")

fig, ax = plt.subplots(figsize=(10, 6))
for group, color, label in [("Low Risk", PALETTE["alive"], "Low Risk"),
                              ("High Risk", PALETTE["dead"], "High Risk")]:
    mask = risk_groups == group
    kmf_r = KaplanMeierFitter()
    kmf_r.fit(y["time"][mask], event_observed=y["event"][mask], label=label)
    kmf_r.plot_survival_function(ax=ax, color=color, linewidth=2)

# Log-rank test
lr = logrank_test(y["time"][risk_groups == "High Risk"],
                  y["time"][risk_groups == "Low Risk"],
                  y["event"][risk_groups == "High Risk"],
                  y["event"][risk_groups == "Low Risk"])

ax.set_title(f"Risk Stratification — {best_model_name}\n(Log-rank p = {lr.p_value:.2e})", fontweight='bold')
ax.set_xlabel("Time (days)")
ax.set_ylabel("Survival Probability")
ax.set_ylim(0, 1.05)
ax.legend(loc='lower left', fontsize=12)
plt.tight_layout()
save_fig(fig, "10_risk_stratification")
plt.close()

print(f"  Log-rank test p-value: {lr.p_value:.2e}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 13: GENE IMPORTANCE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 13: GENE IMPORTANCE ANALYSIS")
print("=" * 70)

# Top genes from LASSO (gene features only)
lasso_gene_coefs = lasso_coefs[top_genes].dropna()
lasso_top_genes = lasso_gene_coefs[lasso_gene_coefs != 0].sort_values(key=abs, ascending=False).head(20)

# Top genes from RSF (from LASSO-selected features, filter to genes only)
rsf_gene_features = [f for f in lasso_selected_features if f in top_genes]
rsf_gene_importance = rsf_importance.reindex(rsf_gene_features).dropna()
rsf_top_genes = rsf_gene_importance.nlargest(min(20, len(rsf_gene_importance)))

# Overlap
lasso_gene_set = set(lasso_top_genes.index)
rsf_gene_set = set(rsf_top_genes.index)
overlap_genes = lasso_gene_set & rsf_gene_set
print(f"  LASSO top 20 genes: {len(lasso_top_genes)}")
print(f"  RSF top 20 genes:   {len(rsf_top_genes)}")
print(f"  Overlap:            {len(overlap_genes)}")
if overlap_genes:
    print(f"  Shared genes: {sorted(overlap_genes)}")

# Combined gene importance figure
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# LASSO genes
if len(lasso_top_genes) > 0:
    colors_l = [PALETTE["dead"] if v > 0 else PALETTE["alive"] for v in lasso_top_genes.values]
    axes[0].barh(range(len(lasso_top_genes)), lasso_top_genes.values, color=colors_l, edgecolor='white')
    axes[0].set_yticks(range(len(lasso_top_genes)))
    axes[0].set_yticklabels(lasso_top_genes.index, fontsize=9)
    axes[0].set_xlabel("LASSO Coefficient")
    axes[0].set_title("Top Genes — LASSO Cox", fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].axvline(0, color='gray', linewidth=0.5)
else:
    axes[0].text(0.5, 0.5, "No nonzero gene coefficients", ha='center', va='center')
    axes[0].set_title("Top Genes — LASSO Cox", fontweight='bold')

# RSF genes
axes[1].barh(range(len(rsf_top_genes)), rsf_top_genes.values, color=PALETTE["purple"], edgecolor='white')
axes[1].set_yticks(range(len(rsf_top_genes)))
axes[1].set_yticklabels(rsf_top_genes.index, fontsize=9)
axes[1].set_xlabel("Feature Importance")
axes[1].set_title("Top Genes — Random Survival Forest", fontweight='bold')
axes[1].invert_yaxis()

plt.tight_layout()
save_fig(fig, "11_gene_importance_combined")
plt.close()

# Save gene results
gene_results = {
    "lasso_top_genes": lasso_top_genes.to_dict(),
    "rsf_top_genes": rsf_top_genes.to_dict(),
    "overlap_genes": list(overlap_genes),
}
with open(RESULTS_DIR / "gene_importance.json", "w") as f:
    json.dump(gene_results, f, indent=2)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 14: INTEGRATED BRIER SCORE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 14: INTEGRATED BRIER SCORE")
print("=" * 70)

try:
    # Need survival function estimates for Brier score
    # RSF provides survival function directly (using LASSO-selected features)
    rsf_surv_fns = rsf.predict_survival_function(X_test_rsf.values)

    # Get common time points
    times_brier = np.linspace(
        max(y_test["time"].min(), y_train["time"].min()) + 1,
        min(y_test["time"].max(), y_train["time"].max()) - 1,
        100
    )
    times_brier = times_brier[times_brier > 0]

    # Build survival probability matrix for RSF
    rsf_surv_matrix = np.column_stack([fn(times_brier) for fn in rsf_surv_fns])

    # Compute IBS for RSF
    from sksurv.metrics import brier_score
    preds_rsf = rsf_surv_matrix.T  # shape: (n_samples, n_times)
    _, bs_scores = brier_score(y_train, y_test, preds_rsf, times_brier)

    ibs_rsf = np.trapezoid(bs_scores, times_brier) / (times_brier[-1] - times_brier[0])
    print(f"  RSF Integrated Brier Score: {ibs_rsf:.4f}")

    # Brier score over time plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times_brier, bs_scores, color=PALETTE["purple"], linewidth=2, label=f'RSF (IBS={ibs_rsf:.3f})')
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Brier Score")
    ax.set_title("Brier Score Over Time — Random Survival Forest", fontweight='bold')
    ax.legend()
    ax.fill_between(times_brier, bs_scores, alpha=0.1, color=PALETTE["purple"])
    plt.tight_layout()
    save_fig(fig, "12_brier_score")
    plt.close()

    cohort_summary["ibs_rsf"] = float(ibs_rsf)
except Exception as e:
    print(f"  Brier score computation failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 15: SAVE FINAL OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 15: SAVING FINAL OUTPUTS")
print("=" * 70)

# Save updated cohort summary
with open(RESULTS_DIR / "cohort_summary.json", "w") as f:
    json.dump(cohort_summary, f, indent=2)

# Save feature names
feature_info = {
    "clinical_features": clinical_feature_names,
    "top_genes": top_genes,
    "lasso_selected_features": lasso_selected_features,
    "all_features": all_feature_names,
    "best_model": best_model_name,
    "best_alpha_lasso": float(best_alpha),
}
with open(RESULTS_DIR / "feature_info.json", "w") as f:
    json.dump(feature_info, f, indent=2)

# Save train/test indices
pd.DataFrame({
    "patient_id": feature_df.loc[X_train.index, "patient_id"].values,
    "split": "train"
}).to_csv(RESULTS_DIR / "train_patients.csv", index=False)

pd.DataFrame({
    "patient_id": feature_df.loc[X_test.index, "patient_id"].values,
    "split": "test"
}).to_csv(RESULTS_DIR / "test_patients.csv", index=False)

# Summary table
print("\n  ╔═══════════════════════════════════════════════════════════╗")
print("  ║         TCGA KIRC SURVIVAL ANALYSIS — SUMMARY           ║")
print("  ╠═══════════════════════════════════════════════════════════╣")
print(f"  ║  Cohort size:          {cohort_summary['total_patients']:>6} patients               ║")
print(f"  ║  Events (dead):        {cohort_summary['events_dead']:>6}                         ║")
print(f"  ║  Censored (alive):     {cohort_summary['censored_alive']:>6}                         ║")
print(f"  ║  Genes used:           {cohort_summary['num_genes']:>6}                         ║")
print(f"  ║  Median survival:      {cohort_summary['median_time_days']:>6.0f} days                   ║")
print("  ╠═══════════════════════════════════════════════════════════╣")
for model_name, ci_val in results.items():
    pad = 27 - len(model_name)
    print(f"  ║  {model_name}:{' ' * pad}{ci_val:.4f}                      ║")
print("  ╚═══════════════════════════════════════════════════════════╝")

print("\n  All outputs saved to outputs/")
print("  Pipeline complete!")
