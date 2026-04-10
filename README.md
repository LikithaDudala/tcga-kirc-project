# TCGA KIRC — Survival Analysis

> **Kidney Renal Clear Cell Carcinoma** survival prediction using multi-model machine learning on TCGA genomic + clinical data.

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

This project builds a complete survival analysis pipeline for TCGA-KIRC (kidney renal clear cell carcinoma):

- **529 patients** from The Cancer Genome Atlas (TCGA)
- **4 survival models**: Cox PH, LASSO-penalized Cox, Random Survival Forest, DeepSurv
- **Best C-index: 0.805** (LASSO Cox, clinical + gene expression features)
- **Interactive Streamlit dashboard** for exploring results and predicting patient risk

---

## Project Structure

```
tcga-kirc-project/
├── app.py                              # Streamlit dashboard
├── requirements.txt                    # Python dependencies
├── data/                               # Raw data (not in git — too large)
│   ├── clinical.tsv                    # TCGA clinical data (537 patients)
│   ├── follow_up.tsv                   # Follow-up records
│   └── kirc_expression.tsv             # RNA-seq expression matrix (20,530 genes × 606 samples)
├── notebooks/
│   ├── 00_data_exploration.ipynb          # Data load, shape inspection & patient ID alignment
│   └── 03_survival_analysis_report.ipynb  # Full analysis report (main deliverable)
├── outputs/
│   └── results/                        # Pre-computed results (JSON / CSV)
│       ├── cohort_summary.json
│       ├── model_comparison.csv
│       ├── model_results.json
│       ├── gene_importance.json
│       ├── lasso_coefficients.csv
│       ├── patient_survival.csv
│       └── rsf_feature_importance.csv
└── src/
    ├── pipeline.py                     # Standalone end-to-end pipeline script
    └── generate_pptx.py                # PowerPoint report generator
```

---

## Dataset

| Source | Description |
|--------|-------------|
| [TCGA-KIRC (GDC Portal)](https://portal.gdc.cancer.gov/projects/TCGA-KIRC) | Clinical TSVs: `clinical.tsv`, `follow_up.tsv` |
| [UCSC Xena](https://xenabrowser.net/datapages/) | RNA-seq gene expression: `kirc_expression.tsv` |

**Data files are not included in the repository** (too large). Download from the links above.

---

## Models & Results

| Model | C-index (test) | Description |
|-------|---------------|-------------|
| **LASSO Cox** ⭐ | **0.8047** | L1-penalized Cox regression, clinical + top 2000 genes |
| Cox PH (clinical) | 0.7809 | Baseline Cox on 3 clinical features |
| DeepSurv | 0.7637 | MLP with Cox partial likelihood loss (PyTorch) |
| Random Survival Forest | 0.7161 | 300 trees, LASSO-selected features |

- **Integrated Brier Score (RSF)**: 0.154 *(lower = better, 0.25 = random)*
- **Cohort**: 529 patients — 173 events (32.7% death rate)
- **Median follow-up**: 1,191 days

### Key Survival-Associated Genes

From LASSO Cox and RSF, the top predictive genes include:

| Gene | Direction | Both models |
|------|-----------|------------|
| PLEKHG4B | ↑ higher risk | No |
| MUC5B | ↑ higher risk | No |
| C8orf47 | ↓ protective | Yes |
| C19orf77 | ↓ protective | Yes |
| ITPKA | ↑ higher risk | Yes |
| EREG | ↑ higher risk | Yes |

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/LikithaDudala/tcga-kirc-project.git
cd tcga-kirc-project
```

### 2. Create virtual environment & install dependencies

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Add data files

Place the following files in `data/`:
- `clinical.tsv`
- `follow_up.tsv`
- `kirc_expression.tsv`

### 4. Run the pipeline (optional — outputs already included)

```bash
python src/pipeline.py
```

This trains all models and saves outputs to `outputs/` (takes ~10–15 minutes).

---

## Running the Dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Dashboard Sections

| Section | Description |
|---------|-------------|
| 📊 Dataset Overview | Cohort statistics, vital status, age, gender, stage distributions |
| 📈 Kaplan-Meier Curves | Overall survival + by cancer stage with log-rank test |
| 🏆 Model Performance | C-index comparison, model descriptions |
| 🔬 Gene Importance | LASSO coefficients, RSF importances, overlap analysis |
| ⚕️ Risk Prediction Demo | Enter clinical profile → predicted survival curve |

---

## Running the Notebooks

```bash
# Full analysis report (main deliverable — trains models, saves all outputs)
jupyter notebook notebooks/03_survival_analysis_report.ipynb

# Data exploration (optional — load/inspect raw files, verify patient ID overlap)
jupyter notebook notebooks/00_data_exploration.ipynb
```

---

## Requirements

Key packages (see `requirements.txt` for full list):

```
pandas >= 2.0
numpy
matplotlib
seaborn
plotly
lifelines
scikit-survival
scikit-learn
streamlit
torch
joblib
```

---

## References

1. The Cancer Genome Atlas Research Network. "Comprehensive molecular characterization of clear cell renal cell carcinoma." *Nature* 499 (2013).
2. Davidson-Pilon, C. et al. *lifelines: Survival analysis in Python*. JOSS, 2019.
3. Pölsterl, S. *scikit-survival: A Library for Time-to-Event Analysis Built on Top of scikit-learn*. JMLR, 2020.
4. Katzman, J.L. et al. "DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network." *BMC Medical Research Methodology* 18 (2018).
5. GDC Data Portal: https://portal.gdc.cancer.gov

---

*MS Elevate Internship Project — TCGA KIRC Survival Analysis*
