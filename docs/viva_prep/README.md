# TCGA-KIRC Viva Preparation Guide

This document is a beginner-friendly but technically accurate viva guide for the repository:

- Repository: `LikithaDudala/tcga-kirc-project`
- Main evidence reviewed for this guide:
  - `/home/runner/work/tcga-kirc-project/tcga-kirc-project/README.md`
  - `/home/runner/work/tcga-kirc-project/tcga-kirc-project/app.py`
  - `/home/runner/work/tcga-kirc-project/tcga-kirc-project/src/pipeline.py`
  - `/home/runner/work/tcga-kirc-project/tcga-kirc-project/src/generate_pptx.py`
  - `/home/runner/work/tcga-kirc-project/tcga-kirc-project/notebooks/00_data_exploration.ipynb`
  - `/home/runner/work/tcga-kirc-project/tcga-kirc-project/notebooks/01_survival_analysis_report.ipynb`
  - `/home/runner/work/tcga-kirc-project/tcga-kirc-project/outputs/results/*`

---

## How to Use This Guide

Use this file in 3 ways:

1. **Read Sections 1 to 7** to understand the full project.
2. **Read Sections 8 to 10** to prepare for viva questions and presentation.
3. **Read the Glossary and Quick Revision** one day before the viva.

Every major concept below follows this pattern:

- **Technical definition**
- **Simple explanation**
- **Why we used it in this project**
- **How to say it in viva**

---

## Important Audit Notes Before Your Viva

These points are very useful because they help you answer confidently if an examiner notices number differences.

### 1. Which files are the most trustworthy?

For viva answers, treat these as the most authoritative:

1. `src/pipeline.py`
2. `notebooks/01_survival_analysis_report.ipynb`
3. `outputs/results/*`

These files reflect the actual modeling workflow and saved results.

### 2. Why do some patient counts differ across files?

There are multiple counts in the repository:

- **537 patients** in raw clinical data mentioned in README/presentation text
- **606 samples** in expression data
- **529 patients** in the final merged survival-analysis cohort
- **441 patients** in the actual modeling subset after dropping rows with missing required clinical fields before train/test split
- **533 rows** in `outputs/results/patient_survival.csv`, which is used by the Streamlit dashboard for some overview pages

### Simple explanation

This is normal in biomedical projects. The number becomes smaller step by step because some records do not align across files, some are duplicated, some have missing survival time, and some do not have all required modeling features.

### Safe viva line

> “The raw dataset had more records, but the final analysis cohort became smaller after patient-ID alignment, removal of invalid survival records, merging with tumor-only expression data, and finally dropping patients with missing required clinical variables for model training.”

### 3. One more useful audit note

`src/generate_pptx.py` contains some hardcoded presentation text that does **not fully match** the current saved outputs and pipeline settings.
Examples:

- It mentions **5-fold CV**, but the current `pipeline.py` chooses LASSO alpha by checking test-set C-index across alpha values.
- It mentions **200 trees** for RSF, but `pipeline.py` uses **300 trees**.
- It mentions **200 epochs** for DeepSurv, but `pipeline.py` trains for **100 epochs**.
- It mentions a **370/159 split**, but the saved train/test CSV files show **308 train / 133 test** for the modeling subset.

### Safe viva line

> “For technical details, I rely on the actual pipeline script and saved results, not only on the generated slide text.”

---

# 1. Project Overview

## 1.1 What the project does

### Technical definition

This project builds an **end-to-end survival analysis pipeline** for **TCGA-KIRC** using clinical data and RNA-seq gene expression data, then compares four survival models:

- Cox Proportional Hazards
- LASSO Cox
- Random Survival Forest
- DeepSurv

It also deploys an interactive **Streamlit** application for result exploration and simple risk prediction.

### Simple explanation

The project tries to answer:

> “Using patient clinical details and gene expression data, can we estimate which kidney cancer patients are at higher survival risk?”

### Why we used this in this project

The goal is not only to train a model but to build a complete research-style workflow:

- collect data
- clean and merge it
- train models
- compare them
- show the results in a usable app

### How to say it in viva

> “My project is a TCGA-KIRC survival prediction system. It combines clinical and genomic data, trains multiple survival models, compares them using survival-specific metrics, identifies important genes, and presents the results through a Streamlit dashboard.”

---

## 1.2 Why kidney cancer survival prediction matters

### Technical definition

Kidney Renal Clear Cell Carcinoma, or **KIRC**, is the most common subtype of kidney cancer. Survival prediction helps estimate prognosis, meaning the likely disease outcome over time.

### Simple explanation

Different patients with the same cancer type may still have very different outcomes. Some patients may remain stable for years, while others may worsen faster.

### Why we used it in this project

If we can estimate risk better, doctors and researchers can:

- identify high-risk patients earlier
- study useful biomarkers
- support personalized treatment planning

### How to say it in viva

> “Kidney cancer survival prediction matters because patients with the same diagnosis can still have different outcomes. A survival model can help identify who may need closer monitoring or more aggressive treatment.”

---

## 1.3 Why TCGA-KIRC is an important dataset

### Technical definition

**TCGA** stands for **The Cancer Genome Atlas**, which is a major public cancer dataset containing molecular and clinical information.
**TCGA-KIRC** is the kidney renal clear cell carcinoma cohort inside TCGA.

### Simple explanation

It is a trusted public cancer dataset that gives real patient information, including:

- survival outcomes
- clinical information
- gene expression measurements

### Why we used it in this project

It is ideal because this project needs both:

- **clinical data** for patient characteristics
- **genomic data** for biomarker discovery

### How to say it in viva

> “I selected TCGA-KIRC because it is a widely used public cancer dataset with both survival labels and genomic features, which makes it suitable for survival modeling.”

---

## 1.4 Why survival analysis is different from normal classification

### Technical definition

**Survival analysis** is a statistical and machine-learning approach for modeling **time-to-event data**, where the target is not just whether an event happened, but also **when** it happened, while handling **censoring**.

### Simple explanation

In normal classification, the model predicts a simple label like:

- yes/no
- alive/dead
- benign/malignant

In survival analysis, the question is richer:

- Did the event happen?
- If yes, after how much time?
- If not observed yet, how long was the patient followed?

### Why we used it in this project

In this project, many patients are still alive at last follow-up. So we cannot treat the problem as simple dead/alive classification.

### How to say it in viva

> “Survival analysis is more appropriate than normal classification because it uses both event status and survival time, and it correctly handles censored patients who were still alive at last follow-up.”

---

## 1.5 Difference between predicting “high risk / low risk” and “who dies first”

### Technical definition

A survival model usually learns a **risk ranking**. It estimates which patients have relatively higher hazard, meaning higher event risk over time. It does not necessarily predict the exact date of death.

### Simple explanation

- **High risk / low risk** means grouping patients by relative danger level.
- **Who dies first** means ordering patients by expected event risk over time.

So the model is mostly learning:

> “Patient A is riskier than Patient B.”

not

> “Patient A will die exactly on day 943.”

### Why we used it in this project

The main evaluation metric, **C-index**, rewards correct ranking, not exact date prediction.

### How to say it in viva

> “The model mainly learns relative risk ranking. It is better at saying which patient is riskier than another patient, rather than giving an exact survival date.”

---

## 1.6 Why this project is meaningful in healthcare AI

### Technical definition

This project combines biomedical data science, interpretable survival modeling, biomarker discovery, and deployment.

### Simple explanation

It is meaningful because it is not only an AI model. It also shows:

- real healthcare data usage
- clinically relevant prediction
- gene-level interpretation
- a usable deployed interface

### Why we used it in this project

A good healthcare AI project should be:

- technically correct
- interpretable
- reproducible
- presentable

This repository shows all of those parts.

### How to say it in viva

> “This project is meaningful in healthcare AI because it combines real patient data, survival-specific modeling, interpretability through important genes, and deployment through a dashboard.”

---

# 2. Dataset Explanation

## 2.1 What TCGA is

### Technical definition

TCGA is a large public cancer genomics program that collected molecular and clinical data from many cancer types.

### Simple explanation

It is a big cancer data library for research.

### Why we used it in this project

It provides reliable, public, research-standard cancer data.

### How to say it in viva

> “TCGA is a public cancer genomics resource. I used it because it provides both survival-related clinical data and molecular features.”

---

## 2.2 What KIRC means

### Technical definition

KIRC stands for **Kidney Renal Clear Cell Carcinoma**, the most common kidney cancer subtype.

### Simple explanation

It is a specific type of kidney cancer.

### Why we used it in this project

The project focuses on survival prediction for this cancer subtype.

### How to say it in viva

> “KIRC is kidney renal clear cell carcinoma, which is the main cancer type studied in my project.”

---

## 2.3 Number of patients and samples

Use this table in viva if an examiner asks why counts differ.

| Stage in workflow | Count | Meaning |
|---|---:|---|
| Raw clinical data | 537 | Mentioned in repository documentation |
| Expression data | 606 samples | RNA-seq matrix samples |
| Final merged cohort | 529 patients | Saved cohort summary used for analysis reporting |
| Modeling subset | 441 patients | After dropping missing required clinical fields before split |
| Train/test split | 308 / 133 | Saved in `train_patients.csv` and `test_patients.csv` |
| Dashboard survival CSV | 533 rows | Used by the app for some descriptive plots |

### Safe viva line

> “The most important count for reported model results is the final analysis and modeling cohort, not the raw file size.”

---

## 2.4 Clinical data

### Technical definition

Clinical data contains patient-level medical fields such as:

- vital status
- days to death
- days to last follow-up
- age
- gender
- AJCC pathologic stage

### Simple explanation

These are the patient details used to describe the case medically.

### Why we used it in this project

Clinical variables often already carry strong prognostic information, especially:

- age
- stage
- gender

### How to say it in viva

> “Clinical data gives medically meaningful baseline features. In my pipeline, age, gender, and stage were especially important for modeling.”

---

## 2.5 RNA-seq gene expression data

### Technical definition

**RNA-seq gene expression** data measures how strongly genes are expressed in tumor samples.

### Simple explanation

You can think of this as a very large table that shows which genes are more active or less active in each patient’s tumor.

### Why we used it in this project

Clinical features alone may miss molecular patterns. Gene expression can reveal hidden biological signals related to tumor aggressiveness and survival.

### How to say it in viva

> “Gene expression data lets the model use molecular information, not just visible clinical features. This can improve risk prediction and help identify biomarkers.”

---

## 2.6 Number of genomic features

### Technical definition

The expression matrix contains about **20,530 genes**.

### Simple explanation

This means each patient potentially has more than 20,000 gene-related numeric inputs.

### Why we used it in this project

This high feature count gives rich biological information, but it also creates modeling difficulty.

### How to say it in viva

> “The genomic data is high-dimensional because each patient has values for over 20,000 genes.”

---

## 2.7 Why high-dimensional data is difficult

### Technical definition

**High-dimensional data** means the number of features is very large compared with the number of patients. This increases the risk of overfitting.

### Simple explanation

There are many more columns than rows. So the model can start memorizing noise instead of learning true patterns.

### Why we used it in this project

Here, there are hundreds of patients but tens of thousands of genes.

### How to say it in viva

> “High-dimensional genomic data is difficult because the model can overfit easily when features greatly outnumber samples.”

---

## 2.8 Why feature selection was required

### Technical definition

**Feature selection** means keeping only the most informative input variables for modeling.

### Simple explanation

Instead of feeding every gene to the model, we keep a smaller and more useful subset.

### Why we used it in this project

The pipeline:

1. keeps the **top 2000 genes by variance**
2. then LASSO shrinks many coefficients to zero
3. finally RSF and DeepSurv use the smaller LASSO-selected set

This improves:

- speed
- stability
- interpretability
- reduced overfitting

### How to say it in viva

> “Feature selection was necessary because using all 20,000+ genes directly would be computationally heavy and prone to overfitting.”

---

## 2.9 Why reducing 20k+ features to smaller subsets is reasonable

### Technical definition

Variance filtering and penalized selection are common dimensionality-reduction strategies in genomics.

### Simple explanation

Many genes are not very informative for this task. Some vary very little, and some may not relate strongly to survival.

### Why we used it in this project

Keeping only the most variable and most predictive genes is a practical way to retain signal while reducing noise.

### How to say it in viva

> “Reducing the feature space does not automatically mean losing important information. In genomic modeling, feature selection often improves generalization by removing noisy or weakly informative genes.”

---

# 3. Complete Pipeline Explanation

## 3.1 Full workflow at a glance

```text
Raw TCGA clinical data + follow-up data + RNA-seq expression
                ↓
Patient-ID alignment and survival target construction
                ↓
Tumor-sample filtering and gene matrix transpose
                ↓
Variance filtering to top 2000 genes
                ↓
Merge survival + clinical + gene expression
                ↓
Encode clinical variables and standardize features
                ↓
Train/test split
                ↓
Train 4 survival models
                ↓
Evaluate with C-index (+ RSF Brier score)
                ↓
Risk stratification and gene importance analysis
                ↓
Save models/results and show them in Streamlit
```

---

## 3.2 Data collection

### What was done technically

- Clinical TSV files were loaded from TCGA/GDC.
- Follow-up TSV files were loaded.
- RNA-seq expression matrix was loaded from UCSC Xena.

### Why it was done

The project needs both:

- survival outcome information
- molecular features

### How to explain it in viva

> “I collected clinical survival information from TCGA/GDC and expression data from UCSC Xena, then aligned them by patient ID.”

---

## 3.3 Preprocessing

### What was done technically

- TCGA missing sentinel values like `'--` were replaced with `NaN`.
- Duplicate clinical patient rows were removed using `cases.submitter_id`.
- Survival fields were converted to numeric values.

### Why it was done

Models cannot work reliably with mixed or invalid raw values.

### How to explain it in viva

> “I cleaned the raw files by handling missing markers, removing duplicates, and converting important fields into machine-readable numeric form.”

---

## 3.4 Survival target construction

### What was done technically

The pipeline created:

- `event = 1` if patient is dead
- `event = 0` if patient is censored or alive at last follow-up
- `time = days_to_death` for deceased patients
- `time = days_to_last_follow_up` for censored patients

If censored follow-up time was missing in clinical data, the script checked `follow_up.tsv` and used the maximum available follow-up time.

### Why it was done

Survival modeling needs two target parts:

1. whether the event happened
2. how long the patient was followed

### How to explain it in viva

> “I converted the raw clinical fields into standard survival-analysis targets: event and time. Dead patients use days to death, and alive patients use days to last follow-up.”

---

## 3.5 Tumor-only selection and expression preparation

### What was done technically

- TCGA expression columns were inspected.
- Only tumor samples were kept using barcode segment `01`.
- The gene-by-sample matrix was transposed so rows became patients and columns became genes.
- Patient IDs were extracted from sample barcodes.
- Duplicate tumor samples per patient were reduced by keeping the first sample.

### Why it was done

The model should learn from tumor biology, not from normal tissue samples.

### How to explain it in viva

> “I kept tumor samples only, because the goal is to predict survival from tumor-related biology.”

---

## 3.6 Normalization and scaling

### What was done technically

The pipeline used `StandardScaler` after the train/test split to standardize features.

### Technical definition

**Standardization** means subtracting the mean and dividing by the standard deviation so features are on a comparable scale.

### Simple explanation

It puts different numeric features on a more balanced scale.

### Why it was done

This is especially important for:

- Cox-based linear models
- neural networks like DeepSurv

### How to explain it in viva

> “I standardized features so that variables with larger numeric ranges would not dominate the models unfairly.”

---

## 3.7 Feature selection

### What was done technically

Step 1:

- Compute variance for all genes
- Keep top **2000** genes with highest variance

Step 2:

- Use LASSO Cox
- Keep **40 non-zero selected features** according to saved outputs

### Why it was done

- reduce dimensionality
- keep informative genes
- improve generalization
- make models faster and easier to interpret

### How to explain it in viva

> “I first applied variance filtering to reduce the gene space, and then LASSO performed a second stage of sparse feature selection.”

---

## 3.8 Train-test split

### What was done technically

- The project uses `train_test_split`
- Test size is **30%**
- Split is **stratified by event status**
- Saved output files show **308 training patients** and **133 test patients**

### Why it was done

The model must be evaluated on unseen data. Stratification keeps the event/censoring balance more stable across splits.

### How to explain it in viva

> “I used a 70/30 split and stratified by event status so that both train and test sets had a similar proportion of observed deaths.”

---

## 3.9 Model training

### What was done technically

Four models were trained:

1. Cox PH on clinical features only
2. LASSO Cox on clinical + top-variance genes
3. Random Survival Forest on LASSO-selected features
4. DeepSurv on LASSO-selected features

### Why it was done

This allows comparison between:

- simple linear survival modeling
- sparse penalized modeling
- non-linear tree-based survival modeling
- deep learning survival modeling

### How to explain it in viva

> “I compared four different survival-model families so I could benchmark interpretability, sparsity, non-linearity, and deep learning on the same cohort.”

---

## 3.10 Evaluation

### What was done technically

Primary metric:

- **Concordance Index (C-index)** for all models

Additional metric:

- **Integrated Brier Score (IBS)** for RSF

Saved results:

- LASSO Cox: **0.8047**
- Cox PH: **0.7809**
- DeepSurv: **0.7637**
- RSF: **0.7161**
- RSF IBS: **0.1542**

### Why it was done

Survival models should be evaluated by ranking quality over censored time-to-event data, not by plain accuracy.

### How to explain it in viva

> “I used C-index because it evaluates whether the model ranks higher-risk patients ahead of lower-risk patients while handling censored survival data.”

---

## 3.11 Risk stratification

### What was done technically

- The best model was identified as **LASSO Cox**
- Risk scores were computed for the full dataset
- Patients were split at the **median risk**
- Kaplan-Meier curves were plotted for:
  - High Risk
  - Low Risk
- A **log-rank test** checked whether the survival difference was statistically significant

### Why it was done

This shows whether the model is clinically useful for separating patients into meaningful groups.

### How to explain it in viva

> “After model comparison, I used the best model for risk stratification. Patients above the median risk score were labeled high risk, and the Kaplan-Meier curves showed whether those groups had clearly different survival outcomes.”

---

## 3.12 Deployment

### What was done technically

- Models and outputs were saved in `outputs/`
- A Streamlit app reads saved JSON, CSV, and model files
- The app shows descriptive analytics, model comparison, biomarker importance, and a simple clinical risk demo

### Why it was done

Deployment makes the work easier to demonstrate, understand, and communicate.

### How to explain it in viva

> “I deployed the results using Streamlit so the project is not just a notebook. It becomes an interactive system for exploring the dataset and model outputs.”

---

# 4. Model Explanations

## 4.1 Cox Proportional Hazards

### Technical explanation

The **Cox Proportional Hazards model** is a semi-parametric survival model that estimates how features affect the hazard over time. It assumes **proportional hazards**, meaning hazard ratios remain constant over time.

### Simple explanation

It is a classic survival model that says:

> “How much does each feature increase or decrease risk?”

### Advantages

- interpretable
- clinically accepted
- simple baseline
- hazard ratios are easy to discuss

### Limitations

- assumes proportional hazards
- mostly linear in effect
- may miss complex gene interactions

### Why used in this project

It provides a strong and interpretable baseline using only age, gender, and stage.

### Performance here

- **C-index = 0.7809**

### How to explain it in viva

> “I used Cox PH as the baseline because it is a standard survival model and easy to interpret. It performed well even with only clinical variables.”

---

## 4.2 LASSO Cox

### Technical explanation

**LASSO Cox** is a Cox model with **L1 regularization**, meaning it penalizes the absolute size of coefficients and pushes many coefficients to zero. This performs embedded feature selection.

### Simple explanation

It is a Cox survival model that automatically chooses the most useful features and removes many unimportant ones.

### Advantages

- excellent for high-dimensional data
- performs feature selection automatically
- more interpretable than many black-box models
- reduces overfitting

### Limitations

- still mostly linear
- feature selection may be unstable across cohorts
- correlated genes can compete with each other

### Why used in this project

This project has thousands of genomic features, so LASSO is a very natural choice.

### Performance here

- **Best model**
- **C-index = 0.8047**
- Saved selected features count: **40**

### Why it likely performed best

- it used both clinical and genomic information
- it reduced noise through regularization
- it kept a sparse, high-signal feature subset

### How to explain it in viva

> “LASSO Cox performed best because it balances predictive power and feature selection. It is especially suitable for genomic data where the number of features is much larger than the number of patients.”

---

## 4.3 Random Survival Forest

### Technical explanation

**Random Survival Forest**, or RSF, is a tree-based ensemble survival model. It extends random forests to censored time-to-event data.

### Simple explanation

It is many survival trees working together to learn complex patterns.

### Advantages

- captures non-linear relationships
- handles interactions automatically
- does not require proportional hazards assumption
- provides feature importance

### Limitations

- less interpretable than Cox models
- may need careful tuning
- can underperform on smaller datasets with noisy high-dimensional inputs

### Why used in this project

It checks whether non-linear tree ensembles can outperform linear survival models.

### Performance here

- **C-index = 0.7161**
- **IBS = 0.1542**
- Used **300 trees** in current `pipeline.py`

### How to explain it in viva

> “I used Random Survival Forest to capture possible non-linear relationships between selected features and survival. It was useful for comparison and feature-importance analysis.”

---

## 4.4 DeepSurv

### Technical explanation

**DeepSurv** is a neural-network-based survival model trained with a Cox partial likelihood objective.

### Simple explanation

It is a deep learning version of survival modeling. Instead of a simple linear formula, it learns a more flexible mapping from features to risk.

### Advantages

- can model complex patterns
- useful when feature interactions are non-linear
- combines survival analysis with deep learning

### Limitations

- less interpretable
- needs careful training and tuning
- can overfit on limited sample sizes

### Why used in this project

It gives a modern deep-learning comparison against classical survival approaches.

### Performance here

- **C-index = 0.7637**
- Architecture in pipeline:
  - input
  - hidden 128
  - hidden 64
  - output 1
- Uses batch normalization, ReLU, and dropout
- Trained for **100 epochs** in current `pipeline.py`

### How to explain it in viva

> “I included DeepSurv to test whether a neural-network survival model could capture more complex structure than linear models. It performed reasonably well but did not beat LASSO Cox in this repository.”

---

## 4.5 Which model performed best and why?

### Best model

- **LASSO Cox**
- **C-index = 0.8047**

### Likely reason

It matched the data situation well:

- high-dimensional genomic features
- moderate sample size
- need for feature selection
- survival-specific modeling

### Safe viva line

> “LASSO Cox performed best because it used both clinical and genomic data while controlling overfitting through sparse regularization.”

---

# 5. Survival Analysis Concepts

## 5.1 Survival analysis

### Technical definition

A framework for modeling time until an event occurs.

### Simple explanation

It predicts not just whether something happens, but how risk changes over time.

### Why used here

Because the target is patient survival time.

### Viva line

> “Survival analysis is used when the outcome is time to an event, not just a class label.”

---

## 5.2 Censoring

### Technical definition

**Censoring** means the event was not observed during the study period, even though the patient was followed.

### Simple explanation

The patient did not die during the available follow-up, so we know they survived at least that long, but not their final event time.

### Why used here

Many TCGA patients are alive at last follow-up.

### Viva line

> “Censoring means we only know that the patient survived up to the last observed time.”

---

## 5.3 Hazard function

### Technical definition

The **hazard** is the instantaneous event risk at a given time, assuming the patient has survived up to that time.

### Simple explanation

It is the patient’s current risk level at that moment.

### Why used here

Cox-based models predict relative hazard.

### Viva line

> “Hazard describes the immediate risk of the event at a given time.”

---

## 5.4 Risk score

### Technical definition

A model-generated numeric value representing relative survival risk.

### Simple explanation

Higher score usually means the patient is more likely to experience the event earlier.

### Why used here

Risk scores were used for model comparison and risk-group stratification.

### Viva line

> “The risk score is a relative ranking score, not an exact date prediction.”

---

## 5.5 Kaplan-Meier curve

### Technical definition

A non-parametric estimator of the survival function.

### Simple explanation

It shows the proportion of patients still surviving over time.

### Why used here

It helps visualize overall survival, stage-wise survival, and risk-group separation.

### Viva line

> “A Kaplan-Meier curve shows survival probability over time while handling censoring.”

---

## 5.6 Concordance Index (C-index)

### Technical definition

The C-index measures how often the model correctly ranks pairs of patients by survival risk.

### Simple explanation

If the model says patient A is riskier than patient B, and A really has the event earlier, that is a concordant pair.

### Why used here

It is one of the most standard metrics for survival analysis.

### Viva line

> “C-index tells us how well the model ranks patients by relative risk.”

---

## 5.7 Time-dependent AUC

### Technical definition

**Time-dependent AUC** measures discriminatory ability at a specific time horizon, such as 1 year or 3 years.

### Simple explanation

It asks:

> “At this time point, how well can the model separate patients who had the event from those who did not?”

### Why mention it here

It is not implemented in the current repository, but it is a strong future extension for survival evaluation.

### Viva line

> “Time-dependent AUC is a survival-specific extension of AUC that evaluates discrimination at selected time points.”

---

## 5.8 ROC curve

### Technical definition

A **ROC curve** plots true positive rate against false positive rate across thresholds.

### Simple explanation

It is commonly used in classification problems.

### Why mention it here

Normal ROC is not enough for censored survival data unless adapted into time-dependent ROC.

### Viva line

> “Standard ROC is mainly for classification, so for survival data we usually prefer time-dependent ROC or C-index.”

---

## 5.9 Feature selection

### Technical definition

The process of choosing useful variables and discarding weak or noisy ones.

### Simple explanation

It reduces unnecessary inputs.

### Why used here

To handle thousands of genes safely.

### Viva line

> “Feature selection improves stability, reduces noise, and makes genomic survival modeling more practical.”

---

## 5.10 Hyperparameter tuning

### Technical definition

Choosing model settings that are not directly learned from the data, such as tree count, depth, or regularization strength.

### Simple explanation

These are the knobs of the model.

### Why mention it here

This project uses selected settings like LASSO alpha path, RSF tree parameters, and DeepSurv training settings.

### Viva line

> “Hyperparameter tuning means selecting the best model configuration, not the learned weights themselves.”

---

## 5.11 Ensemble models

### Technical definition

An **ensemble** combines multiple models to improve prediction robustness.

### Simple explanation

It is like taking the opinion of several models instead of only one.

### Why mention it here

RSF itself is an ensemble of trees, and future work may combine multiple survival models together.

### Viva line

> “An ensemble can improve performance by combining strengths of different models.”

---

## 5.12 Transfer learning for tabular data

### Technical definition

**Transfer learning** means reusing knowledge learned on one dataset or task to help on another.

### Simple explanation

The model first learns from one problem and then uses that experience on a new but related problem.

### Why mention it here

It is a future research direction, especially if larger pan-cancer tabular datasets are available.

### Viva line

> “Transfer learning is common in images and language, but it is harder in tabular biomedical data because features and distributions vary a lot between datasets.”

---

## 5.13 Deep learning for tabular data

### Technical definition

Using neural networks on structured column-based data rather than images or text.

### Simple explanation

This means applying deep learning to spreadsheet-like data.

### Why mention it here

DeepSurv is an example of deep learning for tabular survival data.

### Viva line

> “Deep learning for tabular data can capture complex relationships, but it often needs careful tuning and enough data.”

---

## 5.14 TabNet

### Technical definition

**TabNet** is a deep-learning architecture designed specifically for tabular data using sequential attention.

### Simple explanation

It tries to focus on the most useful columns step by step.

### Why mention it here

It is a possible future extension for tabular biomedical modeling, though it is not implemented in this repository.

### Viva line

> “TabNet is a specialized deep-learning model for tabular data and could be explored in future survival versions.”

---

## 5.15 FT-Transformer

### Technical definition

**FT-Transformer** is a transformer-based architecture adapted for tabular features.

### Simple explanation

It uses attention mechanisms to learn relationships across input columns.

### Why mention it here

It is another future option for advanced tabular survival modeling.

### Viva line

> “FT-Transformer is a transformer model for tabular data and could be adapted for survival prediction in future work.”

---

## 5.16 Why transfer learning is harder for tabular data than images

### Technical definition

Tabular datasets differ strongly in feature meaning, scale, missingness, and schema.

### Simple explanation

In images, the idea of edges, shapes, and textures transfers well. In tabular data, one dataset’s columns may have completely different meaning from another dataset’s columns.

### Why mention it here

This explains why simple transfer-learning ideas from computer vision do not directly work well for genomic survival tables.

### Viva line

> “Transfer learning is harder for tabular data because columns are dataset-specific, so learned representations do not transfer as naturally as image features.”

---

## 5.17 Why C-index is more appropriate than accuracy

### Technical definition

Accuracy assumes a fixed class label, while C-index measures correct risk ranking for censored time-to-event data.

### Simple explanation

Accuracy asks:

> “Did the model say dead or alive correctly?”

C-index asks:

> “Did the model rank risk correctly over survival time?”

### Why used here

The project is about time-to-event survival prediction, not simple classification.

### Viva line

> “C-index is better than accuracy here because survival analysis needs ranking over censored time data, not just a yes/no label.”

---

# 6. Deployment / Streamlit App Walkthrough

File: `/home/runner/work/tcga-kirc-project/tcga-kirc-project/app.py`

## 6.1 App architecture summary

### What the app loads

- `cohort_summary.json`
- `model_comparison.csv`
- `gene_importance.json`
- `lasso_coefficients.csv`
- `rsf_feature_importance.csv`
- `feature_info.json`
- `model_results.json`
- `patient_survival.csv`
- `cox_ph.pkl`
- `scaler.pkl`

### Important point

The **risk prediction demo uses the clinical Cox model**, not the best LASSO model. This is reasonable for demo simplicity because the user only enters clinical inputs.

### Safe viva line

> “The deployed app mainly presents precomputed results, and the live prediction demo uses the clinical Cox model because it can work from simple user-entered clinical inputs.”

---

## 6.2 Sidebar

### What it does

- shows project title
- provides navigation
- lists dataset/model/metric summary

### How to demo it

Say:

> “The sidebar acts as the app’s navigation and quick summary area.”

### Questions examiners may ask

- Why use Streamlit?
- Why use a single-page dashboard with navigation instead of multiple files?

### Good answer

> “Streamlit is fast for data-science deployment and lets me combine charts, model summaries, and a simple prediction interface in one place.”

---

## 6.3 Page 1: Dataset Overview

### What the page does

It shows descriptive statistics of the cohort.

### What is displayed

- KPI cards:
  - total patients
  - deaths
  - censored
  - median follow-up
  - genes selected
- Vital status donut chart
- Age histogram
- Stage distribution bar chart
- Gender distribution bar chart
- Survival/follow-up time histogram

### What the graphs mean

- **Vital status donut**: percentage dead vs alive/censored
- **Age histogram**: patient age spread
- **Stage bar chart**: number of patients in each stage
- **Gender bar chart**: male/female counts
- **Time histogram**: how long patients were followed or survived

### Inputs taken

- none

### Outputs shown

- descriptive cohort understanding

### How to demo this page during viva

1. Start with total cohort summary
2. Mention event-censor balance
3. Point to stage distribution
4. Explain why stage matters for prognosis

### What to say while showing the page

> “This page introduces the dataset. It shows the patient count, number of observed deaths, follow-up duration, and the main demographic and clinical distributions.”

### Questions examiners may ask

- Why is censoring shown separately?
- Why is stage distribution important?
- Why is median follow-up useful?

### Good answer

> “These plots help establish cohort quality and clinical context before modeling. Stage and follow-up length are especially important for survival interpretation.”

---

## 6.4 Page 2: Kaplan-Meier Curves

This section has 3 tabs.

### Tab A: Overall Survival

#### What it does

Shows the Kaplan-Meier survival curve for the full cohort.

#### What the graph means

The curve shows the estimated probability of remaining alive over time.

#### Inputs

- none

#### Outputs

- survival curve
- median survival
- 12-month survival estimate

#### How to demo

Say:

> “This is the cohort-level survival pattern. The curve drops over time as more events occur.”

#### Likely questions

- What is median survival?
- Why does the curve step down?

#### Good answer

> “Median survival is the time at which estimated survival falls to 50%. The curve is stepwise because it updates at observed event times.”

---

### Tab B: By Cancer Stage

#### What it does

Shows separate Kaplan-Meier curves for Stage I to Stage IV.

#### What the graph means

Better stages should usually stay higher on the graph, meaning better survival.

#### Inputs

- none

#### Outputs

- stage-wise survival curves
- log-rank test result for Stage I vs Stage IV

#### How to demo

Say:

> “This tab shows that clinically advanced disease tends to have poorer survival. The log-rank test checks whether the difference between groups is statistically significant.”

#### Likely questions

- What is a log-rank test?
- Why compare Stage I and Stage IV?

#### Good answer

> “The log-rank test compares survival curves statistically. Stage I and Stage IV are the most clinically distinct groups, so this contrast is intuitive.”

---

### Tab C: Stage Comparison Table

#### What it does

Shows survival summary values per stage.

#### What the table means

For each stage it displays:

- patient count
- deaths
- censored
- median survival
- 1-year, 3-year, and 5-year survival estimates

#### How to demo

Say:

> “This table converts the survival curves into numbers that are easier to quote in discussion.”

#### Likely questions

- Why include both graph and table?

#### Good answer

> “The graph is better for pattern recognition, while the table is better for exact comparison.”

---

## 6.5 Page 3: Model Performance

### What the page does

It compares the four survival models.

### What is displayed

- best model callout
- C-index horizontal bar chart
- metrics table
- interpretation band for C-index values
- short expandable descriptions of each model

### What the graph means

The longer the bar, the better the model’s ranking ability.

### Inputs

- none

### Outputs

- ranked model comparison

### How to demo

1. Point out the best model
2. State that C-index is the main metric
3. Explain why LASSO Cox likely won

### What to say while showing the page

> “This page compares the four survival models using C-index. In the current outputs, LASSO Cox gives the best test-set performance.”

### Questions examiners may ask

- Why not use accuracy?
- Why did DeepSurv not win?
- Why include RSF if it performed lower?

### Good answers

- “Accuracy is not ideal for censored time-to-event data.”
- “Deep models need more tuning and sometimes more data.”
- “Including RSF was still valuable because it tests non-linearity and provides feature-importance analysis.”

---

## 6.6 Page 4: Gene Importance

This section has 4 tabs.

### Tab A: LASSO Cox Coefficients

#### What it does

Shows top positive and negative coefficients.

#### What the graph means

- positive coefficient: higher risk
- negative coefficient: protective effect

#### Inputs

- slider for number of top features

#### Outputs

- horizontal bar chart of coefficients

#### How to demo

Say:

> “This tab shows which selected genes and features push risk upward or downward in the LASSO Cox model.”

---

### Tab B: RSF Permutation Importance

#### What it does

Shows feature importance for RSF.

#### What the graph means

Permutation importance measures how much performance drops when a feature is randomly shuffled.

#### Inputs

- same top-N slider

#### Outputs

- RSF importance bar chart

#### How to demo

Say:

> “This tab highlights which features the Random Survival Forest depends on most.”

---

### Tab C: Overlap Analysis

#### What it does

Compares top genes from LASSO and RSF.

#### What the graph means

Shared genes are more convincing candidate biomarkers because two different model families found them useful.

#### Inputs

- none

#### Outputs

- counts of LASSO-only, shared, and RSF-only genes
- table of overlap genes

#### How to demo

Say:

> “Overlap across models increases confidence that these genes are not just artifacts of one method.”

---

### Tab D: Full Table

#### What it does

Provides a searchable feature table.

#### Inputs

- text search box

#### Outputs

- combined feature table with LASSO coefficient and RSF importance

#### How to demo

Search for one overlap gene such as `ITPKA` or `EREG`.

#### Likely questions for the whole page

- Are these genes clinically validated biomarkers?
- Does importance imply causation?

#### Good answer

> “No. These are model-associated predictive features, not proof of biological causation. External validation and biological study would still be needed.”

---

## 6.7 Page 5: Risk Prediction Demo

### What the page does

It lets the user enter a simple clinical profile and get:

- partial hazard
- risk category
- predicted survival curve
- landmark survival probabilities

### Inputs

- age slider
- gender radio button
- stage selection slider

### Outputs

- predicted partial hazard
- low/moderate/high risk badge
- predicted survival function
- table of survival probabilities at landmark times

### What the graph means

- gray dashed curve: population average survival
- colored curve: current patient’s predicted survival

### Important technical note

This demo uses the **clinical Cox model**, because the user provides only clinical inputs. The best LASSO model also needs genomic features, which are not practical for quick manual entry in a demo.

### How to demo during viva

Use two examples:

1. younger Stage I patient
2. older Stage IV patient

Show how the curve shifts and how the risk label changes.

### What to say while showing the page

> “This page demonstrates how the survival model can convert a patient profile into a relative risk estimate and survival curve. For simplicity, the live demo uses the clinical Cox model.”

### Questions examiners may ask

- Why not use the best model in the demo?
- What is partial hazard?
- Is this ready for real hospital use?

### Good answers

- “The best model needs genomic input, so the clinical Cox model is more practical for interactive manual demonstration.”
- “Partial hazard is a relative risk value from the Cox model.”
- “No, this is a research and demonstration tool, not a clinical decision system.”

---

# 7. Codebase Walkthrough

## 7.1 Repository structure

```text
tcga-kirc-project/
├── README.md
├── app.py
├── requirements.txt
├── notebooks/
│   ├── 00_data_exploration.ipynb
│   └── 01_survival_analysis_report.ipynb
├── outputs/
│   ├── TCGA_KIRC_Presentation.pptx
│   ├── models/
│   └── results/
└── src/
    ├── pipeline.py
    └── generate_pptx.py
```

---

## 7.2 What each file/script does

### `README.md`

- project summary
- setup instructions
- dataset description
- main results
- app link

### `app.py`

- main Streamlit dashboard
- loads saved outputs
- shows plots and demo prediction

### `requirements.txt`

- Python package list for the project

### `notebooks/00_data_exploration.ipynb`

- checks data loading
- inspects columns and patient IDs
- verifies cross-file alignment

### `notebooks/01_survival_analysis_report.ipynb`

- main notebook version of the full analysis
- contains the complete modeling workflow

### `src/pipeline.py`

- standalone script version of the full end-to-end pipeline
- most important technical file for viva

### `src/generate_pptx.py`

- generates PowerPoint presentation from outputs
- useful supporting file, but not the main modeling logic

### `outputs/models/`

- saved trained models
- includes `cox_ph.pkl`, `lasso_cox.pkl`, `rsf.pkl`, `deepsurv.pt`, `scaler.pkl`

### `outputs/results/`

- saved metrics, feature lists, and analysis tables

### `outputs/TCGA_KIRC_Presentation.pptx`

- generated presentation artifact

---

## 7.3 Which files are core pipeline files?

Primary core files:

- `src/pipeline.py`
- `notebooks/01_survival_analysis_report.ipynb`

Supporting pipeline understanding:

- `notebooks/00_data_exploration.ipynb`
- `outputs/results/*`

---

## 7.4 Which files are deployment files?

- `app.py`
- `outputs/models/*`
- `outputs/results/*`

The app depends on saved outputs rather than retraining models on the fly.

---

## 7.5 Where preprocessing happens?

Main preprocessing is in:

- `src/pipeline.py`
- notebook equivalent in `01_survival_analysis_report.ipynb`

Key preprocessing tasks there:

- missing value cleanup
- duplicate removal
- survival target construction
- patient-ID alignment
- tumor sample filtering
- feature scaling

---

## 7.6 Where model training happens?

Main training happens in:

- `src/pipeline.py`

Training sections:

- Section 7: Cox PH
- Section 8: LASSO Cox
- Section 9: RSF
- Section 10: DeepSurv

---

## 7.7 Where predictions happen?

### In pipeline

- model prediction on test set for evaluation
- full-dataset risk prediction for risk stratification

### In app

- `app.py` risk demo predicts partial hazard and survival function using saved Cox model

---

## 7.8 Where Streamlit UI logic exists?

All UI logic is mainly in:

- `app.py`

It contains:

- sidebar navigation
- page sections
- tab logic
- plot generation
- live clinical-input prediction

---

## 7.9 Important code blocks in simple language

### A. Survival target construction

In `pipeline.py`, the code converts raw clinical fields into:

- `time`
- `event`

Simple meaning:

> “This block creates the standard survival labels needed by survival models.”

---

### B. Variance-based gene filtering

The code computes gene variance and keeps the top 2000 genes.

Simple meaning:

> “This block removes low-information genes and keeps the most variable ones.”

---

### C. Clinical feature encoding

The code builds:

- `age`
- `is_male`
- `stage_num`

Simple meaning:

> “This block converts medical categories into machine-learning input columns.”

---

### D. StandardScaler

Simple meaning:

> “This block rescales features so the models train more fairly and stably.”

---

### E. LASSO coefficient extraction

The code reads non-zero coefficients after fitting LASSO Cox.

Simple meaning:

> “This block tells us which features survived regularization and therefore matter most.”

---

### F. Risk stratification

The code uses the best model’s risk scores and splits patients at the median.

Simple meaning:

> “This block turns raw risk scores into clear high-risk and low-risk patient groups.”

---

### G. App prediction block

In `app.py`, the user enters age, gender, and stage; the app scales those inputs, uses the saved Cox model, and plots a survival curve.

Simple meaning:

> “This block turns user input into an interactive survival prediction demo.”

---

# 8. Viva Preparation

## 8.1 Common viva questions with simple answers

### Q1. What is the aim of your project?

**Answer:**
To predict survival risk in TCGA-KIRC patients using clinical and gene expression data and compare multiple survival models.

### Q2. Why did you choose survival analysis instead of classification?

**Answer:**
Because the target includes both event status and time, and many patients are censored.

### Q3. What is censoring?

**Answer:**
It means the event was not observed within the follow-up period, but we still know the patient survived at least until the last recorded time.

### Q4. Which model performed best?

**Answer:**
LASSO Cox with a test C-index of about 0.8047.

### Q5. Why did LASSO Cox perform best?

**Answer:**
Because it handled high-dimensional genomic data well, selected a sparse set of informative features, and reduced overfitting.

### Q6. What metric did you use?

**Answer:**
Mainly the Concordance Index, because it is appropriate for censored survival ranking.

### Q7. Why not use accuracy?

**Answer:**
Accuracy ignores survival time and censoring, so it is not appropriate for this task.

### Q8. What are the important clinical features?

**Answer:**
Age, stage, and gender were the main clinical features in the current pipeline.

### Q9. What are the important genes?

**Answer:**
Examples from saved outputs include C8orf47, C19orf77, PLEKHG4B, MUC5B, ITPKA, and EREG.

### Q10. Is this a clinical product?

**Answer:**
No. It is a research and educational prototype based on public retrospective data.

---

## 8.2 Deep technical viva questions

### Q1. What assumption does Cox PH make?

**Answer:**
It assumes proportional hazards, meaning relative hazard ratios remain constant over time.

### Q2. How did you handle missing follow-up time for censored patients?

**Answer:**
The pipeline looks into `follow_up.tsv` and uses the maximum available follow-up value when the clinical field is missing.

### Q3. Why use tumor-only samples?

**Answer:**
Because survival prediction should be based on tumor biology rather than normal tissue expression.

### Q4. Why use variance filtering before LASSO?

**Answer:**
It reduces the genomic feature space first, making downstream modeling more stable and computationally feasible.

### Q5. Why does the app use Cox PH instead of LASSO Cox?

**Answer:**
The app accepts only simple clinical inputs. LASSO Cox also requires genomic inputs, which are not practical for quick manual entry.

### Q6. Why did RSF not outperform LASSO here?

**Answer:**
Possible reasons include limited sample size, noise in high-dimensional biology, and the fact that sparse linear structure may already capture the strongest signal in this cohort.

### Q7. What is permutation importance?

**Answer:**
It measures how much model performance drops when one feature is randomly shuffled. A bigger drop means the feature was more useful.

### Q8. Why is the risk score a relative value?

**Answer:**
Because Cox-type models mainly rank patients by hazard rather than directly predicting an exact event time.

---

## 8.3 Safe fallback answers if you forget details

Use these exact lines if you get stuck.

### Fallback 1

> “I want to answer carefully: in my project the main idea is relative risk ranking over survival time, not simple dead-versus-alive classification.”

### Fallback 2

> “The authoritative technical details come from my pipeline script and saved outputs, because some presentation text is template-based.”

### Fallback 3

> “I may not remember the exact number from memory, but conceptually the count reduces after patient alignment, survival cleaning, and missing-value filtering.”

### Fallback 4

> “The key reason for using LASSO Cox is that genomic data is high-dimensional, so sparse regularization is very useful.”

### Fallback 5

> “This result should be interpreted as predictive association, not biological causation.”

---

## 8.4 Questions specifically about deployment

### Q1. Why did you choose Streamlit?

**Answer:**
Because it is simple, fast for data-science apps, and well suited for interactive visualizations and model demos.

### Q2. What is deployed in the app?

**Answer:**
Saved results, precomputed analysis files, and a simple Cox-model-based risk prediction interface.

### Q3. Does the app retrain models live?

**Answer:**
No. It loads pre-trained artifacts from `outputs/models` and precomputed result files from `outputs/results`.

### Q4. What are the limitations of this deployment?

**Answer:**
It is a research demo, not a hospital-grade clinical system. It uses simplified inputs and has no external cohort validation inside the app.

---

## 8.5 Questions specifically about AI-assisted coding

### Q1. How do you answer if examiner asks whether AI tools were used?

Use a calm, honest, professional answer:

> “Yes, I used AI tools as productivity assistants for help with coding support, documentation refinement, and structuring explanations. But I verified the logic, understood the pipeline, checked the outputs, and I can explain the technical decisions and workflow myself.”

### Stronger version

> “AI tools assisted me in productivity, but the project understanding, validation of outputs, integration of components, and final technical responsibility were mine.”

### If they ask whether AI wrote the whole project

> “No. AI assistance may help speed up development, but I reviewed the code, validated the results, and I understand the architecture, modeling choices, and deployment flow.”

### If they ask why using AI is acceptable

> “In modern software and data-science workflows, AI can assist with productivity. What matters academically is whether I understand the system, can justify the technical choices, and can independently explain the implementation and results.”

---

# 9. Presentation Help

## 9.1 Concise presentation flow

1. Problem statement
2. Why KIRC and why survival analysis
3. Dataset and challenges
4. Pipeline workflow
5. Models used
6. Results and best model
7. Important genes and interpretation
8. Deployment demo
9. Limitations and future scope
10. Conclusion

---

## 9.2 What to say slide-by-slide

### Slide 1: Title

> “Good morning/afternoon. My project is TCGA-KIRC survival prediction using multi-model machine learning and survival analysis.”

### Slide 2: Problem

> “Patients with the same kidney cancer type can still have very different outcomes. So we need better survival-risk estimation than simple staging alone.”

### Slide 3: Dataset

> “I used TCGA clinical data and RNA-seq gene expression data, which together provide both medical and molecular information.”

### Slide 4: Challenge

> “The main challenges were censored survival data and very high-dimensional genomic features.”

### Slide 5: Pipeline

> “My workflow was data collection, preprocessing, survival target construction, feature selection, model training, evaluation, risk stratification, and deployment.”

### Slide 6: Models

> “I compared Cox PH, LASSO Cox, Random Survival Forest, and DeepSurv to evaluate classical, sparse, ensemble, and deep-learning survival methods.”

### Slide 7: Results

> “LASSO Cox achieved the best performance with a C-index of about 0.8047.”

### Slide 8: Gene importance

> “The important genes identified by LASSO and RSF may act as candidate prognostic biomarkers, though they still need external biological validation.”

### Slide 9: Deployment

> “I deployed the results in a Streamlit app to make the project interactive and easier to demonstrate.”

### Slide 10: Conclusion

> “Overall, the project shows that combining clinical and genomic information with survival-specific modeling can produce meaningful risk stratification in TCGA-KIRC.”

---

## 9.3 Simple wording that sounds professional

Use these phrases:

- “end-to-end survival analysis pipeline”
- “time-to-event prediction task”
- “high-dimensional genomic feature space”
- “survival-specific evaluation metric”
- “relative risk stratification”
- “interpretable sparse feature selection”
- “research-oriented deployment dashboard”

---

## 9.4 Good transitions between slides

- “After defining the clinical problem, I will now explain the dataset.”
- “Now that we understand the data, the next step is the preprocessing pipeline.”
- “Once the cohort was prepared, I compared four survival models.”
- “After comparing the models, I analyzed the most important features.”
- “Finally, I deployed the results in a Streamlit application.”

---

## 9.5 Professional thank-you conclusion

> “In conclusion, this project demonstrates a complete survival-analysis workflow for TCGA-KIRC, from raw data integration to model comparison and deployment. Thank you for listening, and I welcome your questions.”

---

# 10. Advanced Research Extensions

## 10.1 Ensemble survival models

### What it means

Combine predictions from multiple survival models.

### Why useful

Different models may capture different parts of the signal.

### Viva line

> “A future extension is to combine complementary survival models through stacking or weighted ensembling.”

---

## 10.2 Survival XGBoost

### What it means

Gradient-boosted tree methods adapted for survival objectives.

### Why useful

Can model complex non-linear patterns and often perform strongly on tabular biomedical data.

### Viva line

> “Survival XGBoost is a promising future direction because boosting often works well on structured tabular data.”

---

## 10.3 Time-dependent AUC optimization

### What it means

Instead of optimizing only global ranking, focus on performance at clinically important times such as 1 year or 5 years.

### Why useful

Sometimes clinicians care about specific horizons.

### Viva line

> “Future work could optimize the model for clinically important time points using time-dependent AUC.”

---

## 10.4 Hyperparameter tuning strategies

Possible improvements:

- nested cross-validation
- randomized search
- Bayesian optimization
- Optuna-style tuning

### Viva line

> “A stronger research extension would be more rigorous hyperparameter tuning with cross-validation or Bayesian optimization.”

---

## 10.5 TabNet / FT-Transformer possibilities

### Why useful

These models are designed for tabular data and may learn more advanced feature interactions.

### Caution

They need careful training and may not always outperform sparse linear baselines on smaller biomedical cohorts.

### Viva line

> “TabNet and FT-Transformer are interesting future options, especially if larger multi-cohort survival data becomes available.”

---

## 10.6 Why transfer learning for tabular survival data is difficult

Main reasons:

- feature mismatch between datasets
- differences in preprocessing pipelines
- different patient populations
- different sequencing platforms
- censoring patterns vary

### Viva line

> “Transfer learning in tabular survival data is difficult because datasets often have different feature definitions and different clinical distributions.”

---

## 10.7 How some research papers improve C-index beyond 0.9

Typical reasons:

- larger datasets
- external harmonized cohorts
- stronger feature engineering
- multi-omics integration
- rigorous tuning
- careful leakage prevention
- cohort-specific tasks that are easier than general prediction

### Important caution

A very high C-index is not automatically better science unless:

- the cohort is large enough
- evaluation is fair
- leakage is avoided
- external validation is done

### Viva line

> “Very high reported C-index values usually depend on stronger data integration, careful tuning, and sometimes external validation, but they must also be checked for robustness and leakage.”

---

# 11. Glossary

| Term | Meaning in simple words |
|---|---|
| Survival analysis | Predicting time until an event happens |
| Event | Outcome of interest, here death |
| Censoring | Event not observed during follow-up |
| Hazard | Instantaneous event risk |
| Risk score | Relative danger level predicted by model |
| Kaplan-Meier curve | Survival probability over time |
| Log-rank test | Statistical test comparing survival curves |
| C-index | How well the model ranks patients by risk |
| Brier score | Prediction error over time |
| RNA-seq | Gene expression measurement technique |
| High-dimensional data | Many more features than samples |
| Feature selection | Keeping useful features and discarding the rest |
| LASSO | L1 regularization that pushes some coefficients to zero |
| RSF | Random Survival Forest |
| DeepSurv | Neural-network survival model |
| Deployment | Making the project usable through an app |
| Streamlit | Python framework for interactive dashboards |

---

# 12. Quick Revision Section

Read this section just before the viva.

## 12.1 30-second summary

> “This project builds an end-to-end survival analysis system for TCGA-KIRC using clinical and gene expression data. I cleaned and merged the data, created survival targets, reduced the genomic feature space, trained four survival models, evaluated them mainly with C-index, found that LASSO Cox performed best, analyzed important genes, and deployed the results in a Streamlit dashboard.”

## 12.2 Must-remember numbers

- Cancer type: **TCGA-KIRC**
- Gene count before filtering: **20,530**
- Variance-filtered genes: **2000**
- LASSO-selected features: **40**
- Final merged cohort reported in outputs: **529**
- Modeling split saved in outputs: **308 train / 133 test**
- Best model: **LASSO Cox**
- Best C-index: **0.8047**
- Cox PH C-index: **0.7809**
- DeepSurv C-index: **0.7637**
- RSF C-index: **0.7161**
- RSF IBS: **0.1542**

## 12.3 Must-remember concept lines

- Survival analysis predicts **time to event**, not just class label.
- Censoring means the event was **not observed yet**.
- C-index measures **correct risk ranking**.
- LASSO is good for **high-dimensional sparse feature selection**.
- The app demo uses **clinical Cox PH** for practical user input.

## 12.4 If you get nervous

Say this:

> “I will explain the workflow step by step: data collection, survival target creation, feature selection, model comparison, result interpretation, and deployment.”

---

# 13. Final Viva Strategy

1. Start with the problem, not with code.
2. Then explain why survival analysis is needed.
3. Then explain the data challenge: censoring + 20k+ genes.
4. Then explain the pipeline in sequence.
5. Then explain the best model and why it won.
6. Then explain the app as a communication layer.
7. If asked very technical questions, come back to:
   - event/time construction
   - feature selection
   - C-index
   - risk stratification
8. If you forget a number, explain the logic confidently first.

---

## Closing Line You Can Reuse

> “Overall, this repository demonstrates a complete survival-analysis workflow for kidney cancer, from raw TCGA data to interpretable modeling and deployment, with LASSO Cox giving the strongest performance in the current results.”
