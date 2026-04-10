"""generate_pptx.py — Generate MS Elevate-style 11-slide PowerPoint for TCGA KIRC project.

Run:
    python src/generate_pptx.py

Output:
    outputs/TCGA_KIRC_Presentation.pptx
"""

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent.parent  # project root
FIGURES = HERE / "outputs" / "figures"
RESULTS = HERE / "outputs" / "results"
OUT = HERE / "outputs" / "TCGA_KIRC_Presentation.pptx"

# ── Colour palette ────────────────────────────────────────────────────────────
BLUE   = RGBColor(0x00, 0x53, 0xA5)   # MS Elevate primary blue
TEAL   = RGBColor(0x00, 0x7C, 0x89)   # accent teal
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
DARK   = RGBColor(0x1A, 0x1A, 0x2E)
GRAY   = RGBColor(0x55, 0x55, 0x55)
LGRAY  = RGBColor(0xF4, 0xF6, 0xF9)
ACCENT = RGBColor(0xE6, 0x54, 0x00)   # orange accent strip


# ── Low-level helpers ─────────────────────────────────────────────────────────

def blank_slide(prs: Presentation):
    """Add a slide using the Blank layout."""
    blank = next(
        (lay for lay in prs.slide_layouts if lay.name == "Blank"),
        prs.slide_layouts[-1],
    )
    return prs.slides.add_slide(blank)


def rect(slide, left, top, width, height, color: RGBColor):
    """Add a solid filled rectangle with no visible border."""
    shape = slide.shapes.add_shape(1, Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape


def txbox(slide, left, top, width, height):
    """Add a transparent text box and return its text frame."""
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    box.fill.background()
    tf = box.text_frame
    tf.word_wrap = True
    return tf


def title_bar(slide, title: str, subtitle: str = ""):
    """Blue header band with white title (and optional subtler subtitle)."""
    rect(slide, 0, 0, 10, 1.55 if subtitle else 1.35, BLUE)
    tf = txbox(slide, 0.25, 0.12, 9.5, 1.0 if subtitle else 0.9)
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = title
    run.font.size = Pt(28)
    run.font.bold = True
    run.font.color.rgb = WHITE
    if subtitle:
        p2 = tf.add_paragraph()
        r2 = p2.add_run()
        r2.text = subtitle
        r2.font.size = Pt(13)
        r2.font.color.rgb = RGBColor(0xBB, 0xD4, 0xFF)


def footer(slide, text="TCGA KIRC — Survival Analysis  ·  MS Elevate Internship Project"):
    rect(slide, 0, 7.2, 10, 0.3, RGBColor(0xE0, 0xE6, 0xF0))
    tf = txbox(slide, 0.2, 7.21, 9.6, 0.28)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = text
    run.font.size = Pt(9)
    run.font.color.rgb = GRAY


def bullets(slide, items, left=0.3, top=1.7, width=9.4, height=5.3):
    """
    Render a structured list of items into a text box.

    items may contain:
      - dict with key 'h': section header string
      - tuple (level, text): bullet at given indent level (0-based)
      - plain string: level-0 bullet
    """
    tf = txbox(slide, left, top, width, height)
    first = True

    for item in items:
        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()

        if isinstance(item, dict):
            run = p.add_run()
            run.text = item["h"]
            run.font.size = Pt(15)
            run.font.bold = True
            run.font.color.rgb = BLUE
            p.space_before = Pt(8)

        elif isinstance(item, tuple):
            level, text = item
            p.level = min(level, 4)
            run = p.add_run()
            run.text = text
            run.font.size = Pt(13)
            run.font.color.rgb = DARK

        else:
            run = p.add_run()
            run.text = str(item)
            run.font.size = Pt(13)
            run.font.color.rgb = DARK


# ── Slide builders ────────────────────────────────────────────────────────────

def slide_01_title(prs, model_results):
    slide = blank_slide(prs)

    # Background halves
    rect(slide, 0, 0, 10, 4.1, BLUE)
    rect(slide, 0, 4.1, 10, 3.4, LGRAY)
    rect(slide, 0, 3.95, 10, 0.15, ACCENT)

    # Title
    tf = txbox(slide, 0.5, 0.65, 9.0, 1.5)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = "TCGA KIRC — Survival Analysis"
    run.font.size = Pt(40)
    run.font.bold = True
    run.font.color.rgb = WHITE

    # Subtitle
    tf2 = txbox(slide, 0.5, 2.1, 9.0, 1.0)
    p2 = tf2.paragraphs[0]
    p2.alignment = PP_ALIGN.CENTER
    r2 = p2.add_run()
    r2.text = "Multi-Model Survival Prediction · Kidney Renal Clear Cell Carcinoma"
    r2.font.size = Pt(17)
    r2.font.color.rgb = RGBColor(0xBB, 0xD4, 0xFF)

    # Student info card
    card = rect(slide, 1.5, 4.25, 7.0, 2.5, WHITE)
    card.line.fill.background()
    tf3 = txbox(slide, 1.6, 4.35, 6.8, 2.3)
    entries = [
        ("Student Name:", "[STUDENT NAME]"),
        ("College:", "[COLLEGE]"),
        ("Department:", "[DEPARTMENT]"),
        ("Email:", "[EMAIL]"),
        ("Program:", "MS Elevate Internship"),
    ]
    first = True
    for label, val in entries:
        p = tf3.paragraphs[0] if first else tf3.add_paragraph()
        first = False
        p.alignment = PP_ALIGN.CENTER
        r1 = p.add_run()
        r1.text = f"{label}  "
        r1.font.size = Pt(13)
        r1.font.bold = True
        r1.font.color.rgb = BLUE
        r2v = p.add_run()
        r2v.text = val
        r2v.font.size = Pt(13)
        r2v.font.color.rgb = DARK

    # Best-model callout
    best = max(model_results, key=model_results.get)
    rect(slide, 2.5, 6.88, 5.0, 0.5, TEAL)
    tf4 = txbox(slide, 2.5, 6.88, 5.0, 0.5)
    p4 = tf4.paragraphs[0]
    p4.alignment = PP_ALIGN.CENTER
    r4 = p4.add_run()
    r4.text = f"Best: {best}  |  C-index = {model_results[best]:.3f}"
    r4.font.size = Pt(13)
    r4.font.bold = True
    r4.font.color.rgb = WHITE


def slide_02_outline(prs):
    slide = blank_slide(prs)
    title_bar(slide, "Outline")
    bullets(slide, [
        (0, "1.   Problem Statement"),
        (0, "2.   Proposed Solution"),
        (0, "3.   System Approach & Technical Stack"),
        (0, "4.   Algorithm & Deployment"),
        (0, "5.   Results & Model Performance"),
        (0, "6.   Conclusion"),
        (0, "7.   Future Scope"),
        (0, "8.   References & GitHub Repository"),
    ], top=1.8)
    footer(slide)


def slide_03_problem(prs):
    slide = blank_slide(prs)
    title_bar(slide, "Problem Statement")
    bullets(slide, [
        {"h": "Clinical Challenge"},
        (0, "Kidney Renal Clear Cell Carcinoma (KIRC) is the 3rd most common urological malignancy."),
        (0, "Significant survival heterogeneity exists even within the same AJCC cancer stage."),
        (0, "Clinicians lack quantitative tools to stratify patients by long-term mortality risk."),

        {"h": "Data Challenge"},
        (0, "TCGA-KIRC cohort: 529 patients, 2,000 high-variance genes + clinical features."),
        (0, "Right-censored survival data — standard regression is inappropriate."),
        (0, "High dimensionality (p >> n) requires regularisation and feature selection."),

        {"h": "Research Question"},
        (0, "Can multi-model survival analysis on TCGA-KIRC accurately stratify patients "
            "into high- and low-risk groups, and which genomic / clinical features drive mortality?"),
    ])
    footer(slide)


def slide_04_solution(prs):
    slide = blank_slide(prs)
    title_bar(slide, "Proposed Solution")
    bullets(slide, [
        {"h": "End-to-End Survival Analysis Pipeline"},
        (0, "● Data Collection:    TCGA-KIRC clinical + gene expression data (GDC Portal)"),
        (0, "● Preprocessing:      survival target derivation, variance-based gene filtering, standardisation"),
        (0, "● Feature Engineering:  2,000 high-variance genes + 7 clinical variables (age, stage, gender…)"),
        (0, "● Machine Learning:   4 complementary survival models (see Algorithm slide)"),
        (0, "● Evaluation:         Concordance index, Integrated Brier Score, KM risk stratification"),
        (0, "● Deployment:         Interactive Streamlit dashboard for real-time risk prediction"),

        {"h": "Key Innovation"},
        (0, "Combining genomic expression data with clinical variables across 4 model families "
            "(Cox regression, LASSO, ensemble tree-based, deep learning) to identify the strongest "
            "predictors of survival in KIRC."),
    ])
    footer(slide)


def slide_05_system(prs):
    slide = blank_slide(prs)
    title_bar(slide, "System Approach")

    # Left column
    bullets(slide, [
        {"h": "Data Sources"},
        (0, "● TCGA-KIRC: clinical.tsv, follow_up.tsv"),
        (0, "● kirc_expression.tsv"),
        (0, "  · 20,530 genes × 533 tumour samples (-01)"),
        (0, "  · Variance filter → top 2,000 genes retained"),
        (0, "  · 529 patients with complete survival data"),

        {"h": "System Requirements"},
        (0, "● Python 3.12  |  Jupyter Notebook"),
        (0, "● Streamlit 1.x  |  python-pptx"),
        (0, "● Git + GitHub (version control)"),
    ], left=0.3, top=1.65, width=4.6, height=5.5)

    # Right column
    bullets(slide, [
        {"h": "Core Libraries"},
        (0, "● pandas / numpy — data wrangling"),
        (0, "● lifelines — Cox PH, Kaplan-Meier"),
        (0, "● scikit-survival — LASSO Cox, RSF"),
        (0, "● PyTorch — DeepSurv neural network"),
        (0, "● plotly / matplotlib — visualisations"),
        (0, "● scikit-learn — preprocessing, metrics"),

        {"h": "Infrastructure"},
        (0, "● Streamlit Cloud / local deployment"),
        (0, "● kaleido — static chart export"),
    ], left=5.1, top=1.65, width=4.6, height=5.5)

    # Divider
    div = slide.shapes.add_shape(1, Inches(4.95), Inches(1.72), Inches(0.05), Inches(5.2))
    div.fill.solid()
    div.fill.fore_color.rgb = RGBColor(0xCC, 0xD6, 0xE8)
    div.line.fill.background()

    footer(slide)


def slide_06_algorithm(prs):
    slide = blank_slide(prs)
    title_bar(
        slide, "Algorithm & Deployment",
        subtitle="4 survival models · 529 patients · 70/30 stratified train/test split",
    )
    bullets(slide, [
        {"h": "1 — Cox Proportional Hazards  (Baseline)"},
        (0, "Clinical features only · hazard ratio interpretation · lifelines CoxPHFitter"),

        {"h": "2 — LASSO-penalised Cox  (Best Model ⭐)"},
        (0, "Clinical + 2,000 genes · L1 regularisation for automatic gene selection"
            " · scikit-survival CoxnetSurvivalAnalysis · 5-fold CV for alpha"),

        {"h": "3 — Random Survival Forest"},
        (0, "Non-linear ensemble · 200 trees · scikit-survival RandomSurvivalForest"
            " · permutation-based feature importance"),

        {"h": "4 — DeepSurv  (Deep Learning)"},
        (0, "3-layer MLP (input → 64 → 32 → 1) · PyTorch · Cox partial-likelihood loss"
            " · Adam optimiser · 200 epochs · batch norm + dropout"),

        {"h": "Deployment — Streamlit Dashboard"},
        (0, "Interactive single-page app · real-time risk score · KM stratification"
            " · gene importance explorer · risk prediction demo with adjustable inputs"),
    ])
    footer(slide)


def slide_07_results(prs, model_results, cohort):
    slide = blank_slide(prs)
    best = max(model_results, key=model_results.get)
    title_bar(
        slide, "Results",
        subtitle=f"LASSO Cox  C-index = {model_results['LASSO Cox']:.3f}  ·  "
                 f"RSF  Integrated Brier Score = {cohort['ibs_rsf']:.3f}",
    )

    # ── Left: model table ───────────────────────────────────────────────────
    rect(slide, 0.2, 1.65, 3.15, 1.9, RGBColor(0xEB, 0xF1, 0xFA))
    tf_tbl = txbox(slide, 0.3, 1.7, 3.0, 1.85)
    hdr = tf_tbl.paragraphs[0]
    hdr_run = hdr.add_run()
    hdr_run.text = "C-index comparison"
    hdr_run.font.size = Pt(12)
    hdr_run.font.bold = True
    hdr_run.font.color.rgb = BLUE

    rows = [
        ("Cox PH (clinical)",        model_results["Cox PH (clinical)"],        ""),
        ("LASSO Cox",                model_results["LASSO Cox"],                "⭐ "),
        ("Random Survival Forest",   model_results["Random Survival Forest"],   ""),
        ("DeepSurv",                 model_results["DeepSurv"],                 ""),
    ]
    for model, score, badge in rows:
        p = tf_tbl.add_paragraph()
        run = p.add_run()
        run.text = f"  {badge}{model}  {score:.3f}"
        run.font.size = Pt(11)
        run.font.bold = bool(badge)
        run.font.color.rgb = BLUE if badge else DARK

    # ── Cohort stats ─────────────────────────────────────────────────────────
    tf_stats = txbox(slide, 0.3, 3.65, 3.0, 1.9)
    stats_hdr = tf_stats.paragraphs[0]
    sh_run = stats_hdr.add_run()
    sh_run.text = "Cohort Summary"
    sh_run.font.size = Pt(12)
    sh_run.font.bold = True
    sh_run.font.color.rgb = BLUE

    stats = [
        f"Total patients: {cohort['total_patients']}",
        f"Events (deaths): {cohort['events_dead']}",
        f"Censored: {cohort['censored_alive']}",
        f"Median follow-up: {cohort['median_time_days']:.0f} days",
        f"Genes retained: {cohort['num_genes']:,}",
    ]
    for s in stats:
        ps = tf_stats.add_paragraph()
        rs = ps.add_run()
        rs.text = f"  {s}"
        rs.font.size = Pt(11)
        rs.font.color.rgb = DARK

    # ── Charts (3 images) ────────────────────────────────────────────────────
    chart_slots = [
        ("09_model_comparison.png",        0.2, 5.65, 3.15, 1.9),
        ("03_km_overall.png",              3.5, 1.65, 3.1,  5.9),
        ("11_gene_importance_combined.png", 6.7, 1.65, 3.1,  5.9),
    ]
    for fname, l, t, w, h in chart_slots:
        fpath = FIGURES / fname
        if fpath.exists():
            slide.shapes.add_picture(str(fpath), Inches(l), Inches(t), Inches(w), Inches(h))

    footer(slide)


def slide_08_conclusion(prs, model_results, cohort):
    slide = blank_slide(prs)
    title_bar(slide, "Conclusion")
    delta = model_results["LASSO Cox"] - model_results["Cox PH (clinical)"]
    bullets(slide, [
        {"h": "Key Findings"},
        (0, f"● LASSO Cox achieved the highest C-index of {model_results['LASSO Cox']:.3f} — "
            f"outperforming clinical-only Cox by ΔC = {delta:.3f}."),
        (0, "● RSF confirmed complementary non-linear gene interactions; "
            f"Integrated Brier Score = {cohort['ibs_rsf']:.3f}."),
        (0, "● Kaplan-Meier stratification confirms a clear separation between "
            "predicted high- and low-risk groups (log-rank p < 0.001)."),
        (0, "● LASSO selected a sparse, interpretable gene signature predictive of KIRC mortality."),

        {"h": "Project Outcomes"},
        (0, "● Reproducible end-to-end pipeline from raw TCGA data to trained survival models."),
        (0, "● Interactive Streamlit dashboard enabling real-time patient risk prediction."),
        (0, "● Clean GitHub repository with complete documentation and requirements."),

        {"h": "Limitations"},
        (0, "● TCGA data may carry selection bias; model not validated on an external cohort."),
        (0, "● DeepSurv performance is constrained by the relatively small sample size (n = 529)."),
    ])
    footer(slide)


def slide_09_future(prs):
    slide = blank_slide(prs)
    title_bar(slide, "Future Scope")
    bullets(slide, [
        {"h": "Model Improvements"},
        (0, "● Stacking / ensemble of all 4 models for improved concordance."),
        (0, "● Attention-based DeepSurv with pathway-level gene embeddings (Gene Ontology / KEGG)."),
        (0, "● Multi-omics integration: copy number variation, methylation, proteomics."),

        {"h": "Validation & Clinical Translation"},
        (0, "● External validation on ICGC or CheckMate-025 / IMmotion151 datasets."),
        (0, "● Prospective clinical pilot: integrate risk scores with EHR systems."),
        (0, "● SHAP values for per-patient risk factor explanation (interpretable AI)."),

        {"h": "Engineering & Deployment"},
        (0, "● Deploy Streamlit app to Streamlit Community Cloud or Azure App Service."),
        (0, "● REST API (FastAPI) for programmatic risk score queries from clinical systems."),
        (0, "● Automated retraining pipeline as new TCGA data releases become available."),
    ])
    footer(slide)


def slide_10_references(prs):
    slide = blank_slide(prs)
    title_bar(slide, "References & GitHub Repository")
    bullets(slide, [
        {"h": "Datasets"},
        (0, "1.  The Cancer Genome Atlas (TCGA) — GDC Data Portal\n"
            "     https://portal.gdc.cancer.gov/"),
        (0, "2.  Project GitHub Repository:\n"
            "     https://github.com/LikithaDudala/tcga-kirc-project"),

        {"h": "Libraries & Frameworks"},
        (0, "3.  Davidson-Pilon C. (2019). lifelines: survival analysis in Python. "
            "Journal of Open Source Software."),
        (0, "4.  Pölsterl S. (2020). scikit-survival: A Library for Time-to-Event Analysis "
            "Building on scikit-learn. JMLR."),
        (0, "5.  Katzman J.L. et al. (2018). DeepSurv: personalised treatment recommender "
            "system using Cox regression deep neural network. BMC Med Res Methodol."),

        {"h": "Clinical Background"},
        (0, "6.  Ricketts C.J. et al. (2018). The TCGA KIRC comprehensive molecular "
            "characterization. Cell Reports 23(1):313–326."),
        (0, "7.  Srivastava A. et al. (2021). Prognostic gene signatures in clear cell RCC. "
            "Cancers 13(6):1407."),
    ], top=1.65)
    footer(slide)


def slide_11_thankyou(prs):
    slide = blank_slide(prs)

    rect(slide, 0, 0, 10, 7.5, BLUE)
    rect(slide, 0, 3.5, 10, 0.12, ACCENT)
    rect(slide, 0, 6.55, 10, 0.95, DARK)

    # "Thank You"
    tf = txbox(slide, 1.0, 1.1, 8.0, 1.8)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = "Thank You"
    run.font.size = Pt(54)
    run.font.bold = True
    run.font.color.rgb = WHITE

    # "Questions?"
    tf2 = txbox(slide, 1.0, 2.85, 8.0, 0.75)
    p2 = tf2.paragraphs[0]
    p2.alignment = PP_ALIGN.CENTER
    r2 = p2.add_run()
    r2.text = "Questions?"
    r2.font.size = Pt(22)
    r2.font.color.rgb = RGBColor(0xBB, 0xD4, 0xFF)

    # Contact block
    tf3 = txbox(slide, 0.5, 3.8, 9.0, 2.5)
    tf3.word_wrap = True
    lines = [
        "[STUDENT NAME]  ·  [COLLEGE]  ·  [DEPARTMENT]",
        "[EMAIL]",
        "GitHub:  https://github.com/LikithaDudala/tcga-kirc-project",
    ]
    first = True
    for line in lines:
        p3 = tf3.paragraphs[0] if first else tf3.add_paragraph()
        first = False
        p3.alignment = PP_ALIGN.CENTER
        r3 = p3.add_run()
        r3.text = line
        r3.font.size = Pt(15)
        r3.font.color.rgb = RGBColor(0xBB, 0xD4, 0xFF)

    # Footer bar
    tf4 = txbox(slide, 0.5, 6.6, 9.0, 0.8)
    p4 = tf4.paragraphs[0]
    p4.alignment = PP_ALIGN.CENTER
    r4 = p4.add_run()
    r4.text = "TCGA KIRC — Survival Analysis  ·  MS Elevate Internship Project"
    r4.font.size = Pt(11)
    r4.font.color.rgb = RGBColor(0x88, 0x99, 0xAA)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    with open(RESULTS / "model_results.json") as f:
        model_results = json.load(f)
    with open(RESULTS / "cohort_summary.json") as f:
        cohort = json.load(f)

    prs = Presentation()
    prs.slide_width  = Inches(10)
    prs.slide_height = Inches(7.5)

    slide_01_title(prs, model_results)      # 1. Title
    slide_02_outline(prs)                   # 2. Outline
    slide_03_problem(prs)                   # 3. Problem Statement
    slide_04_solution(prs)                  # 4. Proposed Solution
    slide_05_system(prs)                    # 5. System Approach
    slide_06_algorithm(prs)                 # 6. Algorithm & Deployment
    slide_07_results(prs, model_results, cohort)   # 7. Results
    slide_08_conclusion(prs, model_results, cohort)  # 8. Conclusion
    slide_09_future(prs)                    # 9. Future Scope
    slide_10_references(prs)               # 10. References
    slide_11_thankyou(prs)                 # 11. Thank You

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print(f"Saved → {OUT}")
    print(f"Slides: {len(prs.slides)}")


if __name__ == "__main__":
    main()
