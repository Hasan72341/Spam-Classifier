# Spam Classifier

Comprehensive spam classification notebook and reproducible pipeline. This repository contains the primary analysis and model development artifacts in `SPAM CLASSIFICATION FINAL.ipynb` along with documentation and dependency hints.

## One-line summary (CV-ready)

Developed an end-to-end spam detection pipeline (data ingestion, advanced text featurization, stacking ensembles, calibration, and reproducible experiment tracking) implemented in a Jupyter notebook. Implemented TF-IDF and character n-gram feature engineering, engineered heuristic features (URL counts, punctuation density, uppercase ratios), and conducted stratified cross-validation and model stacking to optimize ROC-AUC and high-precision operating points.

## Highlights

- Multi-stage preprocessing and robust text featurization (word TF-IDF, char n-grams).
- Comparative model evaluation (logistic regression, tree ensembles) with stacked ensembling and probability calibration.
- Reproducible experiments and recommended packaging for production inference.

## Files added (not exhaustive)

- `SPAM CLASSIFICATION FINAL.ipynb` — primary notebook (not modified by this repo update).
- `DOCUMENTATION.md` — detailed technical documentation suitable for CV/portfolio text.
- `requirements.txt` — dependency hints for reproducing analyses.

## Quick start (Windows / PowerShell)

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

3. Launch Jupyter and open the notebook:

```powershell
jupyter notebook "SPAM CLASSIFICATION FINAL.ipynb"
```

4. For reproducible inference, export the vectorizer and model objects (see notebook) and use a small inference wrapper that accepts newline-delimited text and emits JSONL predictions.

## How to include this project on a CV

Suggested bullet for a CV:

"Developed a high-precision spam detection pipeline using advanced text featurization (TF-IDF, char n-grams, heuristic features) and stacking ensembles; implemented stratified cross-validation, post-hoc calibration, and reproducible experiment logging in a comprehensive Jupyter notebook."

## Repository

Remote: https://github.com/TarunaJ2006/Spam-Classifier

## Next suggested steps

- Add `src/` with lightweight inference utilities and a CLI (`infer.py`) to produce JSONL outputs for production evaluation.
- Add GitHub Actions to run linting and unit tests on push.
- Add CI that runs a small smoke test to verify notebook kernels and imports.

---

If you'd like, I can add a compact `infer.py` and a lightweight test/CI workflow next.
