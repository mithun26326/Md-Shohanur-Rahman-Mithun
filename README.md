# ML Project Pack — Energy Efficiency (UCI)

**Project:** Predict building **Heating Load (HL)** and **Cooling Load (CL)** from design parameters.

**Models (per instructor feedback):**
- Simple baseline: **Ridge Regression (L2)**
- Complex model: **MLP Regressor**

## What’s inside
- `ml_energy_efficiency_project.py` — end-to-end pipeline (download → preprocessing → CV tuning → evaluation → diagnostics)
- `CHECKLIST.md` — guideline checklist mapped to outputs
- `requirements.txt` — minimal Python deps

## How to run

From this folder:

```bash
python ml_energy_efficiency_project.py --data-dir data --out-dir outputs/run1
```

If download is blocked, manually place `ENB2012_data.xlsx` into `data/` and re-run.

## Outputs you’ll get
Inside `outputs/<timestamp>/`:
- `results_metrics.csv` and `.json` (train/dev/test metrics for Ridge and MLP)
- diagnostics plots (loss curve, train-vs-dev RMSE, residual histogram, predicted-vs-true)
- `*_best_params.json` (best hyperparameters)
- `*_ridge_coefficients.csv` (interpretability for Ridge)

## Data & reference
- Dataset: UCI ML Repository — Energy Efficiency (ENB2012)
- Original study referenced by UCI: Tsanas & Xifara (2012), *Energy and Buildings*
