# Course Guideline Checklist (mapped to outputs)

This checklist matches the course guideline slide and the instructor feedback.

## 1) Problem description
- [x] Civil & Environmental Engineering context (energy‑efficient building design)
- [x] Supervised regression formulation (predict Heating Load and Cooling Load)

**Where covered**
- Slides: “Problem description”
- Code: module docstring + README

## 2) Three main components of an ML project

### 2.1 Task (mapping to accomplish)
- [x] Define mapping  f(X) → y  and targets (HL, CL)

**Where covered**
- Slides: “Task: the mapping to learn”
- Code: `load_dataset()` + `run_for_target()`

### 2.2 Experience (explain + analyse data; preprocessing)
- [x] Verify missing values and basic EDA (describe, correlations)
- [x] Handle feature types:
  - numeric → impute + standardize
  - categorical/ordinal → impute + one‑hot encode

**Where covered**
- Slides: “Dataset (Experience)” + “Preprocessing & feature handling”
- Code outputs:
  - `missing_values.csv`
  - `X_describe.csv`
  - `correlation_numeric.csv`

### 2.3 Performance (objective + how you measure performance)
- [x] Objective function: MSE (implicit in Ridge/MLP training)
- [x] Metrics: RMSE, MAE, R² on dev and test sets

**Where covered**
- Slides: “Performance: objective function & metrics”
- Code: `evaluate()` + `results_metrics.csv`

## 3) Cross‑validation + train/dev/test split
- [x] Train/dev/test split (~70/15/15)
- [x] 5‑fold CV for model selection and hyperparameter tuning
- [x] Compare against a simple baseline

**Where covered**
- Slides: “Model selection: train/dev/test + k‑fold CV”
- Code:
  - `split_train_dev_test()`
  - `GridSearchCV` (Ridge)
  - `RandomizedSearchCV` (MLP)

## 4) Diagnostics: convergence, bias, variance
- [x] Convergence diagnostics for iterative model (MLP loss curve)
- [x] Bias/variance signal via train vs dev error
- [x] Residual diagnostics and predicted‑vs‑true plots

**Where covered**
- Slides: “Diagnostics: convergence + bias/variance”
- Code outputs:
  - `*_mlp_loss_curve.png`
  - `*_train_vs_dev_rmse.png`
  - `*_residual_hist.png`
  - `*_pred_vs_true.png`

## 5) Conclusion + improvement ideas
- [x] Conclude with improvement ideas (features, data, model)

**Where covered**
- Slides: “Conclusion, improvements, and checklist”
- Report discussion section (to be written from your results)

---

### Submission checklist (files)
- [x] Presentation: `ML_Project_Presentation_Mithun.pptx`
- [x] Code: `ml_energy_efficiency_project.py`
- [x] Reproducible outputs folder created by running the script