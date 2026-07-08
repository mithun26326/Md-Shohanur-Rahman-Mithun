"""ml_energy_efficiency_project.py

Machine Learning project (Civil & Environmental Engineering)
Predict Heating Load (HL) and Cooling Load (CL) from building design parameters.

Implements the course guideline items:
 - Task definition: mapping f(X)->y (regression)
 - Experience: data exploration + preprocessing (scaling + one-hot)
 - Performance: objective + metrics (RMSE/MAE/R2)
 - Train/dev/test split + 5-fold CV hyperparameter tuning
 - Baseline vs complex model: Ridge vs MLP
 - Diagnostics: convergence (MLP loss curve), bias-variance (train vs dev), residuals

Dataset: UCI Energy Efficiency (ENB2012_data.xlsx)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------
# Configuration
# ---------------------------

RANDOM_SEED = 42


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_dev: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_dev: pd.Series
    y_test: pd.Series


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def download_dataset(dest_dir: Path) -> Path:
    """Download the UCI Energy Efficiency dataset Excel file.

    Notes:
    - This script tries the known UCI file location. If your environment blocks downloads,
      manually download ENB2012_data.xlsx from the dataset page and place it in dest_dir.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / "ENB2012_data.xlsx"
    if out_path.exists():
        return out_path

    urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
        # Some mirrors/alternate paths exist; keep a second attempt.
        "https://archive.ics.uci.edu/static/public/242/ENB2012_data.xlsx",
    ]

    import urllib.request

    last_err = None
    for url in urls:
        try:
            print(f"Downloading dataset from: {url}")
            urllib.request.urlretrieve(url, out_path)  # nosec B310
            if out_path.exists() and out_path.stat().st_size > 10_000:
                return out_path
        except Exception as e:  # noqa: BLE001
            last_err = e

    raise RuntimeError(
        "Failed to download ENB2012_data.xlsx. "
        "Please download it manually from UCI and place it in the data folder. "
        f"Last error: {last_err}"
    )


def load_dataset(xlsx_path: Path) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load dataset and return (X, y_hl, y_cl)."""
    df = pd.read_excel(xlsx_path)

    # UCI file typically uses columns: X1..X8, Y1, Y2.
    # We'll rename to meaningful names if needed.
    expected_cols = [f"X{i}" for i in range(1, 9)] + ["Y1", "Y2"]
    if all(c in df.columns for c in expected_cols):
        df = df.rename(
            columns={
                "X1": "Relative_Compactness",
                "X2": "Surface_Area",
                "X3": "Wall_Area",
                "X4": "Roof_Area",
                "X5": "Overall_Height",
                "X6": "Orientation",
                "X7": "Glazing_Area",
                "X8": "Glazing_Area_Distribution",
                "Y1": "Heating_Load",
                "Y2": "Cooling_Load",
            }
        )

    feature_cols = [
        "Relative_Compactness",
        "Surface_Area",
        "Wall_Area",
        "Roof_Area",
        "Overall_Height",
        "Orientation",
        "Glazing_Area",
        "Glazing_Area_Distribution",
    ]
    target_hl = "Heating_Load"
    target_cl = "Cooling_Load"

    missing = [c for c in feature_cols + [target_hl, target_cl] if c not in df.columns]
    if missing:
        raise ValueError(
            "Unexpected column names in the Excel file. "
            f"Missing: {missing}. Present columns: {list(df.columns)}"
        )

    X = df[feature_cols].copy()
    y_hl = df[target_hl].copy()
    y_cl = df[target_cl].copy()
    return X, y_hl, y_cl


def split_train_dev_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.15,
    dev_size: float = 0.15,
    seed: int = RANDOM_SEED,
) -> SplitData:
    """Split into train/dev/test with sizes ~70/15/15."""
    X_train_dev, X_test, y_train_dev, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    # dev fraction relative to remaining set
    dev_rel = dev_size / (1.0 - test_size)
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train_dev, y_train_dev, test_size=dev_rel, random_state=seed
    )
    return SplitData(X_train, X_dev, X_test, y_train, y_dev, y_test)


def make_preprocessor() -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Create preprocessing: impute, scale numeric; one-hot categorical."""
    numeric_features = [
        "Relative_Compactness",
        "Surface_Area",
        "Wall_Area",
        "Roof_Area",
        "Overall_Height",
        "Glazing_Area",
    ]
    categorical_features = ["Orientation", "Glazing_Area_Distribution"]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    # sklearn 1.2+ uses `sparse_output`; older versions use `sparse`.
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_features, categorical_features


def rmse(y_true, y_pred) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))


def evaluate(model, X, y) -> Dict[str, float]:
    preds = model.predict(X)
    return {
        "RMSE": rmse(y, preds),
        "MAE": mean_absolute_error(y, preds),
        "R2": r2_score(y, preds),
    }


def tune_ridge(preprocessor: ColumnTransformer) -> GridSearchCV:
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", Ridge())])
    grid = {
        "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
    }
    return GridSearchCV(
        pipe,
        param_grid=grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )


def tune_mlp(preprocessor: ColumnTransformer, seed: int = RANDOM_SEED) -> RandomizedSearchCV:
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                MLPRegressor(
                    max_iter=5000,
                    early_stopping=True,
                    n_iter_no_change=30,
                    validation_fraction=0.1,
                    random_state=seed,
                ),
            ),
        ]
    )

    # Randomized search over reasonable ranges (kept small for coursework runtime).
    param_distributions = {
        "model__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
        "model__alpha": np.logspace(-6, -2, 9),
        "model__learning_rate_init": np.logspace(-4, -2, 7),
    }

    return RandomizedSearchCV(
        pipe,
        param_distributions=param_distributions,
        n_iter=20,
        cv=5,
        scoring="neg_root_mean_squared_error",
        random_state=seed,
        n_jobs=-1,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Get expanded feature names after ColumnTransformer."""
    # Numeric
    num_names = list(preprocessor.named_transformers_["num"].feature_names_in_)

    # Categorical
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_in = list(preprocessor.named_transformers_["cat"].feature_names_in_)
    cat_out = list(ohe.get_feature_names_out(cat_in))

    return num_names + cat_out


def save_plots_and_diagnostics(
    out_dir: Path,
    target_name: str,
    ridge_best: Pipeline,
    mlp_best: Pipeline,
    split: SplitData,
) -> None:
    """Create diagnostic plots and save them."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Convergence plot (MLP loss curve) ---
    mlp = mlp_best.named_steps["model"]
    if hasattr(mlp, "loss_curve_") and mlp.loss_curve_:
        plt.figure()
        plt.plot(mlp.loss_curve_)
        plt.xlabel("Epoch")
        plt.ylabel("Training loss")
        plt.title(f"MLP convergence (loss curve) — {target_name}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{target_name}_mlp_loss_curve.png", dpi=160)
        plt.close()

    # --- Bias/variance style diagnostic: train vs dev RMSE ---
    ridge_train = evaluate(ridge_best, split.X_train, split.y_train)["RMSE"]
    ridge_dev = evaluate(ridge_best, split.X_dev, split.y_dev)["RMSE"]
    mlp_train = evaluate(mlp_best, split.X_train, split.y_train)["RMSE"]
    mlp_dev = evaluate(mlp_best, split.X_dev, split.y_dev)["RMSE"]

    plt.figure()
    labels = ["Ridge", "MLP"]
    train_vals = [ridge_train, mlp_train]
    dev_vals = [ridge_dev, mlp_dev]
    x = np.arange(len(labels))
    width = 0.35
    plt.bar(x - width / 2, train_vals, width, label="Train")
    plt.bar(x + width / 2, dev_vals, width, label="Dev")
    plt.xticks(x, labels)
    plt.ylabel("RMSE")
    plt.title(f"Train vs Dev RMSE (bias/variance signal) — {target_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{target_name}_train_vs_dev_rmse.png", dpi=160)
    plt.close()

    # --- Residuals + predicted vs true (best model on dev) ---
    # Choose the model with smaller dev RMSE
    best = mlp_best if mlp_dev <= ridge_dev else ridge_best
    y_hat = best.predict(split.X_dev)
    resid = split.y_dev.values - y_hat

    plt.figure()
    plt.hist(resid, bins=25)
    plt.xlabel("Residual (y - ŷ)")
    plt.ylabel("Count")
    plt.title(f"Residual histogram (dev) — {target_name}")
    plt.tight_layout()
    plt.savefig(out_dir / f"{target_name}_residual_hist.png", dpi=160)
    plt.close()

    plt.figure()
    plt.scatter(split.y_dev.values, y_hat, s=12, alpha=0.75)
    minv = float(min(split.y_dev.min(), y_hat.min()))
    maxv = float(max(split.y_dev.max(), y_hat.max()))
    plt.plot([minv, maxv], [minv, maxv])
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"Predicted vs True (dev) — {target_name}")
    plt.tight_layout()
    plt.savefig(out_dir / f"{target_name}_pred_vs_true.png", dpi=160)
    plt.close()

    # --- Ridge coefficients (feature importance proxy) ---
    try:
        ridge_model = ridge_best.named_steps["model"]
        pre = ridge_best.named_steps["preprocess"]
        feat_names = get_feature_names(pre)
        coefs = ridge_model.coef_.ravel()
        coef_df = pd.DataFrame({"feature": feat_names, "coef": coefs})
        coef_df["abs_coef"] = coef_df["coef"].abs()
        coef_df = coef_df.sort_values("abs_coef", ascending=False).drop(columns=["abs_coef"])
        coef_df.to_csv(out_dir / f"{target_name}_ridge_coefficients.csv", index=False)
    except Exception:
        # Not fatal; feature name extraction can differ across sklearn versions.
        pass


def run_for_target(
    X: pd.DataFrame,
    y: pd.Series,
    target_name: str,
    out_dir: Path,
) -> Dict[str, Dict[str, float]]:
    """Train+evaluate Ridge and MLP for one target."""
    split = split_train_dev_test(X, y)
    preprocessor, _, _ = make_preprocessor()

    # Tune
    ridge_search = tune_ridge(preprocessor)
    mlp_search = tune_mlp(preprocessor)

    ridge_search.fit(split.X_train, split.y_train)
    mlp_search.fit(split.X_train, split.y_train)

    ridge_best = ridge_search.best_estimator_
    mlp_best = mlp_search.best_estimator_

    # Evaluate
    results = {
        "Ridge": {
            "train": evaluate(ridge_best, split.X_train, split.y_train),
            "dev": evaluate(ridge_best, split.X_dev, split.y_dev),
            "test": evaluate(ridge_best, split.X_test, split.y_test),
        },
        "MLP": {
            "train": evaluate(mlp_best, split.X_train, split.y_train),
            "dev": evaluate(mlp_best, split.X_dev, split.y_dev),
            "test": evaluate(mlp_best, split.X_test, split.y_test),
        },
    }

    # Save diagnostics
    save_plots_and_diagnostics(out_dir, target_name, ridge_best, mlp_best, split)

    # Save best params
    params = {
        "Ridge_best_params": ridge_search.best_params_,
        "MLP_best_params": mlp_search.best_params_,
    }
    (out_dir / f"{target_name}_best_params.json").write_text(json.dumps(params, indent=2))

    return results


def results_to_table(all_results: Dict[str, Dict]) -> pd.DataFrame:
    rows = []
    for target, models in all_results.items():
        for model_name, splits in models.items():
            for split_name, metrics in splits.items():
                rows.append(
                    {
                        "target": target,
                        "model": model_name,
                        "split": split_name,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="ML project: energy efficiency (Ridge + MLP)")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory for dataset download / placement.",
    )
    parser.add_argument(
        "--xlsx",
        type=Path,
        default=None,
        help="Optional path to ENB2012_data.xlsx (overrides download).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs") / _timestamp(),
        help="Output directory for results and plots.",
    )
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)

    # Dataset
    xlsx_path = args.xlsx if args.xlsx else download_dataset(args.data_dir)
    X, y_hl, y_cl = load_dataset(xlsx_path)

    # Basic EDA summary saved
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "dataset_head.csv").write_text(X.join(pd.DataFrame({"HL": y_hl, "CL": y_cl})).head().to_csv(index=False))
    X.describe().to_csv(args.out_dir / "X_describe.csv")
    X.isna().sum().to_csv(args.out_dir / "missing_values.csv")
    X.corr(numeric_only=True).to_csv(args.out_dir / "correlation_numeric.csv")

    all_results: Dict[str, Dict] = {}

    all_results["Heating_Load"] = run_for_target(X, y_hl, "Heating_Load", args.out_dir)
    all_results["Cooling_Load"] = run_for_target(X, y_cl, "Cooling_Load", args.out_dir)

    # Save results table
    table = results_to_table(all_results)
    table.to_csv(args.out_dir / "results_metrics.csv", index=False)
    (args.out_dir / "results_metrics.json").write_text(json.dumps(all_results, indent=2))

    print("\nSaved outputs to:", args.out_dir)
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
