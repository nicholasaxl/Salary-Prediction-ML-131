"""Python redo of the PSTAT 131 salary model comparison (see main.Rmd).

Fixes over the R version:
  * duplicate rows are measured and removed (they leak across train/test)
  * every model is scored in USD on held-out folds (the R lm test MSE mixed log and USD)
  * KNN gets scaled features and all categoricals
  * repeated k-fold CV instead of one split; RMSE / MAE / R2 instead of MSE only
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb
from xgboost import XGBRegressor

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

CAT = ["experience_level", "employment_type", "job_title", "company_location", "company_size"]
NUM = ["work_year", "remote_ratio"]
TARGET = "salary_in_usd"
SEED = 123


def load(dedup: bool) -> pd.DataFrame:
    df = pd.read_csv("ds_salaries.csv")
    df = df[CAT + NUM + [TARGET]]
    return df.drop_duplicates().reset_index(drop=True) if dedup else df


def log_target(model):
    return TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)


def models():
    ohe = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CAT), ("num", "passthrough", NUM)]
    )
    ohe_scaled = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
            ("num", StandardScaler(), NUM),
        ]
    )
    return {
        "Median baseline": DummyRegressor(strategy="median"),
        "Ridge (log target)": log_target(make_pipeline(ohe_scaled, RidgeCV(alphas=np.logspace(-2, 3, 20)))),
        "KNN k=15 (scaled)": make_pipeline(ohe_scaled, KNeighborsRegressor(n_neighbors=15, weights="distance")),
        "Random forest": make_pipeline(
            ohe, RandomForestRegressor(n_estimators=400, min_samples_leaf=2, max_features=0.33, n_jobs=-1, random_state=SEED)
        ),
        "XGBoost (log target)": log_target(
            make_pipeline(
                ohe,
                XGBRegressor(
                    n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8,
                    colsample_bytree=0.6, min_child_weight=3, random_state=SEED, n_jobs=-1,
                ),
            )
        ),
        "LightGBM (log target)": log_target(
            make_pipeline(
                ohe,
                LGBMRegressor(
                    n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=10,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.6, random_state=SEED,
                    n_jobs=-1, verbose=-1,
                ),
            )
        ),
    }


def evaluate(df: pd.DataFrame, label: str) -> pd.DataFrame:
    X, y = df[CAT + NUM], df[TARGET]
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }
    rows = []
    for name, model in models().items():
        s = cross_validate(model, X, y, cv=cv, scoring=scoring)
        rows.append(
            {
                "data": label,
                "model": name,
                "rmse": -s["test_rmse"].mean(),
                "rmse_sd": s["test_rmse"].std(),
                "mae": -s["test_mae"].mean(),
                "r2": s["test_r2"].mean(),
            }
        )
        print(f"{label:>13} | {name:<22} RMSE {rows[-1]['rmse']:>9,.0f}  MAE {rows[-1]['mae']:>9,.0f}  R2 {rows[-1]['r2']:.3f}")
    return pd.DataFrame(rows)


def shap_summary(df: pd.DataFrame) -> None:
    X, y = df[CAT + NUM], df[TARGET]
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT), ("num", "passthrough", NUM)]
    )
    Xt = pre.fit_transform(X)
    names = pre.get_feature_names_out()
    model = XGBRegressor(
        n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8,
        colsample_bytree=0.6, min_child_weight=3, random_state=SEED,
    ).fit(Xt, np.log1p(y))
    # xgboost's own TreeSHAP; shap.TreeExplainer can't parse xgboost 3.x's base_score format
    sv = model.get_booster().predict(xgb.DMatrix(Xt, feature_names=list(names)), pred_contribs=True)[:, :-1]
    shap.summary_plot(sv, Xt, feature_names=names, max_display=15, show=False)
    plt.title("What drives log-salary (XGBoost, deduplicated data)")
    plt.tight_layout()
    plt.savefig(OUT / "shap_summary.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    plt.switch_backend("Agg")  # script mode: write files, never open windows (kept out of import so notebooks render inline)
    raw = load(dedup=False)
    dedup = load(dedup=True)
    print(f"rows: raw={len(raw)}, deduplicated={len(dedup)} ({len(raw) - len(dedup)} duplicates)\n")

    results = pd.concat([evaluate(raw, "with dupes"), evaluate(dedup, "deduplicated")], ignore_index=True)
    results.to_csv(OUT / "cv_results.csv", index=False)

    pivot = results.pivot(index="model", columns="data", values="rmse")
    pivot["inflation"] = pivot["deduplicated"] - pivot["with dupes"]
    print("\nRMSE (USD): duplicates make models look better by `inflation` dollars")
    print(pivot.round(0).to_string())

    shap_summary(dedup)
    print("\nwrote outputs/cv_results.csv and outputs/shap_summary.png")
