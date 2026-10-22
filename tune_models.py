"""Nested-CV hyperparameter tuning, plus calibrated prediction intervals.

Outer 5-fold CV gives an honest score for "tune, then evaluate"; the inner
randomized search never sees the outer test fold. Intervals come from
LightGBM quantile models, and their empirical coverage is checked out of fold.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from model_comparison import CAT, NUM, OUT, SEED, TARGET, load, log_target

N_ITER = 25


def search_spaces():
    ohe = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CAT), ("num", "passthrough", NUM)]
    )
    return {
        "Random forest": (
            make_pipeline(ohe, RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED)),
            {
                "randomforestregressor__min_samples_leaf": randint(1, 15),
                "randomforestregressor__max_features": uniform(0.1, 0.7),
                "randomforestregressor__max_depth": [None, 8, 12, 20],
            },
        ),
        "XGBoost (log target)": (
            log_target(make_pipeline(ohe, XGBRegressor(random_state=SEED, n_jobs=-1))),
            {
                "regressor__xgbregressor__n_estimators": randint(150, 700),
                "regressor__xgbregressor__learning_rate": loguniform(0.01, 0.15),
                "regressor__xgbregressor__max_depth": randint(2, 7),
                "regressor__xgbregressor__min_child_weight": randint(1, 12),
                "regressor__xgbregressor__subsample": uniform(0.6, 0.4),
                "regressor__xgbregressor__colsample_bytree": uniform(0.4, 0.6),
                "regressor__xgbregressor__reg_lambda": loguniform(0.1, 30),
            },
        ),
        "LightGBM (log target)": (
            log_target(make_pipeline(ohe, LGBMRegressor(random_state=SEED, n_jobs=-1, verbose=-1, subsample_freq=1))),
            {
                "regressor__lgbmregressor__n_estimators": randint(150, 700),
                "regressor__lgbmregressor__learning_rate": loguniform(0.01, 0.15),
                "regressor__lgbmregressor__num_leaves": randint(4, 40),
                "regressor__lgbmregressor__min_child_samples": randint(5, 40),
                "regressor__lgbmregressor__subsample": uniform(0.6, 0.4),
                "regressor__lgbmregressor__colsample_bytree": uniform(0.4, 0.6),
                "regressor__lgbmregressor__reg_lambda": loguniform(0.1, 30),
            },
        ),
    }


def nested_cv(df: pd.DataFrame) -> pd.DataFrame:
    X, y = df[CAT + NUM], df[TARGET]
    outer = KFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}
    rows = []
    for name, (model, space) in search_spaces().items():
        search = RandomizedSearchCV(
            model, space, n_iter=N_ITER, cv=KFold(3, shuffle=True, random_state=SEED),
            scoring="neg_root_mean_squared_error", random_state=SEED, n_jobs=1,
        )
        s = cross_validate(search, X, y, cv=outer, scoring=scoring)
        rows.append(
            {
                "model": name + " tuned",
                "rmse": -s["test_rmse"].mean(),
                "rmse_sd": s["test_rmse"].std(),
                "mae": -s["test_mae"].mean(),
                "r2": s["test_r2"].mean(),
            }
        )
        print(f"{name:<22} tuned  RMSE {rows[-1]['rmse']:>9,.0f}  MAE {rows[-1]['mae']:>9,.0f}  R2 {rows[-1]['r2']:.3f}")
    return pd.DataFrame(rows)


def interval_coverage(df: pd.DataFrame, lo_q=0.1, hi_q=0.9) -> dict:
    """Out-of-fold coverage of an 80% interval from LightGBM quantile models."""
    X, y = df[CAT + NUM].copy(), df[TARGET]
    for c in CAT:
        X[c] = X[c].astype("category")
    params = dict(n_estimators=250, learning_rate=0.04, num_leaves=8, min_child_samples=15,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, random_state=SEED, verbose=-1)
    lo, hi = np.zeros(len(df)), np.zeros(len(df))
    for tr, te in KFold(5, shuffle=True, random_state=SEED).split(X):
        lo[te] = LGBMRegressor(objective="quantile", alpha=lo_q, **params).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
        hi[te] = LGBMRegressor(objective="quantile", alpha=hi_q, **params).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    cover = float(((y >= lo) & (y <= hi)).mean())
    out = {"nominal": hi_q - lo_q, "coverage": cover, "median_width": float(np.median(hi - lo))}
    print(f"80% interval: coverage {cover:.1%}, median width ${out['median_width']:,.0f}")
    return out


if __name__ == "__main__":
    df = load(dedup=True)
    res = nested_cv(df)
    res.to_csv(OUT / "tuned_results.csv", index=False)
    pd.Series(interval_coverage(df)).to_json(OUT / "interval_coverage.json")
    print("wrote outputs/tuned_results.csv and outputs/interval_coverage.json")
