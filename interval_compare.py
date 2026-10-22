"""Compare quantile-capable models on the 10-90% range (coverage, width, pinball loss).

    python interval_compare.py      # writes outputs/interval_compare.csv
Same cleaned feature pipeline and 15 folds as experiments.py; no log target and no trimming, since
trimming the training tails would bias the upper quantile.
"""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss
from xgboost import XGBRegressor

from experiments import OUT, SEED, TARGET, build, dedup_data, run_cv
from features import CAT_ENG

LO, HI = 0.1, 0.9


class CatBoostQuantile(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=0.1, iterations=300, learning_rate=0.05, depth=5, random_seed=SEED):
        self.alpha = alpha
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.random_seed = random_seed

    def fit(self, X, y):
        self.model_ = CatBoostRegressor(
            loss_function=f"Quantile:alpha={self.alpha}", iterations=self.iterations, learning_rate=self.learning_rate,
            depth=self.depth, random_seed=self.random_seed, thread_count=1, verbose=0, allow_writing_files=False,
            cat_features=[c for c in CAT_ENG if c in X.columns],
        ).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


MAKERS = {
    "LightGBM": lambda a: LGBMRegressor(objective="quantile", alpha=a, n_estimators=250, learning_rate=0.04, num_leaves=8,
                                        min_child_samples=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                                        random_state=SEED, n_jobs=1, verbose=-1),
    "XGBoost": lambda a: XGBRegressor(objective="reg:quantileerror", quantile_alpha=a, n_estimators=300, learning_rate=0.04,
                                      max_depth=3, subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
                                      random_state=SEED, n_jobs=1),
    "HistGradientBoosting": lambda a: HistGradientBoostingRegressor(loss="quantile", quantile=a, max_iter=250,
                                                                    learning_rate=0.05, max_leaf_nodes=8,
                                                                    min_samples_leaf=15, random_state=SEED),
    "CatBoost": lambda a: CatBoostQuantile(alpha=a),
}


def main():
    d = dedup_data()
    y = d[TARGET].to_numpy()
    rows = []
    for name, make in MAKERS.items():
        p = {a: run_cv(f"q{a}_{name}", d, build(name, "eng", False, False, est=make(a)))["preds"][0] for a in (LO, HI)}
        lo, hi = np.minimum(p[LO], p[HI]), np.maximum(p[LO], p[HI])
        rows.append({
            "model": name,
            "coverage": float(((y >= lo) & (y <= hi)).mean()),
            "median_width": float(np.median(hi - lo)),
            "pinball": float((mean_pinball_loss(y, lo, alpha=LO) + mean_pinball_loss(y, hi, alpha=HI)) / 2),
        })
    out = pd.DataFrame(rows).sort_values("pinball")
    out.to_csv(OUT / "interval_compare.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
