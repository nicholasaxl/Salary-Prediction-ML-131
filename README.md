# dsalaries131

PSTAT 131's final project
Topic of choice: Model comparison for predicting the salary of a data science job
Dataset used: https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023

Language used: R, RMarkdown

This notebook explores the usage of several machine learning algorithms to fit trends in a highly leveled categorical dataset.
Focused on predictive optimization through different means and standards.

Final project grade: 91%

## Python redo (branch `python-model-redo`)

`main.Rmd` is the original class project. The Python redo fixes its evaluation issues (duplicate rows leaking across train/test, a log-vs-USD test MSE bug, unscaled KNN):

```
python -m venv .venv && .venv/Scripts/pip install pandas numpy scikit-learn xgboost lightgbm shap matplotlib jupyter
python model_comparison.py   # repeated-CV comparison, with and without duplicates, SHAP plot
python tune_models.py        # nested-CV tuning and 80% prediction-interval coverage
python experiments.py all    # stage-by-stage progression + 10-model search, writes outputs/ tables (slow, cached)
python interval_compare.py   # quantile-model comparison for the 10-90% range
python insights.py           # pay premiums, pay bands, reliability -> outputs/insights.json
python tools/build_notebook.py   # regenerate salary_analysis.ipynb, then execute it with nbconvert
```

Results and discussion are in `salary_analysis.ipynb`; raw numbers are in `outputs/`.

## Salary estimator web app

A FastAPI backend (`app/`) trains 10th/50th/90th-percentile LightGBM models at startup and serves a single-page UI:

- **Estimator:** pick a job title, level, location, etc. to get a salary range, the observed peer range, and the factors moving the estimate.
- **Explorer:** group and filter the data by any field, with distributions and 25-75% ranges.

```
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -m uvicorn app.main:app --port 8000   # then open http://localhost:8000
```

The 10-90 interval covered about 77.5% of held-out salaries in cross-validation, and categories with fewer than 5 rows trigger a low-confidence warning in the UI.

## Model search and decision (summary)

`experiments.py` tests ridge, KNN, random forest, extra trees, HistGradientBoosting, XGBoost, LightGBM, CatBoost, an MLP and a stacking blend under one protocol (5-fold CV x3, cleaning learned on training folds only). Findings: all strong models land within about $150 RMSE (R² about 0.41), so the data, not the model, is the limit. The decision rule and caveats are in `salary_analysis.ipynb` section 6.

- **CatBoost was evaluated at default settings only.** Its nested tuning run was stopped to save about 25 minutes, so there is no tuned CatBoost result.
- TabPFN was not tested (needs a pretrained-weights download); the scikit-learn MLP stands in as the neural option.
- The web app keeps LightGBM quantile models for the 10-90% range, a deliberate exception to the rule's XGBoost pick (see the notebook).
