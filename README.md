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
```

Results and discussion are in `salary_analysis.ipynb`; raw numbers are in `outputs/`.
