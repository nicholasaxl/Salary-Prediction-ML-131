"""Builds salary_analysis.ipynb from outputs/ (run experiments.py and insights.py first).

    python tools/build_notebook.py && python -m jupyter nbconvert --to notebook --execute --inplace salary_analysis.ipynb
"""
import nbformat as nbf

def md(text):
    return nbf.v4.new_markdown_cell(text.replace("$", "\\$"))


code = nbf.v4.new_code_cell
cells = []

# ------------------------------------------------------------------ setup
cells += [
    md("""# Predicting data science salaries: from a messy first pass to a defensible model

A redo of the PSTAT 131 R project (`main.Rmd`). The questions here:

1. How much of the original result survives honest evaluation, and what does each cleaning and optimization step buy?
2. Which model is best, decided by a rule fixed *before* looking at the results?
3. What can a business actually do with a model like this?

Every number below comes from `outputs/` (written by `experiments.py` and `insights.py`), under one protocol: 5-fold cross-validation repeated 3 times, cleaning learned on training folds only, scores always computed on untouched held-out rows."""),
    code("""import json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from IPython.display import Markdown as _Markdown, display

def Markdown(text):
    return _Markdown(text.replace('$', '\\$'))  # a bare $ pair would render as an equation

plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': .25, 'axes.titleweight': 'bold', 'axes.titlesize': 11})
BLUE, RED, GREY, GOLD, GREEN = '#2f6fed', '#d1495b', '#95a0b0', '#f0b429', '#1f9d6b'
OUT = 'outputs'
stage = pd.read_csv(f'{OUT}/stage_results.csv')
search = pd.read_csv(f'{OUT}/model_search.csv')
decision = json.load(open(f'{OUT}/decision.json'))
ins = json.load(open(f'{OUT}/insights.json'))
oof = np.load(f'{OUT}/stage_oof.npz', allow_pickle=True)

from experiments import raw_data, dedup_data, build
from features import TARGET, CAT_RAW, NUM_RAW
raw, dedup = raw_data(), dedup_data()
usd = lambda v: f'${v:,.0f}'
kfmt = FuncFormatter(lambda v, _: f'${v/1000:,.0f}k')
print(f'{len(raw):,} raw rows -> {len(dedup):,} after removing exact duplicates')"""),
]

# ------------------------------------------------------------------ headline (computed)
cells += [
    code("""s = stage.set_index(['family', 'stage'])
best_row = stage[stage.stage == 5].iloc[0]
best_name = decision['accuracy_winner'].split('_', 1)[1]
display(Markdown(f'''### Headline
- **The R-style setup looked better than it was.** Random forest on the raw data scored R\\u00b2 **{s.loc[('RF',0),'r2']:.2f}** / RMSE **{usd(s.loc[('RF',0),'rmse'])}**, but 31% of rows were duplicates leaking across train and test.
- **Honest baseline:** after deduplicating, the same model scores R\\u00b2 **{s.loc[('RF',1),'r2']:.2f}** / RMSE **{usd(s.loc[('RF',1),'rmse'])}**.
- **After cleaning, features and tuning** the best random forest reaches R\\u00b2 **{s.loc[('RF',4),'r2']:.2f}** and the best boosted model R\\u00b2 **{s.loc[('LGBM',4),'r2']:.2f}**. The overall winner, **{best_name}**, reaches R\\u00b2 **{best_row.r2:.2f}** / RMSE **{usd(best_row.rmse)}**.
- **The ceiling is the data, not the method:** the model sees seven columns, and people with identical profiles differ in salary by about 30% (section 4).
'''))"""),
]

# ------------------------------------------------------------------ 1 data
cells += [
    md("## 1. The data"),
    code("""fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
dedup[TARGET].plot.hist(bins=40, ax=ax[0], color=BLUE, title='Salary distribution (USD)')
dedup.groupby('experience_level')[TARGET].median().reindex(['EN', 'MI', 'SE', 'EX']).rename(index={'EN': 'Entry', 'MI': 'Mid', 'SE': 'Senior', 'EX': 'Executive'}).plot.bar(ax=ax[1], color=BLUE, title='Median salary by experience level', rot=0)
loc = dedup.groupby('company_location')[TARGET].agg(['median', 'count']).query('count >= 20')['median'].sort_values()
loc.plot.barh(ax=ax[2], color=BLUE, title='Median salary, locations with 20+ rows')
for a in ax: a.set_xlabel('')
ax[0].xaxis.set_major_formatter(kfmt); ax[1].yaxis.set_major_formatter(kfmt); ax[2].xaxis.set_major_formatter(kfmt); ax[2].set_ylabel('')
ax[0].set_xlabel('Salary (USD)')
plt.tight_layout()"""),
    md("Salaries are right-skewed, rise steeply with experience, and differ hugely by country. Three quarters of rows are US companies, so most other locations are thin."),

    md("""## 2. Data cleaning log
What was changed, why, and what it cost or bought. Steps that learn from the data (rare-category sets, tail quantiles) are fit on training folds only."""),
    code("""n_title_rare = (dedup.job_title.value_counts() < 10).sum()
n_loc_rare = (dedup.company_location.value_counts() < 10).sum()
log = pd.DataFrame([
    ['Drop salary, salary_currency, employee_residence', 'Redundant with salary_in_usd and company_location', 'Not tested separately'],
    ['Remove exact duplicate rows', f'{len(raw) - len(dedup):,} rows ({1 - len(dedup)/len(raw):.0%}) appear more than once and leak across train/test', 'Scores get worse but honest'],
    ['Model log(salary), convert back to USD', 'Errors become proportional to salary instead of dominated by the highest earners', 'Mixed: see stage 2'],
    ['Collapse rare titles and locations (<10 rows)', f'{n_title_rare} of {dedup.job_title.nunique()} titles and {n_loc_rare} of {dedup.company_location.nunique()} locations too thin to learn from', 'Small'],
    ['Add job family, region, level ordinal', 'Groups 93 titles into 7 families and 72 locations into 6 regions', 'Small'],
    ['Trim the top and bottom 1% of salaries in training only', 'Extreme rows pull fits; held-out rows are never trimmed', 'Small'],
], columns=['Step', 'Why', 'Effect on held-out scores'])
log.style.hide(axis='index').set_properties(**{'text-align': 'left', 'white-space': 'normal'})"""),
    md("""A note on trimming: dropping extreme salaries from the **test** rows too makes any model look much better (in a quick test RMSE fell by roughly $5k), but that is a different, easier question. Trimming is therefore applied to training rows only, and every score is measured on the original held-out rows."""),
    code("""ax = stage[stage.stage.isin([0, 1])].pivot(index='family', columns='stage', values='rmse')
ax.columns = ['Raw (duplicates leak)', 'Deduplicated']
ax.rename(index={'RF': 'Random forest', 'LGBM': 'Boosting (LightGBM)'}).plot.barh(figsize=(8, 2.8), color=[RED, BLUE], title='Held-out RMSE (USD): duplicates flatter the score')
plt.xlabel('CV RMSE (USD)'); plt.ylabel(''); plt.legend(loc='upper center', bbox_to_anchor=(.5, -.25), ncol=2); plt.tight_layout()"""),
]

# ------------------------------------------------------------------ 3 progress (presentation)
cells += [
    md("""## 3. The progress, at a glance

**The short version.** The original R report plotted predictions on the *training* rows, where a flexible model looks excellent. Held-out scoring on the cleaned data tells a more modest and more trustworthy story. This section shows that gap, then walks through each step one stage at a time (section 3.3 onwards)."""),
    code("""from sklearn.metrics import r2_score
STAGE_NAMES = {0: 'Raw (R-style, leaky)', 1: 'Deduplicate', 2: 'Log target', 3: 'Clean + features', 4: 'Tuned', 5: 'Best model'}
LIM = 460000

def decile_line(y, p):
    edges = np.quantile(y, np.linspace(0, 1, 11)); idx = np.clip(np.digitize(y, edges[1:-1]), 0, 9)
    return [y[idx == k].mean() for k in range(10)], [p[idx == k].mean() for k in range(10)]

def draw(a, y, p, title, prev=None, dot=BLUE, stats=None):
    a.scatter(y, p, s=9, alpha=.28, color=dot, linewidths=0)
    if prev is not None:
        a.plot(*decile_line(*prev), '--', color=GREY, lw=2, label='previous stage (avg prediction)')
    a.plot(*decile_line(y, p), '-o', color=GOLD, ms=5, lw=2.8, label='this stage (avg prediction)')
    a.plot([0, LIM], [0, LIM], '--', color=RED, lw=1.4, label='perfect prediction')
    a.set_xlim(0, LIM); a.set_ylim(0, LIM); a.set_aspect('equal')
    a.xaxis.set_major_formatter(kfmt); a.yaxis.set_major_formatter(kfmt)
    a.set_xlabel('Actual salary (USD)'); a.set_title(title, fontsize=12)
    rm, r2 = stats if stats else (np.sqrt(np.mean((y - p) ** 2)), r2_score(y, p))
    a.text(.03, .97, ('CV ' if stats else '') + f'R\\u00b2 {r2:.2f}\\n' + ('CV ' if stats else '') + f'RMSE {usd(rm)}', transform=a.transAxes, va='top', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', fc='white', ec=GREY, alpha=.9))
    return r2
print('helpers ready')"""),
    md("### 3.1 Before and after"),
    code("""from sklearn.ensemble import RandomForestRegressor
rf_raw = build('Random forest', 'raw', False, False, est=RandomForestRegressor(300, max_features=1 / 3, random_state=123, n_jobs=-1))
Xr, yr = raw.drop(columns=[TARGET]), raw[TARGET].to_numpy()
fit_in = rf_raw.fit(Xr, yr).predict(Xr)
best_key = decision['accuracy_winner']; best_name = best_key.split('_', 1)[1]
fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharex=True, sharey=True)
r1 = draw(axes[0], yr, fit_in, '1. What the R report plotted\\n(random forest, scored on its own training rows)', dot=RED)
r2_ = draw(axes[1], oof['y_dedup'], oof['stage1_RF'], '2. The same model, honestly scored\\n(duplicates removed, held-out rows only)', stats=(s.loc[('RF', 1)].rmse, s.loc[('RF', 1)].r2))
r3 = draw(axes[2], oof['y_dedup'], oof['stage5_BEST'], f'3. After cleaning, features, tuning\\nand the best model ({best_name})', stats=(s.loc[('BEST', 5)].rmse, s.loc[('BEST', 5)].r2))
axes[0].set_ylabel('Predicted salary (USD)')
axes[0].legend(loc='lower right', fontsize=9)
plt.tight_layout()
display(Markdown(f'''**Left to right:** the training-fit plot hugs the diagonal (R\\u00b2 **{r1:.2f}**) because the model has seen those exact rows. Scored honestly the same model reaches R\\u00b2 **{r2_:.2f}**, and after every improvement in this notebook R\\u00b2 **{r3:.2f}**. The final plot is the one to trust: it uses only rows the model never saw. The gap between the left and middle panel is overfitting; the small gain between the middle and right panel is everything we could do with seven columns.'''))"""),
    md("""### 3.2 Scoreboard
Every stage, scored on held-out rows. **Δ** columns compare with the previous stage of the same model family. Stage 0 is measured on a different, leaky test set (grey, italic), so its Δ to stage 1 is a correction, not a regression."""),
    code("""rows_ = []
for tag, name in (('RF', 'Random forest'), ('LGBM', 'Boosting (LightGBM)')):
    prev = None
    for sidx in range(5):
        r = s.loc[(tag, sidx)]
        rows_.append(dict(Stage=f'{sidx} {STAGE_NAMES[sidx]}', Model=name, RMSE=r.rmse, MAE=r.mae, R2=r.r2,
                          dRMSE=np.nan if prev is None else r.rmse - prev.rmse, dR2=np.nan if prev is None else r.r2 - prev.r2, leaky=sidx == 0))
        prev = r
b_ = s.loc[('BEST', 5)]
rows_.append(dict(Stage='5 Best model', Model=best_name, RMSE=b_.rmse, MAE=b_.mae, R2=b_.r2, dRMSE=np.nan, dR2=np.nan, leaky=False))
sb = pd.DataFrame(rows_)
def paint(row):
    if row.leaky: return ['color:#888;font-style:italic'] * (len(row) - 1)
    if row.Stage.startswith('5'): return ['background-color:#fff3cd;font-weight:bold;color:#222'] * (len(row) - 1)
    return [''] * (len(row) - 1)
good_bad = lambda v: '' if pd.isna(v) else ('color:#1f9d6b' if v < 0 else 'color:#d1495b')
good_bad_r2 = lambda v: '' if pd.isna(v) else ('color:#1f9d6b' if v > 0 else 'color:#d1495b')
styler = (sb.drop(columns='leaky').style.hide(axis='index')
          .apply(lambda r: paint(sb.loc[r.name]), axis=1)
          .map(good_bad, subset=['dRMSE']).map(good_bad_r2, subset=['dR2'])
          .format({'RMSE': '${:,.0f}', 'MAE': '${:,.0f}', 'R2': '{:.3f}', 'dRMSE': '{:+,.0f}', 'dR2': '{:+.3f}'}, na_rep='')
          .set_caption('Held-out scoreboard (green = better than the previous stage, red = worse)'))
styler"""),
    code("""fig, ax = plt.subplots(1, 2, figsize=(15, 4.6))
for tag, name, col in (('RF', 'Random forest', BLUE), ('LGBM', 'Boosting (LightGBM)', GREEN)):
    d_ = stage[(stage.family == tag) & (stage.stage <= 4)]
    for k, m in enumerate(['rmse', 'r2']):
        ax[k].plot(d_.stage, d_[m], '-o', color=col, lw=2.5, ms=8, label=name)
        ax[k].plot(0, d_[m].iloc[0], 'o', mfc='white', mec=col, ms=12)
        for x_, v_ in zip(d_.stage, d_[m]): ax[k].annotate(f'{v_:,.0f}' if m == 'rmse' else f'{v_:.2f}', (x_, v_), textcoords='offset points', xytext=(0, 9 if tag == 'RF' else -16), ha='center', fontsize=8, color=col)
ax[0].plot(5, b_.rmse, '*', color=GOLD, ms=20, label=f'Best: {best_name}'); ax[1].plot(5, b_.r2, '*', color=GOLD, ms=20)
for a, t in zip(ax, ['Held-out RMSE (USD), lower is better', 'Held-out R\\u00b2, higher is better']):
    a.set_title(t + '  (axis does not start at zero)'); a.set_xticks(range(6)); a.set_xticklabels([STAGE_NAMES[i].split(' (')[0] for i in range(6)], fontsize=9)
ax[0].legend(loc='upper center', bbox_to_anchor=(.5, -.14), ncol=3); plt.tight_layout()"""),
    md("The hollow stage-0 markers sit on a leaky test set; stage 1 is the honest starting line. The route back up is real but modest, which is the honest ceiling of a seven-column dataset."),

    md("""### 3.3 Stage by stage
Each large plot below shows that stage's out-of-fold predictions for **random forest (left)** and **boosting (right)**. The **gold line** is the average prediction within deciles of the actual salary; the **grey dashed line** is where that line sat at the previous stage, so you can see what the step moved. The closer the gold line is to the red diagonal, the better the model tracks high and low earners."""),
    code("""def stage_fig(sidx):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.4), sharex=True, sharey=True)
    for a, (tag, name) in zip(axes, [('RF', 'Random forest'), ('LGBM', 'Boosting (LightGBM)')]):
        y = oof['y_raw'] if sidx == 0 else oof['y_dedup']
        prev = None
        if sidx >= 1:
            py = oof['y_raw'] if sidx == 1 else oof['y_dedup']
            prev = (py, oof[f'stage{sidx - 1}_{tag}'])
        draw(a, y, oof[f'stage{sidx}_{tag}'], name, prev=prev, stats=(s.loc[(tag, sidx)].rmse, s.loc[(tag, sidx)].r2))
    axes[0].set_ylabel('Predicted salary (USD)')
    axes[0].legend(loc='lower right', fontsize=9)
    fig.suptitle(f'Stage {sidx}: {STAGE_NAMES[sidx]}' + ('  (leaky test set)' if sidx == 0 else ''), fontsize=14, fontweight='bold')
    plt.tight_layout()

def verdict(sidx, note):
    out = []
    for tag, name in (('RF', 'Random forest'), ('LGBM', 'Boosting')):
        r = s.loc[(tag, sidx)]
        if sidx == 0:
            out.append(f'**{name}:** RMSE {usd(r.rmse)}, R\\u00b2 {r.r2:.3f}')
        else:
            p = s.loc[(tag, sidx - 1)]
            out.append(f'**{name}:** RMSE {usd(r.rmse)} ({r.rmse - p.rmse:+,.0f}), R\\u00b2 {r.r2:.3f} ({r.r2 - p.r2:+.3f})')
    d = [s.loc[(t, sidx)].rmse - s.loc[(t, sidx - 1)].rmse for t in ('RF', 'LGBM')] if sidx > 0 else [0, 0]
    tag_ = 'starting point' if sidx == 0 else ('**helped**' if max(d) < -100 else '**hurt**' if min(d) > 100 else 'mixed or negligible')
    display(Markdown(f'**Result ({tag_}):** ' + ' | '.join(out) + f'  \\n{note}'))
print('stage helpers ready')"""),
    md("""#### Stage 0: the R-style starting point
**What we did:** the data exactly as the R project used it: all 3,755 rows including duplicates, original columns one-hot encoded, salary modelled in dollars, untuned random forest and default boosting.
**Why it is not trustworthy:** copies of the same person can sit in both the training and the test fold, so held-out scores look better than they are."""),
    code("stage_fig(0)"),
    code("verdict(0, 'The model has effectively seen many of the rows it is scored on, so these scores are the most optimistic ones in the notebook.')"),
    md("""#### Stage 1: remove duplicate rows
**What we did:** dropped the 1,172 exact duplicates (31%). Nothing else changed.
**What to expect:** worse scores. This is a correction, not an optimisation."""),
    code("stage_fig(1)"),
    code("verdict(1, 'Scores fall because the leak is gone. This is the honest baseline: every later number is measured against it.')"),
    md("""#### Stage 2: model log-salary
**What we did:** predict log(salary) and convert back to dollars, so errors are proportional to pay instead of dominated by the highest earners. This normally helps linear models."""),
    code("stage_fig(2)"),
    code("verdict(2, 'Tree models gain nothing from the log transform here, and converting back adds a little bias. We keep it because it helps the linear and neural models in the model search.')"),
    md("""#### Stage 3: cleaning and features
**What we did (learned on training folds only):** collapsed job titles and locations with fewer than 10 rows, added job family, region and a level ordinal, and trimmed the top and bottom 1% of salaries **when fitting** (never when scoring)."""),
    code("stage_fig(3)"),
    code("verdict(3, 'Grouping rare categories gives the model fewer, better-supported categories to learn from. The effect is modest.')"),
    md("""#### Stage 4: hyperparameter tuning
**What we did:** randomized search over tree depth, learning rate, regularisation and so on, run *inside* each fold (nested cross-validation) so the search never sees the rows it is scored on."""),
    code("stage_fig(4)"),
    code("verdict(4, 'Tuning helps both families, by a few hundred dollars of RMSE: real, but small next to the noise in the data.')"),
    md("""#### Stage 5: the best model from the search
**What we did:** compared ten model families under the same protocol and picked by the rule in section 6. The plot shows the accuracy winner on the same held-out rows."""),
    code("""fig, ax = plt.subplots(1, 1, figsize=(8, 7.4))
draw(ax, oof['y_dedup'], oof['stage5_BEST'], f'Stage 5: {best_name}', prev=(oof['y_dedup'], oof['stage4_LGBM']), stats=(s.loc[('BEST', 5)].rmse, s.loc[('BEST', 5)].r2))
ax.set_ylabel('Predicted salary (USD)'); ax.legend(loc='lower right', fontsize=9); plt.tight_layout()
b = s.loc[('BEST', 5)]
p = s.loc[('LGBM', 4)]
display(Markdown(f'**Result:** {best_name}: RMSE {usd(b.rmse)} ({b.rmse - p.rmse:+,.0f} vs tuned boosting), R\\u00b2 {b.r2:.3f}. The whole 10-family search changes the answer by roughly the width of the fold-to-fold noise: the data, not the algorithm, sets the limit (section 6).'))"""),
]

# ------------------------------------------------------------------ 4 why diagonal
cells += [
    md("""## 4. Why the R plots hugged the diagonal, and why these do not

**Training fit versus held-out fit.** The original R report plotted predictions on the training data. A flexible model can memorise the rows it trains on, so those plots look excellent. The panels below fit the same models on all rows and plot (left) predictions for rows the model has seen against (right) honest out-of-fold predictions."""),
    code("""from sklearn.metrics import r2_score
fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
X, y = dedup.drop(columns=[TARGET]), dedup[TARGET].to_numpy()
for r, (tag, name, key) in enumerate([('RF', 'Random forest', 'Random forest'), ('LGBM', 'LightGBM', 'LightGBM')]):
    fitted = build(key, 'eng', True, True).fit(X, y)
    pin = fitted.predict(X); pout = oof[f'stage3_{tag}']
    for c, (p, lab) in enumerate([(pin, 'Training rows (seen by model)'), (pout, 'Held-out rows (out of fold)')]):
        a = axes[r, c]; a.scatter(y, p, s=5, alpha=.25, color=RED if c == 0 else BLUE)
        a.plot([0, 450000], [0, 450000], 'k--', lw=1, label='perfect prediction')
        a.xaxis.set_major_formatter(kfmt); a.yaxis.set_major_formatter(kfmt)
        if r == 0 and c == 0: a.legend(loc='upper left', fontsize=8)
        a.set_title(f'{name}: {lab}\\nR\\u00b2 {r2_score(y, p):.2f}')
        if r == 1: a.set_xlabel('Actual salary (USD)')
        if c == 0: a.set_ylabel('Predicted (USD)')
plt.tight_layout()"""),
    md("""**Irreducible noise.** Take every group of people with an identical profile (same title, level, location, company size, employment type, year and remote ratio) and measure how far apart their salaries are. If a model could not tell them apart, this spread is the best it could ever explain."""),
    code("""from features import CAT_RAW, NUM_RAW
g = dedup.groupby(CAT_RAW + NUM_RAW)[TARGET].agg(['count', 'std', 'mean']).query('count >= 5')
cv_ = (g['std'] / g['mean'])
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
cv_.plot.hist(bins=20, ax=ax[0], color=BLUE, title=f'Salary spread inside identical profiles ({len(g)} groups with 5+ rows)')
ax[0].axvline(cv_.median(), color=RED, ls='--', label=f'median {cv_.median():.0%}'); ax[0].legend(); ax[0].set_xlabel('std / mean salary within the group')
big = g.sort_values('count', ascending=False).head(1).index[0]
sub = dedup[(dedup[CAT_RAW + NUM_RAW] == pd.Series(big, index=CAT_RAW + NUM_RAW)).all(axis=1)][TARGET]
sub.plot.hist(bins=20, ax=ax[1], color=GOLD, title=f'Largest identical-profile group (n={len(sub)})'); ax[1].set_xlabel('salary (USD)')
plt.tight_layout()
display(Markdown(f'People with the *same* profile typically differ by **{cv_.median():.0%}** (std/mean). For the largest group above, the same title, level, location, size and year range from {usd(sub.min())} to {usd(sub.max())}. A model that sees only these columns cannot do better than the group average, so held-out predictions will always fan out around the diagonal.'))"""),
    md("""**Why predictions form horizontal bands.** With only seven features there are a few hundred distinct profiles, so many rows receive nearly identical predictions. The bands in the scatter plots are those profile averages, not an artefact."""),
]

# ------------------------------------------------------------------ 5 model search
cells += [
    md("""## 5. Model search
Everything below uses the same cleaned pipeline and the same 15 folds. Each model is shown at sensible defaults and tuned with nested cross-validation (the search never sees the outer test fold).

Candidates: **ridge** (linear), **KNN**, **random forest**, **extra trees**, then the boosting family: **HistGradientBoosting**, **XGBoost**, **LightGBM**, **CatBoost** (built for categorical features), a small **neural net (MLP)**, and a **stacking** blend of ridge, random forest and LightGBM. TabPFN was not tested: it needs a pretrained weights download, so the scikit-learn MLP stands in as the neural option.

> **Note: CatBoost was evaluated at its default settings only.** Its nested tuning run was stopped after about 5 minutes because it needed roughly 25 more (it is the slowest model here), so there is no "tuned" CatBoost point below. Its default is already tied with the best model, so tuning is unlikely to change the conclusion, but a tuned CatBoost remains untested."""),
    code("""s2 = search[search.model != 'Median baseline'].copy()
s2['kind'] = np.where(s2.tuned, 'tuned', 'default')
piv = s2.pivot(index='model', columns='kind', values='rmse')
order = s2[~s2.tuned].set_index('model').rmse.sort_values().index
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
for k, (col, title) in enumerate([('rmse', 'RMSE (USD), lower is better'), ('mae', 'MAE (USD), lower is better'), ('r2', 'R\\u00b2, higher is better')]):
    pv = s2.pivot(index='model', columns='kind', values=col).reindex(order)
    y = np.arange(len(pv))
    ax[k].hlines(y, pv.min(axis=1), pv.max(axis=1), color=GREY, lw=2)
    ax[k].scatter(pv['default'], y, color=GREY, s=55, label='default', zorder=3)
    ax[k].scatter(pv['tuned'], y, color=BLUE, s=55, label='tuned', zorder=3)
    ax[k].set_yticks(y); ax[k].set_yticklabels(pv.index if k == 0 else []); ax[k].invert_yaxis(); ax[k].set_title(title)
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=2, bbox_to_anchor=(.5, -.02))
plt.suptitle('No tuned point: ridge and stacking (no search space) and CatBoost (tuning skipped, see note). Axes are truncated to show the gaps.', y=1.02, fontsize=9)
for a_ in ax: a_.set_xlabel('(axis does not start at zero)', fontsize=8)
plt.tight_layout(rect=(0, .05, 1, 1))"""),
    md("""Tuning moves each model by a few hundred dollars at most, and several tuned models score *worse* than their defaults on held-out folds, a sign that the search is fitting noise rather than signal. The whole top of the table sits within about $1,000 of each other."""),
    code("""base = search.set_index('key')
rows = base[base.model != 'Median baseline'].sort_values('rmse')
fig, ax = plt.subplots(figsize=(9, 7))
y = np.arange(len(rows))
colors = np.where(rows.tied, GREEN, GREY)
ax.barh(y, rows.gap_to_best, xerr=rows.gap_se, color=colors, ecolor='#555', error_kw={'lw': 1.2, 'capsize': 3})
ax.set_yticks(y); ax.set_yticklabels([('tuned ' if t else '') + m for m, t in zip(rows.model, rows.tuned)]); ax.invert_yaxis()
ax.set_xlabel('RMSE gap to the best model (USD), bars = corrected standard error'); ax.set_title('Which models are statistically tied with the best?')
ax.axvline(0, color='k', lw=1)
ax.legend(handles=[Patch(color=GREEN, label='tied with the best (gap smaller than its standard error)'), Patch(color=GREY, label='clearly behind the best')], loc='lower right')
plt.tight_layout()"""),
]

# ------------------------------------------------------------------ 6 decision
cells += [
    md("""## 6. The decision rule (fixed in advance)
1. The **accuracy winner** is the model with the lowest mean held-out RMSE.
2. Any model whose paired fold-by-fold gap to the winner is smaller than its (corrected resampled) standard error is **tied**: the data cannot separate them.
3. Among tied models, prefer **the simplest** (a linear model beats a boosted ensemble when accuracy is equal).
4. For the *product* (the web app), pick the simplest tied model that can output a calibrated 10-90% range natively."""),
    code("""def nm(key): return ('tuned ' if key.startswith('tuned_') else '') + key.split('_', 1)[1]
tied = base[base.tied]
display(Markdown(f'''| Role | Model | Why |
|---|---|---|
| **Accuracy winner** | **{nm(decision['accuracy_winner'])}** | lowest mean RMSE ({usd(base.loc[decision['accuracy_winner'], 'rmse'])}) |
| **Parsimonious pick** | **{nm(decision['parsimonious_pick'])}** | simplest of the {decision['n_tied']} models tied with the winner |
| **Product pick** | **{nm(decision['product_pick'])}** | simplest tied model with native 10-90% quantile output |
'''))
cols = ['model', 'tuned', 'rmse', 'mae', 'r2', 'gap_to_best', 'gap_se', 'tied', 'interval_capable', 'seconds']
(search.sort_values('rmse')[cols].round({'rmse': 0, 'mae': 0, 'r2': 3, 'gap_to_best': 0, 'gap_se': 0, 'seconds': 0})
 .style.hide(axis='index').background_gradient(subset=['rmse'], cmap='Blues_r').format({'rmse': '{:,.0f}', 'mae': '{:,.0f}', 'gap_to_best': '{:,.0f}', 'gap_se': '{:,.0f}'}))"""),
    code("""ic = pd.read_csv(f'{OUT}/interval_compare.csv')
display(ic.round({'coverage': 3, 'median_width': 0, 'pinball': 0}).style.hide(axis='index').format({'coverage': '{:.1%}', 'median_width': '{:,.0f}', 'pinball': '{:,.0f}'}))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
ax[0].barh(ic.model, ic.coverage, color=BLUE); ax[0].axvline(.8, color=RED, ls='--'); ax[0].set_title('Coverage of the 10-90% range (target 80%; axis starts at 60%)'); ax[0].set_xlim(.6, .9)
ax[1].barh(ic.model, ic.median_width, color=GOLD); ax[1].set_title('Median width of the range (USD), narrower is better')
for a in ax: a.invert_yaxis()
plt.tight_layout()"""),
    code("""rd = base.loc['search_Ridge']
ic_best = ic.sort_values('pinball').iloc[0]
app_row = ic[ic.model == 'LightGBM'].iloc[0]
top_span = rows[rows.tied].rmse.max() - rows.rmse.min()
display(Markdown(f'''### Decision
- **No model is clearly better than the others.** {decision['n_tied']} of {len(search)} configurations are statistically tied with the best, spanning only {usd(top_span)} of RMSE ({top_span / rows.rmse.min():.1%}).
- **Accuracy:** the lowest error belongs to **{nm(decision['accuracy_winner'])}**, but a stacked ensemble of three models costs 30 s versus under 1 s for the alternatives and gains {usd(base.loc[decision['parsimonious_pick'], 'gap_to_best'])} over the simplest tied model, **{nm(decision['parsimonious_pick'])}**. By the rule, that is the pick.
- **A near-tie worth knowing about:** plain **ridge regression** misses the cutoff by under a dollar (gap {usd(rd.gap_to_best)} vs standard error {usd(rd.gap_se)}), so by the strict rule it is not tied. In practice it is indistinguishable ({rd.gap_to_best / base.loc[decision['accuracy_winner'], 'rmse']:.2%} behind the best), is far simpler, and is the most interpretable model. If explanations matter more than the last $75, ridge is the better choice; section 8's premium analysis uses a linear model of the same kind.
- **Product (a *range*, not one number):** the rule's pick is **{nm(decision['product_pick'])}**. On the 10-90% range, however, **{ic_best.model}** has the lowest pinball loss ({usd(ic_best.pinball)}) and the web app already uses LightGBM quantile models (coverage {app_row.coverage:.1%}, median width {usd(app_row.median_width)}). The models differ by under 1% on pinball loss, so we **keep LightGBM in the app as a deliberate exception to the rule**: the rule did not score interval quality, and the app's intervals were validated separately.
- **More modelling effort will not help; more information will.** Adding features (skills, industry, tenure, company name) is the only lever left.
'''))"""),
]

# ------------------------------------------------------------------ 7 final diagnostics
cells += [
    md("## 7. The chosen model in detail"),
    code("""key = decision['accuracy_winner']
p = oof[key]; y = oof['y_dedup']
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
ax[0].scatter(y, p, s=6, alpha=.3, color=BLUE)
edges = np.quantile(y, np.linspace(0, 1, 11)); idx = np.clip(np.digitize(y, edges[1:-1]), 0, 9)
ax[0].plot([y[idx == k].mean() for k in range(10)], [p[idx == k].mean() for k in range(10)], '-o', color=GOLD, lw=2)
ax[0].plot([0, 450000], [0, 450000], '--', color=RED); ax[0].set_title(f'{nm(key)}: predicted vs actual (out of fold)'); ax[0].set_xlabel('Actual (USD)'); ax[0].set_ylabel('Predicted (USD)'); ax[0].xaxis.set_major_formatter(kfmt); ax[0].yaxis.set_major_formatter(kfmt)
res = y - p
lvl = pd.Categorical(dedup.experience_level.map({'EN': 'Entry', 'MI': 'Mid', 'SE': 'Senior', 'EX': 'Executive'}), categories=['Entry', 'Mid', 'Senior', 'Executive'], ordered=True)
pd.DataFrame({'level': lvl, 'residual': res}).boxplot(by='level', column='residual', ax=ax[1], showfliers=False, boxprops=dict(color=BLUE), medianprops=dict(color=RED), whiskerprops=dict(color=GREY), capprops=dict(color=GREY)); ax[1].axhline(0, color=RED, ls='--'); ax[1].yaxis.set_major_formatter(kfmt)
ax[1].set_title('Residual (actual - predicted) by level'); ax[1].set_xlabel(''); plt.suptitle('')
ax[2].hist(np.abs(res), bins=40, color=BLUE); ax[2].axvline(np.median(np.abs(res)), color=RED, ls='--')
ax[2].set_title(f'Absolute error, median {usd(np.median(np.abs(res)))}'); ax[2].set_xlabel('USD')
plt.tight_layout()"""),
    code("""from IPython.display import Image
Image('outputs/shap_summary.png', width=620)"""),
    md("A US company location is the largest single effect; entry level pulls salary down, senior and executive levels push it up. These are associations in a survey sample, not causes."),
    code("""import numpy as np
from tune_models import oof_intervals
lo, hi = oof_intervals(dedup)
lo = np.clip(lo, 0, None)  # a salary cannot be negative
yv = dedup[TARGET].to_numpy(); inside = (yv >= lo) & (yv <= hi)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
idx = np.random.default_rng(1).choice(len(yv), 80, replace=False); idx = idx[np.argsort(((lo + hi) / 2)[idx])]
for j, i in enumerate(idx):
    ax[0].vlines(j, lo[i], hi[i], color='#c9d9fb', lw=4); ax[0].plot(j, yv[i], '.', color=BLUE if inside[i] else RED, ms=7)
ax[0].set_title('80 random rows: actual salary vs 80% range (blue = inside, red = outside)'); ax[0].set_ylim(0, None); ax[0].yaxis.set_major_formatter(kfmt); ax[0].set_xlabel('sorted by predicted midpoint'); ax[0].set_ylabel('USD')
pd.Series(hi - lo).plot.hist(bins=30, ax=ax[1], color=BLUE); ax[1].axvline(np.median(hi - lo), color=RED, ls='--')
ax[1].set_title(f'Width of the 80% range, median {usd(np.median(hi - lo))}'); ax[1].set_xlabel('USD')
plt.tight_layout()
display(Markdown(f'The 10-90% quantile range covers **{inside.mean():.1%}** of held-out salaries (target 80%).'))"""),
]

# ------------------------------------------------------------------ 8 business
cells += [
    md("""## 8. What the business can do with this

Raw prediction accuracy is modest (R\\u00b2 about 0.4), so the value is not in quoting one salary. It is in **benchmarking**: knowing the typical range for a role, how big each factor is, and where the data is too thin to trust.

### 8.1 How much does each factor move pay?
Each bar is the percentage difference in salary, holding the other factors constant, from a log-salary linear model with 300 bootstrap resamples (90% interval). Grey bars cross zero: no reliable effect."""),
    code("""pr = pd.DataFrame(ins['premiums'])
pr['group'] = pd.Categorical(pr.factor.str.split(':').str[0], categories=['Level', 'Location', 'Role', 'Company', 'Setup', 'Per year (2020-2023)'], ordered=True)
pr = pr.sort_values(['group', 'pct'])
fig, ax = plt.subplots(figsize=(9, 8))
y = np.arange(len(pr))
col = np.where(~pr.significant, GREY, np.where(pr.pct > 0, GREEN, RED))
ax.hlines(y, pr.lo, pr.hi, color=col, lw=3); ax.scatter(pr.pct, y, color=col, zorder=3, s=40)
ax.axvline(0, color='k', lw=1)
ax.set_yticks(y); ax.set_yticklabels(pr.factor); ax.invert_yaxis()
for yi, (v, n_, f_) in enumerate(zip(pr.pct, pr.n, pr.factor)): ax.text(max(pr.hi) * 1.04, yi, f'{v:+.0f}%' + ('' if 'Per year' in f_ else f'  (n={n_})'), va='center', fontsize=9)
ax.set_xlim(None, max(pr.hi) * 1.45); ax.set_xlabel('% difference in salary'); ax.set_title('Salary premiums and discounts\\ndot = estimate, line = 90% bootstrap interval, reference group named after "vs"', fontsize=10)
plt.tight_layout()
P = {r['factor']: r for r in ins['premiums']}"""),
    code("""display(Markdown(f'''**Reading the chart**
- **Location is the dominant lever.** Compared with a US company, the same role pays about {abs(P['Location: GB vs US']['pct']):.0f}% less in the UK, {abs(P['Location: Europe (other) vs US']['pct']):.0f}% less elsewhere in Europe and {abs(P['Location: IN vs US']['pct']):.0f}% less in India.
- **Seniority compounds:** mid {P['Level: Mid vs Entry']['pct']:+.0f}%, senior {P['Level: Senior vs Entry']['pct']:+.0f}%, executive {P['Level: Executive vs Entry']['pct']:+.0f}% versus entry level.
- **Role matters less than level or location:** analysts earn {abs(P['Role: Analyst vs Engineer']['pct']):.0f}% less than engineers; ML/AI roles {P['Role: ML/AI vs Engineer']['pct']:+.0f}% and leadership {P['Role: Leadership vs Engineer']['pct']:+.0f}%.
- **Remote work shows no pay penalty or premium** (fully remote {P['Setup: Remote vs On-site']['pct']:+.0f}%, interval spans zero): remote is not a cost lever on salary alone.
- **Small companies pay {abs(P['Company: Small vs Medium']['pct']):.0f}% less** than medium ones; large companies are not reliably different.
- **Market drift:** about {P['Per year (2020-2023)']['pct']:+.0f}% per year across 2020-2023.
'''))"""),
    md("""### 8.2 Pay bands by role, level and market
Median salary in thousands of USD where at least 10 people are in the cell. Use the band (25th-75th percentile) in the table below, not the single median, when setting or checking a salary."""),
    code("""bd = pd.DataFrame(ins['bands'])
levels = ['Entry', 'Mid', 'Senior', 'Executive']
fams = sorted(bd.family.unique())  # same rows in both panels, so shared labels stay correct
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
vmax = bd['median'].max() / 1000
for a, mk in zip(ax, ['US', 'Outside US']):
    t = bd[bd.market == mk].pivot(index='family', columns='level', values='median').reindex(index=fams, columns=levels) / 1000
    sns.heatmap(t, annot=True, fmt='.0f', cmap='Blues', vmin=0, vmax=vmax, cbar=mk != 'US', linewidths=.5, ax=a, annot_kws={'size': 9},
                cbar_kws={'label': 'median salary ($k)'} if mk != 'US' else None)
    a.set_title(f'{mk}: median salary ($k), cells with 10+ people'); a.set_xlabel(''); a.set_ylabel('')
plt.tight_layout()"""),
    code("""tab = bd[(bd.market == 'US') & (bd.n >= 30)].sort_values('median', ascending=False)
tab = tab.assign(band=lambda t: t.p25.map(usd) + ' to ' + t.p75.map(usd), median_=lambda t: t['median'].map(usd))
tab[['family', 'level', 'n', 'median_', 'band']].rename(columns={'family': 'Role family', 'level': 'Level', 'n': 'People', 'median_': 'Median', 'band': 'Middle 50% band'}).style.hide(axis='index')"""),
    md("""### 8.3 Where can this be trusted?
Coverage is how often the real salary fell inside the model's 80% range, out of fold. A segment near 80% is well calibrated; the number of people (n) shows how much data stands behind it."""),
    code("""rl = pd.DataFrame(ins['reliability'])
fig, ax = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [3, 2]})
r_ = rl[rl.group != 'All'].copy(); r_['label'] = r_.group + ': ' + r_.segment
r_ = r_.sort_values(['group', 'n'], ascending=[True, False])
ax[0].barh(r_.label, r_.coverage, color=np.where(r_.n < 30, GOLD, BLUE)); ax[0].axvline(.8, color=RED, ls='--')
for yi, (c, n_) in enumerate(zip(r_.coverage, r_.n)): ax[0].text(1.0, yi, f'n={n_}', va='center', ha='right', fontsize=8)
ax[0].set_xlim(0, 1.0); ax[0].invert_yaxis(); ax[0].set_xlabel('share of salaries inside the 80% range (red line = 80% target)'); ax[0].set_title('Coverage of the 80% range by segment (gold = fewer than 30 people)')
ax[1].barh(r_.label, r_.median_width / 1000, color=GOLD); ax[1].invert_yaxis(); ax[1].set_yticklabels([]); ax[1].set_title('Median range width ($k)'); ax[1].set_xlabel('thousand USD')
plt.tight_layout()
weak = r_[(r_.coverage < .76) & (r_.n >= 30)].sort_values('coverage')
display(Markdown('**Least well calibrated segments (30+ people):** ' + (', '.join(f'{a} {c:.0%} (n={n_})' for a, c, n_ in zip(weak.label, weak.coverage, weak.n)) or 'none') + '. The model undercovers where data is thin or pay is unusually low or high.'))"""),
    code("""W = [r for r in ins['reliability'] if r['group'] == 'All'][0]['median_width']
display(Markdown(f'''### 8.4 Decisions this supports

| Business question | How to use the analysis | Confidence |
|---|---|---|
| **Set or review a pay band** for a role | Start from the 25-75% band for the role family, level and market (8.2); adjust for company size (about {P['Company: Small vs Medium']['pct']:.0f}% for small firms) | High where n is large (US engineers, scientists, analysts) |
| **Check an offer** | Enter the profile in the web app; treat the 10-90% range as plausible and an offer below it as a retention or fairness flag | Medium: ranges are wide (median {usd(W)}) |
| **Where to hire** | Location is the biggest cost driver (8.1). The data supports quantifying the gap, not judging output or quality | High for UK, Europe, India; low for thin locations |
| **Remote policy** | Remote work shows no reliable salary penalty or premium ({P['Setup: Remote vs On-site']['pct']:+.0f}%), so going remote does not by itself change what you must pay | Medium: remote is a self-reported field |
| **Career-ladder budgeting** | Plan for about {P['Level: Mid vs Entry']['pct']:+.0f}% (entry to mid) and {P['Level: Senior vs Entry']['pct']:+.0f}% (entry to senior) | High |
| **Forecasting wage inflation** | About {P['Per year (2020-2023)']['pct']:+.0f}% a year in this sample | Low: four years, survey sample |
'''))"""),
    md("""### 8.5 Limits to state out loud
- **Self-reported survey, 2020-2023, USD.** Not a payroll dataset, and not current.
- **Seven columns only.** No skills, tenure, industry, company name or education, which is why R\\u00b2 stops near 0.4.
- **Associations, not causes.** Location "effects" bundle cost of living, taxes and labour markets.
- **Thin segments** (grey bars above) should not drive decisions.
- **A range is not an offer.** Use the model to frame a conversation, not to set an individual's pay."""),
    md("""## 9. Summary
1. The original R result was inflated by duplicate rows and, in places, by evaluating on training data.
2. Cleaning, features and tuning improve held-out error by a few percent, not a transformation. The plots move toward the diagonal but cannot reach it because identical profiles differ by about 30%.
3. Ten model families were tested. Their accuracy is statistically indistinguishable, so the rule picks the simplest model for accuracy and a quantile boosting model for the range-based product.
4. The business value is in benchmarking: location, level and role premiums, pay bands, and an honest view of where the data is too thin to trust."""),
]

# presentation: collapse the code (Jupyter/VS Code honour source_hidden; tools/make_report.py strips it fully)
for c in cells:
    if c.cell_type == "code":
        c.metadata["jupyter"] = {"source_hidden": True}
        c.metadata["tags"] = ["hide-input"]

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "salary_analysis.ipynb")
print(f"wrote salary_analysis.ipynb with {len(cells)} cells")
