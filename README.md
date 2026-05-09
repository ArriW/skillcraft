# SkillCraft Regression

### What in-game habits actually move a StarCraft II player up the ranked ladder?

A regression study of **3,300 ranked StarCraft II players**, told twice — once with the wrong tool (OLS), once with the right one (a cumulative link model). The original was a STAT 757 final from December 2020. The 2026 extension fixes the methodological hole the original flagged but couldn't close: the response is *ordinal*, and OLS doesn't know that.

<p align="center">
  <img src="inline/StarCraft-II-Leagues.png" alt="StarCraft II league icons" width="640">
</p>

## Read the report

- **[skillcraft_regression.html](skillcraft_regression.html)** &mdash; rendered notebook with all 18 plots. Open in any browser.
- **[skillcraft_regression.ipynb](skillcraft_regression.ipynb)** &mdash; executable Jupyter notebook (Python).
- **[skillcraft_regression.Rmd](skillcraft_regression.Rmd)** &mdash; the original R Markdown source, with the 2026 ordinal extension appended.

## Headline result

| Model | Tool | Accuracy | Within-1 league | MAE |
|---|---|---:|---:|---:|
| Base-rate guess | (none) | 19% | — | — |
| `lm_omega` (backward OLS) | `statsmodels.OLS` | 35% | — | — |
| `lm_stepwise` (best-subset OLS) | `statsmodels.OLS` | 40% | 88% | 0.73 |
| **`ord_final` (CLM, probit, 10 predictors)** | **`statsmodels.OrderedModel`** | **43%** | **88%** | **0.70** |
| 5-fold CV mean | (validation) | 43% | 88% | 0.70 |

The accuracy bump is small &mdash; but the interesting property is that **when the ordinal model misses, it misses by a single league**. Confusion mass concentrates on the diagonal and one step off. That's what the OLS rounding hack could never give you.

## What's in the report

1. **Data exploration** &mdash; Shapiro-Wilk on the response, correlation heatmap, violin plots by league for every predictor.
2. **OLS modeling, twice** &mdash; manual backward selection, then exhaustive best-subset by BIC. Confounding diagnostics on APM vs. ActionLatency, hours metrics, hotkey families.
3. **Diagnostics** &mdash; VIF, residuals, Cook's distance, density comparisons.
4. **Original ordinal sketch** &mdash; the half-finished `clm` from the college version, kept as a "before" reference.
5. **Proper ordinal extension (2026)** &mdash; link-function comparison, likelihood-based stepwise selection, forest plot of standardized odds ratios, predicted-probability stacked-area plots, latent-score density map with cutpoints, calibration plots, 5-fold CV.

## What changes when you do it right

- **Coefficients become odds ratios on a common scale.** Standardized predictors mean a forest plot ranks them at a glance.
- **The latent-score density plot makes the cutpoint geometry literal** &mdash; you can see where adjacent leagues are *inherently* hard to separate, regardless of which model you fit.
- **Predictions stay on the league grid.** No more rounding fitted values that came out at 8.4 or 0.6.
- **The Elo/Gumbel motivation from the introduction becomes testable** &mdash; it's a question about which link function fits best (logit vs. probit vs. cloglog), not philosophy.

## The skill takeaway

Once you control for confounding among the speed-of-action predictors, **APM is a follower, not a leader**. The predictors that move rank most are:

1. **TotalHours** &mdash; there is no shortcut for time invested.
2. **MinimapAttacks** &mdash; commanding off-screen is a habit, and a teachable one.
3. **AssignToHotkeys** and **SelectByHotkeys** &mdash; using hotkeys both to *bind* groups and to *recall* them.
4. **ActionLatency** &mdash; short delay between focusing and acting; a sign of confidence and game-state recognition.

Speed (APM) emerges from those habits, not the other way around.

## Reproducing

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn scipy jupyter
python build_notebook.py
jupyter nbconvert --to notebook --execute skillcraft_regression.ipynb \
                  --output skillcraft_regression.ipynb
jupyter nbconvert --to html skillcraft_regression.ipynb
```

The R Markdown version of the report (`skillcraft_regression.Rmd`) requires `tidyverse`, `ordinal`, `broom`, `corrplot`, `car`, `caret`, and `leaps`.

## Repository layout

```
skillcraft/
├── README.md                       # this file
├── SkillCraft1_Dataset.csv         # 3,395 rows × 20 columns (UCI ML Repository)
├── skillcraft_regression.ipynb     # Python notebook (executed)
├── skillcraft_regression.html      # rendered report — open in a browser
├── skillcraft_regression.Rmd       # R Markdown source (original + 2026 extension)
├── build_notebook.py               # generator script for the notebook
├── inline/                         # images and bibliography
└── archive/                        # original R-rendered HTML / PDF / DOCX
```

## Citing the data

> Thompson, J. J., Blair, M. R., Chen, L., & Henrey, A. J. (2013). *SkillCraft1 Master Table Dataset*. UCI Machine Learning Repository.
> <https://archive.ics.uci.edu/ml/datasets/SkillCraft1+Master+Table+Dataset>
