# Sens-AI-tional: How Does AI Adoption Compare with Productivity Change?

**ENGE707 Data Engineering and Machine Learning, Auckland University of Technology (AUT)**
Group project (3 members), 2026

A data engineering and machine learning study of whether companies that adopt more AI report greater productivity gains. Phase 1 covers the data pipeline and exploratory analysis. Phase 2 compares an evolutionary rule-learning system (eLCS) against conventional machine learning models.

## Research question

> How does a company's AI adoption rate compare with the productivity change it subsequently reports?

The project began with several AI-related variables (maturity, ethics governance, displacement, investment) and was narrowed, following lecturer feedback, to one comparison:

| Role | Column | Meaning |
|---|---|---|
| Focus variable 1 | `ai_adoption_rate` | % of a company's operations that use AI |
| Focus variable 2 | `productivity_change_percent` | % change in productivity subsequently reported |

## Dataset

[Global AI Adoption and Workforce Impact Dataset](https://www.kaggle.com/datasets/mohankrishnathalla/global-ai-adoption-and-workforce-impact-dataset) (Kaggle, CC0). The file is included as `ai_company_adoption.csv`.

- 150,000 rows, 43 columns, Q1 2023 to Q4 2026
- One row per company per quarter (`company_id` repeats across quarters)
- No missing values, duplicates or logically inconsistent records, so Phase 1 cleaning was mostly documenting quality rather than fixing errors

## Phase 1: data pipeline and exploratory analysis

Notebook: [`ENGE707_Phase1_Adoption_Vs_Productivity.ipynb`](ENGE707_Phase1_Adoption_Vs_Productivity.ipynb)

1. Problem definition and dataset selection
2. Data acquisition, inspection and documentation
3. Cleansing and transformation, including IQR outlier investigation (high values flagged rather than removed where there was no evidence of error)
4. Exploratory analysis: descriptive statistics, distributions, outliers, relationships, grouped comparisons, correlation heatmap

Headline result: adoption rate and productivity change are positively correlated (Pearson r = 0.675), with a fitted line of `productivity_change_percent = 0.2616 * ai_adoption_rate - 0.26`.

Supporting material: pipeline diagram and EDA charts in [`figures/`](figures/), chart-justification notes and the draft Phase 1 report in [`journal/`](journal/), and earlier exploratory notebooks in [`legacyCode/`](legacyCode/).

## Phase 2: eLCS versus conventional models

The task is a 3-class classification of productivity change (Low / Medium / High, split into tertiles of `productivity_change_percent`). All models share the same set-up so results can be compared with paired statistical tests:

- 10 random seeds
- 80/20 stratified train/test split per seed
- The same fixed 5,000-row test slice per seed
- The same 7 selected features for the improved and conventional models: `ai_adoption_rate`, `ai_maturity_score`, `ai_failure_rate`, `ai_training_hours`, `task_automation_rate`, `ai_adoption_stage`, `industry`

| Task | Model | Script |
|---|---|---|
| 2 | Original eLCS, raw data, all 40 numeric-encoded columns | `task2_raw_lcs Suemon.py` |
| 3 | Original eLCS, preprocessed (about 29 columns, leakage and outliers handled) | `task3_preprocessed_lcs.py`, `task3_elcs_preprocessed_avathanshu.py` |
| 4 | Improved eLCS, 7 selected features | `task4_improved_lcs Suemon.py`, `task4_elcs_avathanshu.py` |
| 5 | Paired t-tests and Wilcoxon tests with Holm correction | `task5_statistical_tests.py` |
| 6 | Logistic Regression, Decision Tree, Random Forest (7 selected features) | `qm_logistic_reg.ipynb`, `task6_decision_tree.py`, `task6_random_forest.py` |

The improved eLCS changes one thing only, feature selection, and keeps population size, iterations, split logic and seeds identical to the raw baseline, so any gain can be attributed to the feature set.

### Results (mean accuracy over 10 seeds)

| Model | Mean accuracy | Std |
|---|---|---|
| Logistic Regression (7 features) | 0.6291 | 0.0019 |
| Random Forest (7 features) | 0.6259 | 0.0088 |
| Decision Tree (7 features) | 0.6227 | 0.0095 |
| Improved eLCS (7 features) | 0.5877 | 0.0129 |
| Original eLCS, preprocessed | 0.4634 | 0.0814 |
| Original eLCS, raw | 0.3916 | 0.0505 |

Using Holm-corrected paired tests at the 0.05 level:

- Logistic Regression, Random Forest and Decision Tree are **not significantly different** from one another.
- All three conventional models are significantly better than every eLCS variant.
- The improved eLCS is significantly better than both original eLCS variants, which supports the claim that feature selection shrinks the rule search space and helps.
- Raw and preprocessed original eLCS are not significantly different from each other after correction.

Full outputs are in `Phase 2 Assignment/` and `Phase 2 Avathanshu/Task5 Results/` (per-seed CSVs, learned rule populations, and `Task 5 results.txt`).

## Repository layout

| Path | Contents |
|---|---|
| `ai_company_adoption.csv` | Dataset (CC0) |
| `ENGE707_Phase1_Adoption_Vs_Productivity.ipynb` | Phase 1 notebook |
| `Phase 2 Assignment/` | eLCS scripts and notebooks (raw, preprocessed, improved), Random Forest, logistic regression notebook, results, planning documents |
| `Phase 2 Avathanshu/` | Alternative eLCS implementations, a custom eLCS module, Decision Tree, statistical tests and their results |
| `figures/` | Pipeline diagram and EDA charts |
| `journal/` | Chart rationale notes, findings summary, meeting notes, draft report |
| `legacyCode/` | Earlier exploratory notebooks |
| `data set website.txt` | Dataset source link |

## Running the code

**Requirements:** Python 3, `pandas`, `numpy`, `scikit-learn`, `scipy`, `jupyter`, and [scikit-eLCS](https://github.com/UrbsLab/scikit-eLCS) (`skeLCS`) for the eLCS scripts.

```bash
pip install pandas numpy scikit-learn scipy jupyter scikit-eLCS
```

Notes:

- The scripts read `ai_company_adoption.csv` by relative path, so run them from the repository root, or edit `RAW_DATA_PATH` at the top of the script.
- Some of the eLCS scripts under `Phase 2 Avathanshu/` expect a local copy of the scikit-eLCS source in a `scikit-eLCS-master` folder next to the script (see `ELCS_REPO` at the top of each file). Either place it there or install the package.
- eLCS is computationally expensive; the scripts train it on a small subsample (population size 1,500, 5,000 learning iterations), whereas the conventional models train on the full training partition.
- Scripts write their result CSVs to the current working directory. `task5_statistical_tests.py` reads all `*seed_results.csv` files in the current directory.

```bash
python "Phase 2 Assignment/task6_random_forest.py"
jupyter notebook ENGE707_Phase1_Adoption_Vs_Productivity.ipynb
```

## Team

| Member | Main areas (from commit history and file names) |
|---|---|
| **Suemon Kwok** | Phase 2 eLCS pipeline (raw, preprocessed and improved), Random Forest, pipeline diagram and several EDA charts |
| **Avathanshu Bhat** | Exploratory notebooks, Decision Tree, statistical testing, alternative eLCS implementations, journal notes |
| **Quentin Masoe** | Logistic regression model, cleaned dataset variant, Phase 1 report draft |

Commit counts at the time of writing: Suemon Kwok 86, Avathanshu Bhat 26, Quentin Masoe 15.

## Licence

The dataset is CC0. No licence has been specified for the code, which is coursework; please contact the authors before reusing it.
