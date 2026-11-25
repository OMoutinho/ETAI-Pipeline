# Baseline Predictive Pipeline -- ETAI

Gonçalo Moutinho - Nº 20260508

### Week 2: Initial Baseline
* **LR (Logistic Regression)**
  * Train accuracy: 0.679
  * Test accuracy: 0.678
  * Gap (train - test): +0.001
* **DT (Decision Tree)**
  * Train accuracy: 0.829
  * Test accuracy: 0.622
  * Gap (train - test): +0.207

> **Overview 2nd week:** Currently the better model is LR, because it's not overfitting like the DT model is.

---

### Week 3: After Data Cleaning
* **LR**
  * Train accuracy: 0.678
  * Test accuracy: 0.655
  * Gap (train - test): +0.023  
  * *Note:* After cleaning, there was an insignificant decrease in train/test accuracy and a slight increase in the gap.
* **DT**
  * Train accuracy: 0.799
  * Test accuracy: 0.608
  * Gap (train - test): +0.192  
  * *Note:* After cleaning, the gap was reduced slightly, suggesting less overfitting, though accuracy dropped a bit.

> **Overview 3rd week:** Currently the better model is still LR, as it still has a better test accuracy.

---

### Week 4: Stratified Cross-Validation
Introduced 5-fold stratified cross-validation on the development set to obtain a more reliable performance estimate and reduce dependence on a single train/validation split.

* **LR:** CV Train: 0.675 | CV Validation: 0.671 | Gap: +0.004
* **DT:** CV Train: 0.694 | CV Validation: 0.611 | Gap: +0.083
* **RF (Random Forest):** CV Train: 0.733 | CV Validation: 0.645 | Gap: +0.089

> **Overview 4th week:** 
> Logistic Regression achieved the best validation accuracy (~67.1%) with almost no gap (+0.004), showing great generalization. Random Forest and Decision Tree showed larger gaps (~8.9% and ~8.3%), indicating overfitting. LR remained the best-performing model (locked test set untouched).

---

### Weeks 5 & 6: Hyperparameter Tuning, Ensembles & Final Evaluation
Tested all pipeline models with Optuna hyperparameter optimization (30 trials x 5 folds nested CV) using robust scaling and target encoding on the development set (5,771 rows), keeping a locked test set (1,443 rows) aside.

* **Tuned Models Summary (Honest Nested CV Accuracy):**
  * **Logistic Regression** (C $\approx$ 0.075): **67.1%** validation accuracy. Highly stable with minimal gap.
  * **Decision Tree** (max_depth: 6, min_samples_leaf: 101): **67.2%** validation accuracy. Strong pruning to control overfitting.
  * **Random Forest** (max_depth: 22, min_samples_leaf: 42, max_features: 0.5): **68.1%** validation accuracy.
  * **Oblique Random Forest** (max_depth: 6, n_components: 5): **67.9%** validation accuracy.
  * **AdaBoost** (max_depth: 4, n_estimators: 220, learning_rate: 0.025): **68.2%** validation accuracy (Top performer).

* **Fairness Audit (False Positive Rate by Race):**
  * Compared to COMPAS's original score (which had an FPR of **0.45** for African-Americans), our tuned models significantly reduced the disparity (e.g., African-American FPR dropped to **0.25 – 0.29**, while Caucasian FPR sat around **0.11 – 0.17**).
  
### Week 7: Advanced Boosting Models & Expanded Evaluation Metrics
Expanded the evaluation framework to include a comprehensive suite of metrics (`accuracy`, `balanced_accuracy`, `precision`, `recall`, `f1`, `fbeta_2`, `roc_auc`, `average_precision`), with model selection guided primarily by **ROC-AUC** (compared against a dummy no-model floor of 0.500).

* **Gradient Boosting**
  * **Honest Nested CV ROC-AUC:** **0.727** (std: 0.014)
  * **Best Hyperparameters:** `n_estimators`: 426, `learning_rate`: 0.012, `max_depth`: 3, `min_samples_leaf`: 2, `subsample`: 0.766
* **Hist Gradient Boosting**
  * **Honest Nested CV ROC-AUC:** **0.728** (std: 0.016)
  * **Best Hyperparameters:** `max_iter`: 92, `learning_rate`: 0.095, `max_leaf_nodes`: 6, `min_samples_leaf`: 84, `l2_regularization`: 0.001

> **Overview 7th week:** 
> Transitioning to boosting algorithms (especially Gradient Boosting and Hist Gradient Boosting optimized for ROC-AUC) noticeably improved ranking performance, achieving an AUC of ~0.728 compared to earlier accuracy-focused runs.

* **Fairness Audit (False Positive Rate by Race — e.g., Hist Gradient Boosting):**
  * **African-American:** FPR = **0.29** (significantly lower than COMPAS's original **0.45**).
  * **Caucasian:** FPR = **0.15** (vs. COMPAS's **0.23**).
  * **Hispanic:** FPR = **0.16**.


### Week 8: XGBoost Integration & Feature Selection Experiments
Integrated **XGBoost** into the pipeline and tested the impact of automated feature selection (`VoteSelector` with `min_votes=2` before tuning) versus running with feature selection turned off.

* **XGBoost (With Feature Selection: `min_votes=1` / kept features):**
  * **Honest Nested CV ROC-AUC:** **0.726** (std: 0.013)
  * **Features kept:** Reduced to key predictive features (e.g., `numeric__age`, `numeric__priors_count`).
* **XGBoost (Feature Selection: `off`):**
  * **Honest Nested CV ROC-AUC:** **0.727** (std: 0.015)
  * **Best Hyperparameters:** `n_estimators`: 287, `learning_rate`: 0.015, `max_depth`: 3, `subsample`: 0.508.

> **Overview 8th week:** 
> Adding XGBoost yielded competitive performance matching the top boosting models (ROC-AUC ~0.727). Interestingly, turning feature selection off performed slightly better or on par with using `VoteSelector`, indicating that the gradient booster handles the full set of features robustly without harming generalization.

* **Fairness Audit (False Positive Rate by Race — XGBoost):**
  * **African-American:** FPR = **0.30** (substantially reduced compared to COMPAS's original **0.45**).
  * **Caucasian:** FPR = **0.16** (vs. COMPAS's **0.23**).
  * **Hispanic:** FPR = **0.16**.

### Week 10: Configurable Numeric Imputers, Model Persistence & Inference on New Data
Upgraded numeric preprocessing to support configurable imputers via `preprocessing.imputation.numeric_strategy` (`median`, `mean`, `knn` via `KNNImputer`, `iterative` via MICE `IterativeImputer`, and `random_forest` via `IterativeImputer` with a random forest estimator per column), with numeric scaling applied before imputation so scalers ignore missing values during fitting and distance-based imputers operate on a common scale. None of the imputers uses the target (`y`), allowing safe inference on unseen data, while the chosen imputer is now printed, logged in `runs.csv`, and included in run report headers. `main.py` also saves the final fitted pipeline (refit on all 5,771 development rows, with the 1,443 test rows still locked) to `models/model.joblib`, paired with `predict.py` for batch inference on new CSVs (`data/compas_new_people.csv`).

- **Oblique Random Forest (`median` imputer, tuned):**
  - **Honest Nested CV ROC-AUC:** **0.727** (std: 0.014; untuned: **0.685**, train-validation gap: +0.150 before tuning)
  - **Best Hyperparameters:** `estimator__tree__max_depth`: 28, `estimator__tree__min_samples_leaf`: 61, `estimator__project__n_components`: 8
  - **New People Inference (`data/compas_new_people.csv`, threshold 0.5):** Predicted **5 positives** (`900004`, `900006`, `900012`, `900014`, `900017`) and **15 negatives** out of 20 new records.
- **Logistic Regression (`median` imputer, tuned):**
  - **Honest Nested CV ROC-AUC:** **0.718** (std: 0.014; train-validation gap: +0.002)
  - **Best Hyperparameters:** `model__C`: ~0.00118, `model__penalty`: `l2`, `model__l1_ratio`: ~0.340
  - **New People Inference (`data/compas_new_people.csv`, threshold 0.5):** More conservative under strong regularization, predicting **2 positives** (`900012` at 0.786, `900014` at 0.679) and **18 negatives** out of 20 new records.
- **XGBoost (`median` imputer, tuned — run `20261119_150109`):**
  - **Honest Nested CV ROC-AUC:** **0.727** (std: 0.015; untuned: **0.724**, train-validation gap: +0.034)
  - **Additional Nested CV Metrics (mean ± std):** accuracy **0.681 ± 0.013**, balanced accuracy **0.670 ± 0.013**, precision **0.674 ± 0.025**, recall **0.567 ± 0.039**, F1 **0.615 ± 0.020**, F2 **0.585 ± 0.031**, average precision **0.691 ± 0.027**
  - **Best Hyperparameters:** `n_estimators`: 287, `learning_rate`: 0.01466, `max_depth`: 3, `min_child_weight`: 1.993, `subsample`: 0.5083, `colsample_bytree`: 0.8338, `reg_lambda`: 0.1488, `reg_alpha`: 0.01377, `gamma`: 0.3906
  - **New People Inference (`data/compas_new_people.csv`, threshold 0.5):** Predicted **5 positives** (`900004` at ~0.623, `900006` at ~0.590, `900012` at ~0.826, `900014` at ~0.718, `900017` at ~0.704) and **15 negatives** out of 20 new records.
- **AdaBoost (`median` imputer, tuned — run `20261119_151539`):**
  - **Honest Nested CV ROC-AUC:** **0.727** (std: 0.015; untuned: **0.723**, train-validation gap: +0.013 in nested CV / +0.005 default CV)
  - **Additional Nested CV Metrics (mean ± std):** accuracy **0.683 ± 0.015**, balanced accuracy **0.673 ± 0.015**, precision **0.677 ± 0.022**, recall **0.569 ± 0.035**, F1 **0.618 ± 0.022**, F2 **0.588 ± 0.030**, average precision **0.689 ± 0.025**
  - **Best Hyperparameters:** `model__estimator__max_depth`: 4, `model__n_estimators`: 55, `model__learning_rate`: ~0.167
  - **New People Inference (`data/compas_new_people.csv`, threshold 0.5):** Predicted **6 positives** (`900004` at ~0.754, `900006` at ~0.643, `900008` at ~0.548, `900012` at ~0.881, `900014` at ~0.721, `900017` at ~0.772) and **14 negatives** out of 20 new records.
  - **Fairness Audit (False Positive Rate by Race, OOF):** African-American FPR = **0.30** (vs. COMPAS **0.45**), Caucasian FPR = **0.17** (vs. **0.23**), Hispanic FPR = **0.15** (vs. **0.23**).

> **Overview 10th week:** 
> Making the numeric imputer configurable and scaling prior to imputation keeps preprocessing leak-free and ready for unseen records without requiring target labels. Saving the fitted pipeline to `models/model.joblib` and running `predict.py` on `data/compas_new_people.csv` completed the end-to-end training-to-inference workflow while keeping the locked test set untouched for final evaluation.

- **Fairness Audit (False Positive Rate by Race — Tuned Logistic Regression, OOF):**
  - **African-American:** FPR = **0.18** (down from COMPAS's original **0.45**).
  - **Caucasian:** FPR = **0.07** (vs. COMPAS's **0.23**).
  - **Hispanic:** FPR = **0.09** (vs. COMPAS's **0.23**)

### Week 11: Unsupervised Anomaly Score Feature Engineering (`AnomalyScore`)
Extended the preprocessing pipeline with an unsupervised `AnomalyScore` transformer (`src/preprocessing.py`) integrated into `build_pipeline` (`src/model.py`) and configurable via `config.yaml` (`anomaly.method`: `"none"`, `"isolation_forest"`, or `"lof"`). When enabled, the step fits strictly on each training fold (preventing data leakage) and appends an `anomaly_score` feature measuring how unusual each record is relative to the training distribution—using Isolation Forest (`-score_samples(X)`) or Local Outlier Factor (`novelty=True`)—before optional feature selection and model training. The selected anomaly method is logged in `runs.csv`, displayed in the scoreboard, and serialized inside `models/model.joblib` for seamless inference with `predict.py` (verified on `data/compas_new_people.csv`, returning all 20 records with 7 predicted positives and 13 negatives at threshold `0.5` on the latest saved `hist_gradient_boosting` pipeline).

- **Initial Guess & Top-5% Anomaly Recidivism Check:**
  - **Initial guess:** Reoffenders are not strictly defined as statistical outliers in COMPAS, so `anomaly_score` was expected to offer at most a marginal lift rather than a major jump in ROC-AUC.
  - **Top 5% most unusual (`isolation_forest`) vs. remaining 95%:** Recidivism rate is **0.595** in the top 5% most anomalous rows versus **0.443** across the remaining 95% (+15.2 percentage points).
- **AdaBoost (`median` imputer, untuned — Anomaly Method Comparison):**
  - **`none` (run `20261125_104541`):** CV ROC-AUC **0.723 ± 0.016** (accuracy **0.676 ± 0.012**, F1 **0.599 ± 0.018**, average precision **0.685 ± 0.022**).
  - **`lof` (run `20261125_105105`):** CV ROC-AUC **0.722 ± 0.016** (accuracy **0.677 ± 0.011**, F1 **0.604 ± 0.017**, average precision **0.684 ± 0.024**).
  - **`isolation_forest` (run `20261125_105144`):** CV ROC-AUC **0.722 ± 0.016** (accuracy **0.677 ± 0.012**, F1 **0.605 ± 0.015**, average precision **0.683 ± 0.025**).
- **Hist Gradient Boosting (`median` imputer, tuned — 30 trials nested CV):**
  - **`isolation_forest` (run `20261125_110105`):**
    - **Honest Nested CV ROC-AUC:** **0.726 ± 0.014** (untuned: **0.707 ± 0.018**, train-validation gap: +0.059)
    - **Additional Nested CV Metrics (mean ± std):** accuracy **0.674 ± 0.016**, balanced accuracy **0.671 ± 0.016**, precision **0.639 ± 0.018**, recall **0.638 ± 0.026**, F1 **0.638 ± 0.020**, F2 **0.638 ± 0.023**, average precision **0.689 ± 0.025**
    - **Best Hyperparameters:** `max_iter`: 494, `learning_rate`: 0.02389, `max_leaf_nodes`: 4, `min_samples_leaf`: 124, `l2_regularization`: 2.034e-05
  - **`none` (run `20261125_110759` — controlled baseline):**
    - **Honest Nested CV ROC-AUC:** **0.727 ± 0.016** (untuned: **0.717 ± 0.020**, train-validation gap: +0.058)
    - **Additional Nested CV Metrics (mean ± std):** accuracy **0.675 ± 0.015**, balanced accuracy **0.671 ± 0.015**, precision **0.639 ± 0.017**, recall **0.638 ± 0.023**, F1 **0.639 ± 0.018**, F2 **0.638 ± 0.021**, average precision **0.692 ± 0.026**
    - **Best Hyperparameters:** `max_iter`: 82, `learning_rate`: 0.1109, `max_leaf_nodes`: 4, `min_samples_leaf`: 130, `l2_regularization`: 0.0006996

> **Overview 11th week:** 
> The initial guess held up: although the top 5% most unusual records reoffend more often (`0.595` vs. `0.443`), `anomaly_score` did not improve cross-validated ROC-AUC over `none` (`0.722` vs. `0.723` on untuned AdaBoost; `0.726` vs. `0.727` on tuned Hist Gradient Boosting, well within fold standard deviation). Because the top-anomaly slice is small (5%), contains both classes, and relies on feature patterns already accessible to tree splits, adding unsupervised outlier scores does not beat the clean `none` baseline on COMPAS.
## Project structure

```text
.
├── main.py                    # entry point: train, cross-validate, tune & save model/report
├── predict.py                 # batch inference on unseen data using the saved pipeline
├── config.yaml                # main configuration (COMPAS dataset, preprocessing, tuning, model)
├── config_adult.yaml          # alternative dataset configuration (Adult dataset)
├── requirements.txt           # Python dependencies
├── predictions.csv            # generated predictions from predict.py
├── src/
│   ├── __init__.py
│   ├── data.py                # dataset loading
│   ├── data_diagnostics.py    # data quality checks and diagnostics
│   ├── preprocessing.py       # cleaning, scaling/imputation pipeline & leak-safe dev/test split
│   ├── model.py               # model & pipeline construction (including optional feature selection)
│   ├── tuning.py              # Optuna hyperparameter tuning & nested cross-validation
│   ├── evaluate.py            # CV metrics, OOF classification report & fairness audit
│   └── results.py             # run reports and scoreboard logging (runs.csv)
├── models/                    # saved fitted pipelines (e.g., models/model.joblib)
├── results/                   # run reports and scoreboard (created automatically)
└── data/
    ├── compas_two_year_recidivism.csv
    ├── compas_new_people.csv
    ├── adult_train.csv
    └── README.md              # problem description + full data dictionary

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |

## Environment setup

You only need to do this once per machine.

### macOS / Linux
```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, loads and preprocesses the data, trains the model, and prints:
- **train accuracy and test accuracy, side by side.** Comparing the two is how you catch overfitting: if the model looks much better on the data it was trained on than on data it's never seen, it has memorised rather than learned something that generalises. 
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

## Missing values
- Numeric features (`age`, `juv_fel_count`, `juv_misd_count`, `juv_other_count`, `priors_count`) are scaled first (`none | standard | minmax | robust`) and then imputed according to `preprocessing.imputation.numeric_strategy` (`median`, `mean`, `knn`, `iterative`, `random_forest`).
- Categorical features (`sex`, `age_cat`, `c_charge_degree`) are imputed (`most_frequent` by default) and encoded (`onehot | ordinal | count | target`), with missingness indicators (`<col>_was_missing`) passed through for `mnar_indicator_sources`.
- All fitting happens strictly inside CV/tuning splits or on the development set (`X_dev`, `y_dev`), and no imputer uses the target, avoiding data leakage.

## Predicting new data
Train and save the final pipeline with `main.py` (writes `models/model.joblib`, `results/runs.csv`, and the run report), then run inference on unseen data with `predict.py`:

```bash
python predict.py data/compas_new_people.csv predictions.csv

## Dataset

See `data/README.md`.
