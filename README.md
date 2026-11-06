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

## Project structure

```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # cleaning + train/test split
│   ├── model.py             # model construction
│   ├── evaluate.py         # accuracy metrics + fairness check
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

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

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Dataset

See `data/README.md`.
