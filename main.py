"""
Entry point for the baseline predictive pipeline.

Run with:
    python main.py

This orchestrates the full (deliberately simple) pipeline:
    load config -> load data -> preprocess -> split -> train
    -> evaluate (train & test) -> save results
"""
import yaml
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from src.data import load_data
from src.preprocessing import (
    clean_dataset, drop_duplicate_rows, split_features_target, build_preprocessor, split_dev_test,
)
from src.model import build_model
from src.evaluate import evaluate, fairness_report, cross_validate_pipeline, oof_classification_report, cv_report
from src.results import save_run



def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    df = load_data(config["data"]["path"])
    df_clean = clean_dataset(df, config["diagnostics"])
    df_clean = drop_duplicate_rows(df_clean, config["diagnostics"].get("id_column"))
    mnar_sources = config["preprocessing"].get("mnar_indicator_sources", [])
    X, y, extras = split_features_target(df_clean, config["data"], mnar_sources)
    # week 4: the final test set is set aside HERE and never used again in this script.
    # Every decision from now on (preprocessing, model, hyperparameters) is made on the
    # development set only. The test set is used only for the final assessment.
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X, y, extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )
    # preprocessing lives INSIDE the pipeline, so cross-validation re-fits it on the
    # training part of every fold -- the validation fold never leaks into its own preprocessing
    pipeline = Pipeline([
        ("prep", build_preprocessor(config["preprocessing"])),
        ("model", build_model(config["model"])),
    ])

    cv_config = config["cv"]
    shuffle = cv_config.get("shuffle", True)
    cv = StratifiedKFold(n_splits= cv_config["n_splits"], shuffle= shuffle,
                         random_state= cv_config.get("random_state") if shuffle else None)
    scoring = cv_config.get("scoring", "accuracy")

    fold_scores, y_oof = cross_validate_pipeline(pipeline, X_dev, y_dev, cv=cv, scoring=scoring,
                                                n_jobs=cv_config.get("n_jobs", 1))

    report = cv_report(fold_scores, scoring)
    report += "\n\n" + oof_classification_report(y_dev, y_oof)
    report += "\n" + fairness_report(
        y_dev, y_oof, extras_dev, sensitive_attr=config["data"]["sensitive_attr"]
    )

    # the model we'd actually use: same pipeline, refit on EVERY development row. CV above
    # estimated how well this recipe does; it didn't produce a model.
    final_model = pipeline.fit(X_dev, y_dev)
    refit = f"Final model: {config['model']['type']} refit on all {len(X_dev)} development rows."
    print(refit)
    report += "\n" + refit + "\n"

    locked = (f"Locked test set: {len(X_test)} rows set aside, not evaluated. "
              f"Development set: {len(X_dev)} rows.")
    print(locked)
    report += "\n" + locked + "\n"

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()

