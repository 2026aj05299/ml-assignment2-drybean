"""
Dry Bean variety classification - training & evaluation script.

Steps
1. Load the UCI Dry Bean dataset (id 602): local CSV first, ucimlrepo as fallback.
2. Clean: drop missing rows and exact duplicates.
3. Stratified 80/20 train/test split; save the test split as ../test_data.csv.
4. Tune each classifier with 5-fold stratified CV on the TRAINING set only.
5. Evaluate the tuned models once on the held-out test set:
   Accuracy, AUC (one-vs-rest, macro), Precision / Recall / F1 (macro), MCC.
6. Save fitted pipelines (.joblib), feature metadata and the metrics table.

Run from the project root:  python model/train_models.py
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
RAW_CSV = ROOT / "data" / "Dry_Bean_Dataset.csv"
TEST_CSV = ROOT / "test_data.csv"

TARGET = "Class"
SEED = 42


# ---------------------------------------------------------------- data
def load_beans() -> pd.DataFrame:
    if RAW_CSV.exists():
        return pd.read_csv(RAW_CSV)
    from ucimlrepo import fetch_ucirepo  # only needed if the CSV is absent

    repo = fetch_ucirepo(id=602)
    frame = pd.concat([repo.data.features, repo.data.targets], axis=1)
    RAW_CSV.parent.mkdir(exist_ok=True)
    frame.to_csv(RAW_CSV, index=False)
    return frame


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    before = len(frame)
    frame = frame.dropna().drop_duplicates().reset_index(drop=True)
    print(f"Rows: {before} raw -> {len(frame)} after dropping NA/duplicates")
    return frame


# ---------------------------------------------------------------- models
def candidate_models():
    """(name, file stem, pipeline, small CV grid) for each classifier."""
    scaled = lambda est: Pipeline([("scale", StandardScaler()), ("clf", est)])
    return [
        (
            "Logistic Regression",
            "logistic_regression",
            scaled(LogisticRegression(max_iter=5000)),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
        (
            "Decision Tree",
            "decision_tree",
            Pipeline([("clf", DecisionTreeClassifier(random_state=SEED))]),
            {"clf__max_depth": [6, 8, 10, 12], "clf__min_samples_leaf": [5, 20]},
        ),
        (
            "KNN",
            "knn",
            scaled(KNeighborsClassifier()),
            {"clf__n_neighbors": [5, 11, 21], "clf__weights": ["uniform", "distance"]},
        ),
        (
            "Naive Bayes",
            "naive_bayes",
            scaled(GaussianNB()),
            {"clf__var_smoothing": [1e-9, 1e-6, 1e-3]},
        ),
        (
            "Support Vector Machine",
            "support_vector_machine",
            scaled(SVC(kernel="rbf", probability=True, random_state=SEED)),
            {"clf__C": [1.0, 10.0], "clf__gamma": ["scale", 0.05]},
        ),
    ]


def score(y_true, y_pred, y_proba, classes) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=classes),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# ---------------------------------------------------------------- main
def main():
    beans = clean(load_beans())
    features = [c for c in beans.columns if c != TARGET]
    print(f"Features: {len(features)} | Classes: {beans[TARGET].nunique()}")
    print("Class share:\n", beans[TARGET].value_counts(normalize=True).round(3).to_string())

    X, y = beans[features], beans[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    print(f"Train: {len(X_tr)} | Test: {len(X_te)}")
    X_te.assign(**{TARGET: y_te}).to_csv(TEST_CSV, index=False)

    classes = sorted(y.unique())
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, chosen = [], {}

    for name, stem, pipe, grid in candidate_models():
        t0 = time.time()
        search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=folds, n_jobs=-1)
        search.fit(X_tr, y_tr)
        best = search.best_estimator_
        pred = best.predict(X_te)
        proba = best.predict_proba(X_te)
        m = score(y_te, pred, proba, classes)
        rows.append({"ML Model Name": name, **m})
        chosen[name] = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        joblib.dump(best, MODEL_DIR / f"{stem}.joblib")
        print(
            f"{name:<24} {time.time()-t0:6.1f}s | CV F1={search.best_score_:.4f} | "
            + " ".join(f"{k}={v:.4f}" for k, v in m.items())
            + f" | {chosen[name]}"
        )

    table = pd.DataFrame(rows).round(4)
    table.to_csv(MODEL_DIR / "metrics.csv", index=False)
    (MODEL_DIR / "feature_columns.json").write_text(
        json.dumps(
            {
                "features": features,
                "target": TARGET,
                "classes": classes,
                "best_params": chosen,
                "model_files": {r[0]: f"{r[1]}.joblib" for r in candidate_models()},
            },
            indent=2,
            default=str,
        )
    )

    print("\nComparison table (paste into README):\n")
    cols = list(table.columns)
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for _, r in table.iterrows():
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")


if __name__ == "__main__":
    main()
