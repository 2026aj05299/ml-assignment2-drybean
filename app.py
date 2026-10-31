"""
Dry Bean Variety Classifier - Streamlit front end.

Loads the five pre-trained pipelines from model/ and evaluates them on a
test CSV uploaded by the user. Nothing is retrained here.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

HERE = Path(__file__).parent
MODEL_DIR = HERE / "model"
SAMPLE_CSV = HERE / "test_data.csv"
POD_GREEN = "#3E6B48"
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

st.set_page_config(page_title="Dry Bean Variety Classifier", page_icon="🫘", layout="wide")


# ---------------------------------------------------------------- loading
@st.cache_resource
def load_bundle():
    meta = json.loads((MODEL_DIR / "feature_columns.json").read_text())
    pipes = {name: joblib.load(MODEL_DIR / fname) for name, fname in meta["model_files"].items()}
    return meta, pipes


def evaluate(pipe, X, y, classes):
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)
    try:
        auc = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=classes)
    except ValueError:  # a variety is missing from the uploaded file
        auc = np.nan
    scores = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": auc,
        "Precision": precision_score(y, pred, average="macro", zero_division=0),
        "Recall": recall_score(y, pred, average="macro", zero_division=0),
        "F1": f1_score(y, pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
    return scores, pred, proba


def check_upload(frame, features, target):
    missing = [c for c in features + [target] if c not in frame.columns]
    if missing:
        return f"The file is missing these columns: {', '.join(missing)}."
    if frame[features].isna().any().any():
        return "Some feature cells are empty. Remove or fill those rows and upload again."
    return None


meta, pipes = load_bundle()
FEATURES, TARGET, CLASSES = meta["features"], meta["target"], meta["classes"]

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Test data")
    upload = st.file_uploader("Upload a CSV with the 16 features and the Class column", type="csv")
    st.download_button(
        "Download sample test_data.csv",
        data=SAMPLE_CSV.read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
        width="stretch",
    )
    st.header("Classifier")
    model_name = st.selectbox("Model to evaluate", list(pipes.keys()))
    params = meta["best_params"].get(model_name, {})
    st.caption("Tuned settings: " + ", ".join(f"{k}={v}" for k, v in params.items()))
    st.divider()
    st.caption(
        "Models were trained once on an 80% stratified split of the UCI Dry Bean "
        "dataset. This app only scores them on the file you upload."
    )

# ---------------------------------------------------------------- header
st.title("🫘 Dry bean variety classifier")
st.write(
    "Seven varieties of dry bean, told apart from 16 shape measurements taken from "
    "kernel images. Upload a labelled test file to see how each model performs."
)

if upload is None:
    st.info(
        "Upload a test CSV in the sidebar to start. No file handy? Download the sample "
        "test set from the sidebar and upload it back."
    )
    st.stop()

data = pd.read_csv(upload)
problem = check_upload(data, FEATURES, TARGET)
if problem:
    st.error(problem)
    st.stop()

unknown = sorted(set(data[TARGET]) - set(CLASSES))
if unknown:
    st.error(f"Unknown variety labels in {TARGET}: {', '.join(map(str, unknown))}. Expected {', '.join(CLASSES)}.")
    st.stop()

X, y = data[FEATURES], data[TARGET]
scores, pred, proba = evaluate(pipes[model_name], X, y, CLASSES)
st.caption(f"Scored {len(data):,} beans across {y.nunique()} varieties.")

tab_model, tab_variety, tab_compare, tab_rows = st.tabs(
    ["Selected model", "By variety", "Compare all models", "Predictions"]
)

# ---------------------------------------------------------------- tab: selected model
with tab_model:
    st.subheader(model_name)
    cols = st.columns(6)
    for col, name in zip(cols, METRIC_NAMES):
        val = scores[name]
        col.metric(name, "n/a" if np.isnan(val) else f"{val:.4f}")
    st.caption("Precision, Recall and F1 are macro-averaged over the 7 varieties; AUC is one-vs-rest, macro.")

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("**Confusion matrix**")
        as_share = st.toggle("Show as share of each true variety", value=False)
        cm = confusion_matrix(y, pred, labels=CLASSES)
        shown = cm / cm.sum(axis=1, keepdims=True).clip(min=1) if as_share else cm
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        sns.heatmap(
            shown, annot=True, fmt=".2f" if as_share else "d", cmap="Greens",
            xticklabels=CLASSES, yticklabels=CLASSES, cbar=False, ax=ax, linewidths=0.5,
        )
        ax.set_xlabel("Predicted variety")
        ax.set_ylabel("True variety")
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
        fig.tight_layout()
        st.pyplot(fig)
    with right:
        st.markdown("**Classification report**")
        report = pd.DataFrame(
            classification_report(y, pred, labels=CLASSES, output_dict=True, zero_division=0)
        ).T
        report["support"] = report["support"].astype(int)
        st.dataframe(report.round(4), width="stretch", height=420)

# ---------------------------------------------------------------- tab: by variety
with tab_variety:
    st.subheader("Which varieties are hardest?")
    per_class = pd.DataFrame(
        {
            "F1": f1_score(y, pred, labels=CLASSES, average=None, zero_division=0),
            "Recall": recall_score(y, pred, labels=CLASSES, average=None, zero_division=0),
            "Beans in file": [int((y == c).sum()) for c in CLASSES],
        },
        index=CLASSES,
    ).sort_values("F1")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(per_class.index, per_class["F1"], color=POD_GREEN)
    ax.set_xlim(max(0.0, per_class["F1"].min() - 0.1), 1.0)
    ax.set_xlabel(f"F1 score ({model_name})")
    for i, v in enumerate(per_class["F1"]):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    cm = confusion_matrix(y, pred, labels=CLASSES)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    i, j = np.unravel_index(off.argmax(), off.shape)
    st.write(
        f"Most frequent mix-up: **{CLASSES[i]}** beans predicted as **{CLASSES[j]}** "
        f"({off[i, j]} times)."
    )
    st.dataframe(per_class.round(4), width="stretch")

# ---------------------------------------------------------------- tab: compare
with tab_compare:
    st.subheader("All five models on this file")
    table = pd.DataFrame(
        {name: evaluate(p, X, y, CLASSES)[0] for name, p in pipes.items()}
    ).T[METRIC_NAMES]
    table.index.name = "ML Model Name"
    st.dataframe(
        table.style.format("{:.4f}").highlight_max(axis=0, color="#CFE3C8"),
        width="stretch",
    )
    st.caption("Shaded cell = best model for that metric.")
    long = table.reset_index().melt(id_vars="ML Model Name", var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    sns.barplot(data=long, x="Metric", y="Score", hue="ML Model Name", palette="Greens_d", ax=ax)
    ax.set_ylim(max(0.0, long["Score"].min() - 0.05), 1.0)
    ax.legend(fontsize=7, loc="lower left", ncol=3)
    ax.set_xlabel("")
    fig.tight_layout()
    st.pyplot(fig)

# ---------------------------------------------------------------- tab: predictions
with tab_rows:
    st.subheader("Row-level predictions")
    out = data.copy()
    out["Predicted"] = pred
    out["Confidence"] = proba.max(axis=1).round(4)
    out["Correct"] = out["Predicted"] == out[TARGET]
    only_wrong = st.checkbox("Show misclassified beans only")
    view = out[~out["Correct"]] if only_wrong else out
    st.dataframe(view, width="stretch", height=380)
    st.download_button(
        "Download predictions as CSV",
        data=out.to_csv(index=False).encode(),
        file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )
