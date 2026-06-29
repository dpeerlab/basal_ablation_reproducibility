#!/usr/bin/env python
"""
Train CellTypist cancer state classifier (classical / basal / mesenchymal).

Strategy
--------
1. Load untreated cancer cells from cancer_final_processed.h5ad which has
   original Palantir diffusion component values (DC_1, DC_2) in .obs.
2. Compute Pearson correlations of every gene (log1p_norm layer) with DC_1
   and DC_2 to identify the most informative features.
3. Select the top 20 most positively and negatively correlated genes for
   each DC (up to 80 genes after deduplication) as the feature set.
4. Assign state labels from DC thresholds:
     mesenchymal  : DC_1 > 0.75
     basal        : DC_2 < -0.3  AND NOT mesenchymal
     classical    : DC_2 > 0.1   AND NOT mesenchymal
     (unlabeled cells between thresholds are excluded from training)
5. Train a CellTypist logistic regression classifier on labeled cells,
   holding out 20% for evaluation.
6. Apply the trained model to all cancer cells in the full compiled adata
   (BAKICA_processed_v2_segger_cancer_refMapped_20260218.h5ad) and save
   per-cell classification probabilities for all three states.

Run with:
    conda activate celltypist
    python scripts/train_cancer_state_classifier.py
"""

import argparse
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import celltypist
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

np.random.seed(42)
sns.set_theme(style="white", palette="muted")
sns.set_context("paper", font_scale=1.5)
plt.rcParams["figure.dpi"] = 150

TODAY = date.today().strftime("%Y%m%d")

# Input files
TRAIN_H5AD = Path(
     "/cancer_final_processed.h5ad"
)
INFERENCE_H5AD = Path(
        # path to full anndata
)

# Output directories
MODEL_DIR = Path("data/models/celltypist")
RESULTS_DIR = Path("/results/celltypist_cancer_states")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / f"celltypist_model_cancer_states_{TODAY}.pkl"
PROBS_FILE = RESULTS_DIR / f"cancer_state_probabilities_{TODAY}.csv"
ANNOT_FILE = RESULTS_DIR / f"cancer_state_annotations_{TODAY}.csv"

# DC thresholds for state labelling
DC1_MES_THRESHOLD = 0.75  # DC_1 > threshold  → mesenchymal
DC2_BAS_THRESHOLD = -0.30  # DC_2 < threshold  → basal        (not mesenchymal)
DC2_CLS_THRESHOLD = 0.20  # DC_2 > threshold  → classical    (not mesenchymal)

# Feature selection
N_TOP_GENES = 20  # top N pos + top N neg per DC → up to 4*N genes

# Training
TEST_SIZE = 0.20
N_JOBS = 10
MAX_ITER = 100
LABEL_COL = "cancer_state"
FIGURE_FORMAT = "png"

# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument(
    "--exclude-genes",
    nargs="+",
    metavar="GENE",
    default=[],
    help=(
        "Genes to exclude from the DC-correlated feature set even if they rank "
        "in the top correlated genes. Space-separated, e.g.: "
        "--exclude-genes Krt19 Epcam Cdh1"
    ),
)
args = parser.parse_args()
EXCLUDE_GENES = set(args.exclude_genes)
if EXCLUDE_GENES:
    print(f"Genes excluded from feature selection: {sorted(EXCLUDE_GENES)}")

# ---------------------------------------------------------------------------
# 1. Load training data
# ---------------------------------------------------------------------------

print("=" * 80)
print("Loading training data...")
print("=" * 80)

adata = sc.read_h5ad(TRAIN_H5AD)
print(f"Loaded: {adata.shape[0]:,} cells × {adata.shape[1]} genes")
print(
    f"All cells are untreated " f"(condition_num unique values: {sorted(adata.obs['condition_num'].unique().tolist())})"
)

# ---------------------------------------------------------------------------
# 2. Compute gene–DC Pearson correlations (log1p_norm layer)
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("Computing gene–DC correlations (log1p_norm layer)...")
print("=" * 80)

X = adata.layers["log1p_norm"]
if sp.issparse(X):
    X = np.asarray(X.todense())  # (cells, genes)

dc1 = adata.obs["DC_1"].values.astype(np.float64)
dc2 = adata.obs["DC_2"].values.astype(np.float64)


def pearson_corr_matrix(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson correlation of each column of X with vector y."""
    X_c = X - X.mean(axis=0)
    y_c = y - y.mean()
    numerator = X_c.T @ y_c  # (n_genes,)
    denom = np.sqrt((X_c**2).sum(axis=0) * (y_c**2).sum())
    return numerator / (denom + 1e-12)


print("  Correlating genes with DC_1...")
cor_dc1 = pearson_corr_matrix(X, dc1)
print("  Correlating genes with DC_2...")
cor_dc2 = pearson_corr_matrix(X, dc2)

gene_names = adata.var_names.tolist()
cor_df = pd.DataFrame(
    {"cor_dc1": cor_dc1, "cor_dc2": cor_dc2},
    index=gene_names,
)
cor_df.index.name = "gene"
cor_df.to_csv(RESULTS_DIR / "gene_dc_correlations.csv")
print(f"  Correlations saved → {RESULTS_DIR / 'gene_dc_correlations.csv'}")

# ---------------------------------------------------------------------------
# 3. Select features: top N pos + top N neg per DC, then take union
# ---------------------------------------------------------------------------

print(f"\nSelecting top {N_TOP_GENES} pos/neg correlated genes per DC...")

top_dc1_pos = cor_df["cor_dc1"].nlargest(N_TOP_GENES).index.tolist()
top_dc1_neg = cor_df["cor_dc1"].nsmallest(N_TOP_GENES).index.tolist()
top_dc2_pos = cor_df["cor_dc2"].nlargest(N_TOP_GENES).index.tolist()
top_dc2_neg = cor_df["cor_dc2"].nsmallest(N_TOP_GENES).index.tolist()

selected_genes = sorted(set(top_dc1_pos + top_dc1_neg + top_dc2_pos + top_dc2_neg))

if EXCLUDE_GENES:
    removed = [g for g in selected_genes if g in EXCLUDE_GENES]
    not_found = EXCLUDE_GENES - set(adata.var_names)
    if not_found:
        print(f"  WARNING: these --exclude-genes were not in the dataset: {sorted(not_found)}")
    if removed:
        print(f"  Removed {len(removed)} excluded gene(s) from feature set: {removed}")
    selected_genes = [g for g in selected_genes if g not in EXCLUDE_GENES]

print(f"  {len(selected_genes)} unique genes selected " f"(up to {N_TOP_GENES * 4} before deduplication)")

# Save selected genes with their correlation values
pd.DataFrame(
    {
        "cor_dc1": cor_df.loc[selected_genes, "cor_dc1"].values,
        "cor_dc2": cor_df.loc[selected_genes, "cor_dc2"].values,
    },
    index=selected_genes,
).to_csv(RESULTS_DIR / "selected_genes.csv")

print(f"  DC1 top pos ({N_TOP_GENES}): {top_dc1_pos[:5]} ...")
print(f"  DC1 top neg ({N_TOP_GENES}): {top_dc1_neg[:5]} ...")
print(f"  DC2 top pos ({N_TOP_GENES}): {top_dc2_pos[:5]} ...")
print(f"  DC2 top neg ({N_TOP_GENES}): {top_dc2_neg[:5]} ...")

# ---------------------------------------------------------------------------
# 4. Assign state labels from DC thresholds
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("Assigning state labels...")
print("=" * 80)

is_mes = dc1 > DC1_MES_THRESHOLD
is_bas = (dc2 < DC2_BAS_THRESHOLD) & ~is_mes
is_cls = (dc2 > DC2_CLS_THRESHOLD) & ~is_mes
is_labeled = is_mes | is_bas | is_cls

labels = np.full(adata.shape[0], "unlabeled", dtype=object)
labels[is_mes] = "mesenchymal"
labels[is_bas] = "basal"
labels[is_cls] = "classical"
adata.obs[LABEL_COL] = labels

label_counts = pd.Series(labels).value_counts()
print("Label distribution (all cells):")
for state, count in label_counts.items():
    pct = count / adata.shape[0] * 100
    print(f"  {state:15s}: {count:>8,}  ({pct:.1f}%)")

adata_labeled = adata[is_labeled].copy()
print(f"\nLabeled cells retained for training: {adata_labeled.shape[0]:,}")

# ---------------------------------------------------------------------------
# 5. Prepare training anndata for CellTypist
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("Preparing training anndata...")
print("=" * 80)

# Subset to selected genes
adata_labeled = adata_labeled[:, selected_genes].copy()

# CellTypist expects normalize_total(1e4) + log1p — use raw counts and normalize
adata_labeled.X = adata_labeled.layers["counts_segger"].copy()
sc.pp.normalize_total(adata_labeled, target_sum=1e4)
sc.pp.log1p(adata_labeled)
print(f"  Normalized X: {adata_labeled.shape[0]:,} cells × {adata_labeled.shape[1]} genes")

# Stratified 80/20 train/test split
cell_indices = np.arange(adata_labeled.shape[0])
train_idx, test_idx = train_test_split(
    cell_indices,
    test_size=TEST_SIZE,
    stratify=adata_labeled.obs[LABEL_COL],
    random_state=42,
)
adata_train = adata_labeled[train_idx].copy()
adata_test = adata_labeled[test_idx].copy()

print(f"  Training set : {len(train_idx):,} cells")
print(f"  Test set     : {len(test_idx):,} cells")
print(f"\n  Training label distribution:")
print(adata_train.obs[LABEL_COL].value_counts().to_string())

# ---------------------------------------------------------------------------
# 6. Train CellTypist model
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("Training CellTypist model...")
print("=" * 80)

new_model = celltypist.train(
    adata_train,
    labels=LABEL_COL,
    genes=selected_genes,
    n_jobs=N_JOBS,
    max_iter=MAX_ITER,
    feature_selection=False,  # feature selection already done above
    check_expression=True,
)

print("\nModel training complete!")
print(f"  Features : {new_model.features.shape[0]}")
print(f"  Classes  : {list(new_model.classifier.classes_)}")

new_model.write(str(MODEL_FILE))
print(f"  Model saved → {MODEL_FILE}")

# ---------------------------------------------------------------------------
# 7. Evaluate on held-out test set
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("Evaluating on held-out test set...")
print("=" * 80)

predictions = celltypist.annotate(adata_test, model=new_model, majority_voting=False)
adata_test_pred = predictions.to_adata()

y_true = adata_test_pred.obs[LABEL_COL]
y_pred = adata_test_pred.obs["predicted_labels"]
conf_scores = adata_test_pred.obs["conf_score"]

accuracy = accuracy_score(y_true, y_pred)
print(f"Overall accuracy: {accuracy:.4f}  ({accuracy * 100:.2f}%)")

class_report_df = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).transpose()
print("\nPer-class metrics:")
print(class_report_df.to_string())
class_report_df.to_csv(RESULTS_DIR / "classification_report.csv")

# Confusion matrices
classes = list(new_model.classifier.classes_)
cm = confusion_matrix(y_true, y_pred, labels=classes)
cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, data, fmt, title in zip(
    axes,
    [cm, cm_norm],
    ["d", ".2f"],
    ["Confusion Matrix (raw counts)", "Confusion Matrix (row-normalized)"],
):
    sns.heatmap(
        data,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
        vmin=(0 if fmt == ".2f" else None),
        vmax=(1 if fmt == ".2f" else None),
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
plt.tight_layout()
plt.savefig(RESULTS_DIR / f"confusion_matrices.{FIGURE_FORMAT}", dpi=300, bbox_inches="tight")
plt.close()

pd.DataFrame(cm, index=classes, columns=classes).to_csv(RESULTS_DIR / "confusion_matrix.csv")
pd.DataFrame(cm_norm, index=classes, columns=classes).to_csv(RESULTS_DIR / "confusion_matrix_normalized.csv")

# Confidence score distribution by true state
fig, ax = plt.subplots(figsize=(8, 4))
for state in classes:
    mask = y_true == state
    ax.hist(conf_scores[mask], bins=40, alpha=0.6, label=state, density=True)
ax.set_xlabel("Confidence score")
ax.set_ylabel("Density")
ax.set_title("Confidence score distribution by true state")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(RESULTS_DIR / f"confidence_distribution.{FIGURE_FORMAT}", dpi=300, bbox_inches="tight")
plt.close()

print(f"\nMean confidence : {conf_scores.mean():.4f}")
print(f"Median confidence: {conf_scores.median():.4f}")

# ---------------------------------------------------------------------------
# 8. Apply model to all cancer cells in inference h5ad
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("Applying model to all cancer cells in inference dataset...")
print("=" * 80)

adata_inf = sc.read_h5ad(INFERENCE_H5AD)
print(f"Loaded: {adata_inf.shape[0]:,} cells × {adata_inf.shape[1]} genes")

# Verify selected genes are present
missing = [g for g in selected_genes if g not in adata_inf.var_names]
if missing:
    print(f"WARNING: {len(missing)} selected genes not in inference data: {missing}")
selected_genes_inf = [g for g in selected_genes if g in adata_inf.var_names]
print(f"  Using {len(selected_genes_inf)} genes for inference")

# Subset and normalize (same pipeline as training)
adata_inf_sub = adata_inf[:, selected_genes_inf].copy()
adata_inf_sub.X = adata_inf_sub.layers["counts_segger"].copy()
sc.pp.normalize_total(adata_inf_sub, target_sum=1e4)
sc.pp.log1p(adata_inf_sub)

# Annotate — majority_voting=False to get per-cell probabilities
inf_preds = celltypist.annotate(adata_inf_sub, model=new_model, majority_voting=False)
adata_inf_pred = inf_preds.to_adata()

print("\nPrediction summary (all cancer cells):")
print(adata_inf_pred.obs["predicted_labels"].value_counts().to_string())

# ---------------------------------------------------------------------------
# 9. Save outputs
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("Saving inference outputs...")
print("=" * 80)

# Probability matrix: rows=cells, columns=states (float, 0–1, rows sum to 1)
prob_matrix = inf_preds.probability_matrix.copy()
prob_matrix.index = adata_inf.obs_names
prob_matrix.to_csv(PROBS_FILE)
print(f"  Per-cell probabilities → {PROBS_FILE}")

# Full annotation table: predicted label + conf score + all probabilities
annot_df = pd.DataFrame(
    {
        "predicted_state": adata_inf_pred.obs["predicted_labels"].values,
        "conf_score": adata_inf_pred.obs["conf_score"].values,
    },
    index=adata_inf.obs_names,
)
annot_df = pd.concat([annot_df, prob_matrix], axis=1)
annot_df.to_csv(ANNOT_FILE)
print(f"  Full annotation table → {ANNOT_FILE}")

# Training summary
summary = {
    "n_training_cells_total": adata_labeled.shape[0],
    "n_train_set": len(train_idx),
    "n_test_set": len(test_idx),
    "n_features": len(selected_genes),
    "test_accuracy": accuracy,
    "mean_conf_score": float(conf_scores.mean()),
    "median_conf_score": float(conf_scores.median()),
    "n_inference_cells": adata_inf.shape[0],
}
pd.Series(summary).to_csv(RESULTS_DIR / "training_summary.csv", header=["value"])

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
for k, v in summary.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v:,}")
print(f"\n  Model     : {MODEL_FILE}")
print(f"  Results   : {RESULTS_DIR}")
