#!/usr/bin/env python
"""
Cluster cells by lineage using RAPIDS-accelerated clustering.

This script supports two modes:

Lineage mode (--input + --lineages):
- Takes in global anndata
- Filters to select predicted_labels for a lineage
- Removes cells that have less than 10 segger counts
- Sets GFP, tagBFP2, CreER to not be highly_variable
- Runs cluster_rapids and phenograph_rapids
- Maps previous cell annotations
- Creates quality metrics and saves outputs

Custom mode (--anndata + --name):
- Takes in any anndata file
- Optionally applies a filter expression to subset cells
- Runs the same clustering and annotation pipeline

Usage:
    # Lineage mode
    python cluster_lineage.py --lineages "epithelial,myeloid" --input /path/to/adata.h5ad

    # Custom mode with filter
    python cluster_lineage.py --anndata /path/to/adata.h5ad --name "my_subset" \\
        --filter_expr "predicted_labels == 'Macrophage' and total_counts_segger > 50"

    # Custom mode without filter (use entire anndata)
    python cluster_lineage.py --anndata /path/to/adata.h5ad --name "all_cells"
"""

import os
import sys


import argparse
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add utils path
sys.path.append("/src")

from xenium_utils.pp.preprocess_rapids import cluster_rapids, phenograph_rapids, nearest_neighbors
from xenium_utils.pl.utils import crop_umap


# Define lineage mappings
LINEAGE_CELL_TYPES = {
    "epithelial": [
        "Cancer_epithelial",
        "Acinar",
        "Ductal",
        "Mesothelial",
        "Epithelial_neuronal-like",
        "Tuft-like",
        "Islet",
        "Glial",
    ],
    "myeloid": ["Macrophage", "Neutrophil", "mregDC", "cDC2", "cDC1", "Monocyte", "pDC", "Granulocyte"],
    "lymphoid": [
        "B_cell",
        "Tconv",
        "CD8_T",
        "Treg",
        "Plasmablast",
        "Lymphoid_proliferating",
        "NK",
        "gdT",
        "ILC2",
        "ILC3",
        "Innate-like_T",
        "MAIT",
        "gdT_CD8",
        "ILC1",
    ],
    "stromal": [
        "Fibroblast",
        "Endothelial_Vascular",
        "PSC",
        "Endothelial_Lymphatic",
        "Adipocyte",
        "Cancer_mesenchymal",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster cells by lineage")
    parser.add_argument(
        "--lineages",
        type=str,
        default=None,
        help='Comma-separated list of lineages to process (e.g., "epithelial,myeloid")',
    )
    parser.add_argument("--input", type=str, default=None, help="Path to input global anndata h5ad file")
    parser.add_argument(
        "--anndata",
        type=str,
        default=None,
        help="Path to an anndata h5ad file to use directly (alternative to --input + --lineages)",
    )
    parser.add_argument(
        "--filter_expr",
        type=str,
        default=None,
        help='Pandas query expression to subset cells (e.g., "predicted_labels == \'Macrophage\' and total_counts_segger > 50")',
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name for outputs when using --anndata mode (used in filenames and cluster keys)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data1/peerd/roses3/basal_ablation/data/raw/xenium/h5ads/lineage",
        help="Base output directory for lineage h5ads",
    )
    parser.add_argument(
        "--previous_checkpoint",
        type=str,
        default="/data1/peerd/roses3/basal_ablation/data/raw/xenium/h5ads/BAKI_processed_all_filtered_v2_noContam_20250829.h5ad",
        help="Path to previous checkpoint anndata for cell annotation mapping",
    )
    parser.add_argument("--min_counts", type=int, default=10, help="Minimum segger counts to keep a cell")
    parser.add_argument("--min_size", type=int, default=150, help="Minimum cluster size for phenograph")
    parser.add_argument("--phenograph_resolution", type=float, default=1.0, help="Resolution for phenograph clustering")
    parser.add_argument("--overcluster_resolution", type=float, default=10.0, help="Resolution for overclustering")
    parser.add_argument("--downsample_n", type=int, default=500000, help="Number of cells to downsample for plotting")
    parser.add_argument("--plot_dpi", type=int, default=200, help="DPI for saved plots")
    return parser.parse_args()


def get_majority_vote(adata, cluster_key, label_key, new_key):
    """
    Assign majority vote label from label_key to each cluster in cluster_key.
    Only considers non-null labels.
    """
    # Get mapping of cluster to most frequent non-null label
    cluster_to_label = {}
    for cluster in adata.obs[cluster_key].cat.categories:
        cluster_mask = adata.obs[cluster_key] == cluster
        labels = adata.obs.loc[cluster_mask, label_key].dropna()
        if len(labels) > 0:
            most_frequent = labels.value_counts().idxmax()
            cluster_to_label[cluster] = most_frequent
        else:
            cluster_to_label[cluster] = None

    # Map to new column
    adata.obs[new_key] = adata.obs[cluster_key].map(cluster_to_label)


def get_putative_low_quality(adata, cluster_key, count_col="total_counts_segger", threshold=25):
    """
    Flag clusters where majority of cells have counts below threshold.
    """
    low_quality_clusters = set()
    for cluster in adata.obs[cluster_key].cat.categories:
        cluster_mask = adata.obs[cluster_key] == cluster
        cluster_counts = adata.obs.loc[cluster_mask, count_col]
        frac_below = (cluster_counts < threshold).mean()
        if frac_below > 0.5:
            low_quality_clusters.add(cluster)

    adata.obs["putative_low_quality"] = adata.obs[cluster_key].isin(low_quality_clusters)


def make_umap_plots(adata, lineage, output_dir, dpi=200, downsample_n=500000):
    """
    Create UMAP plots for various colorings, downsampled to downsample_n cells.
    """
    # Downsample for plotting
    if adata.n_obs > downsample_n:
        np.random.seed(42)
        idx = np.random.choice(adata.n_obs, downsample_n, replace=False)
        adata_ds = adata[idx].copy()
    else:
        adata_ds = adata.copy()

    # Define plot colorings
    plot_configs = [
        (f"{lineage}_phenograph", f"{lineage}_phenograph", {}),
        ("predicted_labels", "predicted_labels", {}),
        ("cell_annotation_majority_vote", "cell_annotation_majority_vote", {}),
        ("total_counts_segger", "total_counts_segger", {"vmax": 200}),
        ("condition", "condition", {}),
        ("condition_num", "condition_num", {}),
        ("slide", "slide", {}),
        ("run", "run", {}),
        ("sample_id", "sample_id", {}),
        ("putative_low_quality", "putative_low_quality", {}),
    ]

    for filename, color_by, kwargs in plot_configs:
        if color_by not in adata_ds.obs.columns:
            print(f"  Warning: {color_by} not found in obs, skipping plot")
            continue

        fig, ax = plt.subplots(figsize=(10, 8))
        sc.pl.umap(adata_ds, color=color_by, ax=ax, show=False, **kwargs)

        plot_path = output_dir / f"{lineage}_{filename}.png"
        fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {plot_path}")


def process_lineage(lineage, adata_global, args):
    """
    Process a single lineage.
    """
    print(f"\n{'='*80}")
    print(f"Processing lineage: {lineage}")
    print(f"{'='*80}")

    # Get cell types for this lineage
    if lineage not in LINEAGE_CELL_TYPES:
        print(f"  Error: Unknown lineage '{lineage}'. Available: {list(LINEAGE_CELL_TYPES.keys())}")
        return

    cell_types = LINEAGE_CELL_TYPES[lineage]
    print(f"  Cell types: {cell_types}")

    # Filter to lineage
    print(f"\n  Filtering to lineage cells...")
    lineage_mask = adata_global.obs["predicted_labels"].isin(cell_types)
    adata = adata_global[lineage_mask].copy()
    print(f"  Cells after lineage filter: {adata.n_obs:,}")

    # Filter by minimum counts
    print(f"  Filtering cells with < {args.min_counts} segger counts...")
    count_mask = adata.obs["total_counts_segger"] >= args.min_counts
    adata = adata[count_mask].copy()
    print(f"  Cells after count filter: {adata.n_obs:,}")

    if adata.n_obs == 0:
        print(f"  Error: No cells remaining after filtering!")
        return

    # Set reporter genes to not be highly variable
    print(f"\n  Setting reporter genes (GFP, tagBFP2, CreER) as not highly_variable...")
    reporter_genes = ["GFP", "tagBFP2", "CreER"]
    for gene in reporter_genes:
        if gene in adata.var_names:
            adata.var.loc[gene, "highly_variable"] = False
            print(f"    Set {gene} highly_variable=False")

    # Run cluster_rapids
    print(f"\n  Running cluster_rapids...")
    phenograph_key = f"{lineage}_phenograph"
    cluster_rapids(
        adata,
        pca_layer="log1p_norm",
        pca_total_var=0.70,
        knn_neighbors=50,
        umap_min_dist=0.1,
        umap_n_epochs=5000,
        phenograph_resolution=args.phenograph_resolution,
        umap_kwargs={"init": "random"},
        use_highly_variable=True,
        min_size=args.min_size,
        phenograph_key=phenograph_key,
    )

    # Run phenograph_rapids for overclustering
    print(f"\n  Running phenograph_rapids for overclustering...")
    overcluster_key = f"{lineage}_phenograph_overcluster"
    phenograph_rapids(
        adata,
        neighbors_key="neighbors",
        key_added=overcluster_key,
        min_size=-1,
        resolution=args.overcluster_resolution,
    )

    # Read previous checkpoint and map cell_annotation
    print(f"\n  Reading previous checkpoint for cell_annotation mapping...")
    if Path(args.previous_checkpoint).exists():
        adata_prev = sc.read_h5ad(args.previous_checkpoint)

        # Create mapping from barcode/cell_id to cell_annotation
        if "cell_annotation" in adata_prev.obs.columns:
            # Use index as the cell identifier
            prev_annotations = adata_prev.obs["cell_annotation"].to_dict()

            # Map to current adata
            adata.obs["cell_annotation_previous"] = adata.obs.index.map(prev_annotations)
            n_mapped = adata.obs["cell_annotation_previous"].notna().sum()
            print(f"    Mapped {n_mapped:,} / {adata.n_obs:,} cells to previous annotations")

            # Clean up
            del adata_prev
        else:
            print(f"    Warning: 'cell_annotation' not found in previous checkpoint")
            adata.obs["cell_annotation_previous"] = None
    else:
        print(f"    Warning: Previous checkpoint not found at {args.previous_checkpoint}")
        adata.obs["cell_annotation_previous"] = None

    # Assign cell_annotation_majority_vote based on overclusters
    print(f"\n  Computing majority vote annotation from overclusters...")
    get_majority_vote(
        adata,
        cluster_key=overcluster_key,
        label_key="cell_annotation_previous",
        new_key="cell_annotation_majority_vote",
    )

    # Create putative_low_quality flag
    print(f"\n  Flagging putative low quality clusters...")
    get_putative_low_quality(adata, cluster_key=overcluster_key, count_col="total_counts_segger", threshold=25)
    n_low_quality = adata.obs["putative_low_quality"].sum()
    print(f"    {n_low_quality:,} cells flagged as putative low quality")

    # Copy UMAP to lineage-specific key
    print(f"\n  Copying UMAP to X_umap_{lineage}_segger...")
    adata.obsm[f"X_umap_{lineage}_segger"] = adata.obsm["X_umap"].copy()

    # Create output directory
    output_dir = Path(args.output_dir) / lineage
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Output directory: {output_dir}")

    # Save anndata
    output_h5ad = output_dir / f"{lineage}_processed.h5ad"
    print(f"\n  Saving anndata to {output_h5ad}...")
    adata.write_h5ad(output_h5ad)

    # Create and save UMAP plots
    print(f"\n  Creating UMAP plots (downsampled to {args.downsample_n:,} cells)...")
    make_umap_plots(adata, lineage=lineage, output_dir=output_dir, dpi=args.plot_dpi, downsample_n=args.downsample_n)

    print(f"\n  Done processing {lineage}!")
    print(f"  Output h5ad: {output_h5ad}")
    print(f"  Output plots: {output_dir}/*.png")

    # Clean up
    del adata


def process_custom(adata, name, args):
    """
    Process a custom anndata (optionally filtered) through the clustering pipeline.
    """
    print(f"\n{'='*80}")
    print(f"Processing custom anndata: {name}")
    print(f"{'='*80}")

    # Apply filter expression if provided
    if args.filter_expr:
        print(f"\n  Applying filter: {args.filter_expr}")
        n_before = adata.n_obs
        try:
            passing_idx = adata.obs.query(args.filter_expr).index
            adata = adata[passing_idx].copy()
        except Exception as e:
            print(f"  Error applying filter expression: {e}")
            return
        print(f"  Cells: {n_before:,} -> {adata.n_obs:,}")

    # Filter by minimum counts (if the column exists)
    if "total_counts_segger" in adata.obs.columns:
        print(f"  Filtering cells with < {args.min_counts} segger counts...")
        count_mask = adata.obs["total_counts_segger"] >= args.min_counts
        adata = adata[count_mask].copy()
        print(f"  Cells after count filter: {adata.n_obs:,}")

    if adata.n_obs == 0:
        print(f"  Error: No cells remaining after filtering!")
        return

    # Set reporter genes to not be highly variable
    if "highly_variable" in adata.var.columns:
        print(f"\n  Setting reporter genes (GFP, tagBFP2, CreER) as not highly_variable...")
        reporter_genes = ["GFP", "tagBFP2", "CreER"]
        for gene in reporter_genes:
            if gene in adata.var_names:
                adata.var.loc[gene, "highly_variable"] = False
                print(f"    Set {gene} highly_variable=False")

    # Run cluster_rapids
    print(f"\n  Running cluster_rapids...")
    phenograph_key = f"{name}_phenograph"
    cluster_rapids(
        adata,
        pca_layer="log1p_norm",
        pca_total_var=0.70,
        knn_neighbors=50,
        umap_min_dist=0.1,
        umap_n_epochs=5000,
        phenograph_resolution=args.phenograph_resolution,
        umap_kwargs={"init": "random"},
        use_highly_variable=True,
        min_size=args.min_size,
        phenograph_key=phenograph_key,
    )

    # Run phenograph_rapids for overclustering
    print(f"\n  Running phenograph_rapids for overclustering...")
    overcluster_key = f"{name}_phenograph_overcluster"
    phenograph_rapids(
        adata,
        neighbors_key="neighbors",
        key_added=overcluster_key,
        min_size=-1,
        resolution=args.overcluster_resolution,
    )

    # Read previous checkpoint and map cell_annotation
    print(f"\n  Reading previous checkpoint for cell_annotation mapping...")
    if Path(args.previous_checkpoint).exists():
        adata_prev = sc.read_h5ad(args.previous_checkpoint)

        if "cell_annotation" in adata_prev.obs.columns:
            prev_annotations = adata_prev.obs["cell_annotation"].to_dict()
            adata.obs["cell_annotation_previous"] = adata.obs.index.map(prev_annotations)
            n_mapped = adata.obs["cell_annotation_previous"].notna().sum()
            print(f"    Mapped {n_mapped:,} / {adata.n_obs:,} cells to previous annotations")
            del adata_prev
        else:
            print(f"    Warning: 'cell_annotation' not found in previous checkpoint")
            adata.obs["cell_annotation_previous"] = None
    else:
        print(f"    Warning: Previous checkpoint not found at {args.previous_checkpoint}")
        adata.obs["cell_annotation_previous"] = None

    # Assign cell_annotation_majority_vote based on overclusters
    print(f"\n  Computing majority vote annotation from overclusters...")
    get_majority_vote(
        adata,
        cluster_key=overcluster_key,
        label_key="cell_annotation_previous",
        new_key="cell_annotation_majority_vote",
    )

    # Create putative_low_quality flag
    if "total_counts_segger" in adata.obs.columns:
        print(f"\n  Flagging putative low quality clusters...")
        get_putative_low_quality(adata, cluster_key=overcluster_key, count_col="total_counts_segger", threshold=25)
        n_low_quality = adata.obs["putative_low_quality"].sum()
        print(f"    {n_low_quality:,} cells flagged as putative low quality")

    # Copy UMAP to name-specific key
    print(f"\n  Copying UMAP to X_umap_{name}_segger...")
    adata.obsm[f"X_umap_{name}_segger"] = adata.obsm["X_umap"].copy()

    # Create output directory
    output_dir = Path(args.output_dir) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Output directory: {output_dir}")

    # Save anndata
    output_h5ad = output_dir / f"{name}_processed.h5ad"
    print(f"\n  Saving anndata to {output_h5ad}...")
    adata.write_h5ad(output_h5ad)

    # Create and save UMAP plots
    print(f"\n  Creating UMAP plots (downsampled to {args.downsample_n:,} cells)...")
    make_umap_plots(adata, lineage=name, output_dir=output_dir, dpi=args.plot_dpi, downsample_n=args.downsample_n)

    print(f"\n  Done processing {name}!")
    print(f"  Output h5ad: {output_h5ad}")
    print(f"  Output plots: {output_dir}/*.png")

    del adata


def main():
    args = parse_args()

    # Validate argument combinations
    if args.anndata and args.lineages:
        print("Error: --anndata and --lineages are mutually exclusive. Use one or the other.")
        sys.exit(1)

    if args.anndata and args.input:
        print("Error: --anndata and --input are mutually exclusive. Use one or the other.")
        sys.exit(1)

    if args.filter_expr and not args.anndata:
        print("Error: --filter_expr requires --anndata.")
        sys.exit(1)

    if args.anndata and not args.name:
        print("Error: --name is required when using --anndata.")
        sys.exit(1)

    if not args.anndata and not args.input:
        print("Error: either --input (with --lineages) or --anndata (with --name) is required.")
        sys.exit(1)

    if args.input and not args.lineages:
        print("Error: --lineages is required when using --input.")
        sys.exit(1)

    if args.anndata:
        # Custom mode
        print(f"\nLoading anndata from {args.anndata}...")
        adata = sc.read_h5ad(args.anndata)
        print(f"  Loaded {adata.n_obs:,} cells x {adata.n_vars:,} genes")

        process_custom(adata, args.name, args)

        del adata
    else:
        # Lineage mode
        lineages = [l.strip() for l in args.lineages.split(",")]
        print(f"Lineages to process: {lineages}")

        print(f"\nLoading global anndata from {args.input}...")
        adata_global = sc.read_h5ad(args.input)
        print(f"  Loaded {adata_global.n_obs:,} cells x {adata_global.n_vars:,} genes")

        for lineage in lineages:
            process_lineage(lineage, adata_global, args)

    print(f"\n{'='*80}")
    print("All done!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
