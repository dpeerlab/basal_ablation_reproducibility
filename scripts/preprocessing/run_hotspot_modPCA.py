#!/usr/bin/env python
"""
Hotspot Analysis Script

This script performs Hotspot analysis on clustered cell type data to identify
spatially variable genes and gene modules.

For modPCA / symNMF I only need the local correlation matrix, so I won't do any of the other steps.

Usage:
    python run_hotspot.py --input <path_to_clustered_h5ad> --output_dir <output_directory> [options]
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import warnings
import seaborn as sns
import hotspot
import matplotlib.pyplot as plt


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Perform Hotspot analysis on clustered cell type data")

    # Required arguments
    parser.add_argument("--input", type=str, required=True, help="Path to input clustered h5ad file")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output files (h5ad with modules, CSV with module assignments, plots)",
    )

    # Hotspot parameters
    parser.add_argument(
        "--layer_key",
        type=str,
        default="counts_nuc",
        help="Layer to use for Hotspot analysis (default: counts_nuc)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="danb",
        choices=["danb", "bernoulli", "normal", "none"],
        help="Hotspot model type (default: danb)",
    )
    parser.add_argument(
        "--latent_obsm_key",
        type=str,
        default="X_spatial",
        help="Key in obsm for spatial coordinates (default: X_spatial)",
    )
    parser.add_argument(
        "--umi_counts_obs_key",
        type=str,
        default="total_counts",
        help="Key in obs for UMI counts (default: total_counts)",
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=30,
        help="Number of neighbors for KNN graph (default: 30)",
    )
    parser.add_argument(
        "--weighted_graph",
        action="store_true",
        default=False,
        help="Use weighted KNN graph (default: False)",
    )

    # Gene filtering parameters
    parser.add_argument(
        "--top_n_genes",
        type=int,
        default=500,
        help="Number of top genes to use for local correlations (default: 500)",
    )
    parser.add_argument(
        "--fdr_threshold",
        type=float,
        default=0.05,
        help="FDR threshold for autocorrelation filtering (default: 0.05)",
    )
    parser.add_argument(
        "--gene_exclude_pattern",
        type=str,
        default="^mt-|^Mtmr|^Mtnd|Neat1|Tmsb4x|Tmsb10|^Rps|^Rpl|^Mrp|^Fau$|Uba52|Malat|Trav|Traj|Trbv|Trbd|Trbj|Hba[12]|^Ig[hkl]|Rp[1-9]+\-|^Mtrnr|pSL21\-VEX",
        help="Regex pattern for genes to exclude (default: mitochondrial, ribosomal, etc.)",
    )

    # Module creation parameters
    parser.add_argument(
        "--min_gene_threshold",
        type=int,
        default=10,
        help="Minimum number of genes per module (default: 10)",
    )
    parser.add_argument(
        "--module_fdr_threshold",
        type=float,
        default=0.05,
        help="FDR threshold for module creation (default: 0.05)",
    )
    parser.add_argument(
        "--core_only",
        action="store_true",
        default=False,  # changed to false for modPCA
        help="Use only core genes for modules (default: False for modPCA)",
    )

    # Parallelization
    parser.add_argument(
        "--autocorr_jobs",
        type=int,
        default=10,
        help="Number of parallel jobs for autocorrelation computation (default: 10)",
    )
    parser.add_argument(
        "--local_corr_jobs",
        type=int,
        default=36,
        help="Number of parallel jobs for local correlation computation (default: 36)",
    )

    # Output options
    parser.add_argument(
        "--cell_type_name",
        type=str,
        default=None,
        help="Cell type name for output files (if not provided, will be inferred from input filename)",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        default=True,
        help="Save local correlation plots (default: True)",
    )

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    # Convert to Path objects
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine cell type name from input filename if not provided
    if args.cell_type_name:
        cell_type_name = args.cell_type_name
    else:
        # Extract from filename (assumes format like "Cancer_epithelial_clustered.h5ad")
        cell_type_name = input_path.stem.replace("_clustered", "")

    print("=" * 80)
    print("Hotspot Analysis")
    print("=" * 80)
    print(f"Input file: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Cell type name: {cell_type_name}")
    print()

    # Load data
    print("Loading data...")
    adata = sc.read_h5ad(input_path)
    print(f"  Loaded {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    print()

    # filter out genes with less than 20 counts in the layer_key
    adata = adata[:, adata.layers[args.layer_key].sum(axis=0) >= 20]
    print(f"  Filtered to {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    print()

    # if umi_counts_obs_key is not present, compute it from summing counts in the layer_key
    if args.umi_counts_obs_key not in adata.obs.columns:
        print(f"Computing {args.umi_counts_obs_key} from {args.layer_key}...")
        adata.obs[args.umi_counts_obs_key] = adata.layers[args.layer_key].sum(axis=1)
        print(f"  Computed {args.umi_counts_obs_key} for {adata.n_obs:,} cells")
        print()

    # Initialize Hotspot
    print("Initializing Hotspot...")
    print(f"  Layer: {args.layer_key}")
    print(f"  Model: {args.model}")
    print(f"  Spatial key: {args.latent_obsm_key}")
    print(f"  UMI counts key: {args.umi_counts_obs_key}")
    print()

    hs = hotspot.Hotspot(
        adata,
        layer_key=args.layer_key,
        model=args.model,
        # latent_obsm_key=args.latent_obsm_key,
        umi_counts_obs_key=args.umi_counts_obs_key,
        distances_obsp_key="distances",  # use precomputed distances
    )

    # Create KNN graph
    print(f"Creating KNN graph (n_neighbors={args.n_neighbors}, weighted={args.weighted_graph})...")
    hs.create_knn_graph(weighted_graph=args.weighted_graph, n_neighbors=args.n_neighbors)
    print()

    # Compute autocorrelations
    print(f"Computing autocorrelations (jobs={args.autocorr_jobs})...")
    hs_results = hs.compute_autocorrelations(jobs=args.autocorr_jobs)
    print(f"  Computed autocorrelations for {len(hs_results):,} genes")
    print()

    # Filter genes
    print("Filtering genes...")
    print(f"  Exclude pattern: {args.gene_exclude_pattern}")
    print(f"  FDR threshold: {args.fdr_threshold}")
    print(f"  Top N genes: {args.top_n_genes}")

    gene_filter = ~(hs_results.index.str.contains(args.gene_exclude_pattern)) & (hs_results.FDR < args.fdr_threshold)
    hs_genes = hs_results[gene_filter].sort_values("Z", ascending=False).head(args.top_n_genes).index
    print(f"  Selected {len(hs_genes):,} genes for local correlation analysis")
    print()

    # Compute local correlations
    print(f"Computing local correlations (jobs={args.local_corr_jobs})...")
    local_correlations = hs.compute_local_correlations(hs_genes, jobs=args.local_corr_jobs)
    print(f"  Computed local correlations for {len(hs_genes):,} genes")
    print()

    # save local correlation results
    local_corr_path = output_dir / f"{cell_type_name}_hotspot_local_correlations.csv"
    print(f"  Local correlation results: {local_corr_path}")
    local_correlations.to_csv(local_corr_path)
    print()

    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
