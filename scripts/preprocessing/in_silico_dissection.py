#!/usr/bin/env python
"""
In silico dissection of pancreatic parenchyma with lymph node removal.

Implements a two-step spatial filtering strategy:

1. Parenchyma definition: Keep all epithelial cells plus any cell within
   a 200-micron radius of an epithelial cell. Cells outside this set
   are excluded.

2. Lymph node removal: Among lymph-node-associated immune cells (B, CD4 T,
   CD8 T), build a spatial neighbor graph (30-micron radius) and identify
   connected components with >250 cells as lymph nodes. Remove those cells.

3. Lymph node expansion: Any cell within 200 microns of an identified
   lymph node cell is also marked as lymph node, to capture non-immune
   cells interspersed within or at the edges of lymph nodes.

All operations are performed per sample (sample_id) to respect coordinate
spaces.

Usage:
    conda run -n squidpy python scripts/in_silico_dissection.py
"""

import argparse
import warnings
import time
import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Config ---
ADATA_PATH = (
    "/data1/peerd/roses3/basal_ablation/data/processed/xenium/compiled_adata/"
    "BAKI_processed_v2_segger_all_20260511.h5ad"
)
OUT_DIR = Path("/data1/peerd/roses3/basal_ablation/data/processed/xenium/dissection_annotations")

# Epithelial cell types that define the pancreatic parenchyma
EPITHELIAL_TYPES = [
    "Acinar",
    "ADM",
    "Cancer_epithelial",
    "Cancer_epithelial_neuronal-like",
    "Cancer_mesenchymal",
    "Ductal",
    "Islet",
    "Tuft-like",
]

# Lymph-node-associated immune cell types (B, CD4 T, CD8 T lineages)
LN_IMMUNE_TYPES = [
    "B_cell",
    "Plasmablast",
    "CD8_T",
    "Tconv",
    "Treg",
]

PARENCHYMA_RADIUS = 200  # microns
LN_NEIGHBOR_RADIUS = 30  # microns
LN_MIN_COMPONENT_SIZE = 250  # cells
LN_EXPANSION_RADIUS = 50  # microns, expand LN annotation to nearby cells


def define_parenchyma(coords, cell_types, sample_ids, epithelial_types, radius):
    """
    For each sample, mark cells that are within `radius` microns of any
    epithelial cell (including epithelial cells themselves).

    Returns a boolean array of length n_cells.
    """
    n_cells = len(cell_types)
    in_parenchyma = np.zeros(n_cells, dtype=bool)
    is_epithelial = cell_types.isin(epithelial_types).values

    for sample in sorted(sample_ids.unique()):
        sample_mask = (sample_ids == sample).values
        sample_idx = np.where(sample_mask)[0]

        epi_in_sample = is_epithelial[sample_mask]
        epi_coords = coords[sample_mask][epi_in_sample]

        if len(epi_coords) == 0:
            continue

        all_coords = coords[sample_mask]

        # Find all cells within radius of any epithelial cell
        epi_tree = KDTree(epi_coords)
        neighbors = epi_tree.query_ball_point(all_coords, r=radius)
        has_epi_neighbor = np.array([len(n) > 0 for n in neighbors])

        in_parenchyma[sample_idx[has_epi_neighbor]] = True

    return in_parenchyma


def find_lymph_nodes(
    coords, cell_types, sample_ids, parenchyma_mask, ln_immune_types, neighbor_radius, min_component_size
):
    """
    For each sample, among LN-associated immune cells within the parenchyma:
    1. Build a spatial graph at `neighbor_radius`
    2. Find connected components
    3. Mark components with > `min_component_size` cells as lymph nodes

    Returns a boolean array (True = lymph node cell to remove).
    """
    n_cells = len(cell_types)
    is_lymph_node = np.zeros(n_cells, dtype=bool)
    is_ln_immune = cell_types.isin(ln_immune_types).values & parenchyma_mask

    for sample in sorted(sample_ids.unique()):
        sample_mask = (sample_ids == sample).values
        sample_idx = np.where(sample_mask)[0]

        ln_in_sample = is_ln_immune[sample_mask]
        ln_local_idx = np.where(ln_in_sample)[0]

        if len(ln_local_idx) < min_component_size:
            continue

        ln_coords = coords[sample_mask][ln_in_sample]

        # Build spatial graph among LN immune cells
        tree = KDTree(ln_coords)
        pairs = tree.query_pairs(r=neighbor_radius)

        if len(pairs) == 0:
            continue

        # Build sparse adjacency matrix
        n = len(ln_coords)
        rows, cols = zip(*pairs)
        rows, cols = list(rows) + list(cols), list(cols) + list(rows)
        adj = csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(n, n),
        )

        # Find connected components
        n_components, comp_labels = connected_components(adj, directed=False)

        # Identify large components (lymph nodes)
        for comp_id in range(n_components):
            comp_mask = comp_labels == comp_id
            if comp_mask.sum() > min_component_size:
                # Map back to global indices
                global_idx = sample_idx[ln_local_idx[comp_mask]]
                is_lymph_node[global_idx] = True

    return is_lymph_node


def main():
    parser = argparse.ArgumentParser(description="In silico dissection of pancreatic parenchyma")
    parser.add_argument("--input", default=ADATA_PATH, help="Path to input h5ad")
    parser.add_argument("--output-dir", default=str(OUT_DIR), help="Output directory")
    parser.add_argument("--parenchyma-radius", type=float, default=PARENCHYMA_RADIUS)
    parser.add_argument("--ln-radius", type=float, default=LN_NEIGHBOR_RADIUS)
    parser.add_argument("--ln-min-size", type=int, default=LN_MIN_COMPONENT_SIZE)
    parser.add_argument("--ln-expansion-radius", type=float, default=LN_EXPANSION_RADIUS)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load ---
    print("Loading adata...")
    t0 = time.time()
    adata = sc.read_h5ad(args.input)
    print(f"  Loaded {adata.shape[0]:,} cells x {adata.shape[1]} genes in {time.time()-t0:.0f}s")

    coords = adata.obsm["X_spatial"]
    cell_types = adata.obs["cell_annotation"]
    sample_ids = adata.obs["sample_id"]

    # --- Step 1: Define parenchyma ---
    print(f"\nStep 1: Defining parenchyma (radius={args.parenchyma_radius} μm)...")
    print(f"  Epithelial types: {EPITHELIAL_TYPES}")
    n_epi = cell_types.isin(EPITHELIAL_TYPES).sum()
    print(f"  Epithelial cells: {n_epi:,}")

    t0 = time.time()
    in_parenchyma = define_parenchyma(coords, cell_types, sample_ids, EPITHELIAL_TYPES, args.parenchyma_radius)
    print(
        f"  Parenchyma cells: {in_parenchyma.sum():,} / {len(in_parenchyma):,} "
        f"({in_parenchyma.mean()*100:.1f}%) in {time.time()-t0:.0f}s"
    )

    # --- Step 2: Identify lymph nodes ---
    print(f"\nStep 2: Identifying lymph nodes (radius={args.ln_radius} μm, " f"min_size={args.ln_min_size})...")
    print(f"  LN immune types: {LN_IMMUNE_TYPES}")
    n_ln_immune = (cell_types.isin(LN_IMMUNE_TYPES) & in_parenchyma).sum()
    print(f"  LN-associated immune cells in parenchyma: {n_ln_immune:,}")

    t0 = time.time()
    is_lymph_node = find_lymph_nodes(
        coords, cell_types, sample_ids, in_parenchyma, LN_IMMUNE_TYPES, args.ln_radius, args.ln_min_size
    )
    print(f"  Lymph node cells identified: {is_lymph_node.sum():,} in {time.time()-t0:.0f}s")

    # --- Step 3: Expand lymph node annotations ---
    print(f"\nStep 3: Expanding LN annotations (radius={args.ln_expansion_radius} μm)...")
    t0 = time.time()
    n_before = is_lymph_node.sum()
    for sample in sorted(sample_ids.unique()):
        sample_mask = (sample_ids == sample).values
        sample_idx = np.where(sample_mask)[0]

        ln_in_sample = is_lymph_node[sample_mask]
        if ln_in_sample.sum() == 0:
            continue

        ln_coords = coords[sample_mask][ln_in_sample]
        all_coords = coords[sample_mask]

        ln_tree = KDTree(ln_coords)
        neighbors = ln_tree.query_ball_point(all_coords, r=args.ln_expansion_radius)
        near_ln = np.array([len(n) > 0 for n in neighbors])
        is_lymph_node[sample_idx[near_ln]] = True

    n_added = is_lymph_node.sum() - n_before
    print(f"  Expanded LN: {n_before:,} -> {is_lymph_node.sum():,} (+{n_added:,} cells) " f"in {time.time()-t0:.0f}s")

    # Per-sample LN summary
    ln_summary = []
    for sample in sorted(sample_ids.unique()):
        s_mask = (sample_ids == sample).values
        n_ln = is_lymph_node[s_mask].sum()
        if n_ln > 0:
            ln_summary.append({"sample_id": sample, "ln_cells_removed": int(n_ln)})
    if ln_summary:
        ln_df = pd.DataFrame(ln_summary)
        print(f"\n  Lymph nodes found in {len(ln_df)} samples:")
        for _, row in ln_df.iterrows():
            print(f"    {row['sample_id']}: {row['ln_cells_removed']:,} cells")
        ln_df.to_csv(out_dir / "lymph_node_removal_summary.csv", index=False)

    # --- Save per-sample annotation CSVs ---
    keep = in_parenchyma & ~is_lymph_node

    print(f"\n--- Summary ---")
    print(f"  Total cells:             {len(adata):,}")
    print(f"  In parenchyma:           {in_parenchyma.sum():,} ({in_parenchyma.mean()*100:.1f}%)")
    print(f"  Lymph node (removed):    {is_lymph_node.sum():,}")
    print(f"  Final (parenchyma - LN): {keep.sum():,} ({keep.mean()*100:.1f}%)")

    print(f"\nSaving per-sample CSVs to: {out_dir}")
    for sample in sorted(sample_ids.unique()):
        s_mask = (sample_ids == sample).values
        df = pd.DataFrame(
            {
                "barcode": adata.obs_names[s_mask],
                "in_parenchyma": in_parenchyma[s_mask],
                "is_lymph_node": is_lymph_node[s_mask],
            }
        )
        safe_sample = sample.replace("/", "_")
        df.to_csv(out_dir / f"{safe_sample}_dissection.csv", index=False)

    print("\nDone!")


if __name__ == "__main__":
    main()
