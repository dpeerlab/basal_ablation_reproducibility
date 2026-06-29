#!/usr/bin/env python
# coding: utf-8

# Updating to pruned dataset. 

import os
from pathlib import Path
import sys

# import colorcet as cc
import matplotlib.pyplot as plt
import scanpy as sc
import umap
from sklearn.decomposition import PCA

from mingle.MINGLE import MINGLE


# import MINGLE package which hosts wasserstein wormhole code in this instance
sc.settings.set_figure_params(figsize=(6, 4), dpi=120)
plt.rcParams["figure.dpi"] = 120

# --- Required file paths ---
input_data = Path("")  # TODO: set AnnData file path
output_dir = Path("")  # TODO: set output directory

# create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
# --- AnnData annotations ---
anchor_inp_layer = "log1p_norm"  # TODO: layer used for anchor input
niche_inp_key = "X_pcs_BA_pleio_0.4_25mods"  # TODO: key in .obsm used for the niche encoder
num_neighbors = 32  # TODO: number of neighbors for the spatial graph
cell_type_key = "cell_annotation"  # TODO: column in .obs for cell types
spatial_key = "X_spatial"  # TODO: key in .obsm with spatial coordinates
sample_key = "sample_id"  # TODO: column in .obs for sample/batch IDs or set to None
test_key = None  # TODO: column in .obs for held-out test partition or keep None

# --- Optional visualization keys (list can be empty) ---
plot_color_keys = [
    "cell_annotation",
    "sample_id"
]

# --- Optional model overrides ---
# Provide additional keyword arguments understood by MINGLE. Leave empty if not needed.
overrides = {
    'v_dim': 24,
    'normalization_strength': 3
}



st_data = sc.read_h5ad(input_data)
print(f"Loaded AnnData with {st_data.n_obs} cells and {st_data.n_vars} genes.")
print(f"Available layers: {list(st_data.layers.keys())}")
print(f"obs columns: {list(st_data.obs.columns)}")
print(f"obsm keys: {list(st_data.obsm.keys())}")


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

model = MINGLE(
    adata=st_data,
    anchor_inp_layer=anchor_inp_layer,
    niche_inp_key=niche_inp_key,
    cell_type_key=cell_type_key,
    spatial_key=spatial_key,
    sample_key=sample_key,
    test_key=test_key,
    num_neighbors=num_neighbors,
    ckpt_dir=str(output_dir),
    **overrides,
)

print("Model initialized. Training configuration summary:")
print(model.config)

model.train()

model.inference()
model.save_ckpt()
print(f"Inference finished. Checkpoints saved in {output_dir}.")



st_data = model.st_data



output_path = output_dir / "st_data.h5ad"
st_data.write_h5ad(output_path)
print(f"Saved AnnData with embeddings to {output_path}")




