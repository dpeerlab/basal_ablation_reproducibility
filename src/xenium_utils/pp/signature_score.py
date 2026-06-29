from tqdm import tqdm
import scanpy as sc
import pandas as pd
import numpy as np
import scipy as sp
import cupy as cp
import warnings
import logging
import cupyx
import cuml
import sys
import os
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
from typing import Optional, Union


def signature_score(adata: sc.AnnData, signature: dict[str, str], layer: str = "scaled") -> pd.DataFrame:
    """
    Calculate signature score for each cell in the AnnData object.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix.
    signature : dict
        Dictionary where keys are names of the gene sets and values are lists of gene names.
    layer : str, optional
        Layer to use for the calculation, by default "scaled".

    Returns
    -------
    pd.Series
        Signature scores for each cell.
    """

    if layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' not found in AnnData object.")

    # Ensure signature is a DataFrame with gene names as index
    if not isinstance(signature, dict):
        raise TypeError("Signature must be a dictionary.")

    # Calculate signature score
    scores = {}
    for name in signature.keys():
        scores[name] = adata[:, signature[name]].layers[layer].mean(axis=1).A1

    scores_df = pd.DataFrame(scores, index=adata.obs_names)

    return scores_df


def scale_expression(
    adata,
    use_rep: str = "lognorm",
    max_value: float = 10,
    key_added: str = "scaled",
    **kwargs,
):
    """
    Scale the expression data in adata.
    """

    # Check if the layer exists
    if use_rep not in adata.layers:
        raise ValueError(f"{use_rep} not found in adata.layers. Please run PCA first.")

    # Check if the layer is sparse
    if sp.sparse.issparse(adata.layers[use_rep]):
        X = adata.layers[use_rep].copy()
        X = X.todense()
        X = np.asarray(X)
    else:
        X = adata.layers[use_rep].copy()
        X = np.asarray(X)

    # get scaled expression
    scaler = StandardScaler()  # center and scale
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.clip(X_scaled, -max_value, max_value)  # clip to max value
    # convert to sparse
    X_scaled = sparse.csr_matrix(X_scaled)

    # add to adata
    adata.layers[key_added] = X_scaled


def module_eigengene_score(
    adata: sc.AnnData,
    modules: dict[str, list],
    layer: str = "log1p_norm",
    neighbors_key: Optional[str] = None,
    smooth: bool = False,
    smooth_lambda: float = 0.9,
    scale: bool = True,
    return_gene_correlations: bool = True,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Calculate module eigengene scores (PC1) for each module and optionally
    compute gene-module correlations (kME scores) for all genes.

    This implements the WGCNA-style module eigengene approach:
    1. For each module, extract expression of member genes
    2. Compute PC1 of module gene expression (the "eigengene")
    3. Optionally smooth across neighbors
    4. Compute correlation of all genes with each module eigengene (kME)

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix.
    modules : dict
        Dictionary where keys are module names and values are lists of gene names.
    layer : str, optional
        Layer to use for the calculation, by default "log1p_norm".
    neighbors_key : str, optional
        Key in adata.obsp for neighbor connectivities. If provided and smooth=True,
        will smooth eigengenes across neighbors. Default None.
    smooth : bool, optional
        Whether to smooth eigengene scores across neighbors. Default False.
    smooth_lambda : float, optional
        Smoothing parameter (0-1). Higher values = more smoothing. Default 0.9.
    scale : bool, optional
        Whether to z-score scale expression before PCA. Default True.
    return_gene_correlations : bool, optional
        Whether to compute and return kME (gene-module correlations). Default True.

    Returns
    -------
    eigengene_scores : pd.DataFrame
        DataFrame with cells as rows and modules as columns, containing PC1 scores.
    gene_correlations : pd.DataFrame or None
        If return_gene_correlations=True, DataFrame with genes as rows and modules
        as columns, containing Pearson correlation (kME) of each gene with each
        module eigengene. None otherwise.

    Examples
    --------
    >>> modules = {'module_0': ['GeneA', 'GeneB', 'GeneC'], 'module_1': ['GeneD', 'GeneE']}
    >>> eigengenes, kME = module_eigengene_score(adata, modules)
    >>> # eigengenes: DataFrame (n_cells x n_modules) with PC1 scores
    >>> # kME: DataFrame (n_genes x n_modules) with correlations
    """
    if layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' not found in AnnData object.")

    # Get expression matrix
    X = adata.layers[layer]
    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X)

    gene_names = adata.var_names.tolist()
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    eigengene_scores = {}
    gene_correlations = {} if return_gene_correlations else None

    for module_name, module_genes in tqdm(modules.items(), desc="Computing eigengenes"):
        # Filter to genes that exist in adata
        valid_genes = [g for g in module_genes if g in gene_to_idx]
        if len(valid_genes) < 2:
            warnings.warn(f"Module '{module_name}' has fewer than 2 valid genes, skipping.")
            continue

        # Get indices for module genes
        gene_indices = [gene_to_idx[g] for g in valid_genes]

        # Extract expression for module genes (cells x genes)
        X_module = X[:, gene_indices]

        # Optionally scale
        if scale:
            scaler = StandardScaler()
            X_module_scaled = scaler.fit_transform(X_module)
        else:
            X_module_scaled = X_module

        # Compute PC1 (module eigengene)
        pca = PCA(n_components=1)
        eigengene = pca.fit_transform(X_module_scaled).flatten()

        # Ensure consistent sign: eigengene should correlate positively with
        # mean expression of module genes on average
        mean_expr = X_module_scaled.mean(axis=1)
        if np.corrcoef(eigengene, mean_expr)[0, 1] < 0:
            eigengene = -eigengene

        # Optionally smooth across neighbors
        if smooth and neighbors_key is not None:
            eigengene = _smooth_scores(eigengene, adata, neighbors_key, smooth_lambda)

        eigengene_scores[module_name] = eigengene

        # Compute gene-module correlations (kME) for ALL genes
        if return_gene_correlations:
            kME = np.zeros(len(gene_names))
            for i in range(len(gene_names)):
                gene_expr = X[:, i]
                # Handle constant genes
                if gene_expr.std() > 0:
                    kME[i] = np.corrcoef(gene_expr, eigengene)[0, 1]
                else:
                    kME[i] = 0.0
            gene_correlations[module_name] = kME

    eigengene_df = pd.DataFrame(eigengene_scores, index=adata.obs_names)

    if return_gene_correlations:
        kME_df = pd.DataFrame(gene_correlations, index=gene_names)
        return eigengene_df, kME_df
    else:
        return eigengene_df, None


def _smooth_scores(
    scores: np.ndarray,
    adata: sc.AnnData,
    neighbors_key: str,
    smooth_lambda: float = 0.9,
) -> np.ndarray:
    """
    Smooth scores across cell neighbors.

    Parameters
    ----------
    scores : np.ndarray
        Array of scores (one per cell).
    adata : sc.AnnData
        AnnData object with neighbor information.
    neighbors_key : str
        Key in adata.obsp for connectivities matrix.
    smooth_lambda : float
        Smoothing parameter (0-1). 0 = no smoothing, 1 = full neighbor average.

    Returns
    -------
    np.ndarray
        Smoothed scores.
    """
    if neighbors_key not in adata.obsp:
        raise ValueError(f"Neighbors key '{neighbors_key}' not found in adata.obsp")

    # Get connectivity matrix
    W = adata.obsp[neighbors_key]
    if sparse.issparse(W):
        W = W.toarray()

    # Row-normalize weights
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    W_norm = W / row_sums

    # Compute neighbor average
    neighbor_avg = W_norm @ scores

    # Blend original and smoothed
    smoothed = (1 - smooth_lambda) * scores + smooth_lambda * neighbor_avg

    return smoothed


def compute_module_eigengenes(
    adata: sc.AnnData,
    modules: dict[str, list],
    layer: str = "log1p_norm",
    scale: bool = True,
    add_to_obs: bool = True,
    obs_prefix: str = "ME_",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Compute module eigengenes and comprehensive gene-module statistics.

    This is a convenience wrapper around module_eigengene_score that also
    computes additional statistics and optionally adds results to adata.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix.
    modules : dict
        Dictionary where keys are module names and values are lists of gene names.
    layer : str, optional
        Layer to use for the calculation, by default "log1p_norm".
    scale : bool, optional
        Whether to z-score scale expression before PCA. Default True.
    add_to_obs : bool, optional
        Whether to add eigengene scores to adata.obs. Default True.
    obs_prefix : str, optional
        Prefix for column names when adding to adata.obs. Default "ME_".

    Returns
    -------
    eigengene_df : pd.DataFrame
        Module eigengene scores (cells x modules).
    kME_df : pd.DataFrame
        Gene-module correlations (genes x modules).
    module_stats : dict
        Dictionary with module statistics including:
        - 'variance_explained': Fraction of variance explained by PC1
        - 'n_genes': Number of genes in each module
        - 'top_genes': Top 10 genes by kME for each module
    """
    if layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' not found in AnnData object.")

    X = adata.layers[layer]
    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X)

    gene_names = adata.var_names.tolist()
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    eigengene_scores = {}
    gene_correlations = {}
    module_stats = {
        "variance_explained": {},
        "n_genes": {},
        "top_genes": {},
    }

    for module_name, module_genes in tqdm(modules.items(), desc="Computing eigengenes"):
        valid_genes = [g for g in module_genes if g in gene_to_idx]
        if len(valid_genes) < 2:
            warnings.warn(f"Module '{module_name}' has fewer than 2 valid genes, skipping.")
            continue

        gene_indices = [gene_to_idx[g] for g in valid_genes]
        X_module = X[:, gene_indices]

        if scale:
            scaler = StandardScaler()
            X_module_scaled = scaler.fit_transform(X_module)
        else:
            X_module_scaled = X_module

        # Compute PC1
        pca = PCA(n_components=1)
        eigengene = pca.fit_transform(X_module_scaled).flatten()

        # Ensure consistent sign
        mean_expr = X_module_scaled.mean(axis=1)
        if np.corrcoef(eigengene, mean_expr)[0, 1] < 0:
            eigengene = -eigengene

        eigengene_scores[module_name] = eigengene

        # Store variance explained
        module_stats["variance_explained"][module_name] = pca.explained_variance_ratio_[0]
        module_stats["n_genes"][module_name] = len(valid_genes)

        # Compute kME for all genes
        kME = np.zeros(len(gene_names))
        for i in range(len(gene_names)):
            gene_expr = X[:, i]
            if gene_expr.std() > 0:
                kME[i] = np.corrcoef(gene_expr, eigengene)[0, 1]
            else:
                kME[i] = 0.0
        gene_correlations[module_name] = kME

        # Get top genes by kME
        top_indices = np.argsort(kME)[::-1][:10]
        module_stats["top_genes"][module_name] = [(gene_names[i], kME[i]) for i in top_indices]

    eigengene_df = pd.DataFrame(eigengene_scores, index=adata.obs_names)
    kME_df = pd.DataFrame(gene_correlations, index=gene_names)

    # Add to adata.obs if requested
    if add_to_obs:
        for col in eigengene_df.columns:
            adata.obs[f"{obs_prefix}{col}"] = eigengene_df[col].values

    return eigengene_df, kME_df, module_stats
