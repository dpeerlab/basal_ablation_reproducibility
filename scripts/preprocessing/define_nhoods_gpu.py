from tqdm import tqdm
import scanpy as sc
import pandas as pd
import numpy as np
import cupy as cp
import warnings
import logging
import cupyx
import cuml

# import squidpy as sq
import sys
import cupyx
import scipy.sparse as sp
import anndata as ad


def aggregate_nhoods(
    adata: sc.AnnData,
    groupby: str = "cell_type",
    layer: str = "counts",
    graph_kwargs: dict = {
        "library_key": "sample_id",
        "coord_type": "generic",
        "radius": 60,
    },
) -> sc.AnnData:
    """
    Aggregate cells of the same type within the same niche.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix.
    groupby : str, optional
        Column name in `nhoods` to group by, by default "nhood".
    layer : str, optional
        Layer to use for aggregation, by default "counts".
    graph_kwargs : dict, optional
        Additional keyword arguments for the spatial neighborhood graph, by default None.

    Returns
    -------
    sc.AnnData
        Aggregated AnnData object for each cell type around each index cell.
    """

    if layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' not found in AnnData object.")

    # Ensure nhoods is a DataFrame with the correct structure
    if not isinstance(groupby, pd.DataFrame):
        raise TypeError("groupby must be a pandas DataFrame.")

    # create spatial neighborhood graph
    if graph_kwargs is None:
        graph_kwargs = {}
    sq.gr.spatial_neighbors(adata, **graph_kwargs)

    adata_nhood = {}

    for cell_type in tqdm(
        adata.obs[groupby].unique(), desc="Aggregating neighborhoods"
    ):
        # Filter cells of the current type
        cells_of_type = adata.obs[groupby] == cell_type

        # Create a new AnnData object for the current cell type
        adata_type = adata[cells_of_type].copy()

        # Aggregate the counts in the specified layer
        if layer in adata_type.layers:
            aggregated_counts = adata_type.layers[layer].sum(axis=0)
            adata_type.layers[layer] = cp.asarray(aggregated_counts)

        # Store the aggregated data
        adata_nhood[cell_type] = adata_type
    return aggregated


##  funcitons for getting neighborhood expression
def create_diagonal_matrix(v, total_cells):
    cell_indexes = np.array(list(range(total_cells)))
    diagonal_matrix = sp.csr_matrix(
        (v, (cell_indexes, cell_indexes)), shape=[total_cells, total_cells]
    )
    return diagonal_matrix


def get_average_neighborhood_expression(
    expression_matrix, spatial_neighborhood, anchor_cells=None, neighbor_cells=None
):
    if anchor_cells is None:
        anchor_cells = np.array([True] * expression_matrix.shape[0])
    if neighbor_cells is None:
        neighbor_cells = np.array([True] * expression_matrix.shape[0])
    X_neighbors = expression_matrix[neighbor_cells, :].copy()
    spatial_neighborhood = (
        spatial_neighborhood[anchor_cells, :].copy()[:, neighbor_cells].copy()
    )
    num_neighbors_per_anchor = np.ravel(spatial_neighborhood.sum(axis=1))
    n_anchors = spatial_neighborhood.shape[0]
    normalization_matrix = create_diagonal_matrix(
        1 / num_neighbors_per_anchor, n_anchors
    )

    # convert all matrices to cupy for GPU computation
    normalization_matrix = cupyx.scipy.sparse.csr_matrix(normalization_matrix)
    spatial_neighborhood = cupyx.scipy.sparse.csr_matrix(spatial_neighborhood)
    X_neighbors = cupyx.scipy.sparse.csr_matrix(X_neighbors)

    # compute average expression matrix
    average_expression_matrix = (
        normalization_matrix @ spatial_neighborhood @ X_neighbors
    )
    # convert back to numpy sparse matrix
    average_expression_matrix = average_expression_matrix.get()
    return average_expression_matrix


def get_total_neighborhood_expression(
    expression_matrix, spatial_neighborhood, anchor_cells=None, neighbor_cells=None
):
    if anchor_cells is None:
        anchor_cells = np.array([True] * expression_matrix.shape[0])
    if neighbor_cells is None:
        neighbor_cells = np.array([True] * expression_matrix.shape[0])
    X_neighbors = expression_matrix[neighbor_cells, :].copy()
    spatial_neighborhood = (
        spatial_neighborhood[anchor_cells, :].copy()[:, neighbor_cells].copy()
    )
    # convert all matrices to cupy for GPU computation
    spatial_neighborhood = cupyx.scipy.sparse.csr_matrix(spatial_neighborhood)
    X_neighbors = cupyx.scipy.sparse.csr_matrix(X_neighbors)
    total_expression_matrix = spatial_neighborhood @ X_neighbors
    return total_expression_matrix


def get_neighborhood_fraction(spatial_neighborhood, boolean_variable):
    n_cells = spatial_neighborhood.shape[0]
    cell_indexes = np.array(list(range(n_cells)))
    masked_matrix = mask_neighborhood_columns(spatial_neighborhood, boolean_variable)
    boolean_fraction = np.ravel(masked_matrix.sum(axis=1)) / np.ravel(
        spatial_neighborhood.sum(axis=1)
    )
    return boolean_fraction


def get_neighborhood_count(spatial_neighborhood, boolean_variable):
    n_cells = spatial_neighborhood.shape[0]
    cell_indexes = np.array(list(range(n_cells)))
    masked_matrix = mask_neighborhood_columns(spatial_neighborhood, boolean_variable)
    boolean_count = np.ravel(masked_matrix.sum(axis=1))
    total_count = np.ravel(spatial_neighborhood.sum(axis=1))
    return (boolean_count, total_count)


def mask_neighborhood_columns(spatial_neighborhood, mask):
    mask_matrix = create_diagonal_matrix(mask, spatial_neighborhood.shape[0])
    masked_neighborhood = spatial_neighborhood @ mask_matrix
    return masked_neighborhood


def mask_neighborhood_rows(spatial_neighborhood, mask):
    mask_matrix = create_diagonal_matrix(mask, spatial_neighborhood.shape[0])
    masked_neighborhood = mask_matrix @ spatial_neighborhood
    return masked_neighborhood


def row_normalize_sparse_matrix(mat):
    normalization_vector = np.ravel(mat.sum(axis=1))
    normalization_factor = create_diagonal_matrix(
        1 / normalization_vector, mat.shape[0]
    )
    normalized_matrix = normalization_factor @ mat * np.median(normalization_vector)
    return normalized_matrix


def get_pseudobulk_neighborhood_expression(
    expression_matrix, spatial_neighborhood, anchor_cells=None, neighbor_cells=None
):
    """
    Pseudobulk neighborhood expression: sum neighbor counts, normalize by total
    counts, scale by median total counts.
    """
    if anchor_cells is None:
        anchor_cells = np.array([True] * expression_matrix.shape[0])
    if neighbor_cells is None:
        neighbor_cells = np.array([True] * expression_matrix.shape[0])

    X_neighbors = expression_matrix[neighbor_cells, :].copy().astype(np.float32)

    spatial_neighborhood_subset = (
        spatial_neighborhood[anchor_cells, :].copy()[:, neighbor_cells].copy()
    ).astype(np.float32)

    # convert to cupy sparse for GPU computation
    spatial_neighborhood_gpu = cupyx.scipy.sparse.csr_matrix(spatial_neighborhood_subset)
    X_neighbors_gpu = cupyx.scipy.sparse.csr_matrix(X_neighbors)

    # sum of neighbor expression for each anchor (excluding anchor itself)
    total_expression = spatial_neighborhood_gpu @ X_neighbors_gpu

    # compute total counts per pseudobulk (anchor + neighbors)
    total_counts = cp.ravel(total_expression.sum(axis=1))
    total_counts = cp.where(total_counts == 0, 1, total_counts)  # avoid div by zero

    # normalize by total counts
    normalization_matrix = cupyx.scipy.sparse.diags(1 / total_counts)
    normalized_expression = normalization_matrix @ total_expression

    # scale by median total counts
    median_counts = float(cp.median(total_counts))
    pseudobulk_expression = normalized_expression * median_counts

    # convert back to scipy sparse
    pseudobulk_expression = pseudobulk_expression.get()
    return pseudobulk_expression


def get_neighborhood_gene_positivity(
    expression_matrix,
    spatial_neighborhood,
    cell_types,
    cell_type_labels,
    threshold=0,
    anchor_cells=None,
):
    """
    Compute number and fraction of cells positive for each gene per cell type within each niche.
    Fractions use total niche size as denominator.

    Parameters
    ----------
    expression_matrix : sparse matrix
        Cells x genes expression matrix.
    spatial_neighborhood : sparse matrix
        Cells x cells adjacency matrix.
    cell_types : list
        List of cell type names to compute positivity for.
    cell_type_labels : array-like
        Cell type label for each cell (length n_cells).
    threshold : float
        Expression threshold for positivity (default 0, i.e. > 0).
    anchor_cells : array-like, optional
        Boolean mask for anchor cells. If None, all cells are anchors.

    Returns
    -------
    counts : dict
        {cell_type: (n_anchors, n_genes) array of positive cell counts}
    fractions : dict
        {cell_type: (n_anchors, n_genes) array of positive cell fractions}
    total_neighbors : array
        (n_anchors,) total neighbor count per anchor
    """
    if anchor_cells is None:
        anchor_cells = np.array([True] * expression_matrix.shape[0])

    spatial_neighborhood_subset = spatial_neighborhood[anchor_cells, :].copy()

    # total neighbors per anchor (denominator for fractions)
    total_neighbors = np.ravel(spatial_neighborhood_subset.sum(axis=1))
    total_neighbors_safe = np.where(total_neighbors == 0, 1, total_neighbors)

    # binarize expression at threshold
    X_positive = (expression_matrix > threshold).astype(np.float32)

    counts = {}
    fractions = {}

    for cell_type in cell_types:
        # mask for cells of this type
        cell_type_mask = (np.array(cell_type_labels) == cell_type).astype(np.float32)

        # mask neighborhood to only include cells of this type
        diag_mask = sp.diags(cell_type_mask)
        neighborhood_masked = spatial_neighborhood_subset @ diag_mask

        # convert to GPU
        neighborhood_gpu = cupyx.scipy.sparse.csr_matrix(neighborhood_masked.astype(np.float32))
        X_positive_gpu = cupyx.scipy.sparse.csr_matrix(X_positive)

        # count positive cells of this type in each neighborhood
        positive_counts = (neighborhood_gpu @ X_positive_gpu).get()
        positive_counts = np.asarray(positive_counts.todense())

        counts[cell_type] = positive_counts
        fractions[cell_type] = positive_counts / total_neighbors_safe[:, np.newaxis]

    return counts, fractions, total_neighbors
