from xenium_utils.tl.phenograph_rapids import phenograph_rapids
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
import rapids_singlecell as rsc


# Class to suppress output from inside functions (e.g., PhenoGraph)
class HiddenPrints:
    def __init__(self, highest_level=logging.CRITICAL):
        self.highest_level = highest_level

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        self.previous_level = logging.root.manager.disable
        logging.disable(self.highest_level)

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        logging.disable(self.previous_level)


def kneepoint(vals, xcoords=None):

    # Convert array of values to 2D vectors by position in array
    vals = np.array(vals)
    descending = vals[-1] < vals[0]
    if descending:
        vals = -vals  # Should go from 0 to 1
    n_pts = vals.shape[0]
    if xcoords is None:
        xcoords = range(n_pts)
    pts = np.vstack([xcoords, vals]).T

    last_vec = pts[-1] - pts[0]  # Get vec b/n first and last points
    last_vec_norm = last_vec / np.linalg.norm(last_vec)  # L2 normalized vector
    all_vecs = pts - pts[0]  # Move to origin at Point 1

    scalars = np.dot(all_vecs, last_vec_norm)  # Scalar of projection to last vector
    projections = np.outer(scalars, last_vec_norm)  # Projection of original vectors onto last vector

    # Get point w/ max difference b/n projection to diagonal and the original vector
    # This occurs at the knee point (similar to ROC graphs)
    vecs = all_vecs - projections
    dists = np.linalg.norm(vecs, axis=1)
    idx = np.argmax(dists)

    return idx


def nearest_neighbors(
    adata,
    n_neighbors: int,
    use_rep: str = "X_pca",
    key_added: str = None,
    **kwargs,
):
    # Run kNN with RAPIDS
    X = cp.array(adata.obsm[use_rep])
    model = cuml.neighbors.NearestNeighbors(n_neighbors=n_neighbors)
    model.fit(X)
    distances, indices = model.kneighbors(X)

    # Index params for sparse matrix
    indptr = np.arange(0, indices.size + 1, indices.shape[1])
    indices = indices.get()

    # Set similar to scanpy
    if key_added is not None:
        neighbors_key = key_added
        prefix = f"{key_added}_"
    else:
        neighbors_key = "neighbors"
        prefix = ""

    adata.obsp[f"{prefix}connectivities"] = sp.sparse.csr_matrix(
        (
            np.ones_like(indices).flatten(),
            indices.flatten(),
            indptr,
        )
    )

    adata.obsp[f"{prefix}distances"] = sp.sparse.csr_matrix(
        (
            distances.get().flatten(),
            indices.flatten(),
            indptr,
        )
    )

    adata.uns[neighbors_key] = dict(
        connectivities_key=f"{prefix}connectivities",
        distances_key=f"{prefix}distances",
        params=dict(distance="euclidean", n_neighbors=n_neighbors, use_rep=use_rep),
    )

    # return distances and indices
    return distances, indices


def preprocess_rapids(
    adata,
    filter_min_counts: int = None,
    pca_layer: str = "norm",
    count_layer: str = "counts",
    # n_pcs: int = None,
    pca_total_var: float = None,
    knn_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_n_epochs: int = 1000,
    phenograph_resolution: float = 1,
    umap_kwargs: dict = None,
    normalize_total_kwargs: dict = None,
    use_highly_variable: bool = True,
    scale_data: bool = True,
):
    with tqdm(total=6) as pbar:
        # Filtering
        pbar.set_description("Filtering")
        adata.X = adata.layers[count_layer].copy()
        if filter_min_counts is None:
            n_counts = np.sort(np.array(adata.X.sum(1)).flatten())
            idx = kneepoint(np.log10(n_counts))
            filter_min_counts = min(max(n_counts[idx], 5), 30)
        sc.pp.filter_cells(adata, min_counts=filter_min_counts)
        pbar.update(1)

        # Median library-size normalization
        pbar.set_description("Normalization")
        adata.layers["norm"] = adata.layers[count_layer].copy()
        if normalize_total_kwargs is None:
            normalize_total_kwargs = dict()
        sc.pp.normalize_total(adata, layer="norm", **normalize_total_kwargs)

        # Log-transformation (natural log, pseudocount of 1)
        adata.layers["lognorm"] = adata.layers["norm"].copy()
        if "log1p" in adata.uns:
            del adata.uns["log1p"]
        sc.pp.log1p(adata, layer="lognorm")

        # scale to unit variance

        pbar.set_description("Scaling")
        if scale_data:
            adata.layers["scaled"] = adata.layers["lognorm"].copy()
            sc.pp.scale(adata, layer="scaled", max_value=10, zero_center=True)
            # convert to sparse matrix
            adata.layers["scaled"] = sp.sparse.csr_matrix(adata.layers["scaled"])
        pbar.update(1)

        # Run PCA using GPU
        pbar.set_description("PCA")
        if use_highly_variable:
            counts_sparse_gpu = cupyx.scipy.sparse.csr_matrix(adata[:, adata.var["highly_variable"]].layers[pca_layer])
        else:
            counts_sparse_gpu = cupyx.scipy.sparse.csr_matrix(adata.layers[pca_layer])
        model = cuml.PCA(n_components=min(counts_sparse_gpu.shape))
        X_pca = model.fit_transform(counts_sparse_gpu).get()
        cumulative_var = model.explained_variance_ratio_.get().cumsum()
        if pca_total_var is None:
            # if n_pcs is None:
            n_pcs = kneepoint(cumulative_var)
        else:
            n_pcs = np.argmin(abs(cumulative_var - pca_total_var))
            print(f"Using {n_pcs} PCs to explain {pca_total_var:.2f} of the variance")

        adata.obsm["X_pca"] = X_pca[:, :n_pcs]
        pbar.update(1)

        # kNN on PCA
        pbar.set_description("kNN")
        nearest_neighbors(adata, knn_neighbors, use_rep="X_pca")
        pbar.update(1)

        # Run UMAP using GPU
        with HiddenPrints():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)

                pbar.set_description("UMAP")
                if umap_kwargs is None:
                    umap_kwargs = dict()
                model = cuml.UMAP(
                    min_dist=umap_min_dist,
                    n_epochs=umap_n_epochs,
                    n_neighbors=knn_neighbors,
                    **umap_kwargs,
                )
                model.fit(X=X_pca[:, :n_pcs])
                X_umap = model.transform(X_pca[:, :n_pcs])
                adata.obsm["X_umap"] = X_umap
                pbar.update(1)

                # Cluster with GPU
                pbar.set_description("Clustering")
                min_size = -1
                kwargs = {"resolution": phenograph_resolution}
                phenograph_rapids(adata, min_size=min_size, **kwargs)

                pbar.update(1)
                pbar.set_description("Done")


#' same as preprocess_rapids but removing normalization and filtering steps
def cluster_rapids(
    adata,
    pca_layer: str = "norm",
    n_pcs: int = None,
    pca_total_var: float = None,
    knn_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_n_epochs: int = 1000,
    phenograph_resolution: float = 1,
    umap_kwargs: dict = None,
    use_highly_variable: bool = True,
    min_size=-1,
    phenograph_key="phenograph_cluster",
):
    with tqdm(total=4) as pbar:

        # Run PCA using GPU
        pbar.set_description("PCA")
        if use_highly_variable:
            counts_sparse_gpu = cupyx.scipy.sparse.csr_matrix(adata[:, adata.var["highly_variable"]].layers[pca_layer])
        else:
            counts_sparse_gpu = cupyx.scipy.sparse.csr_matrix(adata.layers[pca_layer])
        model = cuml.PCA(n_components=min(counts_sparse_gpu.shape))
        X_pca = model.fit_transform(counts_sparse_gpu).get()
        cumulative_var = model.explained_variance_ratio_.get().cumsum()
        if pca_total_var is None:
            # if n_pcs is None:
            # n_pcs = kneepoint(cumulative_var)
            n_pcs = n_pcs
        else:
            n_pcs = np.argmin(abs(cumulative_var - pca_total_var))
            print(f"Using {n_pcs} PCs to explain {pca_total_var:.2f} of the variance")

        adata.obsm["X_pca"] = X_pca[:, :n_pcs]
        pbar.update(1)

        # kNN on PCA
        pbar.set_description("kNN")
        nearest_neighbors(adata, knn_neighbors, use_rep="X_pca")
        pbar.update(1)

        # Run UMAP using GPU
        with HiddenPrints():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)

                pbar.set_description("UMAP")
                if umap_kwargs is None:
                    umap_kwargs = dict()
                model = cuml.UMAP(
                    min_dist=umap_min_dist,
                    n_epochs=umap_n_epochs,
                    n_neighbors=knn_neighbors,
                    random_state=42,
                    **umap_kwargs,
                )
                model.fit(X=X_pca[:, :n_pcs])
                X_umap = model.transform(X_pca[:, :n_pcs])
                adata.obsm["X_umap"] = X_umap
                pbar.update(1)

                # Cluster with GPU
                pbar.set_description("Clustering")
                # min_size = -1
                kwargs = {
                    "resolution": phenograph_resolution,
                    "key_added": phenograph_key,
                }
                phenograph_rapids(adata, min_size=min_size, **kwargs)

                pbar.update(1)
                pbar.set_description("Done")


#' same as preprocess_rapids but just doing PCA and clustering across a range of PCs
#' a list of n_pcs or total variance can be provided and phenograph will be run for each
def cluster_rapids_range(
    adata,
    pca_layer: str = "norm",
    n_pcs: list = None,
    pca_total_var: list = None,
    knn_neighbors: int = 15,
    phenograph_resolution: float = 1,
    use_highly_variable: bool = True,
    min_size=-1,
    phenograph_key="phenograph_cluster",
):
    pbar_len = len(n_pcs) + 1 if n_pcs is not None else len(pca_total_var) + 1
    with tqdm(total=pbar_len) as pbar:

        # Run PCA using GPU
        pbar.set_description("PCA")
        if use_highly_variable:
            counts_sparse_gpu = cupyx.scipy.sparse.csr_matrix(adata[:, adata.var["highly_variable"]].layers[pca_layer])
        else:
            counts_sparse_gpu = cupyx.scipy.sparse.csr_matrix(adata.layers[pca_layer])
        # model = cuml.PCA(n_components=min(adata.shape))
        model = cuml.PCA(n_components=300)
        X_pca = model.fit_transform(counts_sparse_gpu).get()
        cumulative_var = model.explained_variance_ratio_.get().cumsum()
        if pca_total_var is None:
            # if n_pcs is None:
            # n_pcs = kneepoint(cumulative_var)
            n_pcs = n_pcs
        else:
            n_pcs = [np.argmin(abs(cumulative_var - i)) for i in pca_total_var]
            n_pcs = np.argmin(abs(cumulative_var - pca_total_var))
            [
                print(f"Using {n_pcs[i]} PCs to explain {pca_total_var[i]:.2f} of the variance")
                for i in range(len(pca_total_var))
            ]

        pbar.update(1)

        # kNN on PCA
        pbar.set_description("clustering")

        # cluster using GPU
        for i in range(len(n_pcs)):
            with HiddenPrints():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)

                    adata.obsm["X_pca_use"] = X_pca[:, : n_pcs[i]].copy()
                    nearest_neighbors(adata, knn_neighbors, use_rep="X_pca_use")
                    # Cluster with GPU
                    kwargs = {"resolution": phenograph_resolution, "key_added": phenograph_key + f"_{n_pcs[i]}"}
                    phenograph_rapids(adata, min_size=min_size, **kwargs)

                    pbar.update(1)
        pbar.set_description("Done")


# def compute_silhouette(
#     adata,
#     obs_labels: list,
#     use_rep: str = 'X_pca',
#     **kwargs,
# ):
#     res = {}
#     avg_score = []
#     for i in obs_labels:
#         labels = adata.obs[i].values
#         pc_num = int(str.replace(i, 'phenograph_cluster_', ''))
#         X = adata.obsm[use_rep]
#         X = X[:, :pc_num]
#         silhouette_avg = silhouette_score(X, labels, random_state=42)
#         sample_silhouette_values = silhouette_samples(X, labels, random_state = 42)
#         res[i] = (sample_silhouette_values)
#         avg_score.append(silhouette_avg)

#     res_df = pd.DataFrame(res)
#     return avg_score, res_df


def pairwise_adjusted_rand_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise Adjusted Rand Index between all columns of a DataFrame.
    Each column is assumed to be a clustering (label assignments).

    Parameters:
        df (pd.DataFrame): DataFrame where each column is a clustering.

    Returns:
        pd.DataFrame: Symmetric DataFrame of ARI scores.
    """
    columns = df.columns

    result = pd.DataFrame(index=columns, columns=columns, dtype=float)

    for col1, col2 in itertools.combinations_with_replacement(columns, 2):
        ari = adjusted_rand_score(df[col1], df[col2])
        result.at[col1, col2] = ari
        result.at[col2, col1] = ari  # symmetric

    return result


def compute_cluster_entropy(clusters: pd.DataFrame, group_labels: pd.Series):
    """
    Compute the entropy of clusters based on the distribution of group labels.
    Returns data frame with columns: clustering_name, cluster, group_entropy
    """

    result = pd.DataFrame(columns=["clustering_name", "cluster", "group_entropy"])
    columns = clusters.columns
    for col in columns:
        labels = clusters[col]
        cluster_labels = clusters[col].unique()
        for label in cluster_labels:
            label_groups = group_labels.iloc[np.where(labels == label)[0]]
            label_freq = label_groups.value_counts(normalize=True)
            label_entropy = entropy(label_freq)

            result = pd.concat(
                [
                    result,
                    pd.DataFrame({"clustering_name": col, "cluster": label, "group_entropy": label_entropy}, index=[0]),
                ],
                ignore_index=True,
            )

    return result


# def compute_silhouette_score(
#     adata,
#     clusters: pd.DataFrame,
#     n_dim_clusters: pd.Series,
#     use_rep: str = 'X_pca',
#     **kwargs,
# ):
# """ computes silhouette score for a list of clusterings
# """

#     ss_avg = {}


#     # Compute silhouette score
#     for cluster in clusters.columns:

#         silhouette_avg, sample_silhouette_values = silhouette_score(X, clusters[cluster], metric = 'euclidean')

#         ss_avg[cluster] = silhouette_avg.copy()

#     return ss_avg


def compute_knn_label_sharing(
    adata,
    n_pcs: list,
    group_labels: pd.Series,
    k: int = 15,
    use_rep: str = "X_pca",
):
    """
    Compute the label sharing of k-nearest neighbors for each cluster.
    Returns a DataFrame with columns: clustering_name, cluster, knn_label_sharing
    """

    result = pd.DataFrame(columns=["n_pcs", "label", "knn_label_sharing"])

    # if X_pcs is not in adata.obsm, run PCA
    if use_rep not in adata.obsm:
        raise ValueError(f"{use_rep} not found in adata.obsm. Please run PCA first.")

    for pc in n_pcs:

        adata.obsm["X_pca_use"] = adata.obsm[use_rep][:, :pc].copy()

        # Run kNN with RAPIDS
        X = cp.array(adata.obsm["X_pca_use"])
        model = cuml.neighbors.NearestNeighbors(n_neighbors=k)
        model.fit(X)
        distances, indices = model.kneighbors(X)
        # convert to numpy
        indices = cp.asnumpy(indices)

        for label in np.unique(group_labels):

            label_indices = np.where(group_labels == label)[0]
            knn_indices = indices[label_indices]
            knn_labels = group_labels[knn_indices]
            knn_label_sharing = np.mean(knn_labels == label)

            result = pd.concat(
                [
                    result,
                    pd.DataFrame({"n_pcs": pc, "label": label, "knn_label_sharing": knn_label_sharing}, index=[0]),
                ],
                ignore_index=True,
            )

    # this is not necessary for checking relative levels within a label across PCs
    # standardize knn_label_sharing between 0 and 1 within each label
    result["knn_label_sharing_std"] = result.groupby("label")["knn_label_sharing"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )
    # # compute random chance probabilities for each label
    # label_probs = pd.Series(group_labels).value_counts(normalize=True)

    # # standardize the knn_label_sharing by dividing by the random chance probability
    # result['knn_label_sharing_std'] = result['knn_label_sharing'] / result['label'].map(label_probs)

    return result


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


# function for just PCs with GPU
def run_pca_gpu(adata, pca_layer, use_highly_variable, pca_total_var):

    if use_highly_variable:
        counts_sparse_gpu = cupyx.scipy.sparse.csr_matrix(adata[:, adata.var["highly_variable"]].layers[pca_layer])
    else:
        counts_sparse_gpu = cupyx.scipy.sparse.csr_matrix(adata.layers[pca_layer])
    model = cuml.PCA(n_components=min(adata.shape))
    X_pca = model.fit_transform(counts_sparse_gpu).get()
    cumulative_var = model.explained_variance_ratio_.get().cumsum()
    if pca_total_var is None:
        # if n_pcs is None:
        n_pcs = kneepoint(cumulative_var)
    else:
        n_pcs = np.argmin(abs(cumulative_var - pca_total_var))
        print(f"Using {n_pcs} PCs to explain {pca_total_var:.2f} of the variance")

    adata.obsm["X_pca"] = X_pca[:, :n_pcs]


# helper function for rapids single cell clustering and dim reduction
def process_rsc(
    adata,
    pc_var_cutoff=0.6,
    pca_layer="log1p_norm",
    use_highly_variable=True,
    knn_neighbors=30,
    umap_min_dist=0.1,
    umap_n_epochs=1000,
    cluster_resolution=2,
    cluster_key_added="leiden",
):
    rsc.get.anndata_to_GPU(adata, layer="log1p_norm")
    rsc.pp.pca(adata, n_comps=100, layer="log1p_norm", use_highly_variable=True)
    # filter PCs based on variance explained, subset obsm before neighbors
    pc_var = adata.uns["pca"]["variance_ratio"]
    n_pcs = (pc_var.cumsum() < pc_var_cutoff).sum()
    print(f"Using {n_pcs} PCs to explain {pc_var[:n_pcs].sum():.2%} of variance")
    adata.obsm["X_pca"] = adata.obsm["X_pca"][:, :n_pcs]
    rsc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=knn_neighbors)
    rsc.tl.umap(adata, min_dist=umap_min_dist)
    rsc.tl.leiden(adata, resolution=cluster_resolution, key_added=cluster_key_added)
    rsc.get.anndata_to_CPU(adata, layer="log1p_norm")
