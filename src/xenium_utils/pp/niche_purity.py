import numpy as np
import pandas as pd
from anndata import AnnData


def hdbscan_filter(
    adata,
    umap_key: str = 'X_umap_niche',
    min_cluster_size: int = 100,
    cluster_size_threshold: int = None,
    key_added: str = 'hdbscan',
) -> np.ndarray:
    """
    Run HDBSCAN on UMAP coordinates, store cluster labels in adata.obs,
    and return a boolean keep-mask.

    Removes noise points (HDBSCAN label == -1) and any clusters smaller
    than `cluster_size_threshold`.

    Parameters
    ----------
    adata : AnnData
        Must have obsm[umap_key].
    umap_key : str
        Key in adata.obsm containing 2-D UMAP coordinates.
    min_cluster_size : int
        HDBSCAN min_cluster_size parameter (default 100).
    cluster_size_threshold : int, optional
        Drop clusters with fewer than this many cells after clustering.
        Defaults to `min_cluster_size`.
    key_added : str
        Column name written to adata.obs with string cluster labels
        (noise / small clusters stored as '-1').

    Returns
    -------
    np.ndarray of bool, shape (n_obs,)
        True = keep, False = outlier or small cluster.
    """
    import hdbscan

    if cluster_size_threshold is None:
        cluster_size_threshold = min_cluster_size

    labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(
        adata.obsm[umap_key]
    )

    keep = labels >= 0  # drop noise first
    cluster_ids, counts = np.unique(labels[keep], return_counts=True)
    small_clusters = cluster_ids[counts < cluster_size_threshold]
    if len(small_clusters):
        keep &= ~np.isin(labels, small_clusters)

    # store labels; mark removed cells as '-1'
    labels_out = labels.copy()
    labels_out[~keep] = -1
    adata.obs[key_added] = pd.Categorical(labels_out.astype(str))

    n_removed = (~keep).sum()
    print(f"hdbscan_filter: kept {keep.sum():,} / {len(keep):,} cells "
          f"({n_removed:,} removed — noise or clusters < {cluster_size_threshold})")
    print(f"  cluster labels stored in obs['{key_added}']")
    return keep


def select_pure_niche_seeds(
    adata: AnnData,
    target_states: list,
    contam_states: list,
    annotation_col: str = 'cell_annotation',
    max_contam_count: int = 0,
    max_contam_frac: float = None,
    connectivities_key: str = 'spatial_connectivities',
    return_per_cell: bool = False,
) -> dict:
    """
    Select pure-niche seed cells for each target state.

    A seed is 'pure' if the number (or fraction) of its spatial neighbours
    labelled as any contamination state does not exceed the threshold.

    Parameters
    ----------
    adata : AnnData
        Must have ``obs[annotation_col]`` and ``obsp[connectivities_key]``.
    target_states : list[str]
        States to select seeds for (values in ``annotation_col``).
    contam_states : list[str]
        Contaminating states (values in ``annotation_col``).
    annotation_col : str
        Column in ``adata.obs`` with cell-type labels.
    max_contam_count : int
        Max contaminating neighbours allowed (default 0).
        Ignored when ``max_contam_frac`` is set.
    max_contam_frac : float, optional
        Max fraction of neighbours that may be contaminating.
        Takes precedence over ``max_contam_count``.
    connectivities_key : str
        Key in ``adata.obsp`` for the spatial connectivity matrix.
    return_per_cell : bool
        If True, also return per-cell contamination fractions for each
        contamination state.

    Returns
    -------
    dict[str, tuple[pd.Index, pd.DataFrame] or tuple[pd.Index, pd.DataFrame, pd.DataFrame]]
        Keyed by state name. Each value is ``(pure_seeds, summary)`` or
        ``(pure_seeds, summary, per_cell_contam)`` if return_per_cell=True:

        * ``pure_seeds`` — obs_names of seeds passing the purity filter
        * ``summary``    — DataFrame indexed by contam_state with columns:
            - ``mean_contam_frac``: mean fraction of neighbours from this state
                                    across pure seeds
            - ``n_seeds_excluded``: seeds excluded due to this state's contamination
        * ``per_cell_contam`` — DataFrame indexed by obs_names of pure seeds with
            columns for each contam_state containing the fraction of neighbors
            from that state, plus 'total_contam_frac' and 'n_neighbors'.
    """
    conn = adata.obsp[connectivities_key]
    labels = adata.obs[annotation_col].values

    # per-state and combined contamination vectors
    contam_vecs = {s: (labels == s).astype(np.float32) for s in contam_states}
    contam_combined = np.clip(
        sum(contam_vecs[s] for s in contam_states) if contam_states else np.zeros(adata.n_obs),
        0, 1,
    ).astype(np.float32)

    results = {}
    for state in target_states:
        seed_idx = np.where(labels == state)[0]
        n_neighbors = np.maximum(
            np.asarray(conn[seed_idx].sum(axis=1)).flatten(), 1
        )

        n_contam = np.asarray(conn[seed_idx] @ contam_combined).flatten()

        if max_contam_frac is not None:
            pure_mask = (n_contam / n_neighbors) <= max_contam_frac
            filter_desc = f"frac ≤ {max_contam_frac}"
        else:
            pure_mask = n_contam <= max_contam_count
            filter_desc = f"count ≤ {max_contam_count}"

        pure_idx = seed_idx[pure_mask]
        pure_seeds = adata.obs_names[pure_idx]

        # per-contam-state summary over pure seeds
        n_neighbors_pure = np.maximum(
            np.asarray(conn[pure_idx].sum(axis=1)).flatten(), 1
        ) if len(pure_idx) > 0 else np.ones(0)

        rows = {}
        per_cell_data = {'n_neighbors': n_neighbors_pure}
        for cs in contam_states:
            vec = contam_vecs[cs]
            n_contam_cs = np.asarray(conn[seed_idx] @ vec).flatten()
            if max_contam_frac is not None:
                n_excluded = int(((n_contam_cs / n_neighbors) > max_contam_frac).sum())
            else:
                n_excluded = int((n_contam_cs > max_contam_count).sum())

            if len(pure_idx) > 0:
                per_cell_contam_cs = np.asarray(conn[pure_idx] @ vec).flatten() / n_neighbors_pure
                frac = float(per_cell_contam_cs.mean())
                per_cell_data[f'{cs}_contam_frac'] = per_cell_contam_cs
            else:
                frac = float('nan')

            rows[cs] = {'mean_contam_frac': frac, 'n_seeds_excluded': n_excluded}

        summary = pd.DataFrame(rows).T
        summary.index.name = 'contam_state'

        print(f"[{state}] candidates: {len(seed_idx):,}  "
              f"filter: {filter_desc}  "
              f"pure seeds: {len(pure_seeds):,} "
              f"({100 * len(pure_seeds) / max(len(seed_idx), 1):.1f}%)")

        if return_per_cell and len(pure_idx) > 0:
            per_cell_contam = pd.DataFrame(per_cell_data, index=pure_seeds)
            per_cell_contam['total_contam_frac'] = sum(
                per_cell_contam[f'{cs}_contam_frac'] for cs in contam_states
            )
            results[state] = (pure_seeds, summary, per_cell_contam)
        else:
            results[state] = (pure_seeds, summary)

    return results


def get_niche_obs(seed_obs, conn, idx_map, all_obs_names):
    """
    Return obs_names of seed cells and all their direct spatial neighbors.

    Parameters
    ----------
    seed_obs : pd.Index
        obs_names of seed cells.
    conn : scipy.sparse matrix
        Spatial connectivity matrix (n_cells × n_cells).
    idx_map : pd.Series
        Mapping from obs_name to integer position in conn.
    all_obs_names : pd.Index
        Full obs_names of the AnnData (same ordering as conn rows/cols).

    Returns
    -------
    pd.Index
        obs_names of the niche (seeds ∪ neighbors).
    """
    seed_idx = idx_map[seed_obs].values
    neighbor_mask = np.asarray(conn[seed_idx].sum(axis=0)).flatten() > 0
    niche_idx = np.union1d(seed_idx, np.where(neighbor_mask)[0])
    return all_obs_names[niche_idx]
