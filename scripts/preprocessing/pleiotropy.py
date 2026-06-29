
import symNMF
import numpy as np 
import pandas as pd 
import scipy
from sklearn import neighbors
import sklearn
import os

import anndata as ad
import scanpy as sc
from scipy.stats import linregress
import pickle
import gzip
import anndata as ad
from scipy.linalg import norm

import matplotlib.pyplot as plt

import warnings
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import seaborn as sns
from pathlib import Path
from scipy.sparse import lil_matrix
from sklearn.cluster import KMeans
from numpy import concatenate
import itertools
import scipy.sparse as sp
from reconstruction import reconstruct_lr, recon_corr_n_plot




#this function will run the pleiotropy function passed in on the loading matrix and output the module assignments determined by
#that pleiotropy function
#also, outputs diagnostics based off of the module assignments

def assignModules(W, gene_names, name, output_folder, **kwargs):
    """
    Assign modules using specified pleiotropy function. Outputs assignments and diagnostics.
    Parameters:
        W: DataFrame (genes x modules, no gene names)
        gene_names: list of gene names in order of W
        func: pleiotropy assignment function
        name: string to save module assignments
    Returns:
        module_assignments csv: (Gene, Module)
        module_assignments_grouped csv: (Gene, Module) grouped by module 
    """
   

    module_assignments = cumulative_gene_mass(W = W, gene_names=gene_names, output_folder=output_folder, name=name, **kwargs)
    module_assignments.to_csv(f"{output_folder}/module_assignments_{name}.csv", index=False)

    module_assignments_grouped = module_assignments.copy()
    module_assignments_grouped = module_assignments_grouped.sort_values("Module", ascending=[False])
    module_assignments_grouped.to_csv(f"{output_folder}/module_assignments_grouped_{name}.csv", index=False)



    return module_assignments


    

#function to query for info about a gene
#for input of a gene name:
#print out its loading from W, print whether it is unassigned or pleiotropic or neither, and print its module assignemnt

## ADD FUNCTIONALITY TO PLOT GENE ACTIVITY ACROSS UMAP if flag is true 
def queryGene(gene_name, gene_names, module_assignments):
    """
    Query status and loading for a gene. Print diagnostics.
    """
    # get index by position for gene_names in the same order as W
    try:
        index = gene_names.get_loc(gene_name)
    except ValueError:
        raise ValueError(f"Gene {gene_name} not found in gene_names.")
    gene_loading = W.iloc[index, :]
    #print(gene_loading)

    if gene_name.isin(pleiotropic_genes):
        print(f"{gene_name} is pleiotropic")
        print(module_assignments.loc[module_assignments["Gene"] == gene_name])
    elif gene_name.isin(unassigned_genes):
        print(f"{gene_name} is unassigned")
    else:
        print(f"{gene_name} has one module assignment:")
        print(module_assignments.loc[module_assignments["Gene"] == gene_name])

    


#this function runs symmetric NMF
def runSymNMF(A, module_number, name, output_folder, gene_names, save=True, init_matrix = None):

    '''
    parameters: 
        A -> local correlation matrix on which nmf should run
        module_number -> number of desired modules
        init_matrix -> matrix to initialize symNMF with 
        name -> string to save the loading matrix with

    outputs:
        W -> matrix of loadings (dataframe)
    
    '''
    if init_matrix is None:
    
        alpha = 10
        np.random.seed(50)
        k = module_number
        size = len(gene_names)
        init_matrix = np.random.dirichlet(alpha * np.ones(k), size)

    snmf = symNMF.SymNMF(module_number)
    snmf.fit(A, init_matrix)

    #save the matrix of loadings as a dataframe
    W = pd.DataFrame(snmf.W, index=gene_names)
    if save:
        W.to_csv(f"{output_folder}/{name}_loadings_matrix.csv", index=True)
    
    print("Ran symNMF\n")
    return W



def sanitize_slashes(adata):
    import pandas as pd
    from collections.abc import Mapping
    cs = lambda s: s.replace('/', '_') if isinstance(s, str) else s
    def clean_df(x):
        if isinstance(x, pd.DataFrame):
            x.columns = [cs(c) for c in x.columns]
            for c in x.select_dtypes('category').columns:
                x[c] = x[c].cat.rename_categories(lambda v: cs(str(v)))
        return x
    def rec_uns(x):  # recursively clean dict-like objects
        return {cs(k): rec_uns(v) for k, v in x.items()} if isinstance(x, Mapping) else clean_df(x)

    adata.obs, adata.var = clean_df(adata.obs), clean_df(adata.var)
    adata.uns = rec_uns(dict(adata.uns))
    for slot in ("obsm", "varm", "layers", "obsp", "varp"):
        m = getattr(adata, slot, {})
        setattr(adata, slot, {cs(k): clean_df(v) for k, v in dict(m).items()})
    return adata  # usage: sanitize_slashes(adata); adata.write("clean.h5ad")
##-----------------------------------------------------------------------------------------------------------------------------------------

##FUNCTIONS FOR BASIC ANALYSIS 

    
def cumulative_gene_mass(W, alpha, gene_names, output_folder, name):

    """ saves: pleiotropy scores csv: list of genes and associated pleiotropy scores (descending order)"""

    if alpha == 0:
        module_assignments = row_max(W)
        return module_assignments

    H = W.copy()
    
    #take W, and for each row, divide each element in the row by its row sum
    H_sq = H ** 2
    
    # H_sq is a DataFrame indexed by gene names
    scores = H_sq.apply(lambda row: pleiotropy_score(row.to_numpy()), axis=1)

    # 0–1 min-max normalize (order preserved) + round to 4 decimals
    smin, smax = scores.min(), scores.max()
    scores_norm = (scores - smin) / (smax - smin) 

    pleiotropy_scores = scores_norm.round(4).reset_index()
    pleiotropy_scores.columns = ["Gene", "Score"]
    pleiotropy_scores = pleiotropy_scores.sort_values(by="Score", ascending=False)



    pleiotropy_scores.to_csv(f"{output_folder}/pleiotropy_scores_{name}.csv", index=False)


    #normalize with sum of squares 
    
    row_sums = H_sq.sum(axis=1)
    H_norm = H_sq.div(row_sums, axis=0)
    

    #sort each row and take cumulative sum, and keep the top threshold % of each row's cumulative mass
   
    for idx in H_norm.index:
        sorted_vals = H_norm.loc[idx].sort_values(ascending=False)
        cumsum = sorted_vals.cumsum()
        total = cumsum.iloc[-1]
        cutoff_n = (cumsum >= alpha * total).values.searchsorted(True) + 1
        cutoff_idx = sorted_vals.index[:cutoff_n]

        # Set everything below cutoff to 0
        H_norm.loc[idx, ~H_norm.columns.isin(cutoff_idx)] = 0

   

    #convert H_norm to a binary matrix, and multiply by W to get H_thresh
    mask = (H_norm > 0).astype(float)
    H_thresh = W * mask  # Retains original values where thresholded, zeros elsewhere


    #use the kept elements to make module assignments 
    assignments = []

    for gene_idx, gene in enumerate(gene_names):
        for module_idx in range(H_thresh.shape[1]):
            if H_thresh.iloc[gene_idx, module_idx] > 0:
                assignments.append({
                    "Gene": gene,
                    "Module": module_idx + 1  # shift module index to start from 1
                })

    module_assignments = pd.DataFrame(assignments)
    return module_assignments

def pleiotropy_score(x):
    nz = x[x!= 0]
    var = float(np.var(nz, ddof=0))
    return float(nz.size) / (var + 1.0)


#function to run modPCA, returns PCs

def runPCA(adata, module_assignments, variance_threshold=0.70):
    """
    Run PCA for each module subset of genes, keep enough PCs to explain 
    `variance_threshold` of variance, and output concatenated PCs.
    """
    print(module_assignments.shape)
    pcs_by_module = []
    
    for i in sorted(module_assignments['Module'].unique()):
        genes_k = module_assignments.loc[module_assignments['Module'] == i, "Gene"].drop_duplicates()
        adata_k = adata[:, adata.var_names.isin(genes_k)]
        print(f"Module {i} adata shape: ", adata_k.shape)
        
        # Run PCA with full rank
        sc.tl.pca(adata_k, svd_solver='randomized', zero_center=False)
        
        # Compute cumulative explained variance
        explained = adata_k.uns['pca']['variance_ratio']
        cumsum = np.cumsum(explained)
        n_comps = np.searchsorted(cumsum, variance_threshold) + 1

        print(f"Keeping {n_comps} components for module {i} to reach {variance_threshold*100:.1f}% variance")
        
        pcs_k = adata_k.obsm["X_pca"][:, :n_comps].astype(np.float32, copy=False)
        pcs_by_module.append(pcs_k)

    full_pcs = np.ascontiguousarray(np.hstack(pcs_by_module), dtype=np.float32)


    print("Ran PCA\n ")
    print("final shape of cell x pcs matrix: ", full_pcs.shape)



    return full_pcs
        

def umap(name, adata, module_assignments, color, adata_path):

    pcs = runPCA(adata, module_assignments, variance_threshold=0.70)

    adata.obsm[f"X_pcs_{name}"] = pcs 
    adata.write(adata_path)
    print("pcs saved to anndata")
    
    # Compute neighbors based on these PCs
    #sc.pp.neighbors(adata, use_rep=f"X_pcs_{name}", n_neighbors=30, key_added=f"neighbors_{name}")
    
   # Run UMAP embedding using those neighbors
    
    #sc.tl.umap(adata, neighbors_key=f"neighbors_{name}")
    #adata.obsm[f"X_umap_{name}"] = adata.obsm["X_umap"].copy()

    #sc.pl.embedding(adata, basis=f"X_umap_{name}", color=color, title=f"{name}_UMAP", palette=sc.pl.palettes.default_102, save=f"_umap")
    

    print("Done with umap!")


def reconstruction(name, adata_path, nmf_assignments, hotspot_reconstruction_path, output_folder): 

    adata = sc.read_h5ad(adata_path)

    module_genes = nmf_assignments['Gene'].unique().tolist()

    #subset the adata to only genes from module assignments (local corr matrix will leave out a few genes)
    adata_mod = adata[:, module_genes].copy()

    #compute gene means
    gene_means = pd.Series(np.asarray(adata_mod.X.mean(axis=0)).ravel(),index=adata_mod.var_names)

    #original gene expression 
    og_data = adata_mod.X.toarray() 
    # Convert to DataFrame for easier handling (assuming genes are in columns)
    og_df = pd.DataFrame(og_data, columns=adata_mod.var_names)
    n_genes = og_df.shape[1]

    #reconstruct gexp from the nmf pcs
    print("Reconstructing using NMF ModPCs...")
    reconstructed_gex = reconstruct_lr(
        adata=adata_mod,                    
        adata_og=adata_mod,          
        gene_means=gene_means,
        modular_key=f'X_pcs_{name}',
        reprocess=True,
        n_comps=100   # recompute global PCA on subset
    )

    assert list(og_df.columns) == list(reconstructed_gex.columns), "Column mismatch"

    
    #correlate the reconstructed gene expression from nmf with reconstructed gene expression from global pcs
    T_global = adata_mod.obsm["X_pca"]                  # (cells × PCs)
    W_global = adata_mod.varm["PCs"]                    # (genes × PCs)
    recon_global = T_global @ W_global.T + gene_means.values.reshape(1, -1)
    recon_global = np.clip(recon_global, a_min=0, a_max=None)
    reconstructed_glob = pd.DataFrame(recon_global, columns=adata_mod.var_names)

    assert list(reconstructed_glob.columns) == list(reconstructed_gex.columns), "Column mismatch"

    

    #correlate the reconstructed gene expression (nmf) with original gene expression
    correlation_matrix_og = np.corrcoef(og_df.values.T, reconstructed_gex.values.T)  # Transpose to correlate across genes
    corr_og = correlation_matrix_og[:n_genes, n_genes:]

    #correlate the reconstruction gene expression with reconstructed gex from global pca 
    correlation_matrix_global = np.corrcoef(reconstructed_glob.values.T, reconstructed_gex.values.T)  
    corr_global = correlation_matrix_global[:n_genes, n_genes:]

    # Per-gene correlations (diagonals)
    recon_vs_orig_diag   = np.diag(corr_og)         # Recon vs Original
    recon_vs_glob_diag  = np.diag(corr_global)        # Recon vs global


    #compare with hotspot hard module reconstructions 
    reconstructed_HS = pd.read_csv(hotspot_reconstruction_path, header=None)
    

    correlation_matrix_og_HS = np.corrcoef(og_df.values.T, reconstructed_HS.values.T)  # Transpose to correlate across genes
    corr_og_HS = correlation_matrix_og_HS[:n_genes, n_genes:]

    correlation_matrix_global_HS = np.corrcoef(reconstructed_glob.values.T, reconstructed_HS.values.T)  
    corr_global_HS = correlation_matrix_global_HS[:n_genes, n_genes:]

    hs_vs_orig_diag   = np.diag(corr_og_HS)     
    hs_vs_glob_diag  = np.diag(corr_global_HS)


    #output a csv file of each gene with its reconstruction correlation value from nmf for original
    gene_corr_df = pd.DataFrame({
        "gene": adata_mod.var_names,
        "reconstruction_correlation": recon_vs_orig_diag
    })

    # Sort highest → lowest correlation
    gene_corr_df = gene_corr_df.sort_values(
        by="reconstruction_correlation",
        ascending=False
    )

    # Output CSV path
    csv_outpath = os.path.join(
        output_folder,
        f"{name}_NMF_Gene_Reconstruction_Correlations.csv"
    )

    gene_corr_df.to_csv(csv_outpath, index=False)
    print(f"Saved per-gene reconstruction correlation CSV to:\n{csv_outpath}")


    #plot
   

    #plot original gex vs nmf pleio recon, original gex vs hotspot recon
    scatter_corr(
        corr1=recon_vs_orig_diag,
        corr2=hs_vs_orig_diag,
        title="Per-gene correlation: Module Recon Gex vs Original Gex",
        xlabel="NMF Pleio Recon",
        ylabel="Hotspot Recon ",
        outpath=os.path.join(output_folder, f"{name}_mod_vs_original_scatter_nmf.png")
    )

    #plot global recon vs nmf pleio recon, global recon vs hotspot recon
    scatter_corr(
        corr1=recon_vs_glob_diag,
        corr2=hs_vs_glob_diag,
        title="Per-gene correlation: Module Recon Gex vs Global Recon",
        xlabel="NMF Pleio Recon",
        ylabel="Hotspot Recon ",
        outpath=os.path.join(output_folder, f"{name}_mod_vs_global_scatter_nmf.png")
    )


def scatter_corr(corr1, corr2, title, xlabel, ylabel, outpath):
        """
       
        """
        x = np.asarray(corr1, dtype=float)
        y = np.asarray(corr2, dtype=float)

        # Filter NaNs consistently
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        print(f"Skew (slope of best-fit line): {slope:.4f}")

        plt.figure(figsize=(6, 8))
        plt.scatter(x, y, s=20, alpha=0.6)
        plt.plot([0, 1], [0, 1], "--", lw=2, color='red')     # 45° line
        plt.xlim(0, 1); plt.ylim(0, 1)
        plt.xlabel(xlabel, fontsize=16); plt.ylabel(ylabel, fontsize=16)
        plt.title(title, fontsize=16, pad=12)
        plt.tick_params(axis="both", which="major", labelsize=14, length=6, width=1.2)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.tight_layout()

        if outpath:
            plt.savefig(outpath, dpi=300)
        plt.show()

##ADDITIONAL ANALYSIS FUNCTIONS --------------------------------------------------------------------------------------------------------

def plot_module_distribution(module_assignments_path, title):
    # Load CSV
       # Load CSV
    df = pd.read_csv(module_assignments_path)

    # Count # of modules per gene
    module_counts = df.groupby("Gene")["Module"].nunique()

    # Count how many genes fall into each category (#modules = 1, 2, 3, ...)
    count_dist = module_counts.value_counts().sort_index()

    # Plot
    plt.figure(figsize=(6, 4), dpi=150)
    bars = plt.bar(count_dist.index, count_dist.values, edgecolor="black")

    # Label each bar with the count
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(int(height)),
            ha='center',
            va='bottom',
            fontsize=10
        )

    # Labels and title
    plt.xlabel("Number of Modules Assigned to Gene")
    plt.ylabel("Number of Genes")
    plt.title(f"{title}")
    plt.xticks(count_dist.index)
    plt.tight_layout()
    plt.show()


def plot_module_activity_by_celltype(adata, module_csv_path, cell_type_key):
    """
    Plot average module activity per cell type.

    Parameters:
        adata: AnnData object
        module_csv_path: Path to CSV with ['Gene', 'Module']
        cell_type_key: Key in adata.obs with cell type labels
    """
    # ---- Load module assignments ----
    df = pd.read_csv(module_csv_path)
    df["Module"] = df["Module"].astype(int)
    modules = sorted(df["Module"].unique())
    module_to_genes = {
        str(m): sorted(df[df["Module"] == m]["Gene"].tolist()) for m in modules
    }

    # ---- Score modules ----
    adata_tmp = adata.copy()
    score_cols = []
    for m in modules:
        score_name = f"module_{m}_score"
        sc.tl.score_genes(
            adata_tmp,
            gene_list=module_to_genes[str(m)],
            score_name=score_name,
            use_raw=False,
            layer=None,
        )
        score_cols.append(score_name)

    # ---- Compute mean scores per cell type ----
    celltype_groups = adata_tmp.obs.groupby(cell_type_key)
    mean_scores = celltype_groups[score_cols].mean()

    # ---- Convert to heatmap matrix ----
    matrix = mean_scores.T  # shape: (n_modules, n_celltypes)
    matrix.index = [int(s.replace("module_", "").replace("_score", "")) for s in matrix.index]
    matrix = matrix.sort_index()

    fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(matrix.columns)), 10))

    # Plot the heatmap WITHOUT default labels
    sns.heatmap(
        matrix,
        cmap="magma",
        vmin=0.0,
        vmax=0.4,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Avg module score"},
        ax=ax
    )

   
    # Set label positions at cell centers
    ax.set_xticks(np.arange(matrix.shape[1]) + 0.5)
    ax.set_yticks(np.arange(matrix.shape[0]) + 0.5)

    # Set label text
    ax.set_xticklabels(matrix.columns, fontsize=12, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_yticklabels(matrix.index, fontsize=12)

    # Remove tick MARKS but keep tick LABELS
    ax.tick_params(axis='both', length=0)
    # ---------------------------------------------------

    # Axis labels and title
    plt.xlabel("Cell type", fontsize=14, labelpad=10)
    plt.ylabel("Module", fontsize=14, labelpad=10)
    plt.title("Average module activity per cell type", fontsize=16, pad=15)

    # Colorbar formatting
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Avg module score", fontsize=14)

    plt.tight_layout()
    plt.show()


    
def gene_expression_by_celltype(gene_list, adata_path, color, fig_name):
    """
    Plot average expression of a given list of genes by cell type.
    The 'color' parameter should match the column name in adata.obs that encodes cell types.
    """
    # --- Load the data ---
    adata = sc.read_h5ad(adata_path)

    # --- Compute average expression per celltype (using column given by `color`) ---
    celltype_means = (
        adata[:, gene_list].to_df()
        .join(adata.obs[color])
        .groupby(color)[gene_list]
        .mean()
    )

   

    # --- Define color palette for celltypes ---
    unique_types = celltype_means.index.tolist()
    palette = sns.color_palette("tab20", len(unique_types))
    lut = dict(zip(unique_types, palette))

       # --- Instead of a clustermap, build a color-tinted heatmap manually ---
    from matplotlib.colors import to_rgb
    import numpy as np

    # Normalize expression (0–1 per gene)
    data_norm = celltype_means.copy()
    for gene in data_norm.columns:
        col = data_norm[gene]
        data_norm[gene] = (col - col.min()) / (col.max() - col.min())
    

    # Create RGB array: genes × celltypes × 3
    rgb_array = np.zeros((data_norm.shape[1], data_norm.shape[0], 3))
    for j, ct in enumerate(celltype_means.index):
        base_color = np.array(to_rgb(lut[ct]))      # base hue for the cell type
        for i in range(data_norm.shape[1]):
            val = data_norm.iloc[j, i]
            # blend toward white for lower expression
            rgb_array[i, j, :] = base_color * val + (1 - val) * np.array([1, 1, 1])

    # --- Plot the RGB matrix ---
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.imshow(rgb_array, aspect="auto")

    # --- Axis labels and ticks ---
    ax.set_xticks(np.arange(len(celltype_means.index)))
    ax.set_xticklabels(celltype_means.index, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(celltype_means.columns)))
    ax.set_yticklabels(celltype_means.columns, fontsize=10)
    ax.set_xlabel("Cell type", fontsize=12)
    ax.set_ylabel("Gene", fontsize=12)
    plt.title("Gene Expression by Cell Type", pad=20)

    # --- Add a top strip showing base color of each cell type ---
    for j, ct in enumerate(celltype_means.index):
        ax.add_patch(
            plt.Rectangle(
                (j - 0.5, -0.6), 1, 0.4, color=lut[ct],
                transform=ax.transData, clip_on=False
            )
        )

    plt.tight_layout()
    plt.show()
    plt.savefig(f"Pleiotropy_Tests/{fig_name}.png", bbox_inches="tight", dpi=300)
    plt.close()




def plot_module_umaps(adata, adata_global, umap_key, module_assignments, output_folder):

    adata = sc.read_h5ad(adata)
    adata_global = sc.read_h5ad(adata_global)

    adata.obsm["X_umap"] = adata_global.obsm[umap_key]


    modules = sorted(module_assignments["Module"].unique())
    ncols = 4
    nrows = int(np.ceil(len(modules) / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 4*nrows))
    axes = axes.flatten()


    for i, module_id in enumerate(modules):
        ax = axes[i]
        genes_in_module = module_assignments.loc[module_assignments["Module"] == module_id, "Gene"].tolist()
        print(f"Module {module_id} has {len(genes_in_module)} genes in adata")


        score_name = f"module_{module_id}_score"
        sc.tl.score_genes(adata, gene_list=genes_in_module, score_name=score_name)

        # use Scanpy’s built-in plotting function but draw into existing axes
        sc.pl.umap(
            adata,
            color=score_name,
            vmin="p1",
            vmax="p99",
            ax=ax,
            show=False,
            title=f"Module {module_id}",
        )

    # Hide any extra unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    out_path = os.path.join(output_folder, "combined_module_umaps.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved combined UMAP figure to {out_path}")
    



#this function will assign modules based on the maximum of each row (gene) in the loading matrix (hard clustering)
def row_max(W):
    W_array = W.values
    assignments = np.argmax(W_array, axis=1) + 1 #so that module assignments start at 1

    module_assignments =  pd.DataFrame({
        "Gene": W.index,
        "Module": assignments
    })


    return module_assignments



#----------------------------------------------------------------------
# Only run this block when executing pleiotropy_tests.py directly
if __name__ == "__main__":

    # change these depending on run
    adata = sc.read_h5ad('Pleiotropy_Tests/liver_analysis.h5ad')
    #adata = sanitize_slashes(adata)

    adata = adata[adata.obs_names != "UNASSIGNED"].copy()
    output_folder = '/data1/peerd/nandulaa/Pleiotropy_Tests'
    sc.settings.figdir = output_folder
    
    
    local_corr_clipped = pd.read_csv('Liver/fatty_liver_xenium_modpca_danb_unweight_pca_local_corr_clipped.csv', index_col=0)
    local_corr_clipped.index = local_corr_clipped.columns 

    local_corr_unclipped = pd.read_csv('Liver/fatty_liver_xenium_modpca_danb_unweight_pca_local_corr.csv', index_col=0)
    local_corr_unclipped.index = local_corr_unclipped.columns
    
    hotspot_assignments = pd.read_csv("Liver/fatty_liver_xenium_modpca_pca_genemodules_thresh10.csv", usecols=[0,5])

    # Keep only valid genes in Hotspot
    hotspot_assignments = hotspot_assignments.dropna(subset="Module")
    hotspot_assignments = hotspot_assignments[hotspot_assignments["Module"] != -1].copy()

    gene_names = local_corr_clipped.columns


    #CHANGE PARAMETERS
    module_number = 14
    threshold = 0.7
    name = f'liver_genemass_{threshold}_#{module_number}'

    adata_path = Path(output_folder) / "liver_analysis.h5ad"
    color = 'Cell_Type'

    
    
    
    
    
    

    #W = runSymNMF(local_corr_clipped.values, module_number, name, output_folder, gene_names, save=True, init_matrix = None)
    W = pd.read_csv(f"{output_folder}/{name}_loadings_matrix.csv", index_col=0)
    module_assignments, pleiotropic_genes, unassigned_genes  =  assignModules(W = W, gene_names = gene_names, func = cumulative_gene_mass, name = name, output_folder = output_folder, threshold = threshold, squared = True)
    
    umap(name = name, adata = adata, module_assignments = module_assignments, color = color, adata_path = adata_path)
    adata.write(adata_path)



















