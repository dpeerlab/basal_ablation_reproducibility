#!/usr/bin/env python
# coding: utf-8

# ### Import Stuff

import symNMF
import numpy as np 
import pandas as pd 
import scipy
from sklearn import neighbors
import sklearn
import os

import matplotlib.pyplot as plt


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

from pleiotropy import *

plt.rcParams['figure.dpi'] = 300


# ### Set paths       

adata = sc.read_h5ad('BA_mingle_preproc_responsePruned_20260511.h5ad')      ##TO DO: set this to the adata that you want to read from 

output_folder = "mingle_adata/NMF_pleio"          ##TO DO: set the output folder path where you want results to go 
sc.settings.figdir = output_folder

color = 'cell_annotation'                                              ##TO DO: set this to the key for the .obs cell type column you want to use for plotting 

## update
local_corr_unclipped = pd.read_csv('mingle_adata/hotspot/BA_mingle_preproc_20260227_downsampled_allCT_v2_hotspot_local_correlations.csv', index_col=0) ##TO DO: set this to path of the local correlation matrix from hotspot
local_corr_unclipped.index = local_corr_unclipped.columns

local_corr_clipped = local_corr_unclipped.clip(lower=0)

gene_names = local_corr_unclipped.columns


# ### Set NMF parameters
# 
# Set the module_number parameter to the number of modules you would like. Set alpha to a value closer to 1 if you would like more pleiotropic modules, and set it closer to 0 if you would like less pleiotropy and more stringent modules. 

module_number = 25 #TO DO
alpha = 0.4 #TO DO
name = f'BA_pleio_{alpha}_{module_number}mods' ## TO DO: prefix name to use for all outputs 
adata_path = Path(output_folder) / f"BA_mingle_preproc_responsePruned_20260511_a{alpha}_m{module_number}.h5ad"         ##TO DO: set this to the path of the adata that you want to write to (save umaps, neighbors, etc)


# ### Run SymNMF!!

W = runSymNMF(local_corr_clipped.values, module_number, name, output_folder, gene_names, save=True, init_matrix = None)


# ### Assign modules using cumulative sum method based on SymNMF results
# 
# This code cell will save three csv files: 1) module_assignments: list of genes with their associated modules, sorted by gene 2) pleiotropy_scores: a list of genes and an associated Pleiotropy Score for each gene, which reflects how pleiotropic a gene is. This csv is ordered by Pleiotropy Score in descending order. 3) module_assignments_grouped: list of genes with associated modules, grouped by module.
# 
# The loadings were loaded from a previous modPCA run that was done with more cells. 

W = pd.read_csv(f"{output_folder}/{name}_loadings_matrix.csv", index_col=0) ##Read the loadings matrix you just produced using symNMF

module_assignments =  assignModules(W = W, gene_names = gene_names, name = name, output_folder = output_folder, alpha = alpha)

print(module_assignments.head())


# print each module
for module in range(module_number):
    print(f"Module {module}:")
    print(module_assignments[module_assignments['Module'] == module]['Gene'].values)


# This cell will print a distribution of the number of assignments for all the genes in the dataset. 

plot_module_distribution(f"{output_folder}/module_assignments_{name}.csv", title="Distribution of Module Assignments")


# ### Compute modPCA, run neighbors, compute and plot UMAP 
# 

# set .X to log normalized counts
adata.X = adata.layers['log1p_norm'].copy()


pcs = runPCA(adata, module_assignments, variance_threshold=0.70)

adata.obsm[f"X_pcs_{name}"] = pcs 


adata

# write out
adata.write('mingle_adata/BA_mingle_preproc_responsePruned_20260511.h5ad')


