# Basal Ablation Reproducibility

Code and analysis related to Singhal, Ryan, Rose, Styers et al manuscript about ablation of the basal state in PDAC. 

This repo is organized into directories reflecting Rmd or Jupyter notebooks (notebooks), processing scripts (scripts), or functions/modules (src). 

Below is a description of the directory contents. 

# Notebooks

## DE

Code related to differential expression in Fig S4A. 

*niche_emb*

Code related to filtering and construction of niche trajectories from niche embedding outputs of Wormhole. Notebooks relate to either the untreated setting (`notebooks/niche_emb/BA_mingle_untreated_niche_trajectory.ipynb`, Fig. 4B) or after ablation setting (`notebooks/niche_emb/BA_mingle_full_epi_trajectory_responsePruned.ipynb`, `notebooks/niche_emb/BA_mingle_classical_niche_trajectory.ipynb`, Fig. 6A-C). 

*rl*

Code related to ligand and receptor, effector cell frequency or composition, and gene to DC correlation analysis along untreated or classical response axis referenced in Fig. 4C-F, Fig. 5A, S3, S5, S7, S8. 

# Scripts

## Cell annotation

Code used for general cell clustering and annotation of cell lineage (i.e. Myeloid) in Xenium data (`cluster_lineage.py`) or cancer state classifier using CellTypist (`train_cancer_state_classifier.py`). 

## niche_emb

Code used to generate Wormhole embeddings in the untreated (`train_model_BA_untreated_20260514_a0.4_m25.py`) or ablation response (`train_model_BA_responsePruned_20260511_a0.4_m25.py`)settings. 

## preprocessing

Scripts to:

* aggregate information from spatial neighborhoods (`define_nhoods_gpu.py`)
* annotate lymph node and parenchyma (`in_silico_dissection.py`)
* compute modPCs (`NMF_Pleio_BA_untreated.py`, `NMF_Pleio_BA_responsePruned.py`, `pleiotropy.py`, `run_hotspot_modPCA.py`, `symNMF.py`)
* segger preprocessing commands (`sbatch_segger.sh`)

# src

## pl

Helper functions for plotting ligand-receptor heatmaps. 

## pp

Functions related to niche filtering for trajectory construcion (`niche_purity.py`), cell clustering (`preprocess_rapids.py`), or scoring gene set expression (`signature_score.py`). 

