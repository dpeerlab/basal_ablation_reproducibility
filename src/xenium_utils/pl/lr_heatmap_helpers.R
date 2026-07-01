# LR Heatmap Helper Functions
# Plotting utilities for ligand-receptor trajectory analysis

library(ComplexHeatmap)
library(circlize)
library(grid)
library(dplyr)

# --- Utility Functions ---

#' Smooth rows with 3-bin rolling average
#' @param m Matrix to smooth
#' @return Smoothed matrix
smooth_rows <- function(m) {
  n_cols <- ncol(m)
  smoothed <- m
  for (i in seq_len(nrow(m))) {
    for (j in seq_len(n_cols)) {
      neighbors <- max(1, j-1):min(n_cols, j+1)
      smoothed[i, j] <- mean(m[i, neighbors], na.rm = TRUE)
    }
  }
  smoothed
}

#' Z-score normalize rows
#' @param mat Matrix to normalize
#' @return Z-scored matrix
zscore_rows <- function(mat) {
  t(apply(mat, 1, function(x) {
    if (sd(x, na.rm = TRUE) == 0) return(rep(0, length(x)))
    (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
  }))
}

#' Find onset bin where derivative exceeds threshold
#' @param x Numeric vector of values
#' @return Index of onset bin
get_onset_bin <- function(x) {
  if (all(is.na(x)) || length(x) < 2) return(Inf)
  deriv <- diff(x)
  onset <- which(deriv > 0.05 * max(abs(deriv), na.rm = TRUE))[1]
  if (is.na(onset)) return(Inf)
  onset
}

#' Determine trend direction (positive or negative)
#' @param x Numeric vector of values
#' @return "positive" or "negative"
get_trend <- function(x) {
  if (all(is.na(x))) return("positive")
  n <- length(x)
  first_third <- mean(x[1:max(1, floor(n/3))], na.rm = TRUE)
  last_third <- mean(x[max(1, ceiling(2*n/3)):n], na.rm = TRUE)
  if (last_third > first_third) "positive" else "negative"
}

# --- Heatmap Functions ---

#' Cell-type-specific LR heatmap 
#'
#' Creates heatmap with rows = gene (cell_type), columns = bins, row_split by receptor
#' Uses cell type bin data (sender_celltype_bins for ligands, rec_celltype_bins for receptors)
#'
#' @param sig_df Data frame with significant LR pairs
#' @param sender_celltype_bins Named list of ligand bin data by cell type
#' @param rec_celltype_bins Named list of receptor bin data by cell type
#' @param cov_bins Data frame with bin covariates
#' @param DC1_N_BINS Number of DC1 bins
#' @param title_suffix Suffix for heatmap title
#' @param file_suffix Suffix for output filename
#' @param FIG_DIR Output directory for figure
#' @param x_axis_label Label for x-axis (default: "DC1 trajectory")
#' @param normalize Normalization method: "zscore" (default) or "minmax"
#' @param receptor_order Optional vector specifying receptor order
#' @param receptor_groups Optional named vector to merge receptors
#' @param bin_col Column name for bin identifier (default: "DC1_bin30")
#' @param format Output format: "svg" (default) or "pdf"
#' @return ComplexHeatmap object
plot_lr_heatmap_celltype <- function(sig_df, sender_celltype_bins, rec_celltype_bins, cov_bins,
                                      DC1_N_BINS, title_suffix, file_suffix, FIG_DIR,
                                      x_axis_label = "DC1 trajectory",
                                      normalize = "zscore", receptor_order = NULL,
                                      receptor_groups = NULL, bin_col = "DC1_bin30",
                                      format = "svg") {
  format <- match.arg(format, c("svg", "pdf"))
  if (nrow(sig_df) == 0) {
    cat("No pairs for", title_suffix, "— skipping.\n")
    return(NULL)
  }

  # Get bin centers for column labels
  bin_ids <- sort(unique(cov_bins[[bin_col]]))
  n_bins  <- length(bin_ids)
  bin_centres <- (bin_ids - 0.5) / DC1_N_BINS

  # Get unique receptors
  receptors <- if (!is.null(receptor_order)) {
    intersect(receptor_order, unique(sig_df$receptor))
  } else {
    unique(sig_df$receptor)
  }

  # Build row info: gene, celltype, receptor_group, gene_type, bin_key
  row_info <- data.frame(
    gene = character(),
    celltype = character(),
    receptor_group = character(),
    gene_type = character(),
    bin_key = character(),
    stringsAsFactors = FALSE
  )

  for (rec in receptors) {
    rec_df <- sig_df[sig_df$receptor == rec, ]

    # Get unique ligand-sender combinations
    lig_sender <- unique(rec_df[, c("ligand", "sender_celltype")])
    for (i in seq_len(nrow(lig_sender))) {
      lig <- lig_sender$ligand[i]
      ct <- lig_sender$sender_celltype[i]
      bin_key <- paste(ct, lig, sep = "__")
      if (bin_key %in% names(sender_celltype_bins)) {
        row_info <- rbind(row_info, data.frame(
          gene = lig, celltype = ct, receptor_group = rec,
          gene_type = "ligand", bin_key = bin_key, stringsAsFactors = FALSE
        ))
      }
    }

    # Get unique receptor-receiver combinations
    rec_recv <- unique(rec_df[, c("receptor", "receiver_celltype")])
    for (i in seq_len(nrow(rec_recv))) {
      receptor <- rec_recv$receptor[i]
      ct <- rec_recv$receiver_celltype[i]
      bin_key <- paste(receptor, ct, sep = "__")
      if (bin_key %in% names(rec_celltype_bins)) {
        row_info <- rbind(row_info, data.frame(
          gene = receptor, celltype = ct, receptor_group = rec,
          gene_type = "receptor", bin_key = bin_key, stringsAsFactors = FALSE
        ))
      }
    }
  }

  if (nrow(row_info) == 0) {
    cat("No valid rows for", title_suffix, "— skipping.\n")
    return(NULL)
  }

  # Apply receptor grouping if specified
  if (!is.null(receptor_groups)) {
    row_info$receptor_group <- ifelse(
      row_info$receptor_group %in% names(receptor_groups),
      receptor_groups[row_info$receptor_group],
      row_info$receptor_group
    )
  }

  # Build matrix
  mat <- matrix(NA, nrow = nrow(row_info), ncol = n_bins)
  row_labels <- paste0(row_info$gene, " (", row_info$celltype, ")")
  rownames(mat) <- row_labels
  colnames(mat) <- sprintf("%.2f", bin_centres)

  for (i in seq_len(nrow(row_info))) {
    gtype <- row_info$gene_type[i]
    bin_key <- row_info$bin_key[i]

    if (gtype == "ligand") {
      bin_df <- sender_celltype_bins[[bin_key]]
      val_col <- "lig_expr_bin"
    } else {
      bin_df <- rec_celltype_bins[[bin_key]]
      val_col <- "rec_frac_bin"
    }

    if (!is.null(bin_df) && val_col %in% names(bin_df)) {
      for (j in seq_along(bin_ids)) {
        idx <- which(bin_df[[bin_col]] == bin_ids[j])
        if (length(idx) > 0) mat[i, j] <- bin_df[[val_col]][idx]
      }
    }
  }

  # Smooth and normalize
  mat <- smooth_rows(mat)

  if (normalize == "minmax") {
    norm_mat <- t(apply(mat, 1, function(x) {
      rng <- range(x, na.rm = TRUE)
      if (rng[2] == rng[1]) return(rep(0.5, length(x)))
      (x - rng[1]) / (rng[2] - rng[1])
    }))
    legend_name <- "Min-max"
    col_fun <- colorRamp2(c(0, 0.5, 1), c("#f7f7f7", "#fc8d59", "#b30000"))
  } else {
    norm_mat <- zscore_rows(mat)
    legend_name <- "Z-score"
    col_fun <- colorRamp2(c(-2, 0, 2), c("#4575b4", "white", "#d73027"))
  }
  colnames(norm_mat) <- colnames(mat)

  # Determine trend direction and onset bin for ordering
  row_info$onset_bin <- apply(norm_mat, 1, get_onset_bin)
  row_info$trend <- apply(norm_mat, 1, get_trend)
  row_info <- row_info %>%
    group_by(receptor_group) %>%
    arrange(desc(gene_type == "ligand"), desc(trend == "negative"), onset_bin) %>%
    ungroup()
  new_row_labels <- paste0(row_info$gene, " (", row_info$celltype, ")")
  norm_mat <- norm_mat[new_row_labels, , drop = FALSE]

  # Row annotation for gene type
  row_ha <- rowAnnotation(
    type = row_info$gene_type,
    col = list(type = c("ligand" = "forestgreen", "receptor" = "#fc8d59")),
    show_legend = TRUE,
    annotation_name_side = "top"
  )

  # Receptor group levels
  rec_levels <- if (!is.null(receptor_order)) {
    intersect(receptor_order, unique(row_info$receptor_group))
  } else {
    unique(row_info$receptor_group)
  }

  # Create heatmap
  ht <- Heatmap(
    norm_mat,
    col = col_fun,
    name = legend_name,
    row_split = factor(row_info$receptor_group, levels = rec_levels),
    row_title_rot = 0,
    row_title_gp = gpar(fontsize = 10, fontface = "bold"),
    cluster_rows = FALSE,
    cluster_columns = FALSE,
    show_row_names = TRUE,
    row_names_side = "left",
    row_names_gp = gpar(fontsize = 8),
    show_column_names = FALSE,
    column_names_gp = gpar(fontsize = 7),
    column_names_rot = 45,
    column_title = x_axis_label,
    column_title_side = "bottom",
    rect_gp = gpar(col = "lightgrey", lwd = 0.3),
    na_col = "grey90",
    left_annotation = row_ha,
    row_gap = unit(2, "mm")
  )

  # Draw
  draw(ht,
       column_title = paste("LR pairs (cell type) —", title_suffix),
       column_title_gp = gpar(fontsize = 12, fontface = "bold"),
       padding = unit(c(2, 2, 2, 2), "mm"))

  # Save figure
  out_path <- file.path(FIG_DIR, paste0("lr_celltype_heatmap_", file_suffix, ".", format))
  n_rows <- nrow(row_info)
  fig_width <- 10
  fig_height <- max(4, n_rows * 0.25 + 2)

  if (format == "pdf") {
    pdf(out_path, width = fig_width, height = fig_height)
  } else {
    svg(out_path, width = fig_width, height = fig_height)
  }
  draw(ht,
       column_title = paste("LR pairs (cell type) —", title_suffix),
       column_title_gp = gpar(fontsize = 12, fontface = "bold"),
       padding = unit(c(2, 2, 2, 2), "mm"))
  dev.off()
  cat("Cell type heatmap saved to:", out_path, "\n")

  return(ht)
}

#' Visualize gene derivatives along trajectory
#'
#' @param genes Character vector of gene names
#' @param gene_type "ligand" or "receptor"
#' @param sender_celltype_bins Named list of ligand bin data
#' @param rec_celltype_bins Named list of receptor bin data
#' @param cov_bins Data frame with bin covariates
#' @param DC1_N_BINS Number of DC1 bins
#' @param celltype Cell type to plot
#' @param smooth Whether to smooth values
#' @param bin_col Column name for bin identifier
#' @return Grid of ggplots
plot_gene_derivatives <- function(genes, gene_type = "ligand",
                                   sender_celltype_bins, rec_celltype_bins, cov_bins,
                                   DC1_N_BINS, celltype, smooth = TRUE,
                                   bin_col = "DC1_bin30") {
  require(ggplot2)
  require(gridExtra)

  bin_ids <- sort(unique(cov_bins[[bin_col]]))
  bin_centres <- (bin_ids - 0.5) / DC1_N_BINS

  plot_data <- list()

  for (gene in genes) {
    bin_key <- paste(celltype, gene, sep = "__")

    if (gene_type == "ligand" && bin_key %in% names(sender_celltype_bins)) {
      df <- sender_celltype_bins[[bin_key]]
      val_col <- "lig_expr_bin"
    } else if (gene_type == "receptor" && bin_key %in% names(rec_celltype_bins)) {
      df <- rec_celltype_bins[[bin_key]]
      val_col <- "rec_frac_bin"
    } else {
      cat("Gene", gene, "not found in", gene_type, "bins for", celltype, "\n")
      next
    }

    # Extract values
    vals <- sapply(bin_ids, function(b) {
      idx <- which(df[[bin_col]] == b)
      if (length(idx) > 0) df[[val_col]][idx] else NA
    })

    # Smooth if requested
    if (smooth && length(vals) >= 3) {
      smoothed <- vals
      for (j in seq_along(vals)) {
        neighbors <- max(1, j-1):min(length(vals), j+1)
        smoothed[j] <- mean(vals[neighbors], na.rm = TRUE)
      }
      vals <- smoothed
    }

    # Z-score normalize
    if (sd(vals, na.rm = TRUE) > 0) {
      vals <- (vals - mean(vals, na.rm = TRUE)) / sd(vals, na.rm = TRUE)
    }

    # Compute derivatives
    d1 <- c(NA, diff(vals))
    d2 <- c(NA, diff(d1))

    plot_data[[gene]] <- data.frame(
      gene = gene,
      bin = bin_centres,
      value = vals,
      deriv1 = d1,
      deriv2 = d2
    )
  }

  if (length(plot_data) == 0) return(NULL)

  df_all <- do.call(rbind, plot_data)

  p1 <- ggplot(df_all, aes(x = bin, y = value, color = gene)) +
    geom_line(linewidth = 1) +
    geom_point(size = 2) +
    labs(x = "DC1", y = "Z-score", title = "Value") +
    theme_bw() +
    theme(legend.position = "bottom")

  p2 <- ggplot(df_all, aes(x = bin, y = deriv1, color = gene)) +
    geom_line(linewidth = 1) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
    geom_point(size = 2) +
    labs(x = "DC1", y = "dZ/dbin", title = "1st Derivative") +
    theme_bw() +
    theme(legend.position = "none")

  p3 <- ggplot(df_all, aes(x = bin, y = deriv2, color = gene)) +
    geom_line(linewidth = 1) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
    geom_point(size = 2) +
    labs(x = "DC1", y = "d²Z/dbin²", title = "2nd Derivative") +
    theme_bw() +
    theme(legend.position = "none")

  gridExtra::grid.arrange(p1, p2, p3, ncol = 1, heights = c(1.2, 1, 1))
}

#' Plot cell type composition heatmap along DC1
#'
#' @param comp_df Data frame with cell type fractions per bin
#' @param title Heatmap title
#' @param file_suffix Suffix for output filename
#' @param FIG_DIR Output directory
#' @param cluster_rows Whether to cluster rows
#' @param bin_col Column name for bin identifier
#' @return ComplexHeatmap object
plot_celltype_composition_heatmap <- function(comp_df, title, file_suffix, FIG_DIR,
                                               cluster_rows = FALSE, bin_col = "DC1_bin") {
  # Pivot to matrix: rows = cell types, cols = bins
  mat <- comp_df %>%
    tidyr::pivot_wider(names_from = all_of(bin_col), values_from = frac, values_fill = 0) %>%
    tibble::column_to_rownames("cell_annotation") %>%
    as.matrix()

  # Smooth and z-score
  mat <- smooth_rows(mat)
  zmat <- zscore_rows(mat)

  # Order rows by bin of max value
  if (!cluster_rows) {
    max_bins <- apply(zmat, 1, which.max)
    row_order <- order(max_bins)
    zmat <- zmat[row_order, , drop = FALSE]
  }

  col_fun <- colorRamp2(c(-2, 0, 2), c("#4575b4", "white", "#d73027"))

  ht <- Heatmap(
    zmat,
    col = col_fun,
    name = "Z-score",
    cluster_rows = cluster_rows,
    cluster_columns = FALSE,
    show_row_names = TRUE,
    row_names_side = "left",
    row_names_gp = gpar(fontsize = 9),
    show_column_names = TRUE,
    column_names_gp = gpar(fontsize = 7),
    column_names_rot = 45,
    column_title = "DC1 bin",
    column_title_side = "bottom",
    rect_gp = gpar(col = "lightgrey", lwd = 0.3),
    na_col = "grey90"
  )

  draw(ht, column_title = title, column_title_gp = gpar(fontsize = 12, fontface = "bold"))

  # Save
  svg_path <- file.path(FIG_DIR, paste0("celltype_composition_", file_suffix, ".svg"))
  svg(svg_path, width = 10, height = max(4, nrow(zmat) * 0.25 + 2))
  draw(ht, column_title = title, column_title_gp = gpar(fontsize = 12, fontface = "bold"))
  dev.off()
  cat("Composition heatmap saved to:", svg_path, "\n")

  return(ht)
}

#' Plot cytokine program correlation heatmap
#'
#' @param corr_df Data frame with correlation results
#' @param expr_bins Named list of expression bin data
#' @param cov_bins Data frame with bin covariates
#' @param DC1_N_BINS Number of DC1 bins
#' @param title Heatmap title
#' @param file_suffix Suffix for output filename
#' @param FIG_DIR Output directory
#' @param bin_col Column name for bin identifier
#' @return ComplexHeatmap object
plot_cytokine_correlation_heatmap <- function(corr_df, expr_bins, cov_bins,
                                               DC1_N_BINS, title, file_suffix, FIG_DIR,
                                               bin_col = "DC1_bin30") {
  if (nrow(corr_df) == 0) {
    cat("No correlations for", title, "— skipping.\n")
    return(NULL)
  }

  bin_ids <- sort(unique(cov_bins[[bin_col]]))
  n_bins <- length(bin_ids)
  bin_centres <- (bin_ids - 0.5) / DC1_N_BINS

  # Build matrix: rows = gene (celltype), cols = bins
  row_labels <- paste0(corr_df$gene, " (", corr_df$cell_type, ")")
  mat <- matrix(NA, nrow = nrow(corr_df), ncol = n_bins)
  rownames(mat) <- row_labels
  colnames(mat) <- sprintf("%.2f", bin_centres)

  for (i in seq_len(nrow(corr_df))) {
    gene <- corr_df$gene[i]
    ct <- corr_df$cell_type[i]
    bin_key <- paste(ct, gene, sep = "__")

    if (bin_key %in% names(expr_bins)) {
      bin_df <- expr_bins[[bin_key]]
      for (j in seq_along(bin_ids)) {
        idx <- which(bin_df[[bin_col]] == bin_ids[j])
        if (length(idx) > 0) mat[i, j] <- bin_df$lig_expr_bin[idx]
      }
    }
  }

  # Smooth and z-score
  mat <- smooth_rows(mat)
  zmat <- zscore_rows(mat)

  col_fun <- colorRamp2(c(-2, 0, 2), c("#4575b4", "white", "#d73027"))

  # Row annotation for program
  if ("program" %in% names(corr_df)) {
    row_ha <- rowAnnotation(
      program = corr_df$program,
      show_legend = TRUE,
      annotation_name_side = "top"
    )
    row_split <- factor(corr_df$program)
  } else {
    row_ha <- NULL
    row_split <- NULL
  }

  ht <- Heatmap(
    zmat,
    col = col_fun,
    name = "Z-score",
    row_split = row_split,
    cluster_rows = FALSE,
    cluster_columns = FALSE,
    show_row_names = TRUE,
    row_names_side = "left",
    row_names_gp = gpar(fontsize = 8),
    show_column_names = FALSE,
    column_title = "DC1 trajectory",
    column_title_side = "bottom",
    rect_gp = gpar(col = "lightgrey", lwd = 0.3),
    na_col = "grey90",
    left_annotation = row_ha,
    row_gap = unit(2, "mm")
  )

  draw(ht, column_title = title, column_title_gp = gpar(fontsize = 12, fontface = "bold"))

  # Save
  svg_path <- file.path(FIG_DIR, paste0("cytokine_corr_heatmap_", file_suffix, ".svg"))
  svg(svg_path, width = 10, height = max(4, nrow(zmat) * 0.25 + 2))
  draw(ht, column_title = title, column_title_gp = gpar(fontsize = 12, fontface = "bold"))
  dev.off()
  cat("Cytokine correlation heatmap saved to:", svg_path, "\n")

  return(ht)
}
