#!/usr/bin/env Rscript
# Export Papalexi Perturb-seq dataset to Python-readable format.
# Outputs: data/expression_matrix.mtx, genes.csv, cells.csv,
#          cell_perturbations.csv, gene_module_index.csv

library(Matrix)

data_dir <- "/project/xuanyao/jiaming/Getting_started/data/papalexi"
out_dir  <- "/project/xuanyao/jiaming/deeplearning/data"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

cat("Loading cell covariates (for cell barcodes)...\n")
cell_cov <- readRDS(file.path(data_dir, "cell_covariate.rds"))
cell_barcodes <- rownames(cell_cov)
cat(sprintf("Number of cells: %d\n", length(cell_barcodes)))

cat("Loading expression matrix...\n")
expr <- readRDS(file.path(data_dir, "normalized_hallmark_response_matrix.rds"))
cat(sprintf("Expression matrix (genes × cells): %d × %d\n", nrow(expr), ncol(expr)))
cat(sprintf("Value range: [%.4e, %.4f]\n", min(expr@x), max(expr@x)))
stopifnot(ncol(expr) == length(cell_barcodes))

# Transpose to cells × genes for Python convention
expr_t <- t(expr)
cat("Writing expression_matrix.mtx (cells × genes)...\n")
writeMM(expr_t, file.path(out_dir, "expression_matrix.mtx"))
write.csv(data.frame(gene = rownames(expr)), file.path(out_dir, "genes.csv"), row.names = FALSE)
write.csv(data.frame(cell = cell_barcodes), file.path(out_dir, "cells.csv"), row.names = FALSE)

cat("Writing cell covariates...\n")
write.csv(cell_cov, file.path(out_dir, "cell_covariates.csv"))

cat("Loading gRNA assignment matrix...\n")
grna_assign <- readRDS(file.path(data_dir, "union_grna_assignment_matrix.rds"))
cat(sprintf("gRNA assignment matrix (targets × cells): %d × %d\n", nrow(grna_assign), ncol(grna_assign)))
cat("Targets:", rownames(grna_assign), "\n")

# Assign each cell to its perturbation target (argmax across rows)
pert_per_cell <- apply(grna_assign, 2, function(col) {
  nonzero <- which(col > 0)
  if (length(nonzero) == 0) return("unknown")
  rownames(grna_assign)[nonzero[which.max(col[nonzero])]]
})

# Keep individual NTg guides as separate perturbation classes (used as negative controls).
# They share the "NTC" prefix so downstream code can identify them as controls.

cat("Perturbation distribution:\n")
tbl <- sort(table(pert_per_cell), decreasing = TRUE)
print(tbl)

pert_df <- data.frame(cell = cell_barcodes, perturbation = pert_per_cell)
write.csv(pert_df, file.path(out_dir, "cell_perturbations.csv"), row.names = FALSE)

cat("Loading gene module index matrix...\n")
gene_module <- readRDS(file.path(data_dir, "gene_module_index_matrix.rds"))
cat(sprintf("Gene module index (genes × pathways): %d × %d\n", nrow(gene_module), ncol(gene_module)))
write.csv(as.data.frame(gene_module), file.path(out_dir, "gene_module_index.csv"))

cat("\nAll data exported to:", out_dir, "\n")
cat("Files:\n")
cat(paste(" -", list.files(out_dir)), sep = "\n")
