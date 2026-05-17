# Perturb-seq CVAE

A Conditional Variational Autoencoder (CVAE) for disentangling perturbation effects from
cell-intrinsic variation in Perturb-seq single-cell RNA-seq data.

## Project Overview

Perturb-seq couples CRISPR perturbations with single-cell transcriptomics, giving a
gene-expression readout for each perturbed cell. Naive analysis conflates two sources of
variation that are hard to separate:

1. **Perturbation effect** — the transcriptional change caused by knocking out a gene
2. **Cell-intrinsic state** — cell-cycle phase, sequencing depth, batch, and other
   cell-to-cell variation unrelated to the perturbation

This project trains a CVAE that conditions on the perturbation identity (and optional
technical covariates) so the latent space `z` captures only the residual cell-state
variation. The decoder can then be used for in-silico perturbation: fix `z` to a neutral
reference and vary the perturbation condition `c` to read off a clean perturbation
response score per gene.

### Dataset

**Papalexi et al. 2021, *Nature Methods*** — multimodal CRISPR screen targeting immune
checkpoint regulators in human T cells. 26 gene-KO perturbations + `NTg*` non-targeting
controls, with cell-level covariates (UMI count, mitochondrial fraction, cell-cycle phase,
sequencing lane, biological replicate).

---

## Repository Structure

```
.
├── cvae/
│   ├── __init__.py          # Public API: CVAE, load_data, make_dataloaders
│   ├── model.py             # Encoder, Decoder, CVAE (beta-VAE objective)
│   ├── dataset.py           # Data loading, normalisation, DataLoader construction
│   └── plotting.py          # All matplotlib/seaborn figure helpers
│
├── 00_export_data.R         # Export dataset from RDS → Python-readable flat files
├── 01_train_cvae.py         # Train CVAE; saves checkpoints + metadata
├── 02_analysis.py           # Latent-space UMAP, response scores, pathway enrichment
│
├── sweep_papalexi.sh                  # SLURM sweep: all covariates (20 jobs)
├── sweep_papalexi_nophase.sh          # SLURM sweep: phase excluded — z free to capture cell cycle
├── sweep_papalexi_nobatch.sh          # SLURM sweep: lane+bio_rep excluded — z free to capture batch
├── sweep_papalexi_no_perturbation.sh  # SLURM sweep: perturbation one-hot zeroed out (ablation)
├── submit_training.sh                 # Single-run training job (default params)
│
├── environment.yml          # Conda environment specification
├── requirements.txt         # pip-only requirements (alternative to conda)
│
├── data/                    # Dataset files (not tracked in git — see Data Export below)
├── results_papalexi/        # Sweep results: full covariates
├── results_papalexi_nophase/   # Sweep results: phase excluded
└── results_papalexi_nobatch/   # Sweep results: lane + bio_rep excluded
```

---

## Model Architecture

```
Input x  (n_genes)     Condition c = [pert_onehot | extra_covs]
         \                          /
          [x || c]  (n_genes + cond_dim)
               |
           Encoder MLP
         (1024 → 512 → 256)
               |
          (mu, logvar)          ← latent_dim-dimensional
               |
        reparameterize → z
               |
          [z || c]  (latent_dim + cond_dim)
               |
           Decoder MLP
         (256 → 512 → 1024)
               |
          x_recon  (n_genes)
```

**Loss:**
```
L = MSE(x_recon, x)  +  β · KL( N(mu, σ²) ‖ N(0, 1) )
```

The β (beta) hyperparameter controls how much the KL term regularises the latent space.
Large β forces the model to encode perturbation effects entirely through `c`, keeping `z`
perturbation-agnostic.

**Covariate conditioning** (`c` in addition to perturbation one-hot):

| Covariate | Type | Preprocessing |
|-----------|------|---------------|
| `n_umis`, `n_nonzero` | Continuous | log10, then z-standardise |
| `p_mito` | Continuous | z-standardise |
| `lane`, `bio_rep`, `phase` | Categorical | One-hot encode |

Any covariate can be excluded at training time with `--exclude_covs` to study whether
`z` then captures that source of variation.

---

## Reproducing Results

### 1. Environment setup

**Conda (recommended):**
```bash
conda env create -f environment.yml
conda activate cvae_perturb
```

**pip only:**
```bash
pip install -r requirements.txt
```

### 2. Data export

Raw data lives in RDS files. Run the R script to export to the flat-file format expected
by the Python pipeline:

```bash
Rscript 00_export_data.R
```

This writes the following files to `data/`:

| File | Description |
|------|-------------|
| `expression_matrix.mtx` | Sparse gene-expression matrix (cells × genes) |
| `genes.csv` | Gene names |
| `cells.csv` | Cell barcodes |
| `cell_perturbations.csv` | Per-cell perturbation assignment |
| `cell_covariates.csv` | Technical covariates (UMIs, mito fraction, phase, …) |
| `gene_module_index.csv` | Gene → Hallmark pathway membership matrix |

### 3. Single training run

```bash
python 01_train_cvae.py \
  --data_dir    data \
  --results_dir results \
  --latent_dim  64 \
  --beta        0.01 \
  --epochs      100 \
  --batch_size  512 \
  --lr          1e-3 \
  --patience    15

python 02_analysis.py \
  --data_dir    data \
  --results_dir results \
  --checkpoint  model_best.pt \
  --latent_dim  64 \
  --ntc_prefix  NTg
```

To exclude covariates from conditioning (e.g. study whether z captures cell-cycle):
```bash
python 01_train_cvae.py --exclude_covs phase ...
```

### 4. Parameter sweep (SLURM cluster)

Sweeps β ∈ {1e-4, 1e-3, 0.01, 0.1, 1.0} × latent_dim ∈ {32, 64, 128, 256} = **20 jobs**
per configuration.

```bash
bash sweep_papalexi.sh                  # full covariates       → results_papalexi/
bash sweep_papalexi_nophase.sh          # phase excluded        → results_papalexi_nophase/
bash sweep_papalexi_nobatch.sh          # lane+bio_rep excluded → results_papalexi_nobatch/
bash sweep_papalexi_no_perturbation.sh  # pert one-hot zeroed   → results_papalexi_no_pert/
```

After all jobs complete, compare silhouette scores:

```bash
for d in results_papalexi/beta*; do
  echo "$(basename $d): $(cat $d/silhouette.txt 2>/dev/null || echo N/A)"
done | sort -t: -k2 -n -r
```

---

## Outputs

Each results directory contains:

| File/Dir | Description |
|----------|-------------|
| `model_best.pt` | Best checkpoint (lowest validation loss) |
| `model_final.pt` | Final-epoch checkpoint |
| `training_log.csv` | Per-epoch train/val loss (total, recon, KL) |
| `pert_to_idx.json` | Perturbation name → integer index |
| `cov_stats.json` | Covariate normalisation stats (reused at inference) |
| `cov_cols.json` | Covariate columns used during training |
| `latent_embeddings.npy` | z vectors for all cells |
| `umap_embedding.npy` | 2-D UMAP of latent space |
| `response_scores.csv` | Per-gene, per-perturbation response score matrix |
| `pathway_enrichment.csv` | Fisher-exact Hallmark pathway enrichment (−log10 p) |
| `silhouette.txt` | Silhouette score of perturbation clusters in z space |
| `cvae_vs_de_comparison.csv` | Spearman ρ between CVAE response and naive DE |
| `figures/` | All PNG figures (see below) |

### Key figures

| Figure | Description |
|--------|-------------|
| `umap_latent.png` | Latent space coloured by perturbation identity |
| `umap_phase.png` | Same UMAP coloured by cell-cycle phase |
| `umap_lane.png` / `umap_bio_rep.png` | UMAP coloured by batch covariates |
| `gene_response_heatmap.png` | Top differentially-responding genes per perturbation |
| `perturbation_clustermap.png` | Hierarchical clustering of perturbations by response vectors |
| `pathway_enrichment.png` | Hallmark pathway enrichment heatmap |
| `effect_size_vs_null.png` | Per-perturbation mean effect size vs NTC noise floor |
| `top_genes_per_pert.png` | Top up/down genes for each perturbation (bar charts) |
| `umap_per_perturbation.png` | Small-multiples UMAP highlighting each perturbation |

---

## Key Findings

### Covariate ablation

Three sweep configurations let us study what information `z` captures when specific
covariates are removed from the condition vector:

| Config | Modification | What z is expected to capture |
|--------|--------------|-------------------------------|
| Full covariates | — | Residual cell-state variation only |
| No phase | `phase` excluded from c | + Cell-cycle variation (G1/S/G2M visible in z UMAP) |
| No batch | `lane`, `bio_rep` excluded from c | + Batch/replicate variation |
| No perturbation | pert one-hot zeroed out | Everything, including perturbation effects — upper bound on silhouette score |

Comparing `umap_phase.png` across configurations directly tests whether the model correctly
disentangles cell-cycle effects from perturbation effects.

### Silhouette score interpretation

A **low** silhouette score means different perturbations are mixed together in `z` —
the model successfully encoded perturbation identity through `c` alone, leaving `z`
perturbation-agnostic. A **high** score means perturbation signal leaked into `z`.

The parameter sweep identifies which (β, latent_dim) combination achieves the best
disentanglement while maintaining good reconstruction quality (monitored via validation loss).

### CVAE vs differential expression

`cvae_vs_de_comparison.csv` reports the Spearman correlation between CVAE response scores
and naive mean-difference differential expression for each perturbation. High correlation
confirms the CVAE captures the same signal as DE; deviations indicate cases where
conditioning on technical covariates changes the inferred response.

---
