#!/usr/bin/env python
"""Analyze trained CVAE: latent space, in-silico perturbation, clustering, pathway enrichment.

Usage:
    python 02_analysis.py [--checkpoint model_best.pt] [--n_top_genes 40]

Outputs in results/figures/:
    umap_latent.png              UMAP of latent space colored by perturbation
    gene_response_heatmap.png    Top genes per perturbation (decoder Δ)
    perturbation_clustermap.png  Hierarchical clustering of perturbations
    pathway_enrichment.png       Hallmark pathway enrichment heatmap
    response_scores.csv          Full per-gene, per-perturbation response scores
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import fisher_exact
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
import umap

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.dirname(__file__))
from cvae import CVAE, load_data
from cvae.dataset import log1p_scale, PerturbDataset, build_extra_covs
from cvae.plotting import (
    plot_umap,
    plot_umap_by_covariate,
    plot_umap_highlight_perts,
    plot_effect_size_comparison,
    plot_top_genes_per_pert,
    plot_response_heatmap,
    plot_perturbation_clustermap,
    plot_pathway_enrichment,
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=None,
                   help="Path to data directory (default: <script_dir>/data)")
    p.add_argument("--results_dir", default=None,
                   help="Path to results directory (default: <script_dir>/results)")
    p.add_argument("--checkpoint", default="model_best.pt")
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--n_top_genes", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--umap_seed", type=int, default=42)
    p.add_argument("--ntc_prefix", default=None,
                   help="Comma-separated prefixes for NTC perturbation names "
                        "(default: auto-detect from 'NTg','NTC','non-target')")
    p.add_argument("--mask_perturbation", action="store_true",
                   help="Zero out perturbation one-hot during encoding (match training ablation)")
    return p.parse_args()


def get_latent_embeddings(model, expr_norm, pert_onehot, extra_covs, device,
                          batch_size=512, mask_perturbation=False):
    """Encode all cells → latent means."""
    model.eval()
    n = len(expr_norm)
    latent = []
    extra_t = torch.tensor(extra_covs, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            x = torch.from_numpy(expr_norm[start:start + batch_size]).to(device)
            p_oh = pert_onehot[start:start + batch_size]
            if mask_perturbation:
                p_oh = torch.zeros_like(p_oh)
            c = torch.cat([p_oh, extra_t[start:start + batch_size]], dim=-1).to(device)
            mu, _ = model.encoder(x, c)
            latent.append(mu.cpu().numpy())
    return np.concatenate(latent, axis=0)


_DEFAULT_NTC_PREFIXES = ("NTg", "NTC", "non-target", "Non-targeting", "ctrl", "control")


def _find_ntc_keys(pert_to_idx, ntc_prefixes=None):
    prefixes = ntc_prefixes or _DEFAULT_NTC_PREFIXES
    keys = [k for k in pert_to_idx
            if any(k.startswith(p) or k.lower() == p.lower() for p in prefixes)]
    return keys


def _ntc_mean_onehot(pert_to_idx, n_perts, device, ntc_prefixes=None):
    """Average one-hot over all NTC guides → soft NTC baseline for decoding."""
    ntg_keys = _find_ntc_keys(pert_to_idx, ntc_prefixes)
    if not ntg_keys:
        raise ValueError(
            f"No NTC perturbations found. Keys in dataset: {list(pert_to_idx)[:10]}...\n"
            "Use --ntc_prefix to specify the NTC name prefix."
        )
    onehots = torch.stack([
        F.one_hot(torch.tensor(pert_to_idx[k]), num_classes=n_perts).float()
        for k in ntg_keys
    ])
    return onehots.mean(dim=0).to(device)  # soft average


def compute_response_scores(model, expr_norm, pert_labels, pert_to_idx,
                            extra_covs, device, batch_size=512, ntc_prefixes=None):
    """For each perturbation p, compute mean (decode(z, p) - decode(z, NTC_avg)).

    Only the perturbation part of conditioning is swapped; extra_covs are
    held at each cell's actual values so the delta is a pure perturbation effect.
    """
    model.eval()
    pert_arr = np.array(pert_labels)
    n_perts = len(pert_to_idx)

    ntc_pert_onehot = _ntc_mean_onehot(pert_to_idx, n_perts, device, ntc_prefixes)  # (n_perts,)
    ntg_keys = set(_find_ntc_keys(pert_to_idx, ntc_prefixes))
    extra_t = torch.tensor(extra_covs, dtype=torch.float32)

    response_dict = {}

    for pert, p_idx in pert_to_idx.items():
        cell_mask = pert_arr == pert
        if cell_mask.sum() == 0:
            continue

        p_expr  = expr_norm[cell_mask]
        p_extra = extra_t[cell_mask]
        p_pert_vec = F.one_hot(torch.tensor(p_idx), num_classes=n_perts).float()

        all_deltas = []
        with torch.no_grad():
            for start in range(0, len(p_expr), batch_size):
                x_batch     = torch.from_numpy(p_expr[start:start + batch_size]).to(device)
                extra_batch = p_extra[start:start + batch_size].to(device)
                n_batch = len(x_batch)

                # Encode: cell's own pert + UMI + phase
                c_enc = torch.cat([p_pert_vec.unsqueeze(0).expand(n_batch, -1).to(device),
                                   extra_batch], dim=-1)
                mu, _ = model.encoder(x_batch, c_enc)

                # Decode: swap only pert, keep UMI + phase per-cell
                c_p   = c_enc  # same pert as encoding
                c_ntc = torch.cat([ntc_pert_onehot.unsqueeze(0).expand(n_batch, -1),
                                   extra_batch], dim=-1)
                x_p   = model.decoder(mu, c_p)
                x_ntc = model.decoder(mu, c_ntc)
                all_deltas.append((x_p - x_ntc).cpu().numpy())

        response_dict[pert] = np.concatenate(all_deltas, axis=0).mean(axis=0)
        n_cells = cell_mask.sum()
        tag = " [NTC]" if pert in ntg_keys else ""
        print(f"  {pert:<20} ({n_cells:>5} cells) response range: "
              f"[{response_dict[pert].min():.3f}, {response_dict[pert].max():.3f}]{tag}")

    return response_dict


def hallmark_enrichment(response_dict, gene_module, gene_names, top_n=50):
    """Fisher's exact test: top up/down genes vs hallmark pathways.

    Returns DataFrame (perturbations × pathways), values = -log10(p-value).
    """
    pathways = gene_module.columns.tolist()
    gene_arr = np.array(gene_names)
    n_genes = len(gene_arr)
    records = []

    for pert, scores in response_dict.items():
        row = {"perturbation": pert}
        ranked = np.argsort(np.abs(scores))[::-1]
        top_genes = set(gene_arr[ranked[:top_n]])

        for path in pathways:
            path_genes = set(gene_module.index[gene_module[path] == 1])
            a = len(top_genes & path_genes)
            b = len(top_genes) - a
            c = len(path_genes) - a
            d = n_genes - a - b - c
            table = [[a, b], [c, d]]
            _, pval = fisher_exact(table, alternative="greater")
            row[path] = -np.log10(max(pval, 1e-10))

        records.append(row)

    df = pd.DataFrame(records).set_index("perturbation")
    return df


def compare_with_de(response_dict, expr_norm, pert_labels, gene_names,
                    pert_to_idx=None, ntc_prefixes=None, top_k=20):
    """Simple DE comparison: mean expr pert vs pooled NTC mean vs CVAE response scores."""
    from scipy.stats import spearmanr
    pert_arr = np.array(pert_labels)

    # Pool all NTC cells as the NTC baseline for DE
    ntc_set = set(_find_ntc_keys(pert_to_idx, ntc_prefixes))
    ntg_mask = np.array([p in ntc_set for p in pert_arr])
    ntc_mean = expr_norm[ntg_mask].mean(axis=0)

    results = {}
    for pert, scores in response_dict.items():
        pert_mean = expr_norm[pert_arr == pert].mean(axis=0)
        log2fc = pert_mean - ntc_mean  # difference in log1p space ≈ log-fold-change
        rho, p = spearmanr(np.abs(scores), np.abs(log2fc))
        results[pert] = {"spearman_rho": rho, "p_value": p,
                         "is_ntc": pert in ntc_set}

        cvae_top = set(np.array(gene_names)[np.argsort(np.abs(scores))[-top_k:]])
        de_top = set(np.array(gene_names)[np.argsort(np.abs(log2fc))[-top_k:]])
        results[pert]["cvae_de_overlap"] = len(cvae_top & de_top)

    return pd.DataFrame(results).T


def main():
    args = parse_args()
    DATA_DIR    = args.data_dir    or os.path.join(_SCRIPT_DIR, "data")
    RESULTS_DIR = args.results_dir or os.path.join(_SCRIPT_DIR, "results")
    FIG_DIR     = os.path.join(RESULTS_DIR, "figures")
    ntc_prefixes = args.ntc_prefix.split(",") if args.ntc_prefix else None

    os.makedirs(FIG_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load data ---
    print("Loading data...")
    expr, genes, cells, pert_labels, gene_module, cell_cov = load_data(DATA_DIR)
    with open(os.path.join(RESULTS_DIR, "pert_to_idx.json")) as f:
        pert_to_idx = json.load(f)
    with open(os.path.join(RESULTS_DIR, "cov_stats.json")) as f:
        cov_stats = json.load(f)
    with open(os.path.join(RESULTS_DIR, "n_extra_covs.json")) as f:
        n_extra_covs = json.load(f)["n_extra_covs"]

    n_genes = expr.shape[1]
    n_perts = len(pert_to_idx)
    print(f"  {expr.shape[0]} cells, {n_genes} genes, {n_perts} perturbations")

    print("Log1p-normalizing...")
    expr_norm = log1p_scale(expr)

    # Filter out unknown cells (no perturbation assignment)
    pert_arr = np.array(pert_labels)
    known_mask = pert_arr != "unknown"
    cells_arr = np.array(cells)
    if known_mask.sum() < len(pert_arr):
        print(f"  Filtering {(~known_mask).sum()} cells with unknown perturbation.")
        expr_norm = expr_norm[known_mask]
        pert_arr  = pert_arr[known_mask]
        cells_arr = cells_arr[known_mask]

    # Build extra covariates with training-time stats and column lists (no refit)
    cov_cols_path = os.path.join(RESULTS_DIR, "cov_cols.json")
    if os.path.exists(cov_cols_path):
        with open(cov_cols_path) as f:
            _cov_cols = json.load(f)
        _cont_cols, _cat_cols = _cov_cols["cont_cols"], _cov_cols["cat_cols"]
    else:
        _cont_cols, _cat_cols = None, None
    extra_covs, _, _ = build_extra_covs(cell_cov, cells_arr.tolist(), stats=cov_stats,
                                        cont_cols=_cont_cols, cat_cols=_cat_cols)

    # Build one-hot tensor for all cells
    idx_arr = torch.tensor([pert_to_idx[p] for p in pert_arr], dtype=torch.long)
    pert_onehot = F.one_hot(idx_arr, num_classes=n_perts).float()

    # --- Load model ---
    print(f"\nLoading model from {args.checkpoint}...")
    model = CVAE(n_genes=n_genes, n_perts=n_perts, latent_dim=args.latent_dim,
                 n_continuous_covs=n_extra_covs).to(device)
    ckpt_path = os.path.join(RESULTS_DIR, args.checkpoint)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    pert_labels_filtered = list(pert_arr)

    # ======================================================================
    # 1. Latent Space — encode all cells, run UMAP
    # ======================================================================
    print("\n[1/4] Computing latent embeddings...")
    latent = get_latent_embeddings(model, expr_norm, pert_onehot, extra_covs, device,
                                   args.batch_size, mask_perturbation=args.mask_perturbation)
    np.save(os.path.join(RESULTS_DIR, "latent_embeddings.npy"), latent)

    # Silhouette score on latent space (exclude NTC cells — they're not perturbations)
    ntg_keys = _find_ntc_keys(pert_to_idx, ntc_prefixes)
    ntc_set = set(ntg_keys)
    non_ntc_mask = np.array([p not in ntc_set for p in pert_labels_filtered])
    if non_ntc_mask.sum() > 1 and len(set(np.array(pert_labels_filtered)[non_ntc_mask])) > 1:
        le = LabelEncoder()
        labels_encoded = le.fit_transform(np.array(pert_labels_filtered)[non_ntc_mask])
        sil = silhouette_score(latent[non_ntc_mask], labels_encoded,
                               sample_size=min(10000, non_ntc_mask.sum()), random_state=42)
    else:
        sil = float("nan")
    print(f"  Silhouette score (perturbation clusters, excl. NTC): {sil:.4f}")
    with open(os.path.join(RESULTS_DIR, "silhouette.txt"), "w") as f:
        f.write(f"{sil:.6f}\n")

    print("Running UMAP (this may take a few minutes)...")
    pca = PCA(n_components=min(50, latent.shape[1]), random_state=args.umap_seed)
    latent_pca = pca.fit_transform(latent)
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                        random_state=args.umap_seed, verbose=True)
    embedding = reducer.fit_transform(latent_pca)
    np.save(os.path.join(RESULTS_DIR, "umap_embedding.npy"), embedding)

    plot_umap(embedding, pert_labels_filtered, os.path.join(FIG_DIR, "umap_latent.png"),
              title=f"CVAE Latent Space (UMAP)  silhouette={sil:.3f}")
    print(f"  Saved: {FIG_DIR}/umap_latent.png")

    # Color CVAE latent UMAP by each covariate (all conditioned out of latent)
    saved_cov_plots = []
    for col in cell_cov.columns:
        is_num = pd.api.types.is_numeric_dtype(cell_cov[col])
        vals_raw = cell_cov.loc[cells_arr.tolist(), col]
        if is_num:
            vals = np.log10(np.maximum(vals_raw.values.astype(float), 1.0))
            cmap = "plasma"
            title_suffix = f"log₁₀({col})"
        else:
            vals = vals_raw.astype(str).tolist()
            cmap = None
            title_suffix = col
        fname = f"umap_{col}.png"
        plot_umap_by_covariate(embedding, vals,
                               os.path.join(FIG_DIR, fname),
                               title=f"UMAP (CVAE latent) — {title_suffix}",
                               categorical=not is_num, cmap=cmap)
        saved_cov_plots.append(fname)
    print(f"  Saved covariate UMAPs: {', '.join(saved_cov_plots)}")

    # Expression-based UMAP for covariate verification
    print("Running expression-PCA UMAP for covariate verification...")
    pca_expr = PCA(n_components=30, random_state=args.umap_seed)
    expr_pca = pca_expr.fit_transform(expr_norm)
    reducer_expr = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1,
                             random_state=args.umap_seed, verbose=False)
    embedding_expr = reducer_expr.fit_transform(expr_pca)
    np.save(os.path.join(RESULTS_DIR, "umap_expression_embedding.npy"), embedding_expr)
    saved_expr_plots = []
    for col in cell_cov.columns:
        is_num = pd.api.types.is_numeric_dtype(cell_cov[col])
        vals_raw = cell_cov.loc[cells_arr.tolist(), col]
        if is_num:
            vals = np.log10(np.maximum(vals_raw.values.astype(float), 1.0))
            cmap = "plasma"
            title_suffix = f"log₁₀({col})"
        else:
            vals = vals_raw.astype(str).tolist()
            cmap = None
            title_suffix = col
        fname = f"umap_expr_{col}.png"
        plot_umap_by_covariate(embedding_expr, vals,
                               os.path.join(FIG_DIR, fname),
                               title=f"UMAP (Expression PCA) — {title_suffix}",
                               categorical=not is_num, cmap=cmap)
        saved_expr_plots.append(fname)
    print(f"  Saved expression UMAP covariates: {', '.join(saved_expr_plots)}")

    # Small-multiples: each perturbation highlighted
    plot_umap_highlight_perts(embedding, pert_labels_filtered,
                              os.path.join(FIG_DIR, "umap_per_perturbation.png"))
    print(f"  Saved: {FIG_DIR}/umap_per_perturbation.png")

    # ======================================================================
    # 2. In-Silico Perturbation — gene response scores
    # ======================================================================
    ntg_keys = _find_ntc_keys(pert_to_idx, ntc_prefixes)
    print(f"\n[2/4] Computing perturbation response scores "
          f"(decode Δ vs NTC avg of {len(ntg_keys)} guides: {ntg_keys[:5]}{'...' if len(ntg_keys)>5 else ''})...")
    response_dict = compute_response_scores(
        model, expr_norm, pert_labels_filtered, pert_to_idx,
        extra_covs, device, args.batch_size, ntc_prefixes=ntc_prefixes
    )

    # Report NTg null distribution
    ntc_scores = np.array([response_dict[k] for k in ntg_keys if k in response_dict])
    if len(ntc_scores):
        null_std = ntc_scores.std(axis=0).mean()
        print(f"  NTC null: mean |response| per gene = {np.abs(ntc_scores).mean():.4f}  "
              f"std across NTC guides = {null_std:.4f}")

    response_df = pd.DataFrame(response_dict, index=genes)  # genes × perturbations
    response_df.to_csv(os.path.join(RESULTS_DIR, "response_scores.csv"))
    print(f"  Saved: {RESULTS_DIR}/response_scores.csv")

    plot_response_heatmap(response_df, os.path.join(FIG_DIR, "gene_response_heatmap.png"),
                          n_top_genes=args.n_top_genes)
    print(f"  Saved: {FIG_DIR}/gene_response_heatmap.png")

    # Effect size comparison: gene targets vs NTC null
    plot_effect_size_comparison(response_dict,
                                os.path.join(FIG_DIR, "effect_size_vs_null.png"))
    print(f"  Saved: {FIG_DIR}/effect_size_vs_null.png")

    # Top up/down genes per perturbation (bar charts)
    # Noise floor = NTC null mean + 2 SD (computed over per-guide mean |response|)
    ntc_per_guide = np.array([response_dict[k] for k in ntg_keys if k in response_dict])
    ntc_guide_means = np.abs(ntc_per_guide).mean(axis=1)  # one value per NTC guide
    ntc_threshold = ntc_guide_means.mean() + 2 * ntc_guide_means.std() if len(ntc_guide_means) else 0.0
    print(f"  NTC noise floor (mean+2SD): {ntc_threshold:.4f}")
    plot_top_genes_per_pert(response_df,
                            os.path.join(FIG_DIR, "top_genes_per_pert.png"),
                            top_k=10, ntc_threshold=ntc_threshold)
    print(f"  Saved: {FIG_DIR}/top_genes_per_pert.png")

    # ======================================================================
    # 3. Perturbation Clustering
    # ======================================================================
    print("\n[3/4] Clustering perturbations by response vectors...")
    plot_perturbation_clustermap(response_df, os.path.join(FIG_DIR, "perturbation_clustermap.png"))
    print(f"  Saved: {FIG_DIR}/perturbation_clustermap.png")

    # ======================================================================
    # 4. Pathway Enrichment
    # ======================================================================
    print("\n[4/4] Running hallmark pathway enrichment...")
    if gene_module.shape[1] == 0:
        print("  Skipping: gene_module_index has no pathway columns.")
    else:
        enrich_df = hallmark_enrichment(response_dict, gene_module, genes, top_n=50)
        enrich_df.to_csv(os.path.join(RESULTS_DIR, "pathway_enrichment.csv"))
        plot_pathway_enrichment(enrich_df, os.path.join(FIG_DIR, "pathway_enrichment.png"))
        print(f"  Saved: {FIG_DIR}/pathway_enrichment.png")

    # ======================================================================
    # Bonus: Compare CVAE vs naive DE
    # ======================================================================
    print("\n[Bonus] Comparing CVAE response vs naive differential expression...")
    de_compare = compare_with_de(response_dict, expr_norm, pert_labels_filtered, genes,
                                  pert_to_idx=pert_to_idx, ntc_prefixes=ntc_prefixes)
    de_compare.to_csv(os.path.join(RESULTS_DIR, "cvae_vs_de_comparison.csv"))
    mean_rho = de_compare["spearman_rho"].mean()
    mean_overlap = de_compare["cvae_de_overlap"].mean()
    print(f"  Mean Spearman ρ (CVAE vs DE): {mean_rho:.3f}")
    print(f"  Mean top-20 gene overlap:     {mean_overlap:.1f}/20")
    print(f"  Saved: {RESULTS_DIR}/cvae_vs_de_comparison.csv")

    print(f"\nAll figures saved to: {FIG_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
