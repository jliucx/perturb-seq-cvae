import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage


_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff",
    "#9A6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1",
    "#000075", "#a9a9a9", "#ffffff", "#000000", "#e6beff", "#808080",
    "#4169E1", "#FF6347", "#40E0D0",
]


def color_map(labels):
    unique = sorted(set(labels))
    cmap = {l: _PALETTE[i % len(_PALETTE)] for i, l in enumerate(unique)}
    cmap["NTC"] = "#cccccc"
    return cmap


_NTC_PREFIXES = ("NTg", "NTC", "non-targeting", "Non-targeting", "ctrl", "control")


def _is_ntc(label):
    return any(label.startswith(p) or label.lower() == p.lower() for p in _NTC_PREFIXES)


def plot_umap(embedding, pert_labels, save_path,
              title="CVAE Latent Space (UMAP)", show_ntc=False):
    labels = np.array(pert_labels)
    unique_targets = sorted(p for p in set(pert_labels) if not _is_ntc(p))
    ntc_labels = sorted(p for p in set(pert_labels) if _is_ntc(p))
    cmap = color_map(unique_targets)

    fig, ax = plt.subplots(figsize=(11, 8))

    if show_ntc:
        # Draw NTC cells first (background, gray)
        for i, ntc in enumerate(ntc_labels):
            mask = labels == ntc
            shade = 0.55 + 0.04 * i
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       color=(shade, shade, shade), s=3, alpha=0.4,
                       label=ntc if i == 0 else f"_{ntc}", zorder=1)
        if ntc_labels:
            ax.scatter([], [], color="#999999", s=12,
                       label=f"NTC ({len(ntc_labels)}×)", marker="s", alpha=0.7)

    # Draw each gene perturbation
    for p in unique_targets:
        mask = labels == p
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[cmap[p]], s=5, alpha=0.75, label=p, zorder=2)

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(title, fontsize=14)

    handles, lbls = ax.get_legend_handles_labels()
    # Keep only visible labels (skip _-prefixed NTg duplicates)
    handles = [h for h, l in zip(handles, lbls) if not l.startswith("_")]
    lbls   = [l for l in lbls if not l.startswith("_")]
    ax.legend(handles, lbls, bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=7, markerscale=3, ncol=1)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_response_heatmap(response_df, save_path, n_top_genes=100, title="Perturbation Response (Top Genes)"):
    """response_df: genes × perturbations DataFrame of response scores.

    NTg guides are shown at the right of the heatmap as negative controls.
    """
    # Separate gene targets and NTg controls
    ntc_cols = [c for c in response_df.columns if _is_ntc(c)]
    tgt_cols = [c for c in response_df.columns if not _is_ntc(c)]

    # Select top genes by max |response| among gene targets only
    top_genes = response_df[tgt_cols].abs().max(axis=1).nlargest(n_top_genes).index
    data = response_df.loc[top_genes, tgt_cols + ntc_cols]  # targets | NTCs

    vmax = data[tgt_cols].abs().values.max()
    fig, ax = plt.subplots(figsize=(max(10, len(data.columns) * 0.55), max(10, n_top_genes * 0.3)))
    sns.heatmap(data, ax=ax, cmap="RdBu_r", center=0, vmax=vmax, vmin=-vmax,
                xticklabels=True, yticklabels=True, linewidths=0)

    # Draw a vertical separator between targets and NTc controls
    if ntc_cols:
        ax.axvline(x=len(tgt_cols), color="black", linewidth=2)
        ax.text(len(tgt_cols) + len(ntc_cols) / 2, -0.8, "NTC guides",
                ha="center", va="top", fontsize=8, transform=ax.get_xaxis_transform(),
                color="gray", style="italic")

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Perturbation", fontsize=11)
    ax.set_ylabel("Gene", fontsize=11)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_perturbation_clustermap(response_df, save_path):
    """Cluster perturbations (non-NTC only) by their response vectors."""
    tgt_cols = [c for c in response_df.columns if not _is_ntc(c)]
    data = response_df[tgt_cols].T  # perturbations × genes (NTC excluded)
    g = sns.clustermap(data, cmap="RdBu_r", center=0,
                       figsize=(min(30, len(data.columns) * 0.05 + 5), max(6, len(data) * 0.4)),
                       row_cluster=True, col_cluster=True,
                       xticklabels=False, yticklabels=True,
                       dendrogram_ratio=(0.15, 0.05),
                       cbar_pos=(0.02, 0.8, 0.03, 0.15))
    g.fig.suptitle("Perturbation Clustering by Response Vector", y=1.01, fontsize=12)
    g.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(g.fig)


def plot_pathway_enrichment(enrich_df, save_path, top_n=20):
    """enrich_df: perturbations × pathways, values = -log10(p-value)."""
    # Keep top pathways by max enrichment
    top_paths = enrich_df.max(axis=0).nlargest(top_n).index
    data = enrich_df[top_paths]
    # Shorten pathway names
    data.columns = [c.replace("HALLMARK_", "").replace("_", " ").title() for c in data.columns]

    fig, ax = plt.subplots(figsize=(max(12, len(data.columns) * 0.6), max(6, len(data) * 0.4)))
    vmax = min(data.values.max(), 10)
    sns.heatmap(data, ax=ax, cmap="YlOrRd", vmin=0, vmax=vmax,
                xticklabels=True, yticklabels=True, linewidths=0.3,
                cbar_kws={"label": "-log₁₀(p-value)"})
    ax.set_title("Pathway Enrichment of Perturbation Responses", fontsize=13)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_umap_by_covariate(embedding, covariate_values, save_path,
                           title="UMAP", cmap=None, categorical=True):
    """Color UMAP by any covariate (e.g. cell cycle phase, n_umis)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    vals = np.array(covariate_values)

    if categorical:
        unique = sorted(set(vals))
        phase_colors = {"G1": "#4575b4", "S": "#d73027", "G2M": "#1a9850",
                        "G2": "#1a9850"}
        pal = {u: phase_colors.get(u, _PALETTE[i % len(_PALETTE)])
               for i, u in enumerate(unique)}
        for u in unique:
            mask = vals == u
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=[pal[u]], s=3, alpha=0.5, label=u, zorder=2)
        ax.legend(markerscale=4, fontsize=10, loc="best")
    else:
        sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                        c=vals.astype(float), cmap=cmap or "viridis",
                        s=3, alpha=0.5)
        plt.colorbar(sc, ax=ax, shrink=0.7)

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_umap_highlight_perts(embedding, pert_labels, save_path,
                              perts_to_show=None, ncols=5):
    """Small-multiples UMAP: one panel per perturbation, highlighted in red."""
    labels = np.array(pert_labels)
    target_perts = [p for p in (perts_to_show or sorted(set(labels)))
                    if not _is_ntc(p)]
    n = len(target_perts)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3, nrows * 2.8),
                             sharex=True, sharey=True)
    axes = np.array(axes).flatten()

    for ax, pert in zip(axes, target_perts):
        mask = labels == pert
        ax.scatter(embedding[~mask, 0], embedding[~mask, 1],
                   c="#dddddd", s=1, alpha=0.3, rasterized=True)
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c="#d62728", s=4, alpha=0.8, rasterized=True)
        ax.set_title(pert.replace("tar-", ""), fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    # Hide unused panels
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Per-Perturbation UMAP Highlight", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_effect_size_comparison(response_dict, save_path):
    """Boxplot comparing mean |response| for NTg guides vs gene targets.

    Visualises the signal-to-noise ratio: real perturbations should show
    larger effect sizes than NTg negative controls.
    """
    ntg_effects, tgt_effects, tgt_labels = [], [], []
    for pert, scores in response_dict.items():
        mean_abs = np.abs(scores).mean()
        if _is_ntc(pert):
            ntg_effects.append(mean_abs)
        else:
            tgt_effects.append(mean_abs)
            tgt_labels.append(pert.replace("tar-", ""))

    # Sort gene targets by effect size
    order = np.argsort(tgt_effects)[::-1]
    tgt_effects = np.array(tgt_effects)[order]
    tgt_labels = np.array(tgt_labels)[order]

    fig, ax = plt.subplots(figsize=(max(10, len(tgt_labels) * 0.5), 5))
    x = np.arange(len(tgt_labels))
    bars = ax.bar(x, tgt_effects, color="#4575b4", alpha=0.8, label="Gene targets")

    # NTg null band
    ntg_mean = np.mean(ntg_effects)
    ntg_std  = np.std(ntg_effects)
    ax.axhline(ntg_mean, color="#d73027", linewidth=1.5, linestyle="--",
               label=f"NTC mean ({ntg_mean:.4f})")
    ax.axhspan(ntg_mean - ntg_std, ntg_mean + ntg_std,
               alpha=0.15, color="#d73027", label="NTC ±1 SD")

    ax.set_xticks(x)
    ax.set_xticklabels(tgt_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean |response score|", fontsize=11)
    ax.set_title("Perturbation Effect Size vs NTC Null", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_top_genes_per_pert(response_df, save_path, top_k=10,
                            perts_to_show=None, ncols=5, ntc_threshold=None):
    """Horizontal bar charts showing top up/down genes for each perturbation.

    ntc_threshold: if provided, panels whose max |response| falls below this
    value are greyed out with a "below noise floor" label so readers know
    the bars are not meaningful signal.  Compute it as e.g.
    ntc_mean + 2 * ntc_std of the NTg guide mean-|response| values.
    """
    tgt_cols = [c for c in response_df.columns if not _is_ntc(c)]
    if perts_to_show:
        tgt_cols = [c for c in tgt_cols if c in perts_to_show]

    n = len(tgt_cols)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols

    # Compute global x-limit across all non-noise perturbations for a unified scale
    global_max = 0.0
    for pert in tgt_cols:
        scores = response_df[pert]
        max_abs = scores.abs().max()
        if ntc_threshold is None or max_abs >= ntc_threshold:
            global_max = max(global_max, max_abs)
    if global_max == 0.0:
        global_max = response_df[tgt_cols].abs().max().max()
    xlim = global_max * 1.05

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.5, nrows * (top_k * 0.28 + 0.8)),
                             sharex=True)
    axes = np.array(axes).flatten()

    for ax, pert in zip(axes, tgt_cols):
        scores = response_df[pert].sort_values()
        max_abs = scores.abs().max()
        below_noise = ntc_threshold is not None and max_abs < ntc_threshold

        bottom_k = scores.head(top_k)
        top_k_up = scores.tail(top_k)
        combined = pd.concat([bottom_k, top_k_up])

        if below_noise:
            colors = ["#cccccc"] * len(combined)
        else:
            colors = ["#d73027" if v > 0 else "#4575b4" for v in combined.values]

        ax.barh(combined.index, combined.values, color=colors, alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set_xlim(-xlim, xlim)

        title = pert.replace("tar-", "")
        if below_noise:
            ax.set_title(title, fontsize=9, fontweight="bold", color="#999999")
            ax.text(0.5, 0.5, "below noise floor", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7, color="#999999",
                    style="italic")
        else:
            ax.set_title(title, fontsize=9, fontweight="bold")

        ax.tick_params(axis="y", labelsize=7)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="x", alpha=0.3)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(f"Top {top_k} Up/Down Genes per Perturbation", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(train_losses, val_losses, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    keys = ["total", "recon", "kl"]
    titles = ["Total Loss (ELBO)", "Reconstruction Loss (MSE)", "KL Divergence"]
    for ax, key, title in zip(axes, keys, titles):
        ax.plot([l[key] for l in train_losses], label="Train", color="#2196F3")
        ax.plot([l[key] for l in val_losses], label="Val", color="#FF5722")
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
