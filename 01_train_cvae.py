#!/usr/bin/env python
"""Train a Conditional VAE on the Papalexi Perturb-seq dataset.

Usage:
    python 01_train_cvae.py [--latent_dim 64] [--beta 1.0] [--epochs 100] [--batch_size 512]

Outputs in results/:
    model_best.pt       Best checkpoint (lowest val loss)
    model_final.pt      Final epoch checkpoint
    training_log.csv    Per-epoch loss values
    pert_to_idx.json    Perturbation name → integer index mapping
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, os.path.dirname(__file__))
from cvae import CVAE, load_data, make_dataloaders
from cvae.plotting import plot_loss_curves

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=None,
                   help="Path to data directory (default: <script_dir>/data)")
    p.add_argument("--results_dir", default=None,
                   help="Path to results directory (default: <script_dir>/results)")
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--beta", type=float, default=1.0, help="KL weight (beta-VAE)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--exclude_covs", type=str, default=None,
                   help="Comma-separated covariate columns to exclude (e.g. 'phase,lane')")
    p.add_argument("--mask_perturbation", action="store_true",
                   help="Zero out perturbation one-hot (ablation: no pert info in conditioning)")
    return p.parse_args()


def epoch_loop(model, loader, optimizer, device, train=True, mask_perturbation=False):
    model.train(train)
    totals = {"total": 0.0, "recon": 0.0, "kl": 0.0}
    n_batches = 0
    for x, c_pert, extra_covs in loader:
        x = x.to(device)
        if mask_perturbation:
            c_pert = torch.zeros_like(c_pert)
        # Full conditioning: [pert_onehot || log_umi_norm || phase_onehot]
        c = torch.cat([c_pert, extra_covs], dim=-1).to(device)
        if train:
            optimizer.zero_grad()
        x_recon, mu, logvar, _ = model(x, c)
        loss, recon, kl = model.compute_loss(x, x_recon, mu, logvar)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        totals["total"] += loss.item()
        totals["recon"] += recon.item()
        totals["kl"] += kl.item()
        n_batches += 1
    return {k: v / n_batches for k, v in totals.items()}


def main():
    args = parse_args()
    DATA_DIR    = args.data_dir    or os.path.join(_SCRIPT_DIR, "data")
    RESULTS_DIR = args.results_dir or os.path.join(_SCRIPT_DIR, "results")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data ---
    print("Loading data...")
    expr, genes, cells, pert_labels, gene_module, cell_cov = load_data(DATA_DIR)
    print(f"  {expr.shape[0]} cells, {expr.shape[1]} genes, {len(set(pert_labels))} perturbations")

    exclude_covs = args.exclude_covs.split(",") if args.exclude_covs else None
    train_dl, val_dl, pert_to_idx, train_idx, val_idx, cov_stats, n_extra_covs, cont_cols, cat_cols = make_dataloaders(
        expr, pert_labels, cell_cov, cells,
        val_frac=args.val_frac,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        exclude_covs=exclude_covs,
    )
    print(f"  Train: {len(train_idx)} cells | Val: {len(val_idx)} cells")

    # Save mapping and covariate normalization stats for inference
    with open(os.path.join(RESULTS_DIR, "pert_to_idx.json"), "w") as f:
        json.dump(pert_to_idx, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "cov_stats.json"), "w") as f:
        json.dump(cov_stats, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "n_extra_covs.json"), "w") as f:
        json.dump({"n_extra_covs": n_extra_covs}, f)
    with open(os.path.join(RESULTS_DIR, "cov_cols.json"), "w") as f:
        json.dump({"cont_cols": cont_cols, "cat_cols": cat_cols}, f, indent=2)
    np.save(os.path.join(RESULTS_DIR, "train_idx.npy"), train_idx)
    np.save(os.path.join(RESULTS_DIR, "val_idx.npy"), val_idx)

    # --- Model ---
    n_genes = expr.shape[1]
    n_perts = len(pert_to_idx)
    model = CVAE(
        n_genes=n_genes,
        n_perts=n_perts,
        latent_dim=args.latent_dim,
        beta=args.beta,
        dropout=args.dropout,
        n_continuous_covs=n_extra_covs,
    ).to(device)
    print(f"\nModel: {n_genes} genes | {n_perts} perturbations | latent_dim={args.latent_dim}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # --- Training ---
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\nTraining for up to {args.epochs} epochs (early stop patience={args.patience})...\n")
    print(f"{'Epoch':>6}  {'Train Total':>12}  {'Train Recon':>12}  {'Train KL':>10}  "
          f"{'Val Total':>10}  {'Val Recon':>10}  {'LR':>8}  {'Time':>6}")
    print("-" * 90)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = epoch_loop(model, train_dl, optimizer, device, train=True,
                        mask_perturbation=args.mask_perturbation)
        with torch.no_grad():
            va = epoch_loop(model, val_dl, optimizer, device, train=False,
                            mask_perturbation=args.mask_perturbation)
        scheduler.step(va["total"])

        train_losses.append(tr)
        val_losses.append(va)

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(f"{epoch:>6}  {tr['total']:>12.4f}  {tr['recon']:>12.4f}  {tr['kl']:>10.4f}  "
              f"{va['total']:>10.4f}  {va['recon']:>10.4f}  {lr:>8.1e}  {elapsed:>5.1f}s")

        if va["total"] < best_val_loss:
            best_val_loss = va["total"]
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "model_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "model_final.pt"))

    log_df = pd.DataFrame({
        "train_total": [l["total"] for l in train_losses],
        "train_recon": [l["recon"] for l in train_losses],
        "train_kl":    [l["kl"]    for l in train_losses],
        "val_total":   [l["total"] for l in val_losses],
        "val_recon":   [l["recon"] for l in val_losses],
        "val_kl":      [l["kl"]    for l in val_losses],
    })
    log_df.to_csv(os.path.join(RESULTS_DIR, "training_log.csv"), index=False)

    plot_loss_curves(train_losses, val_losses,
                     os.path.join(RESULTS_DIR, "loss_curves.png"))

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints and logs saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
