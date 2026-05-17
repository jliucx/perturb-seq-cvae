import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Default covariate configs — overridden by what's actually present in cell_covariates.csv.
# Columns with "n_umis" or "n_nonzero" in the name get log10-transformed before standardizing.
_LOG_PATTERN = ('n_umis', 'n_nonzero')   # column names containing these are log10-transformed

# Papalexi-style datasets (gene KO, has phase/lane/bio_rep)
_PAPALEXI_CONT = ['n_umis', 'n_nonzero', 'p_mito']
_PAPALEXI_CAT  = ['lane', 'bio_rep', 'phase']

# Morris-style datasets (SNP perturbations, all-continuous covariates)
_MORRIS_CONT = ['response_n_umis', 'response_n_nonzero', 'response_p_mito',
                'grna_n_nonzero', 'grna_n_umis']
_MORRIS_CAT  = []


def _detect_cov_cols(cell_cov: pd.DataFrame):
    """Auto-detect continuous and categorical columns from cell_cov."""
    cont_cols, cat_cols = [], []
    for col in cell_cov.columns:
        if pd.api.types.is_numeric_dtype(cell_cov[col]):
            cont_cols.append(col)
        else:
            cat_cols.append(col)
    return cont_cols, cat_cols


def load_data(data_dir: str):
    """Return (expr_csr, genes, cells, pert_labels, gene_module, cell_cov)."""
    expr = scipy.io.mmread(f"{data_dir}/expression_matrix.mtx").tocsr()
    genes = pd.read_csv(f"{data_dir}/genes.csv")["gene"].tolist()
    cells = pd.read_csv(f"{data_dir}/cells.csv")["cell"].tolist()
    pert_df = pd.read_csv(f"{data_dir}/cell_perturbations.csv")
    pert_labels = pert_df["perturbation"].tolist()
    gene_module = pd.read_csv(f"{data_dir}/gene_module_index.csv", index_col=0)
    cell_cov = pd.read_csv(f"{data_dir}/cell_covariates.csv", index_col=0)
    return expr, genes, cells, pert_labels, gene_module, cell_cov


def log1p_scale(expr_csr, scale: float = 1e4) -> np.ndarray:
    """Log1p-normalize: log(1 + x * scale).  Returns dense float32 array."""
    dense = expr_csr.toarray().astype(np.float32)
    return np.log1p(dense * scale)


def build_extra_covs(cell_cov: pd.DataFrame, cells: list, stats: dict = None,
                     cont_cols: list = None, cat_cols: list = None):
    """Build extra covariate matrix from cell_covariate columns.

    Auto-detects continuous vs categorical columns if cont_cols/cat_cols not given.
    Columns whose names contain 'n_umis' or 'n_nonzero' are log10-transformed.

    Pass stats (returned from a prior call) to reuse training-time
    normalization and category orderings at inference time.

    Returns:
        extra_covs  : np.ndarray  shape (n_cells, n_extra_covs)
        stats       : dict  {col: {"mean":, "std":} or {cat: idx}}  — save to JSON
        n_extra_covs: int
    """
    if stats is None:
        stats = {}
    if cont_cols is None and cat_cols is None:
        cont_cols, cat_cols = _detect_cov_cols(cell_cov)

    parts = []

    # --- continuous ---
    for col in (cont_cols or []):
        vals = cell_cov.loc[cells, col].values.astype(np.float32)
        if any(p in col for p in _LOG_PATTERN):
            vals = np.log10(np.maximum(vals, 1.0))
        if col in stats:
            m, s = stats[col]["mean"], stats[col]["std"]
        else:
            m, s = float(vals.mean()), float(vals.std())
            stats[col] = {"mean": m, "std": s}
        parts.append((vals - m) / (s + 1e-8))

    # --- categorical ---
    for col in (cat_cols or []):
        vals = cell_cov.loc[cells, col].astype(str).values
        if col in stats:
            mapping = stats[col]
        else:
            unique = sorted(set(vals))
            mapping = {c: i for i, c in enumerate(unique)}
            stats[col] = mapping
        n_cats = len(mapping)
        ohe = np.zeros((len(cells), n_cats), dtype=np.float32)
        for i, v in enumerate(vals):
            ohe[i, mapping.get(v, 0)] = 1.0
        parts.append(ohe)

    extra_covs = np.concatenate(
        [p[:, None] if p.ndim == 1 else p for p in parts], axis=1
    ) if parts else np.zeros((len(cells), 0), dtype=np.float32)
    n_extra_covs = extra_covs.shape[1]
    return extra_covs, stats, n_extra_covs


class PerturbDataset(Dataset):
    def __init__(self, expr: np.ndarray, pert_labels: np.ndarray, pert_to_idx: dict,
                 extra_covs: np.ndarray):
        self.expr = torch.from_numpy(expr)
        self.n_perts = len(pert_to_idx)
        idx = torch.tensor([pert_to_idx[p] for p in pert_labels], dtype=torch.long)
        self.pert_onehot = F.one_hot(idx, num_classes=self.n_perts).float()
        self.pert_indices = idx
        self.extra_covs = torch.tensor(extra_covs, dtype=torch.float32)

    def __len__(self):
        return len(self.expr)

    def __getitem__(self, i):
        return self.expr[i], self.pert_onehot[i], self.extra_covs[i]


def make_dataloaders(
    expr_csr,
    pert_labels,
    cell_cov: pd.DataFrame,
    cells: list,
    val_frac: float = 0.15,
    batch_size: int = 512,
    seed: int = 42,
    num_workers: int = 0,
    exclude_covs: list = None,
):
    """Stratified train/val split.

    Returns (train_dl, val_dl, pert_to_idx, train_idx, val_idx, cov_stats, n_extra_covs, cont_cols, cat_cols).
    cov_stats and col lists must be saved and reloaded at inference to apply the same normalization.
    """
    rng = np.random.default_rng(seed)
    pert_arr   = np.array(pert_labels)
    cells_arr  = np.array(cells)

    known_mask = pert_arr != "unknown"
    if known_mask.sum() < len(pert_arr):
        print(f"  Filtering {(~known_mask).sum()} cells with unknown perturbation assignment.")
        known_indices = np.where(known_mask)[0]
        expr_csr  = expr_csr[known_indices]
        pert_arr  = pert_arr[known_indices]
        cells_arr = cells_arr[known_indices]
    else:
        known_indices = np.arange(len(pert_arr))

    known_cells = cells_arr.tolist()
    cont_cols, cat_cols = _detect_cov_cols(cell_cov)
    if exclude_covs:
        cont_cols = [c for c in cont_cols if c not in exclude_covs]
        cat_cols  = [c for c in cat_cols  if c not in exclude_covs]
    extra_covs, cov_stats, n_extra_covs = build_extra_covs(
        cell_cov, known_cells, cont_cols=cont_cols, cat_cols=cat_cols)
    n_cat_dims = sum(len(cov_stats[c]) for c in cat_cols if c in cov_stats)
    print(f"  Extra covariates: {n_extra_covs} dims "
          f"({len(cont_cols)} continuous + {n_cat_dims} categorical OHE)")

    unique_perts = sorted(set(pert_arr))
    pert_to_idx = {p: i for i, p in enumerate(unique_perts)}

    train_idx, val_idx = [], []
    for p in unique_perts:
        mask = np.where(pert_arr == p)[0]
        rng.shuffle(mask)
        n_val = max(1, int(len(mask) * val_frac))
        val_idx.extend(mask[:n_val])
        train_idx.extend(mask[n_val:])

    train_idx = np.array(train_idx)
    val_idx   = np.array(val_idx)

    print("Log1p-normalizing expression data...")
    expr_norm = log1p_scale(expr_csr)

    train_ds = PerturbDataset(expr_norm[train_idx], pert_arr[train_idx], pert_to_idx,
                              extra_covs[train_idx])
    val_ds   = PerturbDataset(expr_norm[val_idx],   pert_arr[val_idx],   pert_to_idx,
                              extra_covs[val_idx])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    orig_train_idx = known_indices[train_idx]
    orig_val_idx   = known_indices[val_idx]

    return train_dl, val_dl, pert_to_idx, orig_train_idx, orig_val_idx, cov_stats, n_extra_covs, cont_cols, cat_cols
