import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim, hidden_dims, dropout):
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    return nn.Sequential(*layers), prev


class Encoder(nn.Module):
    def __init__(self, n_genes, cond_dim, latent_dim, hidden_dims, dropout):
        super().__init__()
        self.net, out_dim = _mlp(n_genes + cond_dim, hidden_dims, dropout)
        self.mu_head = nn.Linear(out_dim, latent_dim)
        self.logvar_head = nn.Linear(out_dim, latent_dim)

    def forward(self, x, c):
        h = self.net(torch.cat([x, c], dim=-1))
        return self.mu_head(h), self.logvar_head(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim, cond_dim, n_genes, hidden_dims, dropout):
        super().__init__()
        self.net, out_dim = _mlp(latent_dim + cond_dim, hidden_dims, dropout)
        self.out_head = nn.Linear(out_dim, n_genes)

    def forward(self, z, c):
        return self.out_head(self.net(torch.cat([z, c], dim=-1)))


class CVAE(nn.Module):
    """Conditional VAE for Perturb-seq data.

    Encoder takes [gene_expr || pert_onehot] → (mu, logvar).
    Decoder takes [z || pert_onehot] → reconstructed gene_expr.
    """

    def __init__(
        self,
        n_genes: int,
        n_perts: int,
        latent_dim: int = 64,
        beta: float = 1.0,
        enc_hidden: tuple = (1024, 512, 256),
        dec_hidden: tuple = (256, 512, 1024),
        dropout: float = 0.1,
        n_continuous_covs: int = 1,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.n_perts = n_perts
        self.n_continuous_covs = n_continuous_covs
        self.cond_dim = n_perts + n_continuous_covs
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = Encoder(n_genes, self.cond_dim, latent_dim, enc_hidden, dropout)
        self.decoder = Decoder(latent_dim, self.cond_dim, n_genes, dec_hidden, dropout)

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return mu

    def forward(self, x, c):
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, c)
        return x_recon, mu, logvar, z

    def compute_loss(self, x, x_recon, mu, logvar):
        recon = F.mse_loss(x_recon, x, reduction="mean")
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        return recon + self.beta * kl, recon, kl

    @torch.no_grad()
    def encode_batch(self, x, c):
        self.eval()
        mu, _ = self.encoder(x, c)
        return mu

    @torch.no_grad()
    def decode_batch(self, z, c):
        self.eval()
        return self.decoder(z, c)
