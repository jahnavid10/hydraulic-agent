"""PyTorch autoencoder used as the anomaly detector."""
import torch
from torch import nn


class Autoencoder(nn.Module):
    """Symmetric MLP autoencoder: in -> hidden... -> latent -> ...hidden -> in."""

    def __init__(self, in_dim: int, hidden: tuple[int, ...] = (64, 32), latent: int = 8):
        super().__init__()
        self.in_dim, self.hidden, self.latent = in_dim, tuple(hidden), latent

        enc, prev = [], in_dim
        for h in hidden:
            enc += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        enc.append(nn.Linear(prev, latent))

        dec, prev = [], latent
        for h in reversed(hidden):
            dec += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec.append(nn.Linear(prev, in_dim))

        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def config(self) -> dict:
        return {"in_dim": self.in_dim, "hidden": list(self.hidden), "latent": self.latent}


@torch.no_grad()
def per_feature_errors(model: Autoencoder, x_scaled, device: str = "cpu"):
    """Squared reconstruction error for every sample and feature -> numpy (n, d).

    The anomaly score of a cycle is the mean over features. Later, the `explain_anomaly`
    tool will reuse the per-feature version to say which sensors look abnormal.
    """
    model.eval()
    x = torch.as_tensor(x_scaled, dtype=torch.float32, device=device)
    return ((model(x) - x) ** 2).cpu().numpy()


def load_model(path, device: str = "cpu") -> Autoencoder:
    ckpt = torch.load(path, map_location=device)
    model = Autoencoder(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()