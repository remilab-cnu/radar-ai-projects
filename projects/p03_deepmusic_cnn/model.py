"""DeepMUSIC CNN -- covariance matrix based DoA estimation.

Architecture:
    Input: (B, 2, N_rx, N_rx) -- real/imag of sample covariance
    -> 4x Conv2d (BN + ReLU)
    -> Flatten -> FC(512) -> ReLU -> Dropout -> FC(grid_size) -> Sigmoid
    Output: (B, grid_size) -- pseudo-spectrum over angle grid
"""

import torch
import torch.nn as nn


class DeepMUSIC(nn.Module):
    """CNN-based DoA pseudo-spectrum estimator.

    Parameters
    ----------
    n_rx : int
        Number of antenna elements (default: 12).
    grid_size : int
        Output spectrum grid points (default: 181 for [-90, 90] at 1 deg).
    dropout : float
        Dropout rate before final FC (default: 0.3).
    """

    def __init__(self, n_rx=12, grid_size=181, dropout=0.3):
        super().__init__()
        self.n_rx = n_rx
        self.grid_size = grid_size

        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        flat_dim = 256 * n_rx * n_rx
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, grid_size),
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor (B, 2, N_rx, N_rx)

        Returns
        -------
        spectrum : Tensor (B, grid_size), values in [0, 1]
        """
        feat = self.encoder(x)
        logits = self.head(feat)
        return torch.sigmoid(logits)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = DeepMUSIC(n_rx=12, grid_size=181)
    x = torch.randn(4, 2, 12, 12)
    y = model(x)
    print(f"DeepMUSIC: input {x.shape} -> output {y.shape}")
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Output range: [{y.min().item():.4f}, {y.max().item():.4f}]")
