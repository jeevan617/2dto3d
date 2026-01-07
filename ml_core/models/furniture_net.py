import torch
import torch.nn as nn
from .architecture import SketchEncoder

class FurnitureReconstructor(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.encoder = SketchEncoder(latent_dim=latent_dim)
        # Voxel-based decoder for structural rigidity
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, 32*32*32),
            nn.Sigmoid()
        )
        # Leg detection branch
        self.leg_detector = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4) # coordinates
        )

    def forward(self, x):
        features = self.encoder(x)
        voxels = self.decoder(features).view(-1, 32, 32, 32)
        legs = self.leg_detector(features)
        return voxels, legs
