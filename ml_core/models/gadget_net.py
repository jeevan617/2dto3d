import torch.nn as nn

class GadgetMicroSurfaceNet(nn.Module):
    """
    High-frequency surface detail reconstruction for electronic gadgets.
    optimized for metallic/glossy texture synthesis.
    """
    def __init__(self):
        super().__init__()
        self.surface_refiner = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def refine_surface(self, coarse_mesh_map):
        return self.surface_refiner(coarse_mesh_map)
