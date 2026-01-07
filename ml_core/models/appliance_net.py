import torch.nn as nn

class IndustrialDesignNet(nn.Module):
    """
    Unified backbone for Electric and Kitchen Appliances.
    Focuses on hard-surface modeling and bevel generation.
    """
    def __init__(self):
        super().__init__()
        self.bevel_generator = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.control_panel_locator = nn.TransformerEncoderLayer(d_model=512, nhead=4)

    def forward(self, sketch_features):
        # Detect knobs, buttons, and screens
        return self.control_panel_locator(sketch_features)
