import torch
import torch.nn as nn
from .attention import MultiHeadLatentAttention

class VehicleReconstructor(nn.Module):
    def __init__(self, num_wheels=4):
        super().__init__()
        self.body_attention = MultiHeadLatentAttention(d_model=1024)
        self.wheel_generator = nn.ModuleList([
            nn.Linear(1024, 256) for _ in range(num_wheels)
        ])
        
    def forward(self, sketch_emb, history_state=None):
        # Attention over vehicle components
        body_features = self.body_attention(sketch_emb, sketch_emb, sketch_emb)
        
        wheels = []
        for gen in self.wheel_generator:
            wheels.append(gen(body_features))
            
        return torch.cat([body_features] + wheels, dim=1)
