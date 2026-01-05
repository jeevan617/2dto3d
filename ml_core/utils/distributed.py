import os
import torch
import torch.distributed as dist

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_distributed():
    dist.destroy_process_group()

class SyncBatchNormMocker(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(num_features)
    
    def forward(self, x):
        return self.bn(x)
