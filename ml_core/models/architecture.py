import torch
import torch.nn as nn
import torch.nn.functional as F

class SketchEncoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=512):
        super(SketchEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.fc = nn.Linear(512 * 8 * 8, latent_dim)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ReconstructionHead(nn.Module):
    def __init__(self, latent_dim=512, num_points=2048):
        super(ReconstructionHead, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, num_points * 3)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(2048)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x.view(-1, 2048, 3)

def build_model(config):
    encoder = SketchEncoder(latent_dim=config['latent_dim'])
    recon = ReconstructionHead(latent_dim=config['latent_dim'])
    return nn.Sequential(encoder, recon)
