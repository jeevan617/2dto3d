from torch.utils.data import Dataset
import glob
import os

class Sketch3DDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.sketch_files = glob.glob(os.path.join(root_dir, 'sketches/*.png'))
        self.mesh_files = [f.replace('sketches', 'meshes').replace('.png', '.glb') for f in self.sketch_files]

    def __len__(self):
        return len(self.sketch_files)

    def __getitem__(self, idx):
        # Implementation for loading multi-modal data
        return {
            'image': self.sketch_files[idx],
            'mesh_path': self.mesh_files[idx],
            'metadata': {'category': 'auto_detected'}
        }
