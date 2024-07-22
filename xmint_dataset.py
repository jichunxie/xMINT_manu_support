import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class xMINTDataset(Dataset):
    def __init__(self, slice_folder, WSI_path, coor_folder, gene_folder, num_known_gene):
        self.slice_folder = slice_folder
        self.WSI_path = WSI_path
        self.num_known_gene = num_known_gene

        # Load coordinate and gene data into memory
        self.coors_data = {}
        self.gene_data = {}
        for f in sorted(os.listdir(coor_folder)):
            file_path = os.path.join(coor_folder, f)
            self.coors_data[f] = pd.read_csv(file_path, index_col=0)
        for f in sorted(os.listdir(gene_folder)):
            file_path = os.path.join(gene_folder, f)
            self.gene_data[f] = pd.read_csv(file_path, index_col=0)

        # Process main image
        self.image = cv2.imread(WSI_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image_tensor = torch.tensor(self.image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        self.image_mean = self.image_tensor.mean([1, 2])
        self.image_std = self.image_tensor.std([1, 2]) + 1e-5

        # Preload slices into memory
        self.slices = self.load_slices()
    
    def load_slices(self):
        slices = {}
        for filename in os.listdir(self.slice_folder):
            if filename.endswith('.png'):
                slice_path = os.path.join(self.slice_folder, filename)
                patch = cv2.imread(slice_path)
                patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                patch_tensor = torch.tensor(patch, dtype=torch.float32).permute(2, 0, 1) / 255.0
                patch_tensor = (patch_tensor - self.image_mean[:, None, None]) / (self.image_std[:, None, None] + 1e-5)
                slice_name = filename[:-4]
                slices[slice_name] = patch_tensor
        return slices

    def __len__(self):
        return len(self.gene_data)

    def __getitem__(self, idx):
        coors_file = list(self.coors_data.keys())[idx]
        gene_file = list(self.gene_data.keys())[idx]

        coors = self.coors_data[coors_file]
        gene = self.gene_data[gene_file]
        cells = gene.index.tolist()
        gene_ids = gene.columns.tolist()
        unknown_gene_ids = gene_ids[self.num_known_gene:]

        coors_tensor = torch.from_numpy(coors.values).float()
        coors_tensor.clamp_min_(0)
        gene_tensor = torch.from_numpy(gene.values).float()

        x, y = os.path.basename(coors_file).split('_')[0], os.path.basename(coors_file).split('_')[1]
        slice_name = x + '_' + y
        image_tensor = self.slices[slice_name]
        known_gene = gene_tensor[:, :self.num_known_gene]
        unknown_gene = gene_tensor[:, self.num_known_gene:]
        sums_known = known_gene.sum(dim=-1, keepdim=True) + 1e-5
        sums_unknown = unknown_gene.sum(dim=-1, keepdim=True) + 1e-5
        normalized_known_gene = known_gene/sums_known 
        normalized_unknown_gene = unknown_gene/sums_unknown

        return image_tensor, coors_tensor, normalized_known_gene, normalized_unknown_gene, slice_name, cells, unknown_gene_ids
