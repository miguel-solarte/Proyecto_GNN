from torch_geometric.data import Dataset

import torch
import os.path as osp

class MyOwnDatasetFixedKnn(Dataset):
    def __init__(self, root, path, enable = False , transform=None, pre_transform=None, pre_filter=None):
      self.path = path
      self.enable = enable
      super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return self.path

    @property
    def processed_file_names(self):
      if self.enable == False:
        return [f'data_{i}.pt' for i in range(10)]
      else: 
        return [f'data_{i}.pt' for i in range(109)]

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data