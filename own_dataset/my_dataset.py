from torch_geometric.data import Dataset, Data
from torch_geometric.nn import knn_graph

import os.path as osp
import pandas as pd
import torch
import h5py



class MyOwnDatasetFixedKnn(Dataset):
    def __init__(self, root, path, enable = False, transform=None, pre_transform=None, pre_filter=None):
      self.path = path
      self.enable = enable
      super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return self.path

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(10)]

    def download(self):
        pass

    def process(self):
      self.df = pd.read_csv(self.raw_paths[0])
      self.data = h5py.File(self.raw_paths[1], 'r')

      X_tensor = torch.tensor(self.data['features'])
      Y_tensor = torch.tensor(self.data['labels'])

      test_i = 1
      val_j = 2
      l = 0  # flag

      if self.enable == False:
         
        for j in range(10):
          print(f"\n valor de i: {test_i}, valor de j: {val_j}")

          Idx_train, Idx_test, Idx_val = self._CrossVal(self.df, test_i, val_j)

          X_tensor_norm = self._normalizacion_tensor(self, X_tensor, Idx_train)
          data = self._get_graph(X_tensor_norm, Y_tensor, Idx_train, Idx_test, Idx_val, k)

          test_i = (test_i + 1)%11
          val_j = (val_j + 1)%11    
          if val_j==0:
            val_j = 1

          torch.save(data, osp.join(self.processed_dir, f'data_{j}.pt'))
      
      else:

        for k in range(5,115,10):

          test_i = 1
          val_j = 2
         
          for j in range(l, l + 10):

            print(f"\n valor de i: {test_i}, valor de j: {val_j}")

            Idx_train, Idx_test, Idx_val = self._CrossVal(self.df, test_i, val_j)

            X_tensor_norm = self._normalizacion_tensor(X_tensor, Idx_train)
            data = self._get_graph(X_tensor_norm, Y_tensor, Idx_train, Idx_test, Idx_val, k)

            test_i = (test_i + 1)%11
            val_j = (val_j + 1)%11    

            if val_j==0:
              val_j = 1

            torch.save(data, osp.join(self.processed_dir, f'data_{j}.pt'))
        
          l = j + 1

    def _normalizacion_tensor(self, X_tensor, Idx_train):
      
      mini = X_tensor[Idx_train].min()
      maxi = X_tensor[Idx_train].max()

      feature_tensor_norm = (X_tensor - mini) / (maxi - mini)

      return feature_tensor_norm
    
    def _CrossVal(self, df, idxtest, idxval):
      df.reset_index(drop=True, inplace=True)
      df2 = df.copy()
    
      Idx_test = df.index[df.fold == idxtest]

      df2.drop(Idx_test, inplace=True)

      Idx_val = df.index[df.fold == idxval]

      df2.drop(Idx_val, inplace=True)
    
      Idx_train = df2.index
    
      return Idx_train, Idx_test, Idx_val

    def _get_graph(self, X_tensor, Y_tensor, Idx_train, Idx_test, Idx_val, val_k):

      x = X_tensor

    #Labels nodes

      y = Y_tensor

    #Datos de entrenamiento y test

      train_mask = Idx_train
      train_mask = torch.tensor(train_mask)

      test_mask = Idx_test
      test_mask = torch.tensor(test_mask)

      val_mask = Idx_val
      val_mask = torch.tensor(val_mask)

    #Se crea las conexines de los nodos

      edge_index = knn_graph(x, k=val_k, batch=y, loop=False)


    #Creacion del grafo 

      data = Data(x=x, edge_index= edge_index, y = y , train_mask = train_mask, test_mask = test_mask, val_mask = val_mask)
  
      return data


    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data