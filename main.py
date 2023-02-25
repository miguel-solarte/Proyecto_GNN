from own_dataset.my_dataset import MyOwnDatasetFixedKnn
from torch_geometric.loader import DataLoader
from models import GAT, GCN, GraphSAGE
from train_test import train, test

import numpy as np

if __name__ == '__main__':
    
    path1 = "UrbanSound8K_8276.csv"
    path2 = "TF_PANNs_8276.hdf5"
    dataset = MyOwnDatasetFixedKnn(root = "./own_dataset/data_encodec_fixed_knn", path = [path1,path2])
    loader_dataset = DataLoader(dataset)

    for graphs in loader_dataset:
       print(graphs)

    acc_gat = []
    acc_gcn =[]
    acc_sage = []

    for i in range(10):
#==========================GAT===============================================

      print("GAT")
      gat = GAT(2048, 20, 10)
      train(gat, dataset[i], epoch = 20)
      acc_gat.append(test(gat, dataset[i], dataset[i].test_mask))

#==========================GCN===============================================  

      print("GCN")
      gcn = GCN(2048, 20, 10)
      train(gcn, dataset[i], epoch = 20)
      acc_gcn.append(test(gcn, dataset[i], dataset[i].test_mask))

#==========================SAGE=============================================== 

      print("SAGE")
      g_sage = GraphSAGE(2048, 20, 20)
      train(g_sage, dataset[i], epoch = 30)
      acc_sage.append(test(g_sage, dataset[i], dataset[i].test_mask))
    

print(f"==============================================================================================================\n"
    f"GAT\n accuracy mean: {np.mean(acc_gat)*100:.2f}% | standard deviation: {np.std(acc_gat):.2f} \n"
        f"GCN\n accuracy mean: {np.mean(acc_gcn)*100:.2f}% | standard deviation: {np.std(acc_gcn):.2f} \n"
        f"SAGE\n accuracy mean: {np.mean(acc_sage)*100:.2f}% | standard deviation: {np.std(acc_sage):.2f} \n"
        f"==============================================================================================================")


