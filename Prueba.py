from own_dataset.my_dataset import MyOwnDatasetFixedKnn
from torch_geometric.loader import DataLoader
from models import GAT, GCN, GraphSAGE
from train_test import train, test

import matplotlib.pyplot as plt
import numpy as np

path1 = "UrbanSound8K_8276.csv"
path2 = "TF_encodec.hdf5"
dataset = MyOwnDatasetFixedKnn(root = "./own_dataset/data_encodec_fixed_knn", path = [path1,path2])
loader_dataset = DataLoader(dataset)

for i in range(30,2030,100):

#GAT

    acc_gat = []
    prom_acc_gat = []
    dev_values_gat = []
    num_epochs_gat = []

#GCN

    acc_gcn = []
    prom_acc_gcn = []
    dev_values_gcn = []
    num_epochs_gcn = []

#SAGE

    acc_sage = []
    prom_acc_sage = []
    dev_values_sage = []
    num_epochs_sage = []
    
    for n,graphs in enumerate(loader_dataset):

        print(n)

#==========================GAT===============================================
      
        gat = GAT(128, 20, 10)
        train(gat, graphs, epoch = i)
        acc_gat.append(test(gat, graphs, graphs.test_mask))

#==========================GCN===============================================  

        gcn = GCN(128, 20, 10)
        train(gcn, graphs, epoch = i)
        acc_gcn.append(test(gcn, graphs, graphs.test_mask))

#==========================SAGE=============================================== 

        g_sage = GraphSAGE(128, 20, 10)
        train(g_sage, graphs, epoch = i)
        acc_sage.append(test(g_sage, graphs, graphs.test_mask))

#==========================GAT===============================================

    prom_acc_gat.append(np.mean(acc_gat))
    dev_values_gat.append(np.std(acc_gat))
    num_epochs_gat.append(i)

#==========================GCN===============================================  

    prom_acc_gcn.append(np.mean(acc_gcn))
    dev_values_gcn.append(np.std(acc_gcn))
    num_epochs_gcn.append(i)

#==========================SAGE===============================================  

    prom_acc_sage.append(np.mean(acc_sage))
    dev_values_sage.append(np.std(acc_sage))
    num_epochs_sage.append(i)

    print(f"GAT\n accuracy mean: {np.mean(acc_gat)*100:.2f}% | standard deviation: {np.std(acc_gat):.2f} | epochs: {i}\n"
        f"GCN\n accuracy mean: {np.mean(acc_gcn)*100:.2f}% | standard deviation: {np.std(acc_gcn):.2f} | epochs: {i}\n"
        f"SAGE\n accuracy mean: {np.mean(acc_sage)*100:.2f}% | standard deviation: {np.std(acc_sage):.2f} | epochs: {i}\n\n"
        f"==============================================================================================================")