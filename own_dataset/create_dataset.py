from my_dataset import MyOwnDatasetFixedKnn

if __name__ == '__main__':

    path1 = "UrbanSound8K_8276.csv"
    path2 = "TF_PANNs_8276.hdf5"
    dataset = MyOwnDatasetFixedKnn(root = "./data_PANNs_fixed_knn", path = [path1,path2], enable = False)

    dataset = MyOwnDatasetFixedKnn(root = "./data_PANNs_varying_knn", path = [path1,path2], enable = True)
