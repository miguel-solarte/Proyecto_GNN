from my_dataset import MyOwnDataset

if __name__ == '__main__':
    path1 = "UrbanSound8K_8276.csv"
    path2 = "TF_PANNs_8276.hdf5"
    dataset = MyOwnDataset(root = "./data", path = [path1,path2])
