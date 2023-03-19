from my_dataset import MyOwnDatasetFixedKnn

if __name__ == '__main__':
    
# ================================================PANNs======================================================
    path1 = "UrbanSound8K_8276.csv"
    path2 = "TF_PANNs_8276.hdf5"

    dataset = MyOwnDatasetFixedKnn(root = "./data_PANNs_fixed_knn", path = [path1,path2], enable = False)
    dataset = MyOwnDatasetFixedKnn(root = "./data_PANNs_varying_knn", path = [path1,path2], enable = True)

# ================================================encodec======================================================

    path3 = "UrbanSound8K.csv"
    path4 = "TF_encodec.hdf5"

    dataset = MyOwnDatasetFixedKnn(root = "./data_encodec_fixed_knn", path = [path3,path4], enable = False)
    dataset = MyOwnDatasetFixedKnn(root = "./data_encodec_varying_knn", path = [path3,path4], enable = True)

# ================================================Vggish======================================================

    #path5 = "UrbanSound8K7780.csv"
    #path6 = "TF_vggish.hdf5"

    #dataset = MyOwnDatasetFixedKnn(root = "./data_vggish_fixed_knn", path = [path5,path6], enable = False)
    #dataset = MyOwnDatasetFixedKnn(root = "./data_vggish_varying_knn", path = [path5,path6], enable = True)

# ================================================encodecKmeans=================================================

    path7 = "UrbanSound8K.csv"
    path8 = "TF_encodec_kmeans.hdf5"

    dataset = MyOwnDatasetFixedKnn(root = "./data_encodecKmeans_fixed_knn", path = [path7,path8], enable = False)
    dataset = MyOwnDatasetFixedKnn(root = "./data_encodecKmeans_varying_knn", path = [path7,path8], enable = True)

# ================================================encodecmaxpooling=================================================

    path9 = "UrbanSound8K.csv"
    path10 = "TF_encodec_maxpooling.hdf5"

    dataset = MyOwnDatasetFixedKnn(root = "./data_encodecmaxpooling_fixed_knn", path = [path9,path10], enable = False)

# ================================================encodecavgpooling=================================================

    path11 = "UrbanSound8K.csv"
    path12 = "TF_encodec_adap_avgpooling.hdf5"

    dataset = MyOwnDatasetFixedKnn(root = "./data_encodec_adap_avgpooling_fixed_knn", path = [path11,path12], enable = False)
