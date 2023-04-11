from encodec_adap_maxpooling import embeddingsLabels
from encodec import EncodecModel


import pandas as pd
import torch
import h5py


if __name__ == '__main__':

    df = pd.read_csv("../../../../../mnt/g/Mi unidad/Datasets/Urbansound8k_8276.csv")

    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(1.5)

    path ='../../../../../mnt/g/Mi unidad/Datasets/Urbansound8k/'

    X, y = embeddingsLabels(df, path, model)

    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)

    print(X_tensor.shape)

    f = h5py.File("../files_h5py/TF_encodec_adap_maxpooling.hdf5", "w")
    dset = f.create_dataset("features", X_tensor.shape, data = X_tensor)
    dset = f.create_dataset("labels", y_tensor.shape, data = y_tensor)
    f.close()

