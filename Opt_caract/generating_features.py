from YaMNet_mean import embeddingsLabels
from encodec import EncodecModel

import torch
import pandas as pd
import numpy as np
import h5py

# Para YaMNet

import tensorflow_hub as hub



# Define a PyTorch model that wraps the TensorFlow Hub model
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super(Wrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':

    df = pd.read_csv("../Urbansound8k/UrbanSound8K.csv")

#==============================Encodec====================================================    

    #model = EncodecModel.encodec_model_24khz()
    #model.set_target_bandwidth(1.5)

#==============================Vggish====================================================  

    #model = torch.hub.load('harritaylor/torchvggish', 'vggish')
    #model.eval()

#==============================YaMNet====================================================  
    yamnet_model_url = "https://tfhub.dev/google/yamnet/1"
    model = hub.KerasLayer(yamnet_model_url)


# Instantiate the PyTorch model
    model = Wrapper(model)

    path = '../Urbansound8k/'

    X, y = embeddingsLabels(df, path, model)

    X_array = np.array(X)
    

    print(X_array.shape)
    print(y.shape)

    f = h5py.File("../files_h5py/TF_YaMNet_mean.hdf5", "w")
    dset = f.create_dataset("features", X_array.shape, data = X_array)
    dset = f.create_dataset("labels", y.shape, data = y)
    f.close()

