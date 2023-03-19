from encodec.utils import convert_audio
from sklearn.cluster import KMeans

import torchaudio
import torch

import numpy as np

from tqdm import tqdm



def embeddingsLabels(df, path, model):

    y = []
    emb = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
      
        ruta = f"{path}fold{row.fold}/{row.slice_file_name}"
        wav, sr = torchaudio.load(ruta)
        
        wav = convert_audio(wav, sr, model.sample_rate, model.channels)
        wav = wav.unsqueeze(0)    
        try:
            embeddings = model.encoder(wav)
            embeddings = embeddings.squeeze().T
            
             
        except:
            print(ruta)
            break

        kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto").fit(embeddings.detach().to('cpu'))
        centers = kmeans.cluster_centers_
        # embeddings = torch.mean(embeddings, 0)

        centers_flat = centers.flatten()

        #print(centers_flat.shape)
        
           
        emb.append(centers_flat.tolist())
        y.append(row["classID"])

        del centers_flat
        
        #if index == 10:
            #break
        
        
    #X = torch.cat(emb,0)
    y = np.array(y)
        
    return emb, y