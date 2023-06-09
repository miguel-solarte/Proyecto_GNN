import torch

import numpy as np

from tqdm import tqdm



def embeddingsLabels(df, path, model):

    y = []
    emb = []

    

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
      
        ruta = f"{path}fold{row.fold}/{row.slice_file_name}"
        embeddings = model(ruta)
            
        if embeddings.shape == (4, 128):
            emb.append(embeddings.detach().to('cpu').tolist())
        else:
            embeddings = embeddings[:,:128]    
            emb.append(embeddings.detach().to('cpu').tolist())
        
        y.append(row["classID"])
        del embeddings
        
        #if index == 10:
            #break
            
        
    emb_tensor = np.array(emb)
    features = [emb_tensor[...,i] for i in range(emb_tensor.shape[2])]
        
    
    y = np.array(y)
        
    return features, y