import torch
import librosa
import resampy
import numpy as np
from tqdm import tqdm



def embeddingsLabels(df, path, model):

    y = []
    emb = []

    

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
      
        ruta = f"{path}fold{row.fold}/{row.slice_file_name}"

        (audio, sr) = librosa.core.load(ruta, sr=None, mono=True)
        audio_rs = resampy.resample(audio, sr, 16_000)

        _, yamnet_emb, _ = model(audio_rs)
        embeddings = torch.from_numpy(yamnet_emb.numpy())
        embeddings = embeddings.T

            
        if embeddings.shape == (1024, 8):
            emb.append(embeddings.tolist())
        else:
            embeddings = embeddings[:,:8]    
            emb.append(embeddings.tolist())
        
        y.append(row["classID"])
        del embeddings
        
        #if index == 10:
            #break
            
        
    emb_tensor = np.array(emb)
    features = [emb_tensor[...,i] for i in range(emb_tensor.shape[2])]
        
    
    y = np.array(y)
        
    return features, y
