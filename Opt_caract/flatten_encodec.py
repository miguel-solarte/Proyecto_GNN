from encodec.utils import convert_audio

import torchaudio
import torch.nn as nn

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
        
        embeddings = model.encoder(wav)
        embeddings = embeddings.squeeze()

        if embeddings.shape != (128, 300):
            embeddings = embeddings[:,:300]

        embeddings_flat = embeddings.detach().to('cpu').flatten()    
        emb.append(embeddings_flat.tolist())
        y.append(row["classID"])

        del embeddings_flat
        
        #if index == 10:
        #    break
        
    y = np.array(y)
        
    return emb, y