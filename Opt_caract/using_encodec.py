from encodec.utils import convert_audio

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
        
        embeddings = torch.mean(embeddings, 0)
        
           
        emb.append(embeddings.detach().to('cpu').tolist())
        y.append(row["classID"])
        del embeddings
        
        #if index == 10:
            #break
        
        
    #X = torch.cat(emb,0)
    y = np.array(y)
        
    return emb, y

