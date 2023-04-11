from encodec.utils import convert_audio

import torchaudio
import torch 
import torch.nn as nn

import numpy as np

from tqdm import tqdm



def embeddingsLabels(df, path, model):

    y = []
    emb = []
    adaptive_pool = nn.AdaptiveMaxPool2d((128,75))
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
      
        ruta = f"{path}fold{row.fold}/{row.slice_file_name}"
        wav, sr = torchaudio.load(ruta)
        
        wav = convert_audio(wav, sr, model.sample_rate, model.channels)
        wav = wav.unsqueeze(0)    
        try:
            embeddings = model.encoder(wav)
            #embeddings = embeddings.squeeze()
            
            #print(embeddings.shape)
        except:
            print(ruta)
            break
        
        
        pool_adap = adaptive_pool(embeddings).squeeze()


        pool_adap_flat = pool_adap.flatten()
        
           
        emb.append(pool_adap_flat.tolist())
        y.append(row["classID"])

        del pool_adap_flat
        
        #if index == 10:
            #break
        
        
    #X = torch.cat(emb,0)
    y = np.array(y)
        
    return emb, y