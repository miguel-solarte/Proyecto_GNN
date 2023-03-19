from encodec.utils import convert_audio
import skimage.measure

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
            embeddings = embeddings.squeeze()
            embeddings = embeddings.detach().numpy()
            
            #print(embeddings.shape)
        except:
            print(ruta)
            break
        
        
        pool = skimage.measure.block_reduce(embeddings, (1,embeddings.shape[1] // 2 + 1), np.max)


        pool_flat = pool.flatten()
        assert 256 == pool_flat.shape[0], print(f"{pool_flat.shape}: {ruta}")
        #print(pool_flat.shape)
        
           
        emb.append(pool_flat.tolist())
        y.append(row["classID"])

        del pool_flat
        
        #if index == 10:
            #break
        
        
    #X = torch.cat(emb,0)
    y = np.array(y)
        
    return emb, y