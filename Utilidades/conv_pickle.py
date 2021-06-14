
#%% 
import numpy as np
import matplotlib.pyplot as plt 
import torch 
import pickle
import os 

PATH = '/home/pakitochus/Descargas/cosas_cristfg'
files = os.listdir(PATH)

for fil in files:
    filename = os.path.join(PATH, fil)
    with open(filename, 'rb') as handle: 
        b = pickle.load(handle)
        

    for k in b.keys():
        # print(f'{k}: {type(b[k])}')
        aux = b[k]
        if type(aux) is torch.Tensor:
            aux = aux.detach().cpu().numpy()
        elif type(aux) is list:
            for i, el in enumerate(aux):
                if type(el) is torch.Tensor:
                    aux[i] = el.detach().cpu().numpy()
        b[k] = aux


    with open(filename, 'wb') as handle:
        pickle.dump(b, handle, protocol=pickle.HIGHEST_PROTOCOL)
	

# %% GUARDAR IMAGENES
from PIL import Image
im = Image.fromarray(b['ypredicha'][-1][0].swapaxes(0,-1))
im.save("your_file.jpeg")
# %%