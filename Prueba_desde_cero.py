#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Como ejercico interesante, vamos a intentar hacer todo desde cero, 
primero para entender todo mejor, y segundo para entender qué fallos 
puede dar directamente. 

@File    :   Prueba_desde_cero.py
@Time    :   2021/03/19 11:43:24
@Author  :   F.J. Martinez-Murcia 
@Version :   1.0
@Contact :   pakitochus@gmail.com
@License :   (C)Copyright 2021, SiPBA-BioSIP
@Desc    :   None

La estructura del script, si queremos hacerlo monolítico sería: 
1. Importar librerías
2. Crear importador de datos (dataloader)
3. Crear el modelo (o importarlo de models.CC)
4. Bucle de entrenamiento/test. 

Voy a ver qué puedo ir haciendo yo: 
'''

# librerías a importar
# (añádelas solo según las vayas usando)
import torch 


#%% CREAMOS DATALOADER
from torch.utils.data import DataLoader
import os
import numpy as np 
from easydict import EasyDict as edict
import torchvision.transforms as standard_transforms
from misc.transforms import Compose, RandomHorizontallyFlip, RandomCrop, \
                            GTScaleDown, LabelNormalize, DeNormalize, \
                            AC_collate

#Path del dataset:
DATA_PATH = '/home/pakitochus/Descargas/propuestas_tfg_cristina/crowd/definitivo/DISCO_dataset'
# para mi: /home/pakitochus/Descargas/propuestas_tfg_cristina/crowd/definitivo/DISCO_dataset o '/Volumes/Cristina /TFG/Data'

mean_std = ([0.410824894905, 0.370634973049, 0.359682112932], [0.278580576181, 0.26925137639, 0.27156367898])
factor = 1  # must be 1
log_para = 100.

train_main_transform =  Compose([
        RandomHorizontallyFlip()
])
img_transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*mean_std)
])
gt_transform = standard_transforms.Compose([
        GTScaleDown(factor),
        LabelNormalize(log_para)
])
restore_transform = standard_transforms.Compose([
        DeNormalize(*mean_std),
    standard_transforms.ToPILImage()
])

train_set=edict()
train_set.img_path=os.path.join(DATA_PATH, 'imgs')
train_set.aud_path=os.path.join(DATA_PATH, 'density')
train_set.mode='train'
train_set.main_transform=train_main_transform
train_set.img_transform=img_transform
train_set.gt_transform=gt_transform
train_set.is_noise=False
train_set.brightness_decay=1.0 
train_set.noise_sigma= 0.
train_set.longest_side= 512


TRAIN_BATCH_SIZE = 1
if TRAIN_BATCH_SIZE == 1:
    train_loader = DataLoader(train_set, batch_size=1, num_workers=8, shuffle=True, drop_last=True)
elif TRAIN_BATCH_SIZE > 1:
    train_loader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, num_workers=8,
                                collate_fn=AC_collate, shuffle=True, drop_last=True)





#%% CREAMOS EL MODELO 
from models.CC import CrowdCounter

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

net_name = 'CANNet'
net = CrowdCounter(gpus=[0], model_name=net_name)
print(net)



#%% ENTRENAMOS EL MODELO 
from torch import optim
from torch.optim.lr_scheduler import StepLR

N_EPOCHS = 200
PRINT_FREQ = 5
LOG_PARA = 100.
LR = 1e-5  # learning rate
LR_DECAY = 0.99  # decay rate
LR_DECAY_START = -1  # when training epoch is more than it, the learning rate will be begin to decay
NUM_EPOCH_LR_DECAY = 1  # decay frequency
MAX_EPOCH = 200

optimizer = optim.Adam(net.CCN.parameters(), lr=LR, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=NUM_EPOCH_LR_DECAY, gamma=LR_DECAY) 

for epoch in range(N_EPOCHS):
    if epoch > LR_DECAY_START:
        #El scheduler.step() va cambiando el Learning Rate en cada epoch:
        scheduler.step()

        net.train()
        for i, data in enumerate(train_loader, 0):
            #Por cada muestra tomamos su imagen, su audio y su mapa de densidad ground-truth:
            img = data[0]
            gt_map = data[1]
            audio_img = data[2]

            #PARA PASAR DE TENSORES A VARIABLES Y PODER TRABAJAR BIEN CON ELLOS:
            #Si hay GPUs:
            if use_cuda:
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()
                audio_img = Variable(audio_img).cuda()

            #No hay GPUs disponibles:
            else:
                img = Variable(img) 
                gt_map = Variable(gt_map)   
                audio_img = Variable(audio_img)

            #En PyTorch tenemos que poner los gradientes a cero antes de la backpropagation
            #porque PyTorch acumula (sumatorio) los gradientes. Por ello, los actualizaré más tarde manualmente.
            optimizer.zero_grad()

            #Si la red que hemos escogido trabaja también con audio, lo metemos junto con la imagen en la red:
            #SE SACA UN MAPA PREDICHO CON LA ESTRUCTURA DE LA RED:
            if 'Audio' in net_name:
                pred_map = net([img, audio_img], gt_map)
            else:
                pred_map = net(img, gt_map)

            loss = net.loss

            #Backpropagation (ajuste de los pesos):
            loss.backward()

            #Usamos el optimizador para calcular mejor la dirección de menor pérdida:
            optimizer.step()

            #Cuando la siguiente iteración sea múltiplo de PRINT_FREQ, se mostrarán las estadísticas del training:
            if (i + 1) % PRINT_FREQ == 0:
                i_tb += 1
                print( '[ep %d][it %d][loss %.4f][lr %.4f]' % \
                        (epoch + 1, i + 1, loss.item(), optimizer.param_groups[0]['lr']*10000) )
                print( '        [cnt: gt: %.1f pred: %.2f]' % (gt_map[0].sum().data/LOG_PARA, pred_map[0].sum().data/LOG_PARA) )           




# %%
