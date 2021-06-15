import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torchvision import models

from datasets import LOG_PARA, load_datasets

# Seed for reproducibility
SEED = 3035
if SEED is not None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

# Nombre de archivo para guardar resultados
SAVE_FILENAME = 'CANNet_Prueba.pickle'
MODEL_FILENAME = 'CANNet_Prueba.pth'

# Para comprobar si tenemos GPUs disponibles para usar o no:
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

train_loader, val_loader, test_loader, restore_transform = load_datasets()


#RED
class ContextualModule(nn.Module):
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(ContextualModule, self).__init__()
        self.scales = []
        self.scales = nn.ModuleList([self._make_scale(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * 2, out_features, kernel_size=1)
        self.relu = nn.ReLU()
        self.weight_net = nn.Conv2d(features,features,kernel_size=1)

    def __make_weight(self,feature,scale_feature):
        weight_feature = feature - scale_feature
        return F.sigmoid(self.weight_net(weight_feature))

    def _make_scale(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        multi_scales = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.scales]
        weights = [self.__make_weight(feats,scale_feature) for scale_feature in multi_scales]
        overall_features = [(multi_scales[0]*weights[0]+multi_scales[1]*weights[1]+multi_scales[2]*weights[2]+multi_scales[3]*weights[3])/(weights[0]+weights[1]+weights[2]+weights[3])]+ [feats]
        bottle = self.bottleneck(torch.cat(overall_features, 1))
        return self.relu(bottle)


class CANNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CANNet, self).__init__()
        self.seen = 0
        self.context = ContextualModule(512, 512)
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512,256,128,64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat,in_channels = 512,batch_norm=True, dilation = True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            self.frontend.load_state_dict(mod.features[0:23].state_dict())
            # for i in range(len(self.frontend.state_dict().items())):
            #     self.frontend.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]

    def forward(self,x):
        x = self.frontend(x)
        x = self.context(x)
        x = self.backend(x)
        x = self.output_layer(x)
        x = F.upsample(x, scale_factor=8)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)



modelo = CANNet()
modelo = modelo.to(device)

#pytorch_total_params = sum(p.numel() for p in modelo.parameters())
#print(pytorch_total_params)

# Definimos el criterion de pérdida:
criterion = nn.MSELoss().to(device)
# criterion = nn.L1Loss(reduction='mean')

# convertimos train_loader en un iterador
# dataiter = iter(train_loader)
# y recuperamos el i-esimo elemento, un par de valores (imagenes, etiquetas)
# x, y = dataiter.next()  # x e y son tensores


# Para predecir y, la normalizaremos. Siempre por el mismo valor:
#Y_NORM = 200
# UPDATE: Ahora se normaliza en dataset.py con LOG_PARA

losses = {'train': list(), 'validacion': list(),
            'val_mae': list(), 'val_mse': list()}

# PRUEBAS:
# print(x.size())
# print(y['map'].size())
# print(torch.max(x))
# print(x)

# Parámetros (de la configuracion original)
LR = 1e-5
LR_DECAY = 0.99
NUM_EPOCH_LR_DECAY = 1
LR_DECAY_START = -1 

# ENTRENAMIENTO 1
n_epochs = 200
optimizador = optim.Adam(modelo.parameters(), lr=LR, weight_decay=1e-4)
scheduler = StepLR(optimizador, step_size=NUM_EPOCH_LR_DECAY, gamma=LR_DECAY)    

train_record = {'best_mae': float('Inf'), 'best_mse': float('Inf')}
# Para early stopping: guarda el modelo si mejora en acumulados, para si se sobrepasa MAX_ACCUM
MAX_ACCUM = 100
accum = 0

for epoch in range(n_epochs):
    print("Entrenando... \n")  # Esta será la parte de entrenamiento
    training_loss = 0.0  # el loss en cada epoch de entrenamiento
    total_iter = 0

    modelo.train()  # Para preparar el modelo para el training
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        total_iter += 1 # como estamos usando el total, auemntamos 1 y.shape[0]

        # ponemos a cero todos los gradientes en todas las neuronas:
        optimizador.zero_grad()

        pred_map = modelo(x)  # forward
        loss = criterion(pred_map.squeeze(), y.squeeze())  # evaluación del loss
        loss.backward()  # backward pass
        optimizador.step()  # optimización

        training_loss += loss.data.cpu().item()  # acumulamos el loss de este batch

    training_loss /= total_iter
    lr_computed = optimizador.param_groups[0]['lr']*10000
    orig_count = y[0].sum().data/LOG_PARA
    pred_count = pred_map[0].sum().data/LOG_PARA 
    print( f'[ep {epoch}][loss {training_loss:.4f}][lr {lr_computed:.4f}][cnt: den: {orig_count:.1f} pred: {pred_count:.1f}]')   
    losses['train'].append(training_loss)  # .item())

    # Para iniciar el scheduler (siempre tras optimizer.step())
    if epoch > LR_DECAY_START:
        scheduler.step()

    val_loss = 0.0
    mae_accum = 0.0
    mse_accum = 0.0
    total_iter = 0
    total = 0

    modelo.eval()  # Preparar el modelo para validación y/o test
    print("Validando... \n")
    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)
        total_iter += 1

        with torch.no_grad():
            # y = y/Y_NORM  # normalizamoss
            pred_map = modelo(x)
            #output = output.flatten()
            loss = criterion(pred_map.squeeze(), y.squeeze())
            val_loss += loss.cpu().item()

            for i_img in range(pred_map.shape[0]):
                pred_num = pred_map[i_img].sum().data/LOG_PARA
                y_num = y[i_img].sum().data/LOG_PARA
                mae_accum += abs(y_num-pred_num)
                mse_accum += (pred_num-y_num)*(pred_num-y_num)
                total += 1

    val_loss /= total
    mae_accum /= total_iter
    mse_accum = torch.sqrt(mse_accum/total_iter)

    losses['validacion'].append(val_loss)  # .item())
    losses['val_mae'].append(mae_accum)  # .item())
    losses['val_mse'].append(mse_accum)  # .item())
    print(
        f'[e {epoch}] \t Train: {training_loss:.4f} \t Val_loss: {val_loss:.4f}, MAE: {mae_accum:.2f}, MSE: {mse_accum:.2f}')

    # EARLY STOPPING
    if (mae_accum <= train_record['best_mae']) or (mse_accum <= train_record['best_mse']):
        print(f'Saving model...')
        torch.save(modelo, MODEL_FILENAME)
        accum = 0
        if mae_accum <= train_record['best_mae']:
            print(f'MAE: ({mae_accum:.2f}<{train_record["best_mae"]:.2f})')
            train_record['best_mae'] = mae_accum 
        if mse_accum <= train_record['best_mse']:
            print(f'MSE: ({mse_accum:.2f}<{train_record["best_mse"]:.2f})')
            train_record['best_mse'] = mse_accum 
    else: 
        accum += 1
        if accum>MAX_ACCUM:
            break

    with open(SAVE_FILENAME, 'wb') as handle:
        pickle.dump(losses, handle, protocol=pickle.HIGHEST_PROTOCOL)

'''
# ENTRENAMIENTO 2
n_epochs = 130
optimizador = optim.SGD(modelo.parameters(), lr=0.0001)

for epoch in range(n_epochs):
    print("Entrenando... \n")  # Esta será la parte de entrenamiento
    training_loss = 0.0  # el loss en cada epoch de entrenamiento
    total = 0

    modelo.train()  # Para preparar el modelo para el training
    for x, y in train_loader:
        # ponemos a cero todos los gradientes en todas las neuronas:
        optimizador.zero_grad()

        # y=y/Y_NORM #normalizamos

        x = x.to(device)
        y = y.to(device)
        total += y.shape[0]

        output = modelo(x)  # forward
        loss = criterion(output, y)  # evaluación del loss
        loss.backward()  # backward pass
        optimizador.step()  # optimización

        training_loss += loss.cpu().item()  # acumulamos el loss de este batch

    training_loss /= total
    losses['train'].append(training_loss)  # .item())

    val_loss = 0.0
    total = 0

    modelo.eval()  # Preparar el modelo para validación y/o test
    print("Validando... \n")
    for x, y in val_loader:

        # y = y/Y_NORM  # normalizamos
        x = x.to(device)
        y = y.to(device)
        total += y.shape[0]

        output = modelo(x)
        #output = output.flatten()
        loss = criterion(output, y)
        val_loss += loss.cpu().item()

    val_loss /= total
    losses['validacion'].append(val_loss)  # .item())

    print(
        f'Epoch {epoch} \t\t Training Loss: {training_loss} \t\t Validation Loss: {val_loss}')

    with open(SAVE_FILENAME, 'wb') as handle:
        pickle.dump(losses, handle, protocol=pickle.HIGHEST_PROTOCOL)
'''

# TEST
modelo.eval()  # Preparar el modelo para validación y/o test
print("Testing... \n")

# definimos la pérdida
total_iter = 0
total = 0
mse = nn.MSELoss()
test_loss = 0.0
test_loss_mse = 0.0
test_loss_mae = 0.0

yreal = list()
ypredicha = list()

with torch.no_grad():
    for x, y in test_loader:

        # y=y/Y_NORM #normalizamos

        x = x.to(device)
        y = y.to(device)
        total_iter += 1

        pred_map = modelo(x)
        #output = output.flatten()
        loss = mse(pred_map.squeeze(), y.squeeze())
        test_loss += loss.cpu().item()

        for i_img in range(pred_map.shape[0]):
            pred_num = pred_map[i_img].sum().data/LOG_PARA
            y_num = y[i_img].sum().data/LOG_PARA
            test_loss_mae += abs(y_num-pred_num)
            test_loss_mse += (pred_num-y_num)*(pred_num-y_num)
            total += 1

        output_num = pred_map.data.cpu().sum()/LOG_PARA
        y_num = y.sum()/LOG_PARA
        test_loss_mae += abs(output_num - y_num)
        test_loss_mse += (output_num-y_num)**2

        # para guardar las etqieutas.
        yreal.append(y.data.cpu().numpy())
        ypredicha.append(pred_map.data.cpu().numpy())

test_loss /= total
test_loss_mae /= total_iter
test_loss_mse = torch.sqrt(test_loss_mse/total_iter)

# yreal = np.array(yreal).flatten()
# ypredicha = np.array(ypredicha).flatten() # comprobar si funciona.

losses['yreal'] = yreal
losses['ypredicha'] = ypredicha

print(f'Test Loss (MSE): {test_loss_mse}')
losses['test_mse'] = test_loss_mse  # .item())
print(f'Test Loss (MAE): {test_loss_mae}')
losses['test_mae'] = test_loss_mae  # .item())

with open(SAVE_FILENAME, 'wb') as handle:
    pickle.dump(losses, handle, protocol=pickle.HIGHEST_PROTOCOL)


#%% VISUALIZATION
# fig, ax = plt.subplots(2, 1, figsize=(30,10))
# img_orig = x.data.cpu().numpy().squeeze().transpose((1,2,0))
# img_orig = (img_orig-img_orig.min())/(img_orig.max()-img_orig.min())
# ax[0].imshow(img_orig)
# ax[1].imshow(y.data.cpu().numpy().squeeze())