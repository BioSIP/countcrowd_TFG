import torch
import torchvision
import torch.nn as nn
import numpy as np
from scipy.io import loadmat
import os
import matplotlib.pyplot as plt
from torch import optim
import pickle
from datasets import load_datasets
from torch.optim.lr_scheduler import StepLR

# Nombre de archivo para guardar resultados
SAVE_FILENAME = 'UNET_MSEsum_(120)Adam0.01_batch2(eval_y_train).pickle'


# Para comprobar si tenemos GPUs disponibles para usar o no:
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

train_loader, val_loader, test_loader, restore_transform = load_datasets()

from torchvision import models
import torch.nn.functional as F

class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            self.frontend.load_state_dict(mod.features[0:23].state_dict())

    def forward(self,x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        x = F.interpolate(x,scale_factor=8)
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



modelo = CSRNet()

#pytorch_total_params = sum(p.numel() for p in modelo.parameters())
#print(pytorch_total_params)

modelo = modelo.to(device)
# Definimos el criterion de pérdida:
criterion = nn.MSELoss(reduction='mean')
# criterion = nn.L1Loss(reduction='sum')

# convertimos train_loader en un iterador
dataiter = iter(train_loader)
# y recuperamos el i-esimo elemento, un par de valores (imagenes, etiquetas)
x, y = dataiter.next()  # x e y son tensores


# PODEMOS NORMALIZAR EL MAPA DE DENSIDAD???!!!??
# Para predecir y, la normalizaremos. Siempre por el mismo valor:
#Y_NORM = 200

losses = {'train': list(), 'validacion': list()}

# PRUEBAS:
# print(x.size())
# print(y['map'].size())
# print(torch.max(x))
# print(x)

# Parámetros (de la configuracion china original)
LR = 1e-5
WD = 1e-4
NUM_EPOCH_LR_DECAY = 1
LR_DECAY = 0.99

# ENTRENAMIENTO 1
n_epochs = 120
optimizador = optim.Adam(modelo.parameters(), lr=LR, weight_decay=1e-4)
scheduler = StepLR(optimizador, step_size=NUM_EPOCH_LR_DECAY, gamma=LR_DECAY)    

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
        loss = criterion(output.squeeze(), y.squeeze())  # evaluación del loss
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
        loss = criterion(output.squeeze(), y.squeeze())
        val_loss += loss.cpu().item()

    val_loss /= total
    losses['validacion'].append(val_loss)  # .item())

    print(
        f'Epoch {epoch} \t\t Training Loss: {training_loss} \t\t Validation Loss: {val_loss}')

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
total = 0

# definimos la pérdida
mse = nn.MSELoss(reduction='sum')
mae = nn.L1Loss(reduction='sum')
test_loss_mse = 0.0
test_loss_mae = 0.0

yreal = list()
ypredicha = list()

for x, y in test_loader:

    # y=y/Y_NORM #normalizamos

    x = x.to(device)
    y = y.to(device)
    total += y.shape[0]

    output = modelo(x)
    #output = output.flatten()
    mse_loss = mse(output, y)
    test_loss_mse += mse_loss.cpu().item()
    mae_loss = mae(output, y)
    test_loss_mae += mae_loss.cpu().item()

    # para guardar las etqieutas.
    yreal.append(y.detach().cpu().numpy())
    ypredicha.append(output.detach().cpu().numpy())

# Esto siemrpe que reduction='sum' -> equiparable a número de personas.
test_loss_mse /= total # *= Y_NORM/total
# Esto siemrpe que reduction='sum' -> equiparable a número de personas.
test_loss_mae /= total # *= Y_NORM/total

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
