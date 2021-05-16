#PROBAR LUEGO EL DATA PARALELL DE CUDA Y MÁS OPTIMIZACIÓN DE ESTE ESTILO!
#PROBAR PONER SCHEDULER TRAS OPTIM PARA QUE NO SE SALTE EL VALOR INICIAL DE LR
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from scipy.io import loadmat
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import optim
from torchvision import models
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F
import pickle

# Nombre de archivo para guardar resultados
SAVE_FILENAME = 'CANNet_MSEmean_(20)Adam1e-5_batch2(eval_y_train).pickle'


# Para comprobar si tenemos GPUs disponibles para usar o no:
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


image_path = '/media/NAS/home/cristfg/datasets/imgs/'
train_density_path = '/media/NAS/home/cristfg/datasets/density/train/'
val_density_path = '/media/NAS/home/cristfg/datasets/density/val/'
test_density_path = '/media/NAS/home/cristfg/datasets/density/test/'

'''
image_path = '/home/pakitochus/Descargas/propuestas_tfg_cristina/crowd/definitivo/DISCO_dataset/imgs/'
train_density_path = '/home/pakitochus/Descargas/propuestas_tfg_cristina/crowd/definitivo/DISCO_dataset/density/train/'
val_density_path = '/home/pakitochus/Descargas/propuestas_tfg_cristina/crowd/definitivo/DISCO_dataset/density/val/'
test_density_path = '/home/pakitochus/Descargas/propuestas_tfg_cristina/crowd/definitivo/DISCO_dataset/density/test/'
'''

'''
image_path = '/Volumes/Cristina /TFG/Data/imgs/'
train_density_path = '/Volumes/Cristina /TFG/Data/density/train/'
val_density_path = '/Volumes/Cristina /TFG/Data/density/val/'
test_density_path = '/Volumes/Cristina /TFG/Data/density/test/'	
'''


class ImageDataset(Dataset):
    def __init__(self, image_path, density_path):

        self.density_path = density_path
        self.image_path = image_path

        self.mapfiles = os.listdir(self.density_path)
        # Para no incluir los archivos con '._':
        self.mapfiles = [
            el for el in self.mapfiles if el.startswith('._') == False]
        self.mapfiles_wo_ext = [el[:-4] for el in self.mapfiles]

        self.imagefiles = os.listdir(image_path)
        self.imagefiles_wo_ext = [el[:-4] for el in self.imagefiles]
        self.imagefiles = [
            el + '.jpg' for el in self.imagefiles_wo_ext if el in self.mapfiles_wo_ext]

    def __len__(self):
        return len(self.imagefiles)

    def __getitem__(self, idx):

        # DENSITY MAP
        map_path = self.density_path + self.mapfiles[idx]
        y = loadmat(map_path)
        y = torch.as_tensor(y['map'], dtype=torch.float32)
        y = y.unsqueeze(0)

        # IMAGES
        filename = str(self.imagefiles[idx])
        filename = filename.lstrip("['")
        filename = filename.rstrip("']")
        img_path = self.image_path + filename
        # Cargamos la imagen:
        x = plt.imread(img_path).copy()

        x = x.transpose((2, 0, 1))  # Cambiar posición
        x = torch.as_tensor(x, dtype=torch.float32)
        # X normalizada a los 255 valores de brillo:
        x = x / 255.0

        return x, y


# CARGAMOS LOS DATOS:
trainset = ImageDataset(image_path, train_density_path)
valset = ImageDataset(image_path, val_density_path)
testset = ImageDataset(image_path, test_density_path)

# PRUEBA
# print(trainset.__getitem__(20))
#torch.set_printoptions(profile="full")
#print(testset.__getitem__(70)[1])
#print(testset.__getitem__(70)[1].sum())


train_batch_size = 2
eval_batch_size = 2
# train BATCH_SIZE: pequeño (1-3)
train_loader = DataLoader(trainset, train_batch_size, shuffle=True)
val_loader = DataLoader(valset, eval_batch_size, shuffle=False)
test_loader = DataLoader(testset, eval_batch_size, shuffle=False)



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

#pytorch_total_params = sum(p.numel() for p in modelo.parameters())
#print(pytorch_total_params)

modelo = modelo.to(device)
# Definimos el criterion de pérdida:
criterion = nn.MSELoss() #(reduction='mean')
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


# ENTRENAMIENTO 1
n_epochs = 20
optimizador = optim.Adam(modelo.parameters(), lr=1e-5, weight_decay=1e-4)
scheduler = StepLR(optimizador, step_size=1, gamma=0.99)
LR_DECAY_START=-1; #Cuando se sobrepasa esta epoch, el LR empezará a decaer.

for epoch in range(n_epochs):
    print("Entrenando... \n")  # Esta será la parte de entrenamiento
    training_loss = 0.0  # el loss en cada epoch de entrenamiento
    total = 0

    if epoch > LR_DECAY_START:
        scheduler.step()
                

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
