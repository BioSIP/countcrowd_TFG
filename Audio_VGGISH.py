from losses import LogCoshLoss
import os
from scipy.io import loadmat
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import torchaudio
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
# https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html

# Para comprobar si tenemos GPUs disponibles para usar o no:
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class AudioDataset(Dataset):
    def __init__(self, audio_path, density_path, transform=None):

        # 3 opciones para el density_path:
        #density_path = '/Volumes/Cristina /TFG/Data/density/train'
        #density_path = '/Volumes/Cristina /TFG/Data/density/test'
        #density_path = '/Volumes/Cristina /TFG/Data/density/val'

        self.density_path = density_path
        self.audio_path = audio_path
        self.transform = transform

        self.mapfiles = os.listdir(self.density_path)
        # Para no incluir los archivos con '._':
        self.mapfiles = [
            el for el in self.mapfiles if el.startswith('._') == False]
        self.mapfiles_wo_ext = [el[:-4] for el in self.mapfiles]

        # list comprehension

        #audio_path = '/Volumes/Cristina /TFG/Data/auds/'
        self.audiofiles = os.listdir(audio_path)
        self.audiofiles_wo_ext = [el[:-4] for el in self.audiofiles]
        self.audiofiles = [
            el + '.wav' for el in self.audiofiles_wo_ext if el in self.mapfiles_wo_ext]

        self.audiofiles = sorted(self.audiofiles)
        self.mapfiles = sorted(self.mapfiles)
        # Añadir extensiones a archivos de audio:
        # for i in range(len(self.audiofiles)):
        # Añado la extensión al nombre del archivo que quiero importar:
        #self.audiofiles[i] = [self.audiofiles[i] + '.wav']

    def __len__(self):
        return len(self.audiofiles)

    def __getitem__(self, idx):

        # DENSITY MAP
        map_path = self.density_path + self.mapfiles[idx]
        mapa = loadmat(map_path)
        y = torch.as_tensor(mapa['map'].sum(), dtype=torch.float32)
        # AUDIO
        # Encuentro el path del archivo:
        filename = str(self.audiofiles[idx])
        filename = filename.lstrip("['")
        filename = filename.rstrip("']")
        aud_path = self.audio_path + filename
        # Cargamos el audio:
        waveform, sample_rate = torchaudio.load(
            aud_path)  # waveform es un tensor
        # SE USARÁ EL SAMPLE_RATE PARA ALGO????????????????

        x = waveform.view((2, 1, -1))  # dimensiones
        if self.transform:
            x = self.transform(x)
        return x, y

# class SpectrogramDataset(Dataset):
    # PROGRAMAR LUEGO!!!


audio_path = '/home/pakitochus/Descargas/propuestas_tfg_cristina/crowd/definitivo/DISCO_dataset/auds/'
train_density_path = '/home/pakitochus/Descargas/propuestas_tfg_cristina/crowd/definitivo/DISCO_dataset/density/train/'
val_density_path = '/home/pakitochus/Descargas/propuestas_tfg_cristina/crowd/definitivo/DISCO_dataset/density/val/'
test_density_path = '/home/pakitochus/Descargas/propuestas_tfg_cristina/crowd/definitivo/DISCO_dataset/density/test/'


trainset = AudioDataset(audio_path, train_density_path)
valset = AudioDataset(audio_path, val_density_path)
testset = AudioDataset(audio_path, test_density_path)

# PRUEBA para ver tensores de audio y de mapas de los conjuntos de train y val:
# print(trainset.__getitem__(20))
# print(valset.__getitem__(20))

#BATCH_SIZE: pequeño (1-3)
batch_size = 48
# BATCH_SIZE: pequeño (1-3)
train_loader = DataLoader(trainset, batch_size, shuffle=True)
val_loader = DataLoader(valset, 32, shuffle=False)
test_loader = DataLoader(testset, 32, shuffle=False)

# RED:
'''
#Por si quiero probar luego con LeNet (CAMBIAR INPUTS!):
class LeNet(nn.Module):
	def __init__(self):
		super(LeNet, self).__init__() # esta linea es siempre necesaria
		self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
		self.mp1 = nn.MaxPool2d(1,2)
		self.conv2 = nn.Conv2d(6, 16, 5, padding=2)
		self.mp2 = nn.MaxPool2d(2)
		self.conv3 = nn.Conv2d(16, 120, 3, padding=1)
		self.fc1 = nn.Linear(7*7*120, 256)#capa oculta
		self.fc2 = nn.Linear(256, 10)#capa de salida
		
	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = self.mp1(x)
		x = F.relu(self.conv2(x))
		x = self.mp2(x)
		x = F.relu(self.conv3(x))
		x = x.view(-1, 7*7*120)
		x = F.relu(self.fc1(x))#Función de activación relu en la salida de la capa oculta
		x = F.softmax(self.fc2(x), dim=1)#Función de activación softmax en la salida de la capa oculta
		return x
'''

# MaxPool2d((1,2))
# torch.nn.Conv2d(in_channels, out_channels, kernel_size) -> kernel_size = (1, 61)
# in_channels ->2, out_channels -> [32,64].
# optim - > adam


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        assert len(x.size()) == 4, x.size()
        B, C, W, H = x.size()
        return F.avg_pool2d(x, (W, H)).view(B, C)


class GlobalMaxPool2d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool2d, self).__init__()

    def forward(self, x):
        assert len(x.size()) == 4, x.size()
        B, C, W, H = x.size()
        return F.max_pool2d(x, (W, H)).view(B, C)


class VGGish(nn.Module):
    """
    PyTorch implementation of the VGGish model.

    Adapted from: https://github.com/harritaylor/torch-vggish
    The following modifications were made: (i) correction for the missing ReLU layers, (ii) correction for the
    improperly formatted data when transitioning from NHWC --> NCHW in the fully-connected layers, and (iii)
    correction for flattening in the fully-connected layers.
    """

    def __init__(self):
        super(VGGish, self).__init__()
        self.pools = [4, 4, 2, 2]
        self.features = nn.Sequential(
            nn.Conv2d(2, 64, (1, 7),  stride=(1, self.pools[0]), padding=(0, 3)),
            nn.ELU(),

            nn.Conv2d(64, 128, (1, 5), stride=(1, self.pools[1]), padding=(0, 2)),
            nn.ELU(),

            nn.Conv2d(128, 256, (1, 3), stride=1, padding=(0, 1)),
            nn.ELU(),
            nn.Conv2d(256, 256, (1, 3), stride=(1, self.pools[2]), padding=(0, 1)),
            nn.ELU(),

            nn.Conv2d(256, 512, (1, 3), stride=1, padding=(0, 1)),
            nn.ELU(),
            nn.Conv2d(512, 1024, (1, 3), stride=1, padding=(0, 1)),
            nn.ELU(),
            # nn.MaxPool2d((1,self.pools[3]), stride=self.pools[3])
        )
        self.avgpool = GlobalAvgPool2d()
        self.fc = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.Linear(4096, 1)
        )
        # así y todo se nos queda en 1572864000

    def forward(self, x):
        x = self.features(x)  # .permute(0, 2, 3, 1).contiguous()
        x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        x = self.fc(x)
        # b, c, w, h = x.shape
        # x = x.view(b, c, -1).mean(-1)
        return x


modelo = VGGish()
modelo = modelo.to(device)
criterion = nn.MSELoss()  # definimos la pérdida
# criterion = LogCoshLoss(reduction='sum')
optimizador = optim.Adam(modelo.parameters(), lr=0.01, weight_decay=1e-4)
# optimizador = optim.SGD(modelo.parameters(), lr=1e-4)
# print(modelo)

# print(train_loader)
# print(type(train_loader))

# TENGO QUE HACER ESTO O NO?
# convertimos train_loader en un iterador
dataiter = iter(train_loader)
# y recuperamos el i-esimo elemento, un par de valores (imagenes, etiquetas)
x, y = dataiter.next()

# print(x)
# print(x.size())
# print(y)
# print(y.size())

# Para predecir y, la normalizaremos. Siempre por el mismo valor:
Y_NORM = 100

losses = {'train': list(), 'validacion': list()}
min_val_loss = float('Inf') 
expcode = 'vggish_adam_mse'

for epoch in range(20):
    print("Entrenando... \n")  # Esta será la parte de entrenamiento
    training_loss = 0.0  # el loss en cada epoch de entrenamiento
    total = 0

    modelo.train()  # Para preparar el modelo para el training
    for x, y in train_loader:

        total += 1
        # ponemos a cero todos los gradientes en todas las neuronas:
        optimizador.zero_grad()

        y = y/Y_NORM  # normalizamos

        x = x.to(device)
        y = y.to(device)

        output = modelo(x)  # forward
        loss = criterion(output.squeeze(), y.squeeze())  # evaluación del loss
        loss.backward()  # backward pass
        optimizador.step()  # optimización

        training_loss += loss.item()  # acumulamos el loss de este batch

    training_loss *= Y_NORM/total
    losses['train'].append(training_loss)  # .item())

    val_loss = 0.0
    total = 0
    modelo.eval()  # Preparar el modelo para validación y/o test
    print("Validando... \n")
    for x, y in val_loader:
        total += 1

        y = y/Y_NORM  # normalizamos ¿AQUÍ TAMBIÉN?

        x = x.to(device)
        y = y.to(device)

        output = modelo(x)
        loss = criterion(output.squeeze(), y.squeeze())
        val_loss += loss.item()

    val_loss *= Y_NORM/total
    if val_loss <= min_val_loss:
        min_val_loss = val_loss
        filename = expcode+'.pt'
        print(f'Saving as {filename}')
        torch.save(modelo, filename)

    losses['validacion'].append(val_loss)  # .item())
    print(
        f'Epoch {epoch} \t\t Training Loss: {training_loss} \t\t Validation Loss: {val_loss}')


# ENTRENAMIENTO
n_epochs = 500

modelo = torch.load(filename)
optimizador = optim.SGD(modelo.parameters(), lr=1e-7, momentum=0.9)
epoch_ni = 0 # epochs not improving. 

for epoch in range(n_epochs):
    print("Entrenando... \n")  # Esta será la parte de entrenamiento
    training_loss = 0.0  # el loss en cada epoch de entrenamiento
    total = 0

    modelo.train()  # Para preparar el modelo para el training
    for x, y in train_loader:

        total += 1
        # ponemos a cero todos los gradientes en todas las neuronas:
        optimizador.zero_grad()

        y = y/Y_NORM  # normalizamos

        x = x.to(device)
        y = y.to(device)

        output = modelo(x)  # forward
        loss = criterion(output.squeeze(), y.squeeze())  # evaluación del loss
        # print(f'loss: {loss}')
        loss.backward()  # backward pass
        optimizador.step()  # optimización

        training_loss += loss.item()  # acumulamos el loss de este batch

    training_loss *= Y_NORM/total
    losses['train'].append(training_loss)  # .item())
    val_loss = 0.0
    total = 0
    modelo.eval()  # Preparar el modelo para validación y/o test
    print("Validando... \n")
    for x, y in val_loader:
        total += 1

        y = y/Y_NORM  # normalizamos ¿AQUÍ TAMBIÉN?

        x = x.to(device)
        y = y.to(device)

        output = modelo(x)  # forward
        loss = criterion(output.squeeze(), y.squeeze())  # evaluación del loss
        val_loss += loss.item()

    val_loss *= Y_NORM/total
    if val_loss <= min_val_loss:
        min_val_loss = val_loss
        filename = expcode+'.pt'
        print(f'Saving as {filename}')
        torch.save(modelo, filename)
        epoch_ni = 0
    else:
        epoch_ni +=1
        if epoch_ni > 100:
            break

    losses['validacion'].append(val_loss)  # .item())
    print(
        f'Epoch {epoch} \t\t Training Loss: {training_loss} \t\t Validation Loss: {val_loss}')



# TEST

modelo = torch.load(filename)
modelo.eval()  # Preparar el modelo para validación y/o test
print("Testing... \n")
total = 0

mse = nn.MSELoss(reduction='sum')  # definimos la pérdida
mae = nn.L1Loss(reduction='sum')
test_loss_mse = 0.0
test_loss_mae = 0.0

yreal = list()
ypredicha = list()

for x, y in test_loader:

    y = y/Y_NORM  # normalizamos

    x = x.to(device)
    y = y.to(device)
    total += y.shape[0]

    output = modelo(x)
    output = output.squeeze()
    mse_loss = mse(output, y)
    test_loss_mse += mse_loss.cpu().item()
    mae_loss = mae(output, y)
    test_loss_mae += mae_loss.cpu().item()

    # para guardar las etqieutas.
    yreal.append(y.detach().cpu().numpy())
    ypredicha.append(output.detach().cpu().numpy())

# Esto siemrpe que reduction='sum' -> equiparable a número de personas.
test_loss_mse *= Y_NORM/total
# Esto siemrpe que reduction='sum' -> equiparable a número de personas.
test_loss_mae *= Y_NORM/total

# yreal = np.array(yreal).flatten()
# ypredicha = np.array(ypredicha).flatten() # comprobar si funciona.

losses['yreal'] = yreal
losses['ypredicha'] = ypredicha

print(f'Test Loss (MSE): {test_loss_mse}')
losses['test_mse'] = test_loss_mse  # .item())
print(f'Test Loss (MAE): {test_loss_mae}')
losses['test_mae'] = test_loss_mae  # .item())
#%% 
'''
Testing... 

Test Loss (MSE): 122.91883238156636
Test Loss (MAE): 74.75587590535481


'''