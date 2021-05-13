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
import pickle

# Nombre de archivo para guardar resultados
SAVE_FILENAME = 'UNET_MSEsum_(15)0.01_(25)0.001_batch1(eval_y_train).pickle'


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
# print(testset.__getitem__(20))


train_batch_size = 1
eval_batch_size = 1
# train BATCH_SIZE: pequeño (1-3)
train_loader = DataLoader(trainset, train_batch_size, shuffle=True)
val_loader = DataLoader(valset, eval_batch_size, shuffle=False)
test_loader = DataLoader(testset, eval_batch_size, shuffle=False)


# RED UNET

# Función necesaria para UNET
def crop_img(tensor, target_tensor):
    target_size = target_tensor.size()[2:]
    tensor_size = tensor.size()[2:]
    delta = torch.as_tensor(tensor_size, dtype=int) - \
        torch.as_tensor(target_size, dtype=int)
    delta = torch.as_tensor(np.c_[delta//2, delta % 2])
    return tensor[:, :, delta[0, 0]+delta[0, 1]:tensor_size[0]-delta[0, 0], delta[1, 0]+delta[1, 1]:tensor_size[1]-delta[1, 0]]



class UNET(nn.Module):

    def __init__(self):
        super(UNET, self).__init__()
        self.contracting_11 = self.conv_block(in_channels=3, out_channels=16)
        self.contracting_12 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_21 = self.conv_block(in_channels=16, out_channels=32)
        self.contracting_22 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_31 = self.conv_block(in_channels=32, out_channels=64)
        self.contracting_32 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_41 = self.conv_block(in_channels=64, out_channels=128)
        self.contracting_42 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.middle = self.conv_block(in_channels=128, out_channels=512)
        self.expansive_11 = nn.ConvTranspose2d(
            in_channels=512, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_12 = self.conv_block(in_channels=256, out_channels=128)
        self.expansive_21 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_22 = self.conv_block(in_channels=128, out_channels=64)
        self.expansive_31 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_32 = self.conv_block(in_channels=64, out_channels=32)
        self.expansive_41 = nn.ConvTranspose2d(
            in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_42 = self.conv_block(in_channels=32, out_channels=16)
        self.output = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=(5,1))

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                              nn.ReLU(),
                              nn.BatchNorm2d(num_features=out_channels),
                              nn.Conv2d(
                                  in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                              nn.ReLU(),
                              nn.BatchNorm2d(num_features=out_channels))
        return block

    def forward(self, x):
        contracting_11_out = self.contracting_11(x)  # [-1, 64, 256, 256]
        x = self.contracting_12(
            contracting_11_out)  # [-1, 64, 128, 128]
        contracting_21_out = self.contracting_21(x)  # [-1, 128, 128, 128]
        x = self.contracting_22(
            contracting_21_out)  # [-1, 128, 64, 64]
        contracting_31_out = self.contracting_31(x)  # [-1, 256, 64, 64]
        x = self.contracting_32(
            contracting_31_out)  # [-1, 256, 32, 32]
        contracting_41_out = self.contracting_41(x)  # [-1, 512, 32, 32]
        x = self.contracting_42(
            contracting_41_out)  # [-1, 512, 16, 16]
        x = self.middle(x)  # [-1, 1024, 16, 16]
        x = self.expansive_11(x)  # [-1, 512, 32, 32]
        contracting_41_out = crop_img(contracting_41_out, x)
        x = self.expansive_12(torch.cat(
            (x, contracting_41_out), dim=1))  # [-1, 1024, 32, 32] -> [-1, 512, 32, 32]
        x = self.expansive_21(x)  # [-1, 256, 64, 64]
        contracting_31_out = crop_img(contracting_31_out, x)
        x = self.expansive_22(torch.cat(
            (x, contracting_31_out), dim=1))  # [-1, 512, 64, 64] -> [-1, 256, 64, 64]
        x = self.expansive_31(x)  # [-1, 128, 128, 128]
        contracting_21_out = crop_img(contracting_21_out, x)
        # [-1, 256, 128, 128] -> [-1, 128, 128, 128]
        x = self.expansive_32(
            torch.cat((x, contracting_21_out), dim=1))
        x = self.expansive_41(x)  # [-1, 64, 256, 256]
        contracting_11_out = crop_img(contracting_11_out, x)
        # [-1, 128, 256, 256] -> [-1, 64, 256, 256]
        x = self.expansive_42(
            torch.cat((x, contracting_11_out), dim=1))
        # [-1, num_classes, 256, 256]
        x = self.output(x)
        return x


modelo = UNET()

#pytorch_total_params = sum(p.numel() for p in modelo.parameters())
#print(pytorch_total_params)

modelo = modelo.to(device)
# Definimos el criterion de pérdida:
criterion = nn.MSELoss(reduction='sum')
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
n_epochs = 15
optimizador = optim.Adam(modelo.parameters(), lr=0.01, weight_decay=1e-4)

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


# ENTRENAMIENTO 2
n_epochs = 25
optimizador = optim.SGD(modelo.parameters(), lr=0.001)

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
