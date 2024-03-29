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
from torch.optim.lr_scheduler import StepLR
import pickle
# https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html

# Para comprobar si tenemos GPUs disponibles para usar o no:
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


SAVE_FILENAME = 'crisnet_spectrogram_mselog_P2.pickle'


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

		x = waveform.view((2, 1, -1))  # dimensiones
		if self.transform:
			x = self.transform(x)
		return x, y


class SpectrogramDataset(Dataset):
	def __init__(self, audio_path, density_path, transform=None):

		#3 opciones para el density_path:
		#density_path = '/Volumes/Cristina /TFG/Data/density/train'
		#density_path = '/Volumes/Cristina /TFG/Data/density/test'
		#density_path = '/Volumes/Cristina /TFG/Data/density/val'

		self.density_path=density_path
		self.audio_path=audio_path
		self.transform=transform

		self.mapfiles = os.listdir(self.density_path)
		#Para no incluir los archivos con '._':
		self.mapfiles = [el for el in self.mapfiles if el.startswith('._')==False]
		self.mapfiles_wo_ext=[el[:-4] for el in self.mapfiles]

		# list comprehension

		#audio_path = '/Volumes/Cristina /TFG/Data/auds/'
		self.audiofiles = os.listdir(audio_path)
		self.audiofiles_wo_ext=[el[:-4] for el in self.audiofiles]
		self.audiofiles = [el + '.wav' for el in self.audiofiles_wo_ext if el in self.mapfiles_wo_ext]

		self.audiofiles = sorted(self.audiofiles)
		self.mapfiles = sorted(self.mapfiles)

		#Anadir extensiones a archivos de audio:
		#for i in range(len(self.audiofiles)):
			#Anado la extension al nombre del archivo que quiero importar:
			#self.audiofiles[i] = [self.audiofiles[i] + '.wav']
		
	
	def __len__(self):
		return len(self.audiofiles)
		
	def __getitem__(self, idx):

		#DENSITY MAP
		map_path = self.density_path + self.mapfiles[idx]
		mapa = loadmat(map_path)
		y = torch.as_tensor(mapa['map'].sum(), dtype=torch.float32) 
		#AUDIO
		#Encuentro el path del archivo:
		filename=str(self.audiofiles[idx])
		filename=filename.lstrip("['")
		filename=filename.rstrip("']")
		aud_path = self.audio_path + filename
		#Cargamos el audio:
		waveform, sample_rate = torchaudio.load(aud_path)   #waveform es un tensor

		x = torchaudio.transforms.Spectrogram()(waveform)

		#print("Shape of spectrogram: {}".format(x.size()))

		#plt.figure()
		#plt.imshow(x.log2()[0,:,:].numpy(), cmap='gray')

		return x, y
'''
audio_path = '/home/pakitochus/Descargas/propuestas_tfg_cristina/crowd/definitivo/DISCO_dataset/auds/'
train_density_path = '/home/pakitochus/Descargas/propuestas_tfg_cristina/crowd/definitivo/DISCO_dataset/density/train/'
val_density_path = '/home/pakitochus/Descargas/propuestas_tfg_cristina/crowd/definitivo/DISCO_dataset/density/val/'
test_density_path = '/home/pakitochus/Descargas/propuestas_tfg_cristina/crowd/definitivo/DISCO_dataset/density/test/'
'''
audio_path = '/media/NAS/home/cristfg/datasets/auds/'
train_density_path = '/media/NAS/home/cristfg/datasets/density/train/'
val_density_path = '/media/NAS/home/cristfg/datasets/density/val/'
test_density_path = '/media/NAS/home/cristfg/datasets/density/test/'

trainset = SpectrogramDataset(audio_path, train_density_path)
valset = SpectrogramDataset(audio_path, val_density_path)
testset = SpectrogramDataset(audio_path, test_density_path)

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
			nn.Conv2d(2, 64, 3,  stride=(1, self.pools[0]), padding=3),
			nn.ReLU(),

			nn.Conv2d(64, 128, 5, stride=(1, self.pools[1]), padding=2),
			nn.ReLU(),

			nn.Conv2d(128, 256, 3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(256, 256, 3, stride=(1, self.pools[2]), padding=1),
			nn.ReLU(),

			nn.Conv2d(256, 512, 3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(512, 1024, 3, stride=1, padding=1),
			nn.ReLU(),
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


class CrisNet(nn.Module):
	def __init__(self):
		super(CrisNet, self).__init__() # esta linea es siempre necesaria
		self.max_pool1 = nn.MaxPool2d((1,2))
		self.max_pool2 = nn.MaxPool2d((1,2))
		self.max_pool3 = nn.MaxPool2d((1,2))
		self.max_pool4 = nn.MaxPool2d((1,2))
		self.max_pool5 = nn.MaxPool2d((1,2))
		self.max_pool6 = nn.MaxPool2d((1,2))

		self.conv1 = nn.Conv2d(2, 32, 3, stride=1, padding=1)
		self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
		self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
		self.conv4 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
		self.conv5 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
		self.conv6 = nn.Conv2d(512, 1024, 3, stride=1, padding=1)

		self.fc1 = nn.Linear(617472,1)
		

	def forward(self, x):
		#Con función de activación ReLu

		#PRIMERA CAPA
		x = F.relu(self.conv1(x))
		x = self.max_pool1(x)


		#SEGUNDA CAPA
		x = F.relu(self.conv2(x))
		x = self.max_pool2(x)

		#TERCERA CAPA
		x = F.relu(self.conv3(x))
		x = self.max_pool3(x)

		#CUARTA CAPA
		x = F.relu(self.conv4(x))
		x = self.max_pool4(x)

		#QUINTA CAPA
		x = F.relu(self.conv5(x))
		x = self.max_pool5(x)


		#SEXTA CAPA
		x = F.relu(self.conv6(x))
		x = self.max_pool6(x)


		x = x.view((x.size(0),-1))
		#print(x.size())
		x = self.fc1(x)


		return x

modelo=CrisNet()
modelo = modelo.to(device)
criterion = nn.MSELoss()  # definimos la pérdida
# criterion = LogCoshLoss(reduction='sum')
optimizador = optim.Adam(modelo.parameters(), lr=1e-3)#, weight_decay=1e-4)
# optimizador = optim.SGD(modelo.parameters(), lr=1e-4)
# print(modelo)

# print(train_loader)
# print(type(train_loader))


# print(x)
# print(x.size())
# print(y)
# print(y.size())

def get_mae_mse(ypred, y, transform):
	mse_loss = 0.0
	mae_loss = 0.0
	total = 0 
	ypred, y = transform(ypred.data), transform(y.data)
	for ix in range(y.shape[0]):
		total += 1
		mae_loss += abs(ypred[ix] - y[ix])
		mse_loss += (ypred[ix] - y[ix])*(ypred[ix] - y[ix])
	return mae_loss/total, torch.sqrt(mse_loss/total)

# Para predecir y, la normalizaremos. Siempre por el mismo valor:
Y_NORM = 10

losses = {'train': list(), 'validacion': list()}
min_val_loss = {'mae': float('Inf'), 'mse': float('Inf'), 'loss': float('Inf') }
expcode = 'crisnet_spectrogram_mselog_P2'

def transform(x):
	return torch.log(x/Y_NORM)

def inverse_transform(x):
	return Y_NORM*torch.exp(torch.as_tensor(x))

for epoch in range(100):
	print("Entrenando... \n")  # Esta será la parte de entrenamiento
	training_loss = 0.0  # el loss en cada epoch de entrenamiento
	total = 0

	modelo.train()  # Para preparar el modelo para el training
	for x, y in train_loader:

		total += 1
		# ponemos a cero todos los gradientes en todas las neuronas:
		optimizador.zero_grad()

		y = transform(y)  # normalizamos

		x = x.to(device)
		y = y.to(device)

		output = modelo(x)  # forward
		loss = criterion(output.squeeze(), y.squeeze())  # evaluación del loss
		# print(f'loss: {loss:.4f}')
		loss.backward()  # backward pass
		optimizador.step()  # optimización

		training_loss += loss.item()  # acumulamos el loss de este batch

	training_loss = training_loss/total
	losses['train'].append(training_loss)  # .item())

	val_loss = 0.0
	mae, mse = 0,0 
	total = 0
	modelo.eval()  # Preparar el modelo para validación y/o test
	print("Validando... \n")
	for x, y in val_loader:
		total += 1

		y = transform(y)  # normalizamos

		x = x.to(device)
		y = y.to(device)

		output = modelo(x)
		loss = criterion(output.squeeze(), y.squeeze())
		val_loss += loss.item()
		mae_accum, mse_accum = get_mae_mse(output.squeeze(), y.squeeze(), inverse_transform)
		mae += mae_accum
		mse += mse_accum

	val_loss  = val_loss/total
	mae = mae/total
	mse = mse/total
	losses['validacion'].append(val_loss)  # .item())
	print(f'[ep {epoch}] [Train: {training_loss:.4f}][Val: {val_loss:.4f}]')
	print(f'\t[Val MAE: {mae:.4f}][Val MSE: {mse:.4f}]')

	# Early stopping
	if (val_loss <= min_val_loss['loss']) or (mse <= min_val_loss['mse']) or (mae<=min_val_loss['mae']):
		filename = expcode+'.pt'
		print(f'Saving as {filename}')
		torch.save(modelo, filename)

		if val_loss <= min_val_loss['loss']:
			min_val_loss['loss'] = val_loss
		if mse <= min_val_loss['mse']:
			min_val_loss['mse'] = mse 
		if mae<=min_val_loss['mae']: 
			min_val_loss['mae'] = mae


last_epoch = epoch
# last_lr = scheduler.get_last_lr()

# ENTRENAMIENTO
n_epochs = 500

LR_DECAY = 0.99
NUM_EPOCH_LR_DECAY = 1
modelo = torch.load(filename)
optimizador = optim.SGD(modelo.parameters(), lr=1e-4, momentum=0.9)
scheduler = StepLR(optimizador, step_size=NUM_EPOCH_LR_DECAY, gamma=LR_DECAY)    

epoch_ni = 0 # epochs not improving. 
MAX_ITER = 100

for epoch in range(last_epoch, n_epochs):
	print("Entrenando... \n")  # Esta será la parte de entrenamiento
	training_loss = 0.0  # el loss en cada epoch de entrenamiento
	total = 0

	modelo.train()  # Para preparar el modelo para el training
	for x, y in train_loader:

		total += 1
		# ponemos a cero todos los gradientes en todas las neuronas:
		optimizador.zero_grad()

		y = transform(y)  # normalizamos

		x = x.to(device)
		y = y.to(device)

		output = modelo(x)  # forward
		loss = criterion(output.squeeze(), y.squeeze())  # evaluación del loss
		# print(f'loss: {loss}')
		loss.backward()  # backward pass
		optimizador.step()  # optimización

		training_loss += loss.item()  # acumulamos el loss de este batch

	training_loss = training_loss/total
	losses['train'].append(training_loss)  # .item())
	val_loss = 0.0
	total = 0
	mae, mse = 0,0 

	scheduler.step() # rebaja un poco la LR del SGD

	modelo.eval()  # Preparar el modelo para validación y/o test
	print("Validando... \n")
	for x, y in val_loader:
		total += 1

		y = transform(y)  # normalizamos

		x = x.to(device)
		y = y.to(device)

		output = modelo(x)  # forward
		loss = criterion(output.squeeze(), y.squeeze()) 
		mae_accum, mse_accum = get_mae_mse(output.squeeze(), y.squeeze(), inverse_transform)
		mae += mae_accum
		mse += mse_accum

	val_loss  = val_loss/total
	mae = mae/total
	mse = mse/total
	losses['validacion'].append(val_loss)  # .item())
	print(f'[ep {epoch}] [Train: {training_loss:.4f}][Val: {val_loss:.4f}]')
	print(f'\t[Val MAE: {mae:.4f}][Val MSE: {mse:.4f}]')

	# Early stopping
	if (val_loss <= min_val_loss['loss']) or (mse <= min_val_loss['mse']) or (mae<=min_val_loss['mae']):
		filename = expcode+'.pt'
		print(f'Saving as {filename}')
		torch.save(modelo, filename)
		
		if val_loss <= min_val_loss['loss']:
			min_val_loss['loss'] = val_loss
		if mse <= min_val_loss['mse']:
			min_val_loss['mse'] = mse 
		if mae<=min_val_loss['mae']: 
			min_val_loss['mae'] = mae


# TEST

modelo = torch.load(filename)
modelo.eval()  # Preparar el modelo para validación y/o test
print("Testing... \n")
total = 0

test_loss_mse = 0.0
test_loss_mae = 0.0

yreal = list()
ypredicha = list()

for x, y in test_loader:

	y = transform(y)  # normalizamos

	x = x.to(device)
	y = y.to(device)
	with torch.no_grad():
		output = modelo(x)

	yreal.append(inverse_transform(y.data.cpu().numpy()))
	ypredicha.append(inverse_transform(output.data.cpu().numpy()))

	mae_accum, mse_accum = get_mae_mse(output.squeeze(), y.squeeze(), inverse_transform)
	mae += mae_accum
	mse += mse_accum
	total += 1

val_loss  = val_loss/total
mae = mae/total
mse = mse/total
print(f'[ep {epoch}][Test MAE: {mae:.4f}][Test MSE: {mse:.4f}]')

# yreal = np.array(yreal).flatten()
# ypredicha = np.array(ypredicha).flatten() # comprobar si funciona.

losses['yreal'] = np.array([el.item() for a in yreal for el in a])
losses['ypredicha'] = np.array([el.item() for a in ypredicha for el in a])

print(f'Test Loss (MSE): {mse}')
losses['test_mse'] = mse  # .item())
print(f'Test Loss (MAE): {mae}')
losses['test_mae'] = mae  # .item())


with open(SAVE_FILENAME, 'wb') as handle:
	pickle.dump(losses, handle, protocol=pickle.HIGHEST_PROTOCOL)
