import os
from scipy.io import loadmat 
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import torchaudio 
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pickle 
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
#import librosa
#from TorchAudioFun import plot_spectrogram

# https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.

# Nombre de archivo para guardar resultados
SAVE_FILENAME = 'crisnet_Adam_0.01_Spec.pickle'

#Para comprobar si tenemos GPUs disponibles para usar o no:
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class AudioDataset(Dataset):
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
		waveform, sample_rate = torchaudio.load(aud_path)	#waveform es un tensor
		
		x = waveform.view((2,1,-1)) # dimensiones
		if self.transform:
			x = self.transform(x)
		return x, y
		
class SpectrogramDataset(Dataset):
	def __init__(self, audio_path, density_path, transform):

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
		waveform, sample_rate = torchaudio.load(aud_path)	#waveform es un tensor

		if(transform=='MEL'):
			x = torchaudio.transforms.MelScale()(waveform)
		elif(transform==None):
			x = torchaudio.transforms.Spectrogram()(waveform)

		#print("Shape of spectrogram: {}".format(x.size()))

		#plt.figure()
		#plt.imshow(x.log2()[0,:,:].numpy(), cmap='gray')

		return x, y


#PINTAR EL 20esimo ESPECTROGRAMA:
#waveform_list=trainset.__getitem__(20)
#waveform=[t.numpy() for t in waveform_list]
#plot_spectrogram(waveform[0].reshape(2,48000))


audio_path = '/media/NAS/home/cristfg/datasets/auds/'
train_density_path = '/media/NAS/home/cristfg/datasets/density/train/'
val_density_path = '/media/NAS/home/cristfg/datasets/density/val/'
test_density_path = '/media/NAS/home/cristfg/datasets/density/test/'



#trainset = AudioDataset(audio_path, train_density_path)
#valset = AudioDataset(audio_path, val_density_path)
#testset = AudioDataset(audio_path, test_density_path)

trainset = SpectrogramDataset(audio_path, train_density_path,'MEL')
valset = SpectrogramDataset(audio_path, val_density_path,'MEL')
testset = SpectrogramDataset(audio_path, test_density_patH,'MEL')

#PRUEBA para ver tensores de audio y de mapas de los conjuntos de train y test:
#print(trainset.__getitem__(20))
#print(testset.__getitem__(20))

#BATCH_SIZE: pequeno (1-3)
batch_size=3
train_loader = DataLoader(trainset,batch_size,shuffle=True) #BATCH_SIZE: pequeno (1-3)
val_loader = DataLoader(valset,batch_size,shuffle=False)
test_loader = DataLoader(testset,batch_size,shuffle=False)

#RED:

# MaxPool2d((1,2))
# torch.nn.Conv2d(in_channels, out_channels, kernel_size) -> kernel_size = (1, 61)
# in_channels ->2, out_channels -> [32,64]. 
# optim - > adam

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
		self.features = nn.Sequential(
			nn.Conv2d(1, 64, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2),

			nn.Conv2d(64, 128, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2),

			nn.Conv2d(128, 256, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2),

			nn.Conv2d(256, 512, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2)
		)
		self.fc = nn.Sequential(
			nn.Linear(512 * 24, 4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096, 128),
			nn.ReLU(inplace=True),
		)

	def forward(self, x):
		x = self.features(x).permute(0, 2, 3, 1).contiguous()
		x = x.view(x.size(0), -1)
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
		#Con funcion de activacion ReLu

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
modelo=modelo.to(device)
criterion = nn.MSELoss(reduction='sum') # definimos la perdida
# criterion = nn.L1Loss(reduction='sum') 
#print(modelo)

#print(train_loader)
#print(type(train_loader))


# convertimos train_loader en un iterador
dataiter = iter(train_loader) 
# y recuperamos el i-esimo elemento, un par de valores (imagenes, etiquetas)
x, y = dataiter.next() 

#print(x)
#print(x.size())
#print(y)
#print(y.size())

#Para predecir y, la normalizaremos. Siempre por el mismo valor:
Y_NORM = 500

losses = {'train': list(), 'validacion': list()}


#ENTRENAMIENTO 
n_epochs = 20
optimizador = optim.Adam(modelo.parameters(), lr=0.01, weight_decay=1e-4) 

for epoch in range(n_epochs):
	print("Entrenando... \n") # Esta sera la parte de entrenamiento
	training_loss = 0.0 # el loss en cada epoch de entrenamiento
	total = 0

	modelo.train() #Para preparar el modelo para el training	
	for x,y in  train_loader:
		# ponemos a cero todos los gradientes en todas las neuronas:
		optimizador.zero_grad()

		y=y/Y_NORM #normalizamos

		x = x.to(device)
		y = y.to(device)
		total += y.shape[0]
	
		output = modelo(x) # forward 
		output = output.flatten()
		loss = criterion(output,y) # evaluacion del loss
		loss.backward()# backward pass
		optimizador.step() # optimizacion 

		training_loss += loss.cpu().item() # acumulamos el loss de este batch
	
	training_loss /= total
	losses['train'].append(training_loss)#.item())

	val_loss = 0.0
	total = 0

	modelo.eval() #Preparar el modelo para validacion y/o test
	print("Validando... \n")
	for x,y in val_loader:
		
		y=y/Y_NORM #normalizamos 
		x = x.to(device)
		y = y.to(device)
		total += y.shape[0]

		output = modelo(x) 
		output = output.flatten()
		loss = criterion(output,y)
		val_loss += loss.cpu().item()

	val_loss /= total
	losses['validacion'].append(val_loss)#.item())

	print(f'Epoch {epoch} \t\t Training Loss: {training_loss} \t\t Validation Loss: {val_loss}')

	with open(SAVE_FILENAME, 'wb') as handle:
		pickle.dump(losses, handle, protocol=pickle.HIGHEST_PROTOCOL)
	

#TEST
modelo.eval() #Preparar el modelo para validacion y/o test
print("Testing... \n")
total = 0

mse = nn.MSELoss(reduction='sum') # definimos la perdida
mae = nn.L1Loss(reduction='sum')
test_loss_mse = 0.0
test_loss_mae = 0.0

yreal = list()
ypredicha = list()

for x,y in test_loader:
	
	y=y/Y_NORM #normalizamos 

	x = x.to(device)
	y = y.to(device)
	total += y.shape[0]

	output = modelo(x)
	output = output.flatten() 
	mse_loss = mse(output,y)
	test_loss_mse += mse_loss.cpu().item()
	mae_loss = mae(output,y)
	test_loss_mae += mae_loss.cpu().item()

	# para guardar las etqieutas. 
	yreal.append(y.detach().cpu().numpy())
	ypredicha.append(output.detach().cpu().numpy())

test_loss_mse *= Y_NORM/total # Esto siemrpe que reduction='sum' -> equiparable a numero de personas. 
test_loss_mae *= Y_NORM/total # Esto siemrpe que reduction='sum' -> equiparable a numero de personas. 

# yreal = np.array(yreal).flatten()
# ypredicha = np.array(ypredicha).flatten() # comprobar si funciona. 

losses['yreal'] = yreal
losses['ypredicha'] = ypredicha

print(f'Test Loss (MSE): {test_loss_mse}')
losses['test_mse'] = test_loss_mse #.item())
print(f'Test Loss (MAE): {test_loss_mae}')
losses['test_mae'] = test_loss_mae #.item())

with open(SAVE_FILENAME, 'wb') as handle:
	pickle.dump(losses, handle, protocol=pickle.HIGHEST_PROTOCOL)


#PARA LEER FICHEROS .pickle:
#import pickle
# with open('filename.pickle', 'rb') as handle:
#     b = pickle.load(handle)
# import matplotlib.pyplot as plt 
# plt.plot()


