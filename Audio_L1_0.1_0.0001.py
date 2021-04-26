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

# https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html

# Nombre de archivo para guardar resultados
SAVE_FILENAME = 'crisnet_L1_0.1_0.0001.pickle'

#Para comprobar si tenemos GPUs disponibles para usar o no:
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class AudioDataset(Dataset):
	def __init__(self, audio_path, density_path, transform=None, ynorm=100):

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


		#Añadir extensiones a archivos de audio:
		#for i in range(len(self.audiofiles)):
			#Añado la extensión al nombre del archivo que quiero importar:
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
		#SE USARÁ EL SAMPLE_RATE PARA ALGO????????????????
		
		x = waveform.view((2,1,-1)) # dimensiones
		if self.transform:
			x = self.transform(x)
		return x, y
		
#class SpectrogramDataset(Dataset):
	#PROGRAMAR LUEGO!!!




audio_path = '/media/NAS/home/cristfg/datasets/auds/'
train_density_path = '/media/NAS/home/cristfg/datasets/density/train/'
val_density_path = '/media/NAS/home/cristfg/datasets/density/val/'
test_density_path = '/media/NAS/home/cristfg/datasets/density/test/'	


trainset = AudioDataset(audio_path, train_density_path)
valset = AudioDataset(audio_path, val_density_path)
testset = AudioDataset(audio_path, test_density_path)

#PRUEBA para ver tensores de audio y de mapas de los conjuntos de train y val:
#print(trainset.__getitem__(20))
#print(valset.__getitem__(20))

#BATCH_SIZE: pequeño (1-3)
batch_size=3
train_loader = DataLoader(trainset,batch_size,shuffle=True) #BATCH_SIZE: pequeño (1-3)
val_loader = DataLoader(valset,batch_size,shuffle=False)
test_loader = DataLoader(testset,batch_size,shuffle=False)

#RED:
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
class CrisNet(nn.Module):
	def __init__(self):
		super(CrisNet, self).__init__() # esta linea es siempre necesaria
		self.max_pool1 = nn.MaxPool2d((1,2))

		self.conv1 = nn.Conv2d(2, 32, (1,5))
		self.conv2 = nn.Conv2d(32, 64, (1,5))
		self.conv3 = nn.Conv2d(64, 128, (1,5))
		self.conv4 = nn.Conv2d(128, 256, (1,5))
		self.conv5 = nn.Conv2d(256, 512, (1,5))
		self.conv6 = nn.Conv2d(512, 1024, (1,5))

		self.fc1 = nn.Linear(763904,1)
		'''
		self.conv2 = nn.Conv2d()
		self.max_pool2 = nn.MaxPool2d((1,2))
		self.fc2 = nn.Linear()
		'''

	def forward(self, x):
		#Con función de activación ReLu

		#PRIMERA CAPA
		x = F.relu(self.conv1(x))
		x = self.max_pool1(x)


		#SEGUNDA CAPA
		x = F.relu(self.conv2(x))
		x = self.max_pool1(x)

		#TERCERA CAPA
		x = F.relu(self.conv3(x))
		x = self.max_pool1(x)

		#CUARTA CAPA
		x = F.relu(self.conv4(x))
		x = self.max_pool1(x)

		#QUINTA CAPA
		x = F.relu(self.conv5(x))
		x = self.max_pool1(x)


		#SEXTA CAPA
		x = F.relu(self.conv6(x))
		x = self.max_pool1(x)


		x = x.view((x.size(0),-1))
		x = self.fc1(x)


		return x

modelo=CrisNet()
modelo=modelo.to(device)
#criterion = nn.MSELoss(reduction='sum') # definimos la pérdida
criterion = nn.L1Loss(reduction='sum')
optimizador = optim.Adam(modelo.parameters(), lr=0.1, weight_decay=1e-4) 
#print(modelo)

#print(train_loader)
#print(type(train_loader))



"""
 COSAS A PROBAR: 
 - modificar learning rates (adam, sgd)
 - prueba las dos loses: mse, l1loss, con y sin redución. (reduction='sum')
 - extraer PARA TODAS LAS COMBINACIONES QUE HAGAS, guarda los datos, especialmente el de test. 
"""

#TENGO QUE HACER ESTO O NO?
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

for epoch in range(n_epochs):
	print("Entrenando... \n") # Esta será la parte de entrenamiento
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
		loss = criterion(output,y) # evaluación del loss
		loss.backward()# backward pass
		optimizador.step() # optimización 

		training_loss += loss.cpu().item() # acumulamos el loss de este batch
	
	training_loss /= total
	losses['train'].append(training_loss)#.item())

	val_loss = 0.0
	total = 0

	modelo.eval() #Preparar el modelo para validación y/o test
	print("Validando... \n")
	for x,y in val_loader:
		
		y=y/Y_NORM #normalizamos 

		x = x.to(device)
		y = y.to(device)
		total += y.shape[0]

		output = modelo(x) 
		loss = criterion(output,y)
		val_loss += loss.cpu().item()

	val_loss /= total
	losses['validacion'].append(val_loss)#.item())

	print(f'Epoch {epoch} \t\t Training Loss: {training_loss} \t\t Validation Loss: {val_loss}')

	with open(SAVE_FILENAME, 'wb') as handle:
		pickle.dump(losses, handle, protocol=pickle.HIGHEST_PROTOCOL)
	

#ENTRENAMIENTO
n_epochs = 200
optimizador = optim.SGD(modelo.parameters(), lr=0.0001) 

for epoch in range(n_epochs):
	print("Entrenando... \n") # Esta será la parte de entrenamiento
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
		loss = criterion(output,y) # evaluación del loss
		loss.backward()# backward pass
		optimizador.step() # optimización 

		training_loss += loss.cpu().item() # acumulamos el loss de este batch
	
	training_loss /= total
	losses['train'].append(training_loss)#.item())

	val_loss = 0.0
	total = 0

	modelo.eval() #Preparar el modelo para validación y/o test
	print("Validando... \n")
	for x,y in val_loader:
		
		y=y/Y_NORM #normalizamos 
		x = x.to(device)
		y = y.to(device)
		total += y.shape[0]

		output = modelo(x) 
		loss = criterion(output,y)
		val_loss += loss.cpu().item()

	val_loss /= total
	losses['validacion'].append(val_loss)#.item())

	print(f'Epoch {epoch} \t\t Training Loss: {training_loss} \t\t Validation Loss: {val_loss}')

	with open(SAVE_FILENAME, 'wb') as handle:
		pickle.dump(losses, handle, protocol=pickle.HIGHEST_PROTOCOL)
	

#TEST
modelo.eval() #Preparar el modelo para validación y/o test
print("Testing... \n")
total = 0

mse = nn.MSELoss(reduction='sum') # definimos la pérdida
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
	mse_loss = mse(output,y)
	test_loss_mse += mse_loss.cpu().item()
	mae_loss = mae(output,y)
	test_loss_mae += mae_loss.cpu().item()

	# para guardar las etqieutas. 
	yreal.append(y.detach().cpu().numpy())
	ypredicha.append(output.detach().cpu().numpy())

test_loss_mse *= Y_NORM/total # Esto siemrpe que reduction='sum' -> equiparable a número de personas. 
test_loss_mae *= Y_NORM/total # Esto siemrpe que reduction='sum' -> equiparable a número de personas. 

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


