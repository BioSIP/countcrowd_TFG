import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from scipy.io import loadmat 
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import optim
import pickle 

# Para comprobar si tenemos GPUs disponibles para usar o no:
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#image_path = '/media/NAS/home/cristfg/datasets/imgs/'
#train_density_path = '/media/NAS/home/cristfg/datasets/density/train/'
#val_density_path = '/media/NAS/home/cristfg/datasets/density/val/'
#test_density_path = '/media/NAS/home/cristfg/datasets/density/test/'

image_path = '/Volumes/Cristina /TFG/Data/imgs/'
train_density_path = '/Volumes/Cristina /TFG/Data/density/train/'
val_density_path = '/Volumes/Cristina /TFG/Data/density/val/'
test_density_path = '/Volumes/Cristina /TFG/Data/density/test/'	

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
		#mapa = loadmat(map_path)
		#y = torch.as_tensor(mapa['map'].sum(), dtype=torch.float32)

		# IMAGES
		filename = str(self.imagefiles[idx])
		filename = filename.lstrip("['")
		filename = filename.rstrip("']")
		img_path = self.image_path + filename
		# Cargamos la imagen:
		x = plt.imread(img_path)
	   
		return x, y


#CARGAMOS LOS DATOS:
trainset = ImageDataset(image_path, train_density_path)
valset = ImageDataset(image_path, val_density_path)
testset = ImageDataset(image_path, test_density_path)

#PRUEBA
#print(trainset.__getitem__(20))
#print(testset.__getitem__(20))


batch_size=3
train_loader = DataLoader(trainset,batch_size,shuffle=True) #train BATCH_SIZE: pequeño (1-3)
val_loader = DataLoader(valset,batch_size,shuffle=False)
test_loader = DataLoader(testset,batch_size,shuffle=False)


#RED UNET

#Función necesaria para UNET
def crop_img(tensor, target_tensor):
	target_size = target_tensor.size()[2:]
	tensor_size = tensor.size()[2:]
	delta = torch.Tensor(tensor_size) - torch.Tensor(target_size)
	delta = delta // 2
	return tensor[:, :, delta[0]:tensor_size[0]-delta[0], delta[1]:tensor_size[1]-delta[1]]


class UNET(nn.Module):
	# https://lmb.informatik.uni-freiburg.de/research/funded_projects/bioss_deeplearning/unet.png
	def __init__(self):
		super(UNET, self).__init__()
		#ENCODER
		self.conv1 =  nn.Sequential(nn.Conv2d(1080, 8, 11, padding = 5), 
									nn.ReLU(inplace=True),
									nn.Conv2d(8, 8, 11, padding = 5),
									nn.ReLU(inplace=True))
		self.mp1 = nn.MaxPool2d((1,2))

		self.conv2 =  nn.Sequential(nn.Conv2d(8, 16, 7, padding = 3), 
									nn.ReLU(inplace=True),
									nn.Conv2d(16, 16, 7, padding = 3),
									nn.ReLU(inplace=True))
		self.mp2 = nn.MaxPool2d((1,2))

		self.conv3 =  nn.Sequential(nn.Conv2d(16, 32, 5, padding = 2), 
									nn.ReLU(inplace=True),
									nn.Conv2d(32, 32, 5, padding = 2),
									nn.ReLU(inplace=True))
		  
		
		#DECODER
		self.up_conv5 = nn.ConvTranspose2d(32, 16, 5, stride=2, padding=2, output_padding=1)
		
		self.up_conv6 = nn.Sequential(nn.Conv2d(32, 16, 5, padding = 2),
									  nn.ReLU(inplace=True),
									  nn.Conv2d(16, 16, 5, padding = 2), 
									  nn.ReLU(inplace=True))

		self.up_conv7 = nn.ConvTranspose2d(16, 8, 7, stride=2, padding=3, output_padding=1)
		
		self.up_conv8 = nn.Sequential(nn.Conv2d(16, 8, 7, padding = 3),
									  nn.ReLU(inplace=True),
									  nn.Conv2d(8, 8, 7, padding = 3), 
									  nn.ReLU(inplace=True))

		self.out = nn.Conv2d(8, 2, 11, padding = 5)
		

	def forward(self, x):
		
		#ENCODER
		x1 = self.conv1(x)
		#print(x1.size())
		x2 = self.mp1(x1)

		x3 = self.conv2(x2)
		x4 = self.mp2(x3)

		x5 = self.conv3(x4)
		#print(x5.size())

		#DECODER
		x6 = self.up_conv5(x5)
		y1 = crop_img(x3, x6)
		x7 = self.up_conv6(torch.cat([x6, y1], 1))

		x8 = self.up_conv7(x7)
		y2 = crop_img(x1, x8)
		x9 = self.up_conv8(torch.cat([x8, y2], 1))
		#print(x9.size())

		x = self.out(x9)
		#print(x.size())
		
		return x

modelo=UNET()
modelo=modelo.to(device)
#Definimos el criterion de pérdida:
criterion = nn.MSELoss(reduction='sum')
# criterion = nn.L1Loss(reduction='sum') 

# convertimos train_loader en un iterador
dataiter = iter(train_loader) 
# y recuperamos el i-esimo elemento, un par de valores (imagenes, etiquetas)
x, y = dataiter.next() #x e y son tensores


#PODEMOS NORMALIZAR EL MAPA DE DENSIDAD???!!!??
#Para predecir y, la normalizaremos. Siempre por el mismo valor:
#Y_NORM = 200

losses = {'train': list(), 'validacion': list()}

#PRUEBA:
#print(x.size())
#print(y['map'].size())


#ENTRENAMIENTO 
n_epochs = 200
optimizador = optim.Adam(modelo.parameters(), lr=0.01, weight_decay=1e-4) 

for epoch in range(n_epochs):
	print("Entrenando... \n") # Esta será la parte de entrenamiento
	training_loss = 0.0 # el loss en cada epoch de entrenamiento
	total = 0

	modelo.train() #Para preparar el modelo para el training	
	for x,y in  train_loader:
		# ponemos a cero todos los gradientes en todas las neuronas:
		optimizador.zero_grad()

		#y=y/Y_NORM #normalizamos

		x = x.to(device)
		y = y['map'].to(device)
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
		output = output.flatten()
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

# definimos la pérdida
mse = nn.MSELoss(reduction='sum') 
mae = nn.L1Loss(reduction='sum')
test_loss_mse = 0.0
test_loss_mae = 0.0

yreal = list()
ypredicha = list()

for x,y in test_loader:
	
	#y=y/Y_NORM #normalizamos 

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

