import os
from scipy.io import loadmat 
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import torchaudio 
# https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html

class AudioDataset(Dataset):
	def __init__(self, audio_path, density_path, transform=None, ynorm=100):

		#3 opciones para el density_path:
		#density_path = '/Volumes/Cristina /TFG/Data/density/train'
		#density_path = '/Volumes/Cristina /TFG/Data/density/test'
		#density_path = '/Volumes/Cristina /TFG/Data/density/val'

		self.density_path=density_path
		self.audio_path=audio_path
		self.transform=transform

		self.mapfiles = os.listdir(density_path)

		self.mapfiles_wo_ext=[el[:-4] for el in self.mapfiles]

		#Para no incluir los archivos con 'stage':
		#self.mapfiles_wo_ext=[el[:-4] for el in self.mapfiles if el.startswith('stage')==False]
		# list comprehension

		#audio_path = '/Volumes/Cristina /TFG/Data/auds/'
		self.audiofiles = os.listdir(audio_path)
		self.audiofiles_wo_ext=[el[:-4] for el in self.audiofiles]
		self.audiofiles = [el for el in self.audiofiles_wo_ext if el in self.mapfiles_wo_ext]


		#Añadir extensiones a archivos de audio:
		for i in range(len(self.audiofiles)):
			#Añado la extensión al nombre del archivo que quiero importar:
			self.audiofiles[i] = [self.audiofiles[i] + '.wav']
		
	
	def __len__(self):
		return len(self.audiofiles)
		
	def __getitem__(self, idx):

		#DENSITY MAP
		map_path = self.density_path + self.mapfiles[idx]
		mapa = loadmat(map_path)
		y = torch.as_tensor(mapa['map'].sum()) #<-- Así está bien? Por qué tiene que ser un tensor?
		#y_norm=y/ynorm <--- Normalizo entonces o no?

		#AUDIO
		#Encuentro el path del archivo:
		filename=str(self.audiofiles[idx])
		filename=filename.lstrip("['")
		filename=filename.rstrip("']")
		aud_path = self.audio_path + filename
		#Cargamos el audio:
		waveform, sample_rate = torchaudio.load(aud_path)	#waveform es un tensor
		#SE USARÁ EL SAMPLE_RATE PARA ALGO????????????????
		
		x = waveform.view((1,-1,2)) # dimensiones
		if self.transform:
			x = self.transform(x)
		return x, y
		
#class SpectrogramDataset(Dataset):
	#alsdjfañlskdfj a




audio_path = '/Volumes/Cristina /TFG/Data/auds/'
train_density_path = '/Volumes/Cristina /TFG/Data/density/train/'
test_density_path = '/Volumes/Cristina /TFG/Data/density/test/'
#val_density_path = '/Volumes/Cristina /TFG/Data/density/val'	--> Lo obviamos por ahora



trainset = AudioDataset(audio_path, train_density_path)
testset = AudioDataset(audio_path, test_density_path)

#PRUEBA para ver tensores de audio y de mapas de los conjuntos de train y test:
#print(trainset.__getitem__(20))
#print(testset.__getitem__(20))

train_loader = DataLoader(trainset,batch_size=3) #BATCH_SIZE: pequeño (1-3)


#RED:
# MaxPool2d((1,2))
# torch.nn.Conv2d(in_channels, out_channels, kernel_size) -> kernel_size = (1, 61)
# in_channels ->2, out_channels -> [32,64]. 
# optim - > adam

'''
for x,y in train_loader:
	x = x.to(device)
	y = y.to(device)
	
	yhat = model(x) # forward 
	# loss estimation -> MSE, MAE
	# optimizer step  
'''