# countcrowd_TFG
TFG para conteo de multitudes basado en audio

# TODO
1. Leer el paper
2. Clonar los repositorios de los chinos y el nuestro. 
3. Programar nuestra maravillosa red. 

- crear dataloader (yo) 
## pseudocodigo
```python
import os
from scipy.io import loadmat 
#algo para cargar audio como vector. buscar torchaudio. 
# https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html

class AudioDataset(Dataset):
	def __init__(self, audio_path, density_path, transform=None, ynorm=100):
		self.mapfiles = os.listdir(density_path)
		# [el[:-4] for el in self.mapfiles]
		# list comprehension
		# [el[:-4] for el in mapfiles if el.startswith('stage')==False]
		self.audiofiles = os.listdir(audio_path)
		self.audiofiles = [el for el in self.audiofiles if el in self.mapfiles]
		# faltaría algo
	
	def __len__(self):
		return len(self.audiofiles)
		
	def __getitem__(self, idx):
		# cargar densitymap (scipy.io.loadmat())
		# from scipy.io import loadmat
		mapa = loadmat(self.mapfiles[idx])
		y = torch.Tensor(mapa['map'].sum()) #/ynorm
		x = loadaudio(self.audiofiles[idx]) # asegurate de que carge un tensor
		x = x.view((1,-1,2)) # dimensiones
		if self.transform:
			x = self.transform(x)
		return x, y
		
class SpectrogramDataset(Dataset):
	#alsdjfañlskdfj a

# creas dataset -> trainset = AudioDataset(audio_path, train_density_path)
# testset = AudioDataset(audio_path, test_density_path)
# train_loader = DataLoader(trainset), BATCH_SIZE: pequeño (1-3)
	
# MaxPool2d((1,2))
# torch.nn.Conv2d(in_channels, out_channels, kernel_size) -> kernel_size = (1, 61)
# in_channels ->2, out_channels -> [32,64]. 
# optim - > adam

for x,y in train_loader:
	x = x.to(device)
	y = y.to(device)
	
	yhat = model(x) # forward 
	# loss estimation -> MSE, MAE
	# optimizer step  
	
	
```
- contar lo de los spectrogramas. 
