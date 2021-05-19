#CAMBIAR EL NOMBRE DEL ARCHIVO A ABRIR!
import pickle
import matplotlib.pyplot as plt

with open('CSRNet_prueba.pickle', 'rb') as handle: 
	b = pickle.load(handle)
	#b = torch.load(handle,map_location=torch.device('cpu'))

fig,ax=plt.subplots() 
ax.plot(b['train'],label='train') 
ax.plot(b['validacion'],label='validacion') 
plt.legend()
plt.show()