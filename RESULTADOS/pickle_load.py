#CAMBIAR EL NOMBRE DEL ARCHIVO A ABRIR!
import pickle
import matplotlib.pyplot as plt

with open('crisnet_L1_0.01_0.0001.pickle', 'rb') as handle: 
	b = pickle.load(handle) 


fig,ax=plt.subplots() 
ax.plot(b['train'],label='train') 
ax.plot(b['validacion'],label='validacion') 
plt.legend()
plt.show()