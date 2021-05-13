#CAMBIAR EL NOMBRE DEL ARCHIVO A ABRIR!
import pickle
import matplotlib.pyplot as plt

with open('UNET_MSEsum_(15)0.1_(25)0.0001_batch1(eval_y_train).pickle', 'rb') as handle: 
	b = pickle.load(handle) 


fig,ax=plt.subplots() 
ax.plot(b['train'],label='train') 
ax.plot(b['validacion'],label='validacion') 
plt.legend()
plt.show()