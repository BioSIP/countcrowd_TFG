#CAMBIAR EL NOMBRE DEL ARCHIVO A ABRIR!
import pickle
import matplotlib.pyplot as plt

with open('CANNet_MSEmean_(20)Adam1e-5_batch2(eval_y_train).pickle', 'rb') as handle: 
	b = pickle.load(handle) 


fig,ax=plt.subplots() 
ax.plot(b['train'],label='train') 
ax.plot(b['validacion'],label='validacion') 
plt.legend()
plt.show()