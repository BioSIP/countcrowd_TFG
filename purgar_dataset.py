from scipy.io import loadmat
import os
import shutil 
import pickle

# criterio: ningun mapa puede tener menos que 1. 

audio_path = '/home/pakitochus/Descargas/propuestas_tfg_cristina/crowd/definitivo/DISCO_dataset/auds/'
train_density_path = '/home/pakitochus/Descargas/propuestas_tfg_cristina/crowd/definitivo/DISCO_dataset/density/train/'
val_density_path = '/home/pakitochus/Descargas/propuestas_tfg_cristina/crowd/definitivo/DISCO_dataset/density/val/'
test_density_path = '/home/pakitochus/Descargas/propuestas_tfg_cristina/crowd/definitivo/DISCO_dataset/density/test/'

dest_path = '/home/pakitochus/Descargas/propuestas_tfg_cristina/crowd/definitivo/DISCO_dataset/density/mal/'

train_files = os.listdir(train_density_path)
val_files = os.listdir(val_density_path)
test_files = os.listdir(test_density_path)

lista_moved = {'train': list(), 'val': list(), 'test': list()}

print('train')

for fil in train_files:
    mapa = loadmat(os.path.join(train_density_path, fil))
    y = mapa['map'].sum()
    if(y<1):
        print(fil)
        lista_moved['train'].append(fil)
        shutil.move(os.path.join(train_density_path, fil),
                    os.path.join(dest_path, fil))

print('validacion')

for fil in val_files:
    mapa = loadmat(os.path.join(val_density_path, fil))
    y = mapa['map'].sum()
    if(y<1):
        print(fil)
        lista_moved['val'].append(fil)
        shutil.move(os.path.join(val_density_path, fil),
                    os.path.join(dest_path, fil))

print('test')

for fil in test_files:
    mapa = loadmat(os.path.join(test_density_path, fil))
    y = mapa['map'].sum()
    if(y<1):
        print(fil)
        lista_moved['test'].append(fil)
        shutil.move(os.path.join(test_density_path, fil),
                    os.path.join(dest_path, fil))


filename = os.path.join('/'.join(train_density_path.split('/')[:-2]),'lista_moved.pickle')
with open(filename, 'wb') as handle:
    pickle.dump(lista_moved, handle, protocol=pickle.HIGHEST_PROTOCOL)



