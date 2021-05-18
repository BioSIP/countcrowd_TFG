#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   datasets.py
@Time    :   2021/05/15 09:28:00
@Author  :   F.J. Martinez-Murcia 
@Version :   1.0
@Contact :   pakitochus@gmail.com
@License :   (C)Copyright 2021, SiPBA-BioSIP
@Desc    :   Archivo para cargar datasets a la manera de los chinos.
'''

import random
import os
import numpy as np
from PIL import Image
import torch
from scipy.io import loadmat
from torch.utils import data
import torchvision.transforms as standard_transforms

# VARIABLES PARA INICIAR
# PATH = '/media/NAS/home/cristfg/datasets/'
# PATH = '/Volumes/Cristina /TFG/Data/'
PATH = '/home/pakitochus/Descargas/propuestas_tfg_cristina/crowd/definitivo/DISCO_dataset/'

STD_SIZE = (768, 1024)
TRAIN_SIZE = (576, 768)  # 2D tuple or 1D scalar
IMAGE_PATH = os.path.join(PATH, 'imgs')
DENSITY_PATH = os.path.join(PATH, 'density')

IS_CROSS_SCENE = False
LONGEST_SIDE = 512
BLACK_AREA_RATIO = 0

MEAN_STD = ([0.410824894905, 0.370634973049, 0.359682112932], [0.278580576181, 0.26925137639, 0.27156367898])

LABEL_FACTOR = 1  # must be 1
LOG_PARA = 100.

RESUME_MODEL = ''  # model path
TRAIN_BATCH_SIZE = 48  # imgs
VAL_BATCH_SIZE = 1  # must be 1


# Transformaciones que usan en imagenes / density maps. 

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, bbx=None):
        if bbx is None:
            for t in self.transforms:
                img, mask = t(img, mask)
            return img, mask
        for t in self.transforms:
            img, mask, bbx = t(img, mask, bbx)
        return img, mask, bbx

class RandomHorizontallyFlip(object):
    def __call__(self, img, mask, bbx=None):
        if random.random() < 0.5:
            if bbx is None:
                return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
            w, h = img.size
            xmin = w - bbx[:,3]
            xmax = w - bbx[:,1]
            bbx[:,1] = xmin
            bbx[:,3] = xmax
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT), bbx
        if bbx is None:
            return img, mask
        return img, mask, bbx

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

class LabelNormalize(object):
    def __init__(self, para):
        self.para = para

    def __call__(self, tensor):
        # tensor = 1./(tensor+self.para).log()
        tensor = torch.from_numpy(np.array(tensor))
        tensor = tensor*self.para
        return tensor

class GTScaleDown(object):
    def __init__(self, factor=8):
        self.factor = factor

    def __call__(self, img):
        w, h = img.size
        if self.factor==1:
            return img
        tmp = np.array(img.resize((int(w/self.factor), int(h/self.factor)), Image.BICUBIC))*self.factor*self.factor
        img = Image.fromarray(tmp)
        return img


def share_memory(batch):
    out = None
    if False:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum([x.numel() for x in batch])
        storage = batch[0].storage()._new_shared(numel)
        out = batch[0].new(storage)
    return out

def AC_collate(batch):
    # @GJY
    r"""Puts each data field into a tensor with outer dimension batch size"""

    transposed = list(zip(*batch))  # imgs, dens and raw audios
    # imgs, dens, aud = [transposed[0], transposed[1], transposed[2]]
    imgs, dens = [transposed[0], transposed[1]]

    error_msg = "batch must contain tensors; found {}"
    if isinstance(imgs[0], torch.Tensor) and isinstance(dens[0], torch.Tensor):

        cropped_imgs = []
        cropped_dens = []
        # cropped_auds = []
        for i_sample in range(len(batch)):
            # _img, _den = random_crop(imgs[i_sample], dens[i_sample], [min_ht, min_wd])
            cropped_imgs.append(imgs[i_sample])
            cropped_dens.append(dens[i_sample])
            # cropped_auds.append(aud[i_sample])

        cropped_imgs = torch.stack(cropped_imgs, 0, out=share_memory(cropped_imgs))
        cropped_dens = torch.stack(cropped_dens, 0, out=share_memory(cropped_dens))
        # cropped_auds = torch.stack(cropped_auds, 0, out=share_memory(cropped_auds))

        return [cropped_imgs, cropped_dens] # , cropped_auds]

    raise TypeError((error_msg.format(type(batch[0]))))

#%% CLASE DEL DATASET

class DISCO(data.Dataset):

    def __init__(self, image_path, density_path, mode='train', main_transform=None,
                img_transform=None, den_transform=None, longest_side=1024, black_area_ratio=0):
        
        self.density_path = os.path.join(density_path, mode) # directamente le decimos el "modo" y no hay que especificarlo
        self.image_path = image_path # ruta a las imágenes

        self.mapfiles = os.listdir(self.density_path) 
        # Para no incluir los archivos con '._':
        self.mapfiles = [
            el for el in self.mapfiles if el.startswith('._') == False]
        self.mapfiles_wo_ext = [el[:-4] for el in self.mapfiles]
        self.num_samples = len(self.mapfiles_wo_ext)
        self.imagefiles = os.listdir(image_path)
        self.imagefiles_wo_ext = [el[:-4] for el in self.imagefiles]
        self.imagefiles = [
            el + '.jpg' for el in self.imagefiles_wo_ext if el in self.mapfiles_wo_ext]

        self.imagefiles = sorted(self.imagefiles)
        self.mapfiles_wo_ext = sorted(self.mapfiles_wo_ext)

        self.main_transform = main_transform # se aplica a todos: mapas e imágenes. Solo en training para generar más sample
        self.img_transform = img_transform # se aplica solo a imágenes
        self.den_transform = den_transform # se aplica solo a mapas de densidad
        self.longest_side = longest_side # se usa para resize() luego, porque no entrean con las imágenes grandes
        self.black_area_ratio = black_area_ratio # lo he dejado por si queremos hacer pruebas de oclusión más adelante

    def __getitem__(self, index):
        # Esta rutina (heredadad de data.dataset) crea las imágenes y aplica las transformaciones.
        img, den = self.load_image_den(self.imagefiles[index], self.mapfiles_wo_ext[index])
        if self.main_transform is not None:
            img, den = self.main_transform(img, den)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.den_transform is not None:
            den = self.den_transform(den)
        return img, den

    def __len__(self):
        return self.num_samples

    def load_image_den(self, img, den):
        # para replicar lo del paper, vamos a usar su smismas rutinas, así que no convertimos a tensor aquí.
        img = Image.open(os.path.join(self.image_path, img))
        if img.mode == 'L':
            img = img.convert('RGB')
        img = self.random_black(img, self.black_area_ratio) # he copiado esta rutina que les funciona bien
        w, h = img.size
        if w > h: # cambian el tamaño y lo reducen con interpolación bicúbica
            factor = w / self.longest_side
            img = img.resize((self.longest_side, int(h / factor)), Image.BICUBIC)
        else:
            factor = h / self.longest_side
            img = img.resize((int(w / factor), self.longest_side), Image.BICUBIC)

        den = loadmat(os.path.join(self.density_path, den)) # esto es loq ue hacíamos nosotros
        den = den['map']
        den = den.astype(np.float32, copy=False)
        den = Image.fromarray(den)  # salvo converitrlo a imágenes. Nosotros lo hacíamos a tensores. 
        if w > h: # otra vez cambian el tamaño
            den = np.array(den.resize((self.longest_side, int(h / factor)), Image.BICUBIC)) * factor * factor
        else:
            den = np.array(den.resize((int(w / factor), self.longest_side), Image.BICUBIC)) * factor * factor
        den = Image.fromarray(den)
        
        return img, den

    def random_black(self, image, ratio):
        # genera un cuadrado negro en nmedio de la imagen para ver oclusiones
        if ratio < 0:
            ratio = 0
        if ratio > 1:
            ratio = 1
        if ratio == 0:
            return image
        image = np.array(image).astype(float)
        row, col, channel = image.shape
        if ratio == 1:
            return Image.fromarray(np.uint8(np.zeros([row, col, channel])))
        r = np.sqrt(ratio)
        black_area_row = int(row * r)
        black_area_col = int(col * r)
        remain_row = row - black_area_row
        remain_col = col - black_area_col
        x = np.random.randint(low=0, high=remain_row)
        y = np.random.randint(low=0, high=remain_col)
        image[x:(x + black_area_row), y:(y + black_area_col), :] = np.zeros([black_area_row, black_area_col, channel])
        return Image.fromarray(np.uint8(image))

#%% GENERAR LAS TRANSFORMACIONES Y LOS DATOS ENTEROS:

def load_datasets():
    """Para cargar los datasets directamente desde el script de lanzamiento. 

    Returns:
        tupla: Tupla con las diferentes bases de datos. 
    """
    # Primero creamos las transformaciones a aplicar. 
    # Ten cuidado que unas son Compose definidas en este script,
    # pero las otras son las de standard_transforms. 
    train_main_transform = Compose([
        RandomHorizontallyFlip()
    ])
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*MEAN_STD)
    ])
    den_transform = standard_transforms.Compose([
        GTScaleDown(LABEL_FACTOR),
        LabelNormalize(LOG_PARA)
    ])
    restore_transform = standard_transforms.Compose([
        DeNormalize(*MEAN_STD),
        standard_transforms.ToPILImage()
    ])

    train_set = DISCO(image_path=IMAGE_PATH, density_path=DENSITY_PATH, mode='train', main_transform=train_main_transform,
                    img_transform=img_transform, den_transform=den_transform, longest_side=LONGEST_SIDE,
                    black_area_ratio=BLACK_AREA_RATIO)
    val_set = DISCO(image_path=IMAGE_PATH, density_path=DENSITY_PATH, mode='val', main_transform=None,
                    img_transform=img_transform, den_transform=den_transform, longest_side=LONGEST_SIDE,
                    black_area_ratio=BLACK_AREA_RATIO)
    test_set = DISCO(image_path=IMAGE_PATH, density_path=DENSITY_PATH, mode='test', main_transform=None,
                    img_transform=img_transform, den_transform=den_transform, longest_side=LONGEST_SIDE,
                    black_area_ratio=BLACK_AREA_RATIO)

    train_loader = data.DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, num_workers=8,
                                  collate_fn=AC_collate, shuffle=True, drop_last=True)
    val_loader = data.DataLoader(val_set, batch_size=VAL_BATCH_SIZE, num_workers=1, shuffle=False, drop_last=False)
    test_loader = data.DataLoader(test_set, batch_size=VAL_BATCH_SIZE, num_workers=1, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader, restore_transform



# TODO: añadir soporte para Audio (ver read_image_and_den en loading_data)