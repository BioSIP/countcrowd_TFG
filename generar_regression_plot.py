#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   generar_regression_plot.py
@Time    :   2021/06/09 19:51:14
@Author  :   F.J. Martinez-Murcia 
@Version :   1.0
@Contact :   pakitochus@gmail.com
@License :   (C)Copyright 2021, SiPBA-BioSIP
@Desc    :   None
'''
#%% 
# here put the import lib

import numpy as np 
import pickle
import os
import matplotlib.pyplot as plt 

# PATH = '/home/pakitochus/Dropbox/Docencia/TFGs/Cristina/countcrowd_TFG/RESULTADOS'
PATH = '/home/pakitochus/Descargas/cosas_cristfg/Pickles'
files = os.listdir(PATH)
files = [el for el in files if el.endswith('pickle')]
for fil in files:
    filename = os.path.join(PATH, fil)
    title = fil.split('.')[0]
    with open(filename, 'rb') as handle: 
        b = pickle.load(handle)


    #%% 

    if type(b['yreal']) is list:
        yreal = np.array([el.sum() for el in b['yreal']])
        ypredicha = np.array([el.sum() for el in b['ypredicha']])
    else:
        yreal = b['yreal']
        ypredicha = b['ypredicha']

    fig, ax = plt.subplots()
    ax.scatter(yreal, ypredicha)
    ax.set_title(title)
    ax.set_xlabel('# personas etiquetadas')
    ax.set_ylabel('# personas predichas')
    xmin, xmax = 0, yreal.max()
    ax.set_ylim(xmin-xmax/50, xmax+xmax/50)
    fig.savefig(title+'.jpg')
# %%
