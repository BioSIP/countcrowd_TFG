#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   losses.py
@Time    :   2021/04/30 14:00:06
@Author  :   F.J. Martinez-Murcia 
@Version :   1.0
@Contact :   pakitochus@gmail.com
@License :   (C)Copyright 2021, SiPBA-BioSIP
@Desc    :   Biblioteca de funciones de loss. Para importar, en el
             mismo script donde estes haces:
             from losses import LogCoshLoss 
'''

# here put the import lib

import torch 
from torch import nn


class RMSLELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

class LogCoshLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        val =  torch.sum(torch.log(torch.cosh(ey_t + 1e-12)))
        if self.reduction=='mean':
            val /= y_t.shape[0]
        return val

class XTanhLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        val = torch.mean(ey_t * torch.tanh(ey_t))
        if self.reduction=='mean':
            val /= y_t.shape[0]
        return val


class XSigmoidLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        val = torch.sum(2 * ey_t / (1 + torch.exp(-ey_t)) - ey_t)
        if self.reduction=='mean':
            val /= y_t.shape[0]
        return val


 