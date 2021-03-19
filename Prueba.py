#¡NO FUNCIONA!



#DEJAR LOS MODELOS EN LA CARPETA TAL CUAL LA TENÍAN ELLOS PARA IMPORTAR EL QUE HAGA FALTA.

from easydict import EasyDict as edict
import os
import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.nn.functional as F
import pdb
import argparse
import time
from timeit import Timer
import matplotlib.pyplot as pyplot
import numbers
import random
from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import shutil


#Declaro la variable global cfg con edict para ir añadiéndole atributos sobre la marcha:
cfg = edict()
cfg_data = edict() 

#Para comprobar si tenemos GPUs disponibles para usar o no:
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def config():

	cfg.SEED = 3035 # random seed,  for reproduction --> ¿PARA QUÉ ES ESTO?????????????????????????????????????????????????????????
	""" Los métodos que producen números aleatorios, en realidad no son aleatorios del todo. Implementan
	un algoritmo que, a partir de una semilla (seed) generan un número de forma "pseudoaleatoria". Habitualmente
	se toman cosas fuera del alcance del usuario, por ejemplo, el timestamp (número de segundos desde no se qué día de 1970),
	pero si queremos replicar exactamente los mismos resultados, ponemos una semilla y de esa manera siempre 
	saldrán los mismos resultados.  
	"""

	#Lo comento porque dudo que usemos más de un dataset:
	#cfg.DATASET = 'AC' # dataset selection: GCC, SHHA, SHHB, UCF50, QNRF, WE, Mall, UCSD

	'''
	if cfg.DATASET == 'UCF50':# only for UCF50
	from datasets.UCF50.setting import cfg_data
	cfg.VAL_INDEX = cfg_data.VAL_INDEX 

	if cfg.DATASET == 'GCC':  # only for GCC
		from datasets.GCC.setting import cfg_data
		cfg.VAL_MODE = cfg_data.VAL_MODE 

	'''



	# Net selection: MCNN, AlexNet, VGG, VGG_DECODER, Res50, CSRNet, SANet, CSRNet_IN, CANNet, CSRNet_Audio, CANNet_Audio
	# CSRNet_Audio_Concat, CANNet_Audio_Concat, CSRNet_Audio_Guided, CANNet_Audio_Guided
	cfg.NET = 'CANNet' 



	#SÓLO PARA GCC, no creo que usemos otro DATASET que no sea el AC:
	#cfg.PRE_GCC = False  # use the pretrained model on GCC dataset
	#cfg.PRE_GCC_MODEL = 'path to model' # path to model


    #Para poder continuar el training si se interrumpe, poner en True:
	cfg.RESUME = False # contine training

    #¿QUÉ PONGO EXACTAMENTE AQUÍ? ES ESTO LA CLAVE DE QUE ME FUNCIONE   ???????????????????????????????????????????????????????????
    # Yo diría que no, porque apenas varía la ejecución cuando cambio la ruta. 
    
	cfg.RESUME_PATH = '/home/pakitochus/' #
	# paths /home/pakitochus/Dropbox/Docencia/TFGs/Cristina/countcrowd_TFG/Resultados /Users/cristinareyes/Desktop/TFG/Resultados/CANNet-noise-False-1.0-0-512-0-False-denoise-False


	#No usaremos GPU, pero habilitaremos la opción de hacerlo con CUDA:
	cfg.GPU_ID = [0, 1, 2, 3, 4, 5, 6, 7]  # sigle gpu: [0], [1] ...; multi gpus: [0,1]



	# learning rate settings
	cfg.LR = 1e-5  # learning rate
	cfg.LR_DECAY = 0.99  # decay rate
	cfg.LR_DECAY_START = -1  # when training epoch is more than it, the learning rate will be begin to decay
	cfg.NUM_EPOCH_LR_DECAY = 1  # decay frequency
	cfg.MAX_EPOCH = 200
	#Learning rate inicial de 10^-5, decayendo 0.99 cada epoch


	# multi-task learning weights, no use for single model, such as MCNN, VGG, VGG_DECODER, Res50, CSRNet, and so on
	cfg.LAMBDA_1 = 1e-4  # SANet:0.001 CMTL 0.0001 other: 1e-4
	#Para aliviar el overfitting, el decaimiento de pesos se hace con este Lambda.



	# print 
    #Para mostrar periódicamente las estadísticas del entrenamiento:
	cfg.PRINT_FREQ = 10



	#Variable que almacena la fecha y hora local del momento de ejecución, NO SE USA:
	#now = time.strftime("%m-%d_%H-%M", time.localtime())




	#CREAR NOMBRE SEGÚN EL EXPERIMENTO (he quitado DATASET porque solo usaremos el que tenemos):
	# settings = 'image-noise(0.2, 25)-audio-wo'  # image-clean/(0.3, 50)_audio-w/wo
	cfg.SETTINGS = 'image-clean-audio-wo'  # image-clean/(0.3, 50)_audio-w/wo

	cfg.EXP_NAME = cfg.SETTINGS + '_' + cfg.NET + '_' + str(cfg.LR)

	#Lo comento porque dudo que usemos estos dataset:

	#if cfg.DATASET == 'UCF50':
	#	cfg.EXP_NAME += '_' + str(cfg.VAL_INDEX)	
	#
	#if cfg.DATASET == 'GCC':
	#	cfg.EXP_NAME += '_' + cfg.VAL_MODE

	

	cfg.EXP_PATH = '/home/pakitochus/Dropbox/Docencia/TFGs/Cristina/countcrowd_TFG' # the path of logs, checkpoints, and current codes
	# paths: '/Users/cristinareyes/Desktop/TFG/trained_models/exp' /home/pakitochus/Dropbox/Docencia/TFGs/Cristina/countcrowd_TFG


	#------------------------------VAL------------------------
	cfg.VAL_DENSE_START = 50
	cfg.VAL_FREQ = 10 # Before __C.VAL_DENSE_START epoches, the freq is set as __C.VAL_FREQ

	#------------------------------VIS------------------------

	#Tener en cuenta que tendremos también imágenes de baja resolución (128. x72):
	cfg.VISIBLE_NUM_IMGS = 1 #  must be 1 for training images with the different sizes



def setting():

	#Path del dataset:
	DATA_PATH = '/home/pakitochus/Descargas/propuestas_tfg_cristina/crowd/definitivo/DISCO_dataset'
	# para mi: /home/pakitochus/Descargas/propuestas_tfg_cristina/crowd/definitivo/DISCO_dataset o '/Volumes/Cristina /TFG/Data'

	#Tamaño estándar de las fotos (¿Seguro que es este tamaño?):????????????????????????????????????????????????????????
	cfg_data.STD_SIZE = (768, 1024)

	cfg_data.TRAIN_SIZE = (576, 768)  # 2D tuple or 1D scalar

	cfg_data.IMAGE_PATH = os.path.join(DATA_PATH, 'imgs')
	cfg_data.DENSITY_PATH = os.path.join(DATA_PATH, 'density')
	cfg_data.AUDIO_PATH = os.path.join(DATA_PATH, 'auds')

	cfg_data.IS_CROSS_SCENE = False
	cfg_data.IS_NOISE = False
	cfg_data.IS_DENOISE = False
	cfg_data.BRIGHTNESS = 1.0  # if is_noise, this param works
	cfg_data.NOISE_SIGMA = 0  # if is_noise, this param works
	cfg_data.LONGEST_SIDE = 512
	cfg_data.BLACK_AREA_RATIO = 0
	cfg_data.IS_RANDOM = False  # if is_noise, this param works

	cfg_data.MEAN_STD = ([0.410824894905, 0.370634973049, 0.359682112932], [0.278580576181, 0.26925137639, 0.27156367898])

	cfg_data.LABEL_FACTOR = 1  # must be 1
	cfg_data.LOG_PARA = 100.

	cfg_data.RESUME_MODEL = ''  # model path

    #Cambiar a un batch size más grande para aliviar la computación:
	cfg_data.TRAIN_BATCH_SIZE = 48  # imgs

	cfg_data.VAL_BATCH_SIZE = 1  # must be 1

#EJECUTAMOS LOS SETTINGS Y LOS CONFIGS:
setting()
config()

#La clase que importa el modelo (Usada en la clase Tester):
class CrowdCounter(nn.Module):
    def __init__(self,gpus,model_name):
        super(CrowdCounter, self).__init__()        

        if model_name == 'AlexNet':
            from models.SCC_Model.AlexNet import AlexNet as net        
        elif model_name == 'VGG':
            from models.SCC_Model.VGG import VGG as net
        elif model_name == 'VGG_DECODER':
            from models.SCC_Model.VGG_decoder import VGG_decoder as net
        elif model_name == 'MCNN':
            from models.SCC_Model.MCNN import MCNN as net
        elif model_name == 'CSRNet':
            from models.SCC_Model.CSRNet import CSRNet as net
        elif model_name == 'Res50':
            from models.SCC_Model.Res50 import Res50 as net
        elif model_name == 'Res101':
            from models.SCC_Model.Res101 import Res101 as net            
        elif model_name == 'Res101_SFCN':
            from models.SCC_Model.Res101_SFCN import Res101_SFCN as net
        elif model_name == 'CSRNet_IN':  # Qingzhong
            from models.SCC_Model.CSRNet_IN import CSRNet as net
        elif model_name == 'CSRNet_Audio':  # Qingzhong
            from models.SCC_Model.CSRWaveNet import CSRNet as net
        elif model_name == 'CANNet':
            from models.SCC_Model.CACC import CANNet as net
        elif model_name == 'CANNet_Audio':
            from models.SCC_Model.CANWaveNet import CANNet as net
        elif model_name == 'CSRNet_Audio_Concat':
            from models.SCC_Model.CSRWaveNet import CSRNetConcat as net
        elif model_name == 'CANNet_Audio_Concat':
            from models.SCC_Model.CANWaveNet import CANNetConcat as net
        elif model_name == 'CSRNet_Audio_Guided':
            from models.SCC_Model.CSRWaveNet import CSRNetGuided as net

        self.model_name = model_name

        #Se asigna a self.CNN la clase con el modelo de la red:
        self.CCN = net()


		#Si hay GPUs:
        if device == 'True':
            if len(gpus)>1:
	           #Multiples GPU
               self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
            else:
	       #1 GPU
                self.CCN=self.CCN.cuda()
                self.loss_mse_fn = nn.MSELoss().cuda()

	    #Si no hay GPUs, lo hacemos con la CPU:
        else:
            self.loss_mse_fn = nn.MSELoss()



    @property   #¿PARA QUÉ ES ESTO?????????????????????????????????????????????????????????
    def foo(self):
        return self._foo
    
    def loss(self):
        return self.loss_mse
    
    def forward(self, img, gt_map):   
    	#El modelo de la red se importa desde un fichero externo dependiendo de la red que escojamos.  

        #En el forward, se pasa como parámetro la imágen y se calcula el MSE. Luego, se devuelve el density map predicho:                       
        density_map = self.CCN(img)                          
        #GT= Ground-Truth --> gt_map = Mapa de densidad original
        self.loss_mse= self.build_loss(density_map.squeeze(), gt_map.squeeze())               
        return density_map
    
    #Calcula el MSE:
    def build_loss(self, density_map, gt_data):
        loss_mse = self.loss_mse_fn(density_map, gt_data)  
        return loss_mse

    #ESTO EN UN PRINCIPIO NO SE USA y no sé para qué es, pero como no se usa...:
    '''
    def test_forward(self, img):                               
        density_map = self.CCN(img)                    
        return density_map
    '''


#MÉTODO NECESARIO PARA TESTER (no me he parado mucho en detalle a verlo):
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cur_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val
        self.count += 1
        self.avg = self.sum / self.count


#La clase que llevará nuestro entrenamiento:
class Tester():

	#Método para inicializar la clase (no hace falta llamarlo, se inicializa solo):
    def __init__(self, dataloader, cfg_data, pwd):


    	#Para guardar los resultados del test (si no existe la carpeya, se crea, y si no, se reescribe (se borra y se vuelve a crear)):
    	# paths: /Users/cristinareyes/Desktop/TFG/Resultados /home/pakitochus/Dropbox/Docencia/TFGs/Cristina/countcrowd_TFG/Resultados
        self.save_path = os.path.join('/home/pakitochus/Dropbox/Docencia/TFGs/Cristina/countcrowd_TFG/Resultados',
                                      str(cfg.NET) + '-' + 'noise-' + str(cfg_data.IS_NOISE) + '-' + str(
                                          cfg_data.BRIGHTNESS) +
                                      '-' + str(cfg_data.NOISE_SIGMA) + '-' + str(cfg_data.LONGEST_SIDE) + '-' + str(
                                          cfg_data.BLACK_AREA_RATIO) +
                                      '-' + str(cfg_data.IS_RANDOM) + '-' + 'denoise-' + str(cfg_data.IS_DENOISE))
        if not os.path.exists(self.save_path):
            os.system('mkdir ' + self.save_path)
        else:
            os.system('rm -rf ' + self.save_path)
            os.system('mkdir ' + self.save_path)




        self.cfg_data = cfg_data
        self.cfg = cfg
        #self.data_mode = cfg.DATASET
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd
        self.net_name = cfg.NET

        #Guardamos como "net" el contador de personas con la red escogida:
        #Si hay GPUs:
        if device == 'True':
            self.net = CrowdCounter(cfg.GPU_ID, self.net_name).cuda()

        #Si no hay GPUs disponibles:
        else:
            self.net = CrowdCounter(cfg.GPU_ID, self.net_name)


        #Se usa el optimizador Adam:
        self.optimizer = optim.Adam(self.net.CCN.parameters(), lr=cfg.LR, weight_decay=1e-4)
        # self.optimizer = optim.SGD(self.net.CCN.parameters(), cfg.LR, momentum=0.9, weight_decay=5e-4)

        #Usamos un tipo scheduler para ir ajustando el Learning Rate (LR)
        self.scheduler = StepLR(self.optimizer, step_size=cfg.NUM_EPOCH_LR_DECAY, gamma=cfg.LR_DECAY) 

        #Para guardar las estadísticas y verlas más tarde:
        self.train_record = {'best_mae': 1e20, 'best_mse': 1e20, 'best_model_name': ''}
        self.timer = {'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}

        self.epoch = 0
        self.i_tb = 0

    
        #NO USAREMOS EL DATASET GCC EN PRINCIPIO:
        '''
        if cfg.PRE_GCC:
            self.net.load_state_dict(torch.load(cfg.PRE_GCC_MODEL))

        '''

        #Llama a la función dataloader(), que carga en estas variables los conjuntos de data, entre otras cosas:
        self.train_loader, self.val_loader, self.test_loader, self.restore_transform = dataloader()

        #Si se ha interrumpido el test, se intenta retomar desde donde se dejó:
        #EN PRINCIPIO SE PODRÁ RETOMAR EL TEST LUEGO PONIENDO cfg.RESUME=TRUE en config() --> No estoy segura   ?????????????????????????????????????????????????????????
        if cfg.RESUME:
            # latest_state = torch.load(cfg.RESUME_PATH)
            # self.net.load_state_dict(latest_state['net'])
            # self.optimizer.load_state_dict(latest_state['optimizer'])
            # self.scheduler.load_state_dict(latest_state['scheduler'])
            # self.epoch = latest_state['epoch'] + 1
            # self.i_tb = latest_state['i_tb']
            # self.train_record = latest_state['train_record']
            # self.exp_path = latest_state['exp_path']
            # self.exp_name = latest_state['exp_name']

            latest_state = torch.load(cfg.RESUME_PATH)
            try:
                self.net.load_state_dict(latest_state)
            except:
                #Si curre un error al intentar cargar los datos anteriores para retomar el training:
                #¿QUÉ HACE ESTOE EXACTAMENTE?????????????????????????????????????
                self.net.load_state_dict({k.replace('module.', ''): v for k, v in latest_state.items()})

        # self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, 'exp', resume=cfg.RESUME)



    def forward(self):

        #Para apagar partes que no interesan durante el test (como Dropout Layers o BatchNorm Layers):
        #Porque no estamos entrenando:
        self.net.eval()

        #Todas las estadísticas de este tipo son del tipo AverageMeter:
        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()

       
        for vi, data in enumerate(self.val_loader, 0):
            #Nos va a mostrar el número por donde va la data guardándose en memoria:
            print(vi)
            #Separamos la data en imágenes, ground-truth maps y audios:
            img = data[0]
            gt_map = data[1]
            audio_img = data[2]

            #Si torch.no_grad usa junto con el net.eval() para apagar cosas que no interesan al hacer test.
            #Apaga gradientes de computación:
            #PARA NO APRENDER DEL TEST:
            with torch.no_grad():

                #PARA PASAR DE TENSORES A VARIABLES Y PODER TRABAJAR BIEN CON ELLOS:
                #Si hay GPUs:
                if device == 'True':
                    img = Variable(img).cuda()
                    gt_map = Variable(gt_map).cuda()
                    audio_img = Variable(audio_img).cuda()

                #No hay GPUs disponibles:
                else:
                    img = Variable(img) 
                    gt_map = Variable(gt_map)   
                    audio_img = Variable(audio_img)


                #Si la red que hemos escogido trabaja también con audio, lo metemos junto con la imagen en la red:
                #SE SACA UN MAPA PREDICHO CON LA ESTRUCTURA DE LA RED:
                if 'Audio' in self.net_name:
                    pred_map = self.net([img, audio_img], gt_map)
                else:
                    pred_map = self.net(img, gt_map)

                #¿PARA QUÉ ES NUMPY??????????????¿Que hace esto?¿Cargar la info del mapa de densidad?
                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()


                #Para cada mapa predicho se calculan el número de cabezas (personas) en la foto predicho y el original (y se dividen por cfg_data.LOG_PARA = 100 en este caso):
                for i_img in range(pred_map.shape[0]):
                    pred_cnt = np.sum(pred_map[i_img]) / self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img]) / self.cfg_data.LOG_PARA

                    #Actualizamos las pérdidas, MAE Y MSE para cada mapa:
                    losses.update(self.net.loss.item())
                    maes.update(abs(gt_count - pred_cnt))
                    mses.update((gt_count - pred_cnt) * (gt_count - pred_cnt))
                # if vi == 0:
                #     vis_results(self.exp_name, self.epoch, self.writer, self.restore_transform, img, pred_map, gt_map)

                #Guardamos resultados en imagen:
                save_img_name = 'val-' + str(vi) + '.jpg'
                raw_img = self.restore_transform(img.data.cpu()[0, :, :, :])
                log_mel = audio_img.data.cpu().numpy()

                raw_img.save(os.path.join(self.save_path, 'raw_img' + save_img_name))
                pyplot.imsave(os.path.join(self.save_path, 'log-mel-map' + save_img_name), log_mel[0, 0, :, :],
                              cmap='jet')

                pred_save_img_name = 'val-' + str(vi) + '-' + str(pred_cnt) + '.jpg'
                gt_save_img_name = 'val-' + str(vi) + '-' + str(gt_count) + '.jpg'
                pyplot.imsave(os.path.join(self.save_path, 'gt-den-map' + '-' + gt_save_img_name), gt_map[0, :, :],
                              cmap='jet')
                pyplot.imsave(os.path.join(self.save_path, 'pred-den-map' + '-' + pred_save_img_name),
                              pred_map[0, 0, :, :],
                              cmap='jet')


        #Hacemos una media de las estadísticas de todas las predicciones:
        mae = maes.avg
        mse = np.sqrt(mses.avg)
        loss = losses.avg

        # self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        # self.writer.add_scalar('test_mae', mae, self.epoch + 1)
        # self.writer.add_scalar('test_mse', mse, self.epoch + 1)
        print('test_mae: %.5f, test_mse: %.5f, test_loss: %.5f' % (mae, mse, loss))


  

#CLASES PARA LAS TRANSFORMACIONES DE IMAGEN Y ETIQUETAS (no lo he mirado mucho en profundidad):
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

#Métodos necesarios para loading_data():
def AC_collate(batch):
    # @GJY
    r"""Puts each data field into a tensor with outer dimension batch size"""

    transposed = list(zip(*batch))  # imgs, dens and raw audios
    imgs, dens, aud = [transposed[0], transposed[1], transposed[2]]

    error_msg = "batch must contain tensors; found {}"
    if isinstance(imgs[0], torch.Tensor) and isinstance(dens[0], torch.Tensor):

        # min_ht, min_wd = get_min_size(imgs)

        # print min_ht, min_wd

        # pdb.set_trace()

        cropped_imgs = []
        cropped_dens = []
        cropped_auds = []
        for i_sample in range(len(batch)):
            # _img, _den = random_crop(imgs[i_sample], dens[i_sample], [min_ht, min_wd])
            cropped_imgs.append(imgs[i_sample])
            cropped_dens.append(dens[i_sample])
            cropped_auds.append(aud[i_sample])

        cropped_imgs = torch.stack(cropped_imgs, 0, out=share_memory(cropped_imgs))
        cropped_dens = torch.stack(cropped_dens, 0, out=share_memory(cropped_dens))
        cropped_auds = torch.stack(cropped_auds, 0, out=share_memory(cropped_auds))

        return [cropped_imgs, cropped_dens, cropped_auds]

    raise TypeError((error_msg.format(type(batch[0]))))


def loading_data():

    #Cargamos todo lo que vamos a usar:
    mean_std = cfg_data.MEAN_STD
    log_para = cfg_data.LOG_PARA
    factor = cfg_data.LABEL_FACTOR
    train_main_transform =  Compose([
         RandomHorizontallyFlip()
    ])
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    gt_transform = standard_transforms.Compose([
         GTScaleDown(factor),
         LabelNormalize(log_para)
    ])
    restore_transform = standard_transforms.Compose([
         DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])


    train_set=edict()
    train_set.img_path=cfg_data.IMAGE_PATH
    train_set.aud_path=cfg_data.AUDIO_PATH
    train_set.mode='train'
    train_set.main_transform=train_main_transform
    train_set.img_transform=img_transform
    train_set.gt_transform=gt_transform
    train_set.is_noise=cfg_data.IS_NOISE
    train_set.brightness_decay=cfg_data.BRIGHTNESS
    train_set.noise_sigma=cfg_data.NOISE_SIGMA
    train_set.longest_side=cfg_data.LONGEST_SIDE


    val_set=edict()
    val_set.img_path=cfg_data.IMAGE_PATH
    val_set.aud_path=cfg_data.AUDIO_PATH
    val_set.mode='val'
    val_set.main_transform=None
    val_set.img_transform=img_transform
    val_set.gt_transform=gt_transform
    val_set.is_noise=cfg_data.IS_NOISE
    val_set.brightness_decay=cfg_data.BRIGHTNESS
    val_set.noise_sigma=cfg_data.NOISE_SIGMA
    val_set.longest_side=cfg_data.LONGEST_SIDE

    test_set=edict()
    test_set.img_path=cfg_data.IMAGE_PATH
    test_set.aud_path=cfg_data.AUDIO_PATH
    test_set.mode='test'
    test_set.main_transform=None
    test_set.img_transform=img_transform
    test_set.gt_transform=gt_transform
    test_set.is_noise=cfg_data.IS_NOISE
    test_set.brightness_decay=cfg_data.BRIGHTNESS
    test_set.noise_sigma=cfg_data.NOISE_SIGMA
    test_set.longest_side=cfg_data.LONGEST_SIDE


    if cfg_data.IS_CROSS_SCENE:
        '''
        train_set = AC(img_path=cfg_data.IMAGE_PATH, den_path=cfg_data.DENSITY_PATH + '/cross_scene_train',
                       aud_path=cfg_data.AUDIO_PATH,
                       mode='train', main_transform=train_main_transform, img_transform=img_transform,
                       gt_transform=gt_transform, is_noise=cfg_data.IS_NOISE, brightness_decay=cfg_data.BRIGHTNESS,
                       noise_sigma=cfg_data.NOISE_SIGMA, longest_side=cfg_data.LONGEST_SIDE
                       )
        ''' 

        train_set.den_path=cfg_data.DENSITY_PATH + '/cross_scene_train'
        val_set.den_path=cfg_data.DENSITY_PATH + '/cross_scene_val'
        test_set.den_path=cfg_data.DENSITY_PATH + '/cross_scene_test'

    else:
        '''
        train_set = AC(img_path=cfg_data.IMAGE_PATH, den_path=cfg_data.DENSITY_PATH + '/train',
                       aud_path=cfg_data.AUDIO_PATH,
                       mode='train', main_transform=train_main_transform, img_transform=img_transform,
                       gt_transform=gt_transform, is_noise=cfg_data.IS_NOISE, brightness_decay=cfg_data.BRIGHTNESS,
                       noise_sigma=cfg_data.NOISE_SIGMA, longest_side=cfg_data.LONGEST_SIDE,
                       black_area_ratio=cfg_data.BLACK_AREA_RATIO, is_random=cfg_data.IS_RANDOM, is_denoise=cfg_data.IS_DENOISE
                       )
        '''


        train_set.den_path=cfg_data.DENSITY_PATH + '/train'
        train_set.black_area_ratio=cfg_data.BLACK_AREA_RATIO
        train_set.is_random=cfg_data.IS_RANDOM
        train_set.is_denoise=cfg_data.IS_DENOISE


        val_set.den_path=cfg_data.DENSITY_PATH + '/val'
        val_set.black_area_ratio=cfg_data.BLACK_AREA_RATIO 
        val_set.is_random=cfg_data.IS_RANDOM
        val_set.s_denoise=cfg_data.IS_DENOISE

        test_set.den_path=cfg_data.DENSITY_PATH + '/test'
        test_set.black_area_ratio=cfg_data.BLACK_AREA_RATIO
        test_set.is_random=cfg_data.IS_RANDOM 
        test_set.is_denoise=cfg_data.IS_DENOISE



    #Limpiamos antes train_loader para cargar correctamente aquí:
    train_loader = None

    if cfg_data.TRAIN_BATCH_SIZE == 1:
        train_loader = DataLoader(train_set, batch_size=1, num_workers=8, shuffle=True, drop_last=True)
    elif cfg_data.TRAIN_BATCH_SIZE > 1:
        train_loader = DataLoader(train_set, batch_size=cfg_data.TRAIN_BATCH_SIZE, num_workers=8,
                                  collate_fn=AC_collate, shuffle=True, drop_last=True)

    '''
    if cfg_data.IS_CROSS_SCENE:

        val_set = AC(img_path=cfg_data.IMAGE_PATH, den_path=cfg_data.DENSITY_PATH + '/cross_scene_val',
                     aud_path=cfg_data.AUDIO_PATH,
                     mode='val', main_transform=None, img_transform=img_transform, gt_transform=gt_transform,
                     is_noise=cfg_data.IS_NOISE, brightness_decay=cfg_data.BRIGHTNESS,
                     noise_sigma=cfg_data.NOISE_SIGMA, longest_side=cfg_data.LONGEST_SIDE
                     )
    else:
        val_set = AC(img_path=cfg_data.IMAGE_PATH, den_path=cfg_data.DENSITY_PATH + '/val',
                     aud_path=cfg_data.AUDIO_PATH,
                     mode='val', main_transform=None, img_transform=img_transform, gt_transform=gt_transform,
                     is_noise=cfg_data.IS_NOISE, brightness_decay=cfg_data.BRIGHTNESS,
                     noise_sigma=cfg_data.NOISE_SIGMA, longest_side=cfg_data.LONGEST_SIDE,
                     black_area_ratio=cfg_data.BLACK_AREA_RATIO, is_random=cfg_data.IS_RANDOM, is_denoise=cfg_data.IS_DENOISE
                     )
    '''

    val_loader = DataLoader(val_set, batch_size=cfg_data.VAL_BATCH_SIZE, num_workers=1, shuffle=False, drop_last=False)

    '''
    if cfg_data.IS_CROSS_SCENE:
        test_set = AC(img_path=cfg_data.IMAGE_PATH, den_path=cfg_data.DENSITY_PATH + '/cross_scene_test',
                      aud_path=cfg_data.AUDIO_PATH,
                     mode='test', main_transform=None, img_transform=img_transform, gt_transform=gt_transform,
                     is_noise=cfg_data.IS_NOISE, brightness_decay=cfg_data.BRIGHTNESS,
                     noise_sigma=cfg_data.NOISE_SIGMA, longest_side=cfg_data.LONGEST_SIDE
                     )
    else:
        test_set = AC(img_path=cfg_data.IMAGE_PATH, den_path=cfg_data.DENSITY_PATH + '/test',
                      aud_path=cfg_data.AUDIO_PATH,
                      mode='test', main_transform=None, img_transform=img_transform, gt_transform=gt_transform,
                      is_noise=cfg_data.IS_NOISE, brightness_decay=cfg_data.BRIGHTNESS,
                      noise_sigma=cfg_data.NOISE_SIGMA, longest_side=cfg_data.LONGEST_SIDE,
                      black_area_ratio=cfg_data.BLACK_AREA_RATIO, is_random=cfg_data.IS_RANDOM, is_denoise=cfg_data.IS_DENOISE
                      )
    '''

    test_loader = DataLoader(test_set, batch_size=cfg_data.VAL_BATCH_SIZE, num_workers=1, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, restore_transform

'''
#PARA QUÉ ES ESTOO? --> está fuera del método, cuidao --> Tampoco se usa
if __name__ == '__main__':
    train_loader, val_loader, test_loader, restore_transform = loading_data()

    for i, data in enumerate(train_loader):
        im = data[0]
        den = data[1]
        aud = data[2]
        print('Training dataset', im.shape, den.shape, aud.shape, den.sum(-1).sum(-1)/100)
        print(im)

    for i, data in enumerate(val_loader):
        im = data[0]
        den = data[1]
        aud = data[2]
        print('Validation dataset', im.shape, den.shape, aud.shape, den.sum()/100)
'''

def Test():
  
    #------------prepare enviroment------------

    #Para sintetizar más el código, poner la SEED deseada desde config y ya está.
    '''
    seed = cfg.SEED
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    '''

    #¿ESTÁ ESTO BIEN ASÍ??????????????????????????????????????????????????????¿QUÉ HACE EXACTAMENTE???????????????????????
    if device=='True':
        cfg.GPU_ID = [0,1]
        gpus = cfg.GPU_ID
        if len(gpus)==1:
            torch.cuda.set_device(gpus[0])
        torch.backends.cudnn.benchmark = True


    #Solo usaremos 1 dataset, lo comento:
    '''
        #------------prepare data loader------------
    data_mode = cfg.DATASET
    if data_mode is 'SHHA':
        from datasets.SHHA.loading_data import loading_data
        from datasets.SHHA.setting import cfg_data
    elif data_mode is 'SHHB':
        from datasets.SHHB.loading_data import loading_data
        from datasets.SHHB.setting import cfg_data
    elif data_mode is 'QNRF':
        from datasets.QNRF.loading_data import loading_data
        from datasets.QNRF.setting import cfg_data
    elif data_mode is 'UCF50':
        from datasets.UCF50.loading_data import loading_data
        from datasets.UCF50.setting import cfg_data
    elif data_mode is 'WE':
        from datasets.WE.loading_data import loading_data
        from datasets.WE.setting import cfg_data
    elif data_mode is 'GCC':
        from datasets.GCC.loading_data import loading_data
        from datasets.GCC.setting import cfg_data
    elif data_mode is 'Mall':
        from datasets.Mall.loading_data import loading_data
        from datasets.Mall.setting import cfg_data
    elif data_mode is 'UCSD':
        from datasets.UCSD.loading_data import loading_data
        from datasets.UCSD.setting import cfg_data
    elif data_mode is 'AC':  # Qingzhong
        from datasets.AC.loading_data import loading_data
        from datasets.AC.setting import cfg_data
    '''


    #Muestra todos los parámetros escogidos en Setting() y Config()
    print(cfg, cfg_data)

    #------------Prepare Tester------------
    net = cfg.NET


    #------------Start Test------------
    pwd = os.path.split(os.path.realpath(__file__))[0]
    cc_trainer = Tester(loading_data, cfg_data, pwd)
    cc_trainer.forward()




#Necesario para el método logger():
def copy_cur_env(work_dir, dst_dir, exception):

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    for filename in os.listdir(work_dir):

        file = os.path.join(work_dir,filename)
        dst_file = os.path.join(dst_dir,filename)


        if os.path.isdir(file) and exception not in filename:
            shutil.copytree(file, dst_file)
        elif os.path.isfile(file):
            shutil.copyfile(file,dst_file)


#Métodos necesarios para la clase TRAINER() (No me he parado aquí en detalle):

def vis_results(exp_name, epoch, writer, restore, img, pred_map, gt_map):

    pil_to_tensor = standard_transforms.ToTensor()

    x = []
    
    for idx, tensor in enumerate(zip(img.cpu().data, pred_map, gt_map)):
        if idx>1:# show only one group
            break
        pil_input = restore(tensor[0])
        pil_output = torch.from_numpy(tensor[1]/(tensor[2].max()+1e-10)).repeat(3,1,1)
        pil_label = torch.from_numpy(tensor[2]/(tensor[2].max()+1e-10)).repeat(3,1,1)
        x.extend([pil_to_tensor(pil_input.convert('RGB')), pil_label, pil_output])
    x = torch.stack(x, 0)
    x = vutils.make_grid(x, nrow=3, padding=5)
    x = (x.numpy()*255).astype(np.uint8)

    writer.add_image(exp_name + '_epoch_' + str(epoch+1), x)

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

#Crea un fichero de logs:
def logger(exp_path, exp_name, work_dir, exception, resume=False):

    from tensorboardX import SummaryWriter
    
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    writer = SummaryWriter(exp_path+ '/' + exp_name)
    log_file = exp_path + '/' + exp_name + '/' + exp_name + '.txt'
    
    
    with open(log_file, 'a') as f:
        f.write(''.join('cfg.NET = ' + cfg.NET + '\n' 
        	+ 'cfg.RESUME = ' + str(cfg.RESUME) + '\n'
        	+ 'cfg.RESUME_PATH =' + cfg.RESUME_PATH + '\n'
        	+ 'cfg.LR = '+ str(cfg.LR) + '\n'
        	+ 'cfg.LR_DECAY =' + str(cfg.LR_DECAY) + '\n'
        	+ 'cfg.LR_DECAY_START =' + str(cfg.LR_DECAY_START) + '\n'
        	+ 'cfg.NUM_EPOCH_LR_DECAY =' + str(cfg.NUM_EPOCH_LR_DECAY) + '\n'
        	+ 'cfg.MAX_EPOCH =' + str(cfg.MAX_EPOCH) + '\n'
        	+ 'cfg.LAMBDA_1 =' + str(cfg.LAMBDA_1) + '\n'
        	+ 'cfg.SETTINGS =' + cfg.SETTINGS + '\n'
        	+ 'cfg.EXP_NAME =' + cfg.EXP_NAME + '\n'
        	+ 'cfg.EXP_PATH =' + cfg.EXP_PATH + '\n'
        	+ 'cfg.VAL_DENSE_START =' + str(cfg.VAL_DENSE_START) + '\n'
        	+ 'cfg.VAL_FREQ =' + str(cfg.VAL_FREQ) + '\n'
        	+ 'cfg.VISIBLE_NUM_IMGS =' + str(cfg.VISIBLE_NUM_IMGS) + '\n'
        	) + '\n\n\n\n')

    if not resume:
        #Se copia la carpeta del código dentro del EXP_PATH: ¿ES IMPORTANTE, PARA QUÉ???????????????????????????????????????????????
        copy_cur_env(work_dir, exp_path+ '/' + exp_name + '/code', exception)


    return writer, log_file


#CLASE PARA ENTRENAR LA RED (SE PARECE MUCHO A TESTER() ):
class Trainer():
    def __init__(self, dataloader, cfg_data, pwd):

    	#Guardar los resulstados del train:

    	# paths: /Users/cristinareyes/Desktop/TFG/Resultados /home/pakitochus/Dropbox/Docencia/TFGs/Cristina/countcrowd_TFG/Resultados
        self.save_path = os.path.join('/home/pakitochus/Dropbox/Docencia/TFGs/Cristina/countcrowd_TFG/Resultados',
                                      str(cfg.NET) + '-' + 'noise-' + str(cfg_data.IS_NOISE) + '-' + str(
                                          cfg_data.BRIGHTNESS) +
                                      '-' + str(cfg_data.NOISE_SIGMA) + '-' + str(cfg_data.LONGEST_SIDE) + '-' + str(
                                          cfg_data.BLACK_AREA_RATIO) +
                                      '-' + str(cfg_data.IS_RANDOM) + '-' + 'denoise-' + str(cfg_data.IS_DENOISE))
        if not os.path.exists(self.save_path):
            os.system('mkdir '+self.save_path)
        else:
            os.system('rm -rf ' + self.save_path)
            os.system('mkdir ' + self.save_path)


        self.cfg_data = cfg_data
        self.cfg = cfg

        #Lo deshabilito porque de momento sólo usaremos un dataset:
        #self.data_mode = cfg.DATASET
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd

        self.net_name = cfg.NET

        #Si hay GPUs:
        if device == 'True':
            self.net = CrowdCounter(cfg.GPU_ID,self.net_name).cuda()

        #Si no hay GPUs disponibles:
        else:
            self.net = CrowdCounter(cfg.GPU_ID,self.net_name)  


        self.optimizer = optim.Adam(self.net.CCN.parameters(), lr=cfg.LR, weight_decay=1e-4)
        # self.optimizer = optim.SGD(self.net.CCN.parameters(), cfg.LR, momentum=0.9, weight_decay=5e-4)
        self.scheduler = StepLR(self.optimizer, step_size=cfg.NUM_EPOCH_LR_DECAY, gamma=cfg.LR_DECAY)          

        self.train_record = {'best_mae': 1e20, 'best_mse':1e20, 'best_model_name': ''}
        self.timer = {'iter time' : Timer(),'train time' : Timer(),'val time' : Timer()} 

        self.epoch = 0
        self.i_tb = 0
        
        #No creo que acabe usando esto:
        '''
        if cfg.PRE_GCC:
            self.net.load_state_dict(torch.load(cfg.PRE_GCC_MODEL))
        '''
        
        self.train_loader, self.val_loader, self.test_loader, self.restore_transform = dataloader()
		

		#Para continuar el entrenamiento si se interrumpe:
        if cfg.RESUME:
            latest_state = torch.load(cfg.RESUME_PATH)
            self.net.load_state_dict(latest_state['net'])
            self.optimizer.load_state_dict(latest_state['optimizer'])
            self.scheduler.load_state_dict(latest_state['scheduler'])
            self.epoch = latest_state['epoch'] + 1
            self.i_tb = latest_state['i_tb']
            self.train_record = latest_state['train_record']
            self.exp_path = latest_state['exp_path']
            self.exp_name = latest_state['exp_name']

        #Creamos un fichero de logs:
        self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, 'exp', resume=cfg.RESUME)


    def forward(self):

        #Para cada época:
        for epoch in range(self.epoch,self.cfg.MAX_EPOCH):
            self.epoch = epoch
            
            #Calculamos el tiempo del entrenamiento (entre tic y toc)
            # training    
            self.timer['train time'].tic()
            #Ejecutamos el métdo train() de Trainer() definido más abajo:
            self.train()
            self.timer['train time'].toc(average=False)

            #Se imprime cuánto se ha tardado en entrenar:
            print( 'train time: {:.2f}s'.format(self.timer['train time'].diff) )
            print( '='*20 )


            #HE CAMBIADO ESTO DEBAJO DEL TRAIN PORQUE ME DABA UN WARNING --> Optimizer debe ejecutarse antes del lr_scheduler   #¿ES CORRECTO ENTONCES HACER ESTO???????????????
            #Sólo si estamos en una epoch después de la que hemos decidido en "cfg.LR_DECAY_START", el Learning Rate comenzará a decaer:
            if epoch > self.cfg.LR_DECAY_START:
                #El scheduler.step() va cambiando el Learning Rate en cada epoch:
                self.scheduler.step()
                


            # validation

            #Antes de VAL_DENSE-START se valida con VAL_FREQ:
            #Si la epoch es múltiplo de VAL_FREQ o la epoch es posterior a VAL_DENSE_START: #¿NO SERÍA UN AND?????????????????????????????????
            if epoch%self.cfg.VAL_FREQ==0 or epoch>self.cfg.VAL_DENSE_START:

                #Se calcula el tiempo de la validación (entre tic y toc)
                self.timer['val time'].tic()

                #PARA OTROS DATASET:
                '''
                if self.data_mode in ['SHHA', 'SHHB', 'QNRF', 'UCF50', 'AC']:  # Qingzhong
                    self.validate_V1()
                    self.test_V1()
                elif self.data_mode is 'WE':
                    self.validate_V2()
                elif self.data_mode is 'GCC':
                    self.validate_V3()
                '''
                #Como usamos 'AC':
                self.validate_V1()

                #He quitado este porque lo veo redundante con el de arriba:
                #self.test_V1()

                self.timer['val time'].toc(average=False)
                print( 'val time: {:.2f}s'.format(self.timer['val time'].diff) )


    def train(self): # training for all datasets

        #La clase CrowdCounter(nn.module),por su tipo, tiene un módulo .train() que avisa al modelo de que estás entrenando.
        #OJO: esto NO significa que ponga el modelo a entrenar, sino que antes de programar el entrenamiento avises al modelo de que
        #debe hacer los Dropout, Batchnorm y otros procesos que sólo se hacen en training y no en test por ejemplo:
        self.net.train()
        for i, data in enumerate(self.train_loader, 0):
            #Por cada muestra tomamos su imagen, su audio y su mapa de densidad ground-truth:
            self.timer['iter time'].tic()
            img = data[0]
            gt_map = data[1]
            audio_img = data[2]

            #PARA PASAR DE TENSORES A VARIABLES Y PODER TRABAJAR BIEN CON ELLOS:
            #Si hay GPUs:
            if device == 'True':
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()
                audio_img = Variable(audio_img).cuda()

            #No hay GPUs disponibles:
            else:
                img = Variable(img) 
                gt_map = Variable(gt_map)   
                audio_img = Variable(audio_img)

            #En PyTorch tenemos que poner los gradientes a cero antes de la backpropagation
            #porque PyTorch acumula (sumatorio) los gradientes. Por ello, los actualizaré más tarde manualmente.
            self.optimizer.zero_grad()

            #Si la red que hemos escogido trabaja también con audio, lo metemos junto con la imagen en la red:
            #SE SACA UN MAPA PREDICHO CON LA ESTRUCTURA DE LA RED:
            if 'Audio' in self.net_name:
                pred_map = self.net([img, audio_img], gt_map)
            else:
                pred_map = self.net(img, gt_map)

            loss = self.net.loss

            #Backpropagation (ajuste de los pesos):
            loss.backward()

            #Usamos el optimizador para calcular mejor la dirección de menor pérdida:
            self.optimizer.step()

            #Cuando la siguiente iteración sea múltiplo de PRINT_FREQ, se mostrarán las estadísticas del training:
            if (i + 1) % self.cfg.PRINT_FREQ == 0:
                self.i_tb += 1
                self.writer.add_scalar('train_loss', loss.item(), self.i_tb)
                self.timer['iter time'].toc(average=False)
                print( '[ep %d][it %d][loss %.4f][lr %.4f][%.2fs]' % \
                        (self.epoch + 1, i + 1, loss.item(), self.optimizer.param_groups[0]['lr']*10000, self.timer['iter time'].diff) )
                print( '        [cnt: gt: %.1f pred: %.2f]' % (gt_map[0].sum().data/self.cfg_data.LOG_PARA, pred_map[0].sum().data/self.cfg_data.LOG_PARA) )           


    def validate_V1(self):# validate_V1 for SHHA, SHHB, UCF-QNRF, UCF50, AC

        #Para apagar partes que no interesan durante el test (como Dropout Layers o BatchNorm Layers) --> contrario de net.train():
        #Porque no estamos entrenando:
        self.net.eval()

        #Todas las estadísticas de este tipo son del tipo AverageMeter:
        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()

        #Vamos a guardar resultados de la validación en la carpeta "Resultados"--> si no existe la crea y si existe, la reemplaza:
        if not os.path.exists(self.save_path):
            os.system('mkdir '+self.save_path)
        else:
            os.system('rm -rf ' + self.save_path)
            os.system('mkdir ' + self.save_path)

        #Por cada muestra en validación cogemos el audio, imagen y mapa:
        for vi, data in enumerate(self.val_loader, 0):
            img = data[0]
            gt_map = data[1]
            audio_img = data[2]


            #torch.no_grad usa junto con el net.eval() para apagar cosas que no interesan al hacer test.
            #Apaga gradientes de computación:
            #PARA NO APRENDER DEL TEST:
            with torch.no_grad():

                #PARA PASAR DE TENSORES A VARIABLES Y PODER TRABAJAR BIEN CON ELLOS:

                #Si hay GPUs:
                if device == 'True':
                    img = Variable(img).cuda()
                    gt_map = Variable(gt_map).cuda()
                    audio_img = Variable(audio_img).cuda()

                #No hay GPUs disponibles:
                else:
                    img = Variable(img) 
                    gt_map = Variable(gt_map)   
                    audio_img = Variable(audio_img) 

                #Si la red que hemos escogido trabaja también con audio, lo metemos junto con la imagen en la red:
                #SE SACA UN MAPA PREDICHO CON LA ESTRUCTURA DE LA RED:
                if 'Audio' in self.net_name:
                    pred_map = self.net([img, audio_img], gt_map)
                else:
                    pred_map = self.net(img, gt_map)

                #¿Para qué es esto????????????????
                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()


                #Para cada mapa predicho se calculan el número de cabezas (personas) en la foto predicho y el original (y se dividen por cfg_data.LOG_PARA = 100 en este caso):
                for i_img in range(pred_map.shape[0]):
                
                    pred_cnt = np.sum(pred_map[i_img])/self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img])/self.cfg_data.LOG_PARA

                    #Actualizamos las pérdidas, MAE Y MSE para cada mapa:
                    losses.update(self.net.loss.item())
                    maes.update(abs(gt_count-pred_cnt))
                    mses.update((gt_count-pred_cnt)*(gt_count-pred_cnt))


                #Para la primera muestra de audio, vídeo y mapa, visualizamos los resultados (creo que pinta la imagen y todo):
                if vi==0:
                    vis_results(self.exp_name, self.epoch, self.writer, self.restore_transform, img, pred_map, gt_map)

                # print('---------------------val-----------------------')
                # print('gt_cnt: %.3f, pred_cnt: %.3f'%(gt_count, pred_count))


                #Guardamos resultados en imagen:
                save_img_name = 'val-' + str(vi) + '.jpg'
                raw_img = self.restore_transform(img.data.cpu()[0, :, :, :])
                log_mel = audio_img.data.cpu().numpy()

                raw_img.save(os.path.join(self.save_path, 'raw_img' + save_img_name))
                pyplot.imsave(os.path.join(self.save_path, 'log-mel-map' + save_img_name), log_mel[0, 0, :, :],
                              cmap='jet')

                pred_save_img_name = 'val-' + str(vi) + '-' + str(pred_cnt) + '.jpg'
                gt_save_img_name = 'val-' + str(vi) + '-' + str(gt_count) + '.jpg'
                pyplot.imsave(os.path.join(self.save_path, 'gt-den-map' + '-' + gt_save_img_name), gt_map[0, :, :],
                              cmap='jet')
                pyplot.imsave(os.path.join(self.save_path, 'pred-den-map' + '-' + pred_save_img_name),
                              pred_map[0, 0, :, :],
                              cmap='jet')
            
        mae = maes.avg
        mse = np.sqrt(mses.avg)
        loss = losses.avg

        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('val_mae', mae, self.epoch + 1)
        self.writer.add_scalar('val_mse', mse, self.epoch + 1)

        self.train_record = update_model(self.net,self.optimizer,self.scheduler,self.epoch,self.i_tb,self.exp_path,self.exp_name, \
            [mae, mse, loss],self.train_record,self.log_txt)
        print_summary(self.exp_name,[mae, mse, loss],self.train_record)
        print('val_mae: %.5f, val_mse: %.5f, val_loss: %.5f' % (mae, mse, loss))



    #ESTO NO LO USO:
    '''
    def test_V1(self):  # test_v1 for SHHA, SHHB, UCF-QNRF, UCF50, AC

        #Para apagar partes que no interesan durante el test (como Dropout Layers o BatchNorm Layers) --> contrario de net.train():
        #Porque no estamos entrenando:
        self.net.eval()

        #Todas las estadísticas de este tipo son del tipo AverageMeter:
        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()

        #Por cada muestra en test cogemos el audio, imagen y mapa:
        for vi, data in enumerate(self.test_loader, 0):
            img = data[0]
            gt_map = data[1]
            audio_img = data[2]


            #torch.no_grad usa junto con el net.eval() para apagar cosas que no interesan al hacer test.
            #Apaga gradientes de computación:
            #PARA NO APRENDER DEL TEST:
            with torch.no_grad():

                #Si hay GPUs:
                if device == 'True':
                    img = Variable(img).cuda()
                    gt_map = Variable(gt_map).cuda()
                    audio_img = Variable(audio_img).cuda()

                #No hay GPUs disponibles:
                else:
                    img = Variable(img) 
                    gt_map = Variable(gt_map)   
                    audio_img = Variable(audio_img) 


                if 'Audio' in self.net_name:
                    pred_map = self.net([img, audio_img], gt_map)
                else:
                    pred_map = self.net(img, gt_map)

                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()

                for i_img in range(pred_map.shape[0]):
                    pred_cnt = np.sum(pred_map[i_img]) / self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img]) / self.cfg_data.LOG_PARA

                    losses.update(self.net.loss.item())
                    maes.update(abs(gt_count - pred_cnt))
                    mses.update((gt_count - pred_cnt) * (gt_count - pred_cnt))
                if vi == 0:
                    vis_results(self.exp_name, self.epoch, self.writer, self.restore_transform, img, pred_map, gt_map)
                # print('------------------------test-------------------------')
                # print('gt_cnt: %.3f, pred_cnt: %.3f' % (gt_count, pred_count))


        mae = maes.avg
        mse = np.sqrt(mses.avg)
        loss = losses.avg

        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('test_mae', mae, self.epoch + 1)
        self.writer.add_scalar('test_mse', mse, self.epoch + 1)
        print('test_mae: %.5f, test_mse: %.5f, test_loss: %.5f' % (mae, mse, loss))
    


	#Es para otros dataset, nosotros no lo usaremos:

    def validate_V2(self):# validate_V2 for WE

        self.net.eval()

        losses = AverageCategoryMeter(5)
        maes = AverageCategoryMeter(5)

        roi_mask = []
        from datasets.WE.setting import cfg_data 
        from scipy import io as sio
        for val_folder in cfg_data.VAL_FOLDER:

            roi_mask.append(sio.loadmat(os.path.join(cfg_data.DATA_PATH,'test',val_folder + '_roi.mat'))['BW'])
        
        for i_sub,i_loader in enumerate(self.val_loader,0):

            mask = roi_mask[i_sub]
            for vi, data in enumerate(i_loader, 0):
                img, gt_map = data

                with torch.no_grad():
                    img = Variable(img).cuda()
                    gt_map = Variable(gt_map).cuda()

                    pred_map = self.net.forward(img,gt_map)

                    pred_map = pred_map.data.cpu().numpy()
                    gt_map = gt_map.data.cpu().numpy()

                    for i_img in range(pred_map.shape[0]):
                    
                        pred_cnt = np.sum(pred_map[i_img])/self.cfg_data.LOG_PARA
                        gt_count = np.sum(gt_map[i_img])/self.cfg_data.LOG_PARA

                        losses.update(self.net.loss.item(),i_sub)
                        maes.update(abs(gt_count-pred_cnt),i_sub)
                    if vi==0:
                        vis_results(self.exp_name, self.epoch, self.writer, self.restore_transform, img, pred_map, gt_map)
            
        mae = np.average(maes.avg)
        loss = np.average(losses.avg)

        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('mae', mae, self.epoch + 1)
        self.writer.add_scalar('mae_s1', maes.avg[0], self.epoch + 1)
        self.writer.add_scalar('mae_s2', maes.avg[1], self.epoch + 1)
        self.writer.add_scalar('mae_s3', maes.avg[2], self.epoch + 1)
        self.writer.add_scalar('mae_s4', maes.avg[3], self.epoch + 1)
        self.writer.add_scalar('mae_s5', maes.avg[4], self.epoch + 1)

        self.train_record = update_model(self.net,self.optimizer,self.scheduler,self.epoch,self.i_tb,self.exp_path,self.exp_name, \
            [mae, 0, loss],self.train_record,self.log_txt)
        print_WE_summary(self.log_txt,self.epoch,[mae, 0, loss],self.train_record,maes)





    def validate_V3(self):# validate_V3 for GCC

        self.net.eval()
        
        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()

        c_maes = {'level':AverageCategoryMeter(9), 'time':AverageCategoryMeter(8),'weather':AverageCategoryMeter(7)}
        c_mses = {'level':AverageCategoryMeter(9), 'time':AverageCategoryMeter(8),'weather':AverageCategoryMeter(7)}


        for vi, data in enumerate(self.val_loader, 0):
            img, gt_map, attributes_pt = data

            with torch.no_grad():
                img = Variable(img).cuda()
                gt_map = Variable(gt_map).cuda()


                pred_map = self.net.forward(img,gt_map)

                pred_map = pred_map.data.cpu().numpy()
                gt_map = gt_map.data.cpu().numpy()

                for i_img in range(pred_map.shape[0]):
                
                    pred_cnt = np.sum(pred_map[i_img])/self.cfg_data.LOG_PARA
                    gt_count = np.sum(gt_map[i_img])/self.cfg_data.LOG_PARA

                    s_mae = abs(gt_count-pred_cnt)
                    s_mse = (gt_count-pred_cnt)*(gt_count-pred_cnt)

                    losses.update(self.net.loss.item())
                    maes.update(s_mae)
                    mses.update(s_mse)   
                    attributes_pt = attributes_pt.squeeze() 
                    c_maes['level'].update(s_mae,attributes_pt[i_img][0])
                    c_mses['level'].update(s_mse,attributes_pt[i_img][0])
                    c_maes['time'].update(s_mae,attributes_pt[i_img][1]/3)
                    c_mses['time'].update(s_mse,attributes_pt[i_img][1]/3)
                    c_maes['weather'].update(s_mae,attributes_pt[i_img][2])
                    c_mses['weather'].update(s_mse,attributes_pt[i_img][2])


                if vi==0:
                    vis_results(self.exp_name, self.epoch, self.writer, self.restore_transform, img, pred_map, gt_map)
            
        loss = losses.avg
        mae = maes.avg
        mse = np.sqrt(mses.avg)


        self.writer.add_scalar('val_loss', loss, self.epoch + 1)
        self.writer.add_scalar('mae', mae, self.epoch + 1)
        self.writer.add_scalar('mse', mse, self.epoch + 1)

        self.train_record = update_model(self.net,self.optimizer,self.scheduler,self.epoch,self.i_tb,self.exp_path,self.exp_name, \
            [mae, mse, loss],self.train_record,self.log_txt)


        print_GCC_summary(self.log_txt,self.epoch,[mae, mse, loss],self.train_record,c_maes,c_mses)

	'''

#Ejecutar este método para entrenar la red:
def Train():
	

    #------------prepare enviroment------------

    #Para sintetizar más el código, poner la SEED deseada desde config y ya está.
    '''
    seed = cfg.SEED
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    '''

    #¿ESTÁ ESTO BIEN ASÍ??????????????????????????????????????????????????????¿QUÉ HACE EXACTAMENTE???????????????????????
    if device=='True':
        cfg.GPU_ID = [0,1]
        gpus = cfg.GPU_ID
        if len(gpus)==1:
            torch.cuda.set_device(gpus[0])
        torch.backends.cudnn.benchmark = True


    #Solo usaremos 1 dataset, lo comento:
    '''
    
    #------------prepare data loader------------
    data_mode = cfg.DATASET
    if data_mode is 'SHHA':
        from datasets.SHHA.loading_data import loading_data
        from datasets.SHHA.setting import cfg_data
    elif data_mode is 'SHHB':
        from datasets.SHHB.loading_data import loading_data
        from datasets.SHHB.setting import cfg_data
    elif data_mode is 'QNRF':
        from datasets.QNRF.loading_data import loading_data
        from datasets.QNRF.setting import cfg_data
    elif data_mode is 'UCF50':
        from datasets.UCF50.loading_data import loading_data
        from datasets.UCF50.setting import cfg_data
    elif data_mode is 'WE':
        from datasets.WE.loading_data import loading_data
        from datasets.WE.setting import cfg_data
    elif data_mode is 'GCC':
        from datasets.GCC.loading_data import loading_data
        from datasets.GCC.setting import cfg_data
    elif data_mode is 'Mall':
        from datasets.Mall.loading_data import loading_data
        from datasets.Mall.setting import cfg_data
    elif data_mode is 'UCSD':
        from datasets.UCSD.loading_data import loading_data
        from datasets.UCSD.setting import cfg_data
    elif data_mode is 'AC':  # Qingzhong
        from datasets.AC.loading_data import loading_data
        from datasets.AC.setting import cfg_data

    '''

    # cfg_data.IS_NOISE = (opt.is_noise == 1)
    # cfg_data.BRIGHTNESS = opt.brightness
    # cfg_data.NOISE_SIGMA = opt.noise_sigma
    # cfg_data.LONGEST_SIDE = opt.longest_side
    # cfg_data.BLACK_AREA_RATIO = opt.black_area_ratio
    # cfg_data.IS_RANDOM = (opt.is_random == 1)

    print(cfg, cfg_data)


    #------------Prepare Trainer------------

    #CAMBIAR!!! QUE SE PUEDA ENTRENAR TAMBIÉN CON 'SANet', 'SANet_Audio', 'CMTL' y 'PCCNet'!!!!!!! CAMBIAR!
    '''
    net = cfg.NET
    if net in ['MCNN', 'AlexNet', 'VGG', 'VGG_DECODER', 'Res50', 'Res101', 'CSRNet','Res101_SFCN',
               'CSRNet_IN', 'CSRNet_Audio', 'CANNet', 'CANNet_Audio', 'CSRNet_Audio_Concat', 'CANNet_Audio_Concat',
               'CSRNet_Audio_Guided', 'CANNet_Audio_Guided'
               ]:
        from trainer import Trainer
    elif net in ['SANet', 'SANet_Audio']:
        from trainer_for_M2TCC import Trainer # double losses but signle output
    elif net in ['CMTL']: 
        from trainer_for_CMTL import Trainer # double losses and double outputs
    elif net in ['PCCNet']:
        from trainer_for_M3T3OCC import Trainer
	'''

    #------------Start Training------------
    pwd = os.path.split(os.path.realpath(__file__))[0]
    cc_trainer = Trainer(loading_data, cfg_data, pwd)
    cc_trainer.forward()


#PRUEBA
Train()
