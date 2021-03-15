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
import matplotlib.pyplot as pyplot

#Declaro la variable global cfg con edict para ir añadiéndole atributos sobre la marcha:
cfg = edict()
cfg_data = edict() 

#Para comprobar si tenemos GPUs disponibles para usar o no:
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def config():

	cfg.SEED = 3035 # random seed,  for reproduction

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


    #Para poder continuar el training si se interrumpe:
	cfg.RESUME = False # contine training
	cfg.RESUME_PATH = '../trained_models/exp/image-noise-0.2-25-denoise-audio-wo_AC_CSRNet_1e-05/all_ep_274_mae_29.8_mse_48.5.pth' #



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
	#cfg.PRINT_FREQ = 10



	#Variable que almacena la fecha y hora local del momento de ejecución:
	#now = time.strftime("%m-%d_%H-%M", time.localtime())




	#CREAR NOMBRE SEGÚN EL EXPERIMENTO (red,dataset(ESO LO HE QUITADO)...):
	# settings = 'image-noise(0.2, 25)-audio-wo'  # image-clean/(0.3, 50)_audio-w/wo
	cfg.SETTINGS = 'image-clean-audio-wo'  # image-clean/(0.3, 50)_audio-w/wo

	cfg.EXP_NAME = cfg.SETTINGS + '_' + cfg.NET + '_' + str(cfg.LR)

	#Lo comento porque dudo que usemos estos dataset:

	#if cfg.DATASET == 'UCF50':
	#	cfg.EXP_NAME += '_' + str(cfg.VAL_INDEX)	
	#
	#if cfg.DATASET == 'GCC':
	#	cfg.EXP_NAME += '_' + cfg.VAL_MODE

	

	cfg.EXP_PATH = '../trained_models/exp' # the path of logs, checkpoints, and current codes


	#------------------------------VAL------------------------
	cfg.VAL_DENSE_START = 50
	cfg.VAL_FREQ = 10 # Before __C.VAL_DENSE_START epoches, the freq is set as __C.VAL_FREQ

	#------------------------------VIS------------------------

	#Tener en cuenta que tendremos también imágenes de baja resolución (128. x72):
	cfg.VISIBLE_NUM_IMGS = 1 #  must be 1 for training images with the different sizes



def setting():

	#Path del dataset:
	DATA_PATH = '/Volumes/Cristina /TFG/Data'

	#Tamaño estándar de las fotos (¿Seguro que es este tamaño?):
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
            from .SCC_Model.AlexNet import AlexNet as net        
        elif model_name == 'VGG':
            from .SCC_Model.VGG import VGG as net
        elif model_name == 'VGG_DECODER':
            from .SCC_Model.VGG_decoder import VGG_decoder as net
        elif model_name == 'MCNN':
            from .SCC_Model.MCNN import MCNN as net
        elif model_name == 'CSRNet':
            from .SCC_Model.CSRNet import CSRNet as net
        elif model_name == 'Res50':
            from .SCC_Model.Res50 import Res50 as net
        elif model_name == 'Res101':
            from .SCC_Model.Res101 import Res101 as net            
        elif model_name == 'Res101_SFCN':
            from .SCC_Model.Res101_SFCN import Res101_SFCN as net
        elif model_name == 'CSRNet_IN':  # Qingzhong
            from .SCC_Model.CSRNet_IN import CSRNet as net
        elif model_name == 'CSRNet_Audio':  # Qingzhong
            from .SCC_Model.CSRWaveNet import CSRNet as net
        elif model_name == 'CANNet':
            from .SCC_Model.CACC import CANNet as net
        elif model_name == 'CANNet_Audio':
            from .SCC_Model.CANWaveNet import CANNet as net
        elif model_name == 'CSRNet_Audio_Concat':
            from .SCC_Model.CSRWaveNet import CSRNetConcat as net
        elif model_name == 'CANNet_Audio_Concat':
            from.SCC_Model.CANWaveNet import CANNetConcat as net
        elif model_name == 'CSRNet_Audio_Guided':
            from .SCC_Model.CSRWaveNet import CSRNetGuided as net

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



    @property
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

    #ESTO EN UN PRINCIPIO NO SE USA y no sé para qué es¿?:
    '''
    def test_forward(self, img):                               
        density_map = self.CCN(img)                    
        return density_map
    '''


#La clase que llevará nuestro entrenamiento:
class Tester():

	#Método para inicializar la clase (no hace falta llamarlo, se inicializa solo):
    def __init__(self, dataloader, cfg_data, pwd):


    	#Para guardar los resultados del entrenamiento (si no existe la carpeya, se crea, y si no, se reescribe (se borra y se vuelve a crear)):
        self.save_path = os.path.join('/Users/cristinareyes/Desktop/TFG/Resultados',
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

        #Ejecutamos el contador de personas con la red escogida:
        self.net = CrowdCounter(cfg.GPU_ID, self.net_name)  #.cuda()

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

    
        #NO USAREMOS GCC EN PRINCIPIO:
        '''
        if cfg.PRE_GCC:
            self.net.load_state_dict(torch.load(cfg.PRE_GCC_MODEL))

        '''

        #Qué hace esto exactamente?
        self.train_loader, self.val_loader, self.test_loader, self.restore_transform = dataloader()

        #Si se ha interrumpido el training, se intenta retomar desde donde se dejó:
        #¿SI SALIMOS DE LA CLASE TESTER, SE PODRÁ RETOMAR EL TRAINING LUEGO?
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


    #¿Por qué no usa directamente test_V1????????????????????????????????????????
    def forward(self):

        self.test_V1()


    def test_V1(self):  # test_v1 for SHHA, SHHB, UCF-QNRF, UCF50, AC

        #Para apagar partes que no interesan durante el test (como Dropout Layers o BatchNorm Layers):
        #¿PERO PARA QUÉ EXACTAMENTE EN NUESTRO CASO?:
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
            #¿PARA QUÉ????¿PARA NO APRENDER DEL TEST??
            with torch.no_grad():
                #¿QUÉ HACE VARIABLE()????????????????????
                img = Variable(img) #.cuda()
                gt_map = Variable(gt_map) #.cuda()
                audio_img = Variable(audio_img) #.cuda()

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

                #Guardamos resultados:
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

         ############################################################################# ME HE QUEDADO POR AQUÍ --> ¡Lo de abajo es copiado!

def loading_data():
    mean_std = cfg_data.MEAN_STD
    log_para = cfg_data.LOG_PARA
    factor = cfg_data.LABEL_FACTOR
    train_main_transform = own_transforms.Compose([
        own_transforms.RandomHorizontallyFlip()
    ])
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    gt_transform = standard_transforms.Compose([
        own_transforms.GTScaleDown(factor),
        own_transforms.LabelNormalize(log_para)
    ])
    restore_transform = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

    if cfg_data.IS_CROSS_SCENE:
        train_set = AC(img_path=cfg_data.IMAGE_PATH, den_path=cfg_data.DENSITY_PATH + '/cross_scene_train',
                       aud_path=cfg_data.AUDIO_PATH,
                       mode='train', main_transform=train_main_transform, img_transform=img_transform,
                       gt_transform=gt_transform, is_noise=cfg_data.IS_NOISE, brightness_decay=cfg_data.BRIGHTNESS,
                       noise_sigma=cfg_data.NOISE_SIGMA, longest_side=cfg_data.LONGEST_SIDE
                       )
    else:
        train_set = AC(img_path=cfg_data.IMAGE_PATH, den_path=cfg_data.DENSITY_PATH + '/train',
                       aud_path=cfg_data.AUDIO_PATH,
                       mode='train', main_transform=train_main_transform, img_transform=img_transform,
                       gt_transform=gt_transform, is_noise=cfg_data.IS_NOISE, brightness_decay=cfg_data.BRIGHTNESS,
                       noise_sigma=cfg_data.NOISE_SIGMA, longest_side=cfg_data.LONGEST_SIDE,
                       black_area_ratio=cfg_data.BLACK_AREA_RATIO, is_random=cfg_data.IS_RANDOM, is_denoise=cfg_data.IS_DENOISE
                       )
    train_loader = None
    if cfg_data.TRAIN_BATCH_SIZE == 1:
        train_loader = DataLoader(train_set, batch_size=1, num_workers=8, shuffle=True, drop_last=True)
    elif cfg_data.TRAIN_BATCH_SIZE > 1:
        train_loader = DataLoader(train_set, batch_size=cfg_data.TRAIN_BATCH_SIZE, num_workers=8,
                                  collate_fn=AC_collate, shuffle=True, drop_last=True)

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
    val_loader = DataLoader(val_set, batch_size=cfg_data.VAL_BATCH_SIZE, num_workers=1, shuffle=False, drop_last=False)

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
    test_loader = DataLoader(test_set, batch_size=cfg_data.VAL_BATCH_SIZE, num_workers=1, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, restore_transform


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


def Test():
  
  
    #------------prepare enviroment------------
    #¿QUÉ HACE ESTO????
    '''
    seed = cfg.SEED
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.manual_seed(seed) #cuda

    cfg.GPU_ID = [0,1]
    gpus = cfg.GPU_ID
    if len(gpus)==1:
        torch.cuda.set_device(gpus[0])

    torch.backends.cudnn.benchmark = True
    '''

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


    #¿Qué muestra esto?
    print(cfg, cfg_data)

    #------------Prepare Trainer------------
    net = cfg.NET


    #------------Start Training------------
    pwd = os.path.split(os.path.realpath(__file__))[0]
    cc_trainer = Tester(loading_data, cfg_data, pwd)
    cc_trainer.forward()

