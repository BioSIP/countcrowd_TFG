#Bibliotecas
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Creacion del Modelo de Deep Learning
#3 capas convolucionales y 1 capa densa (la capa densa tendrá 25 neuronas)

#definimos la clase Encoder

def crop_img(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]


class UNET(nn.Module):
    # https://lmb.informatik.uni-freiburg.de/research/funded_projects/bioss_deeplearning/unet.png
    # torch.concat / cat()
    # nn.Conv2d, conv2dtranspoese
    def __init__(self, n_neuronas):
        super(UNET, self).__init__()

        #ENCODER v1
        '''
        self.conv_1 = double_conv(1,8)
        self.conv_2 = double_conv(8, 16)
        self.conv_3 = double_conv(16, 32)
        self.mp_1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        #DECODER V1
        self.up_conv5 = nn.ConvTranspose2d(32, 16, kernel_size = 2, stride = 2)
        self.up_conv6 = double_conv(32, 16)
        self.up_conv7 = nn.ConvTranspose2d(16, 8, kernel_size = 2, stride = 2)
        self.up_conv8 = double_conv(16, 8)

        self.out = nn.Conv2d(8, 2, kernel_size = 1)

        '''

        #?--------------------------------------------------------------------------------------------------------
        
        #ENCODER
        self.conv1 =  nn.Sequential(nn.Conv2d(1, 8, (1, 11), padding = (0, 5)), 
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(8, 8, (1, 11), padding = (0, 5)),
                                    nn.ReLU(inplace=True))
        self.mp1 = nn.MaxPool2d((1, 2))

        self.conv2 =  nn.Sequential(nn.Conv2d(8, 16, (1, 7), padding = (0, 3)), 
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(16, 16, (1, 7), padding = (0, 3)),
                                    nn.ReLU(inplace=True))
        self.mp2 = nn.MaxPool2d((1, 2)) 

        self.conv3 =  nn.Sequential(nn.Conv2d(16, 32, (1, 5), padding = (0, 2)), 
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 32, (1, 5), padding = (0, 2)),
                                    nn.ReLU(inplace=True))
          
        
        #DECODER
        self.up_conv5 = nn.ConvTranspose2d(32, 16, (1,5), stride=(1,2), padding=(0,2), output_padding=(0,1))
        
        self.up_conv6 = nn.Sequential(nn.Conv2d(32, 16, (1, 5), padding = (0, 2)),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(16, 16, (1, 5), padding = (0, 2)), 
                                      nn.ReLU(inplace=True))

        self.up_conv7 = nn.ConvTranspose2d(16, 8, (1, 7), stride=(1,2), padding=(0,3), output_padding=(0,1))
        
        self.up_conv8 = nn.Sequential(nn.Conv2d(16, 8, (1, 7), padding = (0, 3)),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(8, 8, (1, 7), padding = (0, 3)), 
                                      nn.ReLU(inplace=True))

        self.out = nn.Conv2d(8, 2, (1, 11), padding = (0, 5))
        

    def forward(self, x):
        #ENCODER V1
        '''
        x1 = self.conv_1(x) #x1 se concatena
        x2 = self.mp_1(x1)

        x3 = self.conv_2(x2) #x3 se concatena
        x4 = self.mp_1(x3)

        x5 = self.conv_3(x4)

        #DECODER V1
        x6 = self.up_conv5(x5)
        y1 = crop_img(x3, x6)
        x7 = self.up_conv6(torch.cat([x6, y1], 1))
       

        x8 = self.up_conv7(x7)
        y2 = crop_img(x1, x8)
        x9 = self.up_conv8(torch.cat([x8, y2], 1))
        print(x9.size())

        x = self.out(x9)
        print(x.size())

        '''

        #?--------------------------------------------------------------------------------------------------------
        
        #ENCODER
        x1 = self.conv1(x)
        #print(x1.size())
        x2 = self.mp1(x1)

        x3 = self.conv2(x2)
        x4 = self.mp2(x3)

        x5 = self.conv3(x4)
        #print(x5.size())

        #DECODER
        x6 = self.up_conv5(x5)
        y1 = crop_img(x3, x6)
        x7 = self.up_conv6(torch.cat([x6, y1], 1))

        x8 = self.up_conv7(x7)
        y2 = crop_img(x1, x8)
        x9 = self.up_conv8(torch.cat([x8, y2], 1))
        #print(x9.size())

        x = self.out(x9)
        print(x.size())
        
        return x

#Pruebas
if __name__ == "__main__":
    #image = torch.rand((8618, 1, 1, 512))
    image = torch.rand((1, 1, 572, 572))
    print(image.size())
    model = UNET(25)
    print(model(image))

