import numpy as np
import matplotlib.pyplot as plt

import h5py

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

import sys
# Import FBP
sys.path.append('../FBPConvNet/')
#from FBPConvNet import Generator,DiscriminativeNet, Discriminator2
from pix2pixModels import NLayerDiscriminator, UnetGenerator, FacadeDataset

sys.path.append('../')
from net_utils import train_facade 


batch_size = 40
# Prepare data
print('Preparing Data')
pathtodata = '../data/facades/train/'
dataset = FacadeDataset(pathtodata)
trainloader = DataLoader(dataset, batch_size=batch_size,shuffle=True)

# Define Net
print('Creating Models')
D = NLayerDiscriminator(1,n_layers=4,use_sigmoid=False)

#checkpoint = torch.load('./D_init.weights')
#D.load_state_dict(checkpoint)
#G = Generator()
G = UnetGenerator(input_nc=1, output_nc=1, num_downs=8, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=True)

torch.cuda.empty_cache()
num_epochs = 200

GPU = torch.cuda.is_available()
if GPU:
    D = D.cuda()
    G = G.cuda()

print('Beginning Training')
d_losses, g_losses = train_facade(G,D,trainloader,num_epochs=num_epochs,save_epoch=20,GPU=GPU,saveweights=False)
