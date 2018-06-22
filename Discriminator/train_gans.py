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
from pix2pixModels import NLayerDiscriminator, UnetGenerator

sys.path.append('../')
from net_utils import train_GANs, preprocess, EllipseDataset, RealDataset, EllipseTransformPair


# Prepare data
print('Preparing Data')
pathtodata = '../data/NormRandomLineEllipses15.hdf5'
dataset_size = 200
batch_size = 50 

f = h5py.File(pathtodata,'r')
fakeinput1 = f['ellip/training_data'][0:dataset_size]
fakesynthinput = f['synthetic/training_data'][0:2300]
fakeinput = np.concatenate((fakeinput1,fakesynthinput))

fakelabels1 = f['ellip/training_labels'][0:dataset_size]
fakesynthlabels = f['synthetic/training_labels'][0:2300]
fakelabels = np.concatenate((fakelabels1,fakesynthlabels))

reallabels1 = f['ellip/training_labels'][dataset_size:2*dataset_size]
realsynthlabels = f['synthetic/training_labels'][2300:]
reallabels = np.concatenate((reallabels1,realsynthlabels))

f.close()

faketrainset = EllipseDataset(fakeinput,fakelabels,transform=EllipseTransformPair())
realtrainset = RealDataset(reallabels)

faketrainloader = DataLoader(faketrainset,batch_size=batch_size,shuffle=True)
realtrainloader = DataLoader(realtrainset, batch_size=batch_size,shuffle=True)

# Define Net
print('Creating Models')
D = NLayerDiscriminator(1,n_layers=4,use_sigmoid=False)

checkpoint = torch.load('./D_init.weights')
D.load_state_dict(checkpoint)
#G = Generator()
G = UnetGenerator(input_nc=1, output_nc=1, num_downs=8, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=True)

torch.cuda.empty_cache()
num_epochs = 20

GPU = torch.cuda.is_available()
if GPU:
    D = D.cuda()
    G = G.cuda()

print('Beginning Training')
d_losses, g_losses = train_GANs(G,D,faketrainloader,realtrainloader,num_epochs=num_epochs,save_epoch=2,GPU=GPU,saveweights=True)
