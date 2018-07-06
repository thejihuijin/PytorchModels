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
from net_utils import train_GANs, preprocess, train_net, EllipseDataset, EllipseTransformPair


# Prepare data
print('Preparing Data')
pathtodata = '../data/RandomLineEllipses15.hdf5'
dataset_size = 200
batch_size = 38 

f = h5py.File(pathtodata,'r')
fakeinput1 = f['ellip/training_data'][0:]
fakesynthinput = f['synthetic/training_data'][0:]
fakeinput = np.concatenate((fakeinput1,fakesynthinput))

fakelabels1 = f['ellip/training_labels'][0:]
fakesynthlabels = f['synthetic/training_labels'][0:]
fakelabels = np.concatenate((fakelabels1,fakesynthlabels))

#del fakeinput1, fakeinput2, fakelabels1, fakelabels2, reallabels1, reallabels2, fakesynthlabels, realsynthlabels
f.close()

faketrainset = EllipseDataset(fakeinput,fakelabels,transform=EllipseTransformPair())
#realset = TensorDataset(reallabels)

faketrainloader = DataLoader(faketrainset,batch_size=batch_size,shuffle=True)

# Define Net
print('Creating Models')
#G = Generator()
G = UnetGenerator(input_nc=1, output_nc=1, num_downs=8, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=True)

torch.cuda.empty_cache()
num_epochs = 250

#checkpoint = './weights/2018-06-14_13-07-25/epoch_250.weights'
#checkpoint = torch.load(checkpoint)
#G.load_state_dict(checkpoint)

GPU = torch.cuda.is_available()
if GPU:
    G = G.cuda()

print('Beginning Training')
train_net(G,faketrainloader,num_epochs=num_epochs,save_epoch=25,weight_decay=1e-5,GPU=GPU)
