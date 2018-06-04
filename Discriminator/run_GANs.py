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
from FBPConvNet import FBPConvNet, Discriminator

sys.path.append('../')
from net_utils import train_GANs, preprocess


# Prepare data
print('Preparing Data')
pathtodata = '../EllipseGeneration/RandomLineEllipses15.hdf5'
dataset_size = 200
batch_size = 1

f = h5py.File(pathtodata,'r')
fakeinput = preprocess(f['ellip/training_data'][0:dataset_size])
fakelabels = preprocess(f['ellip/training_labels'][0:dataset_size])
reallabels = preprocess(f['ellip/training_labels'][dataset_size:2*dataset_size])
f.close()

faketrainset = TensorDataset(fakeinput,fakelabels)
#realset = TensorDataset(reallabels)

faketrainloader = DataLoader(faketrainset,batch_size=batch_size,shuffle=True)
realtrainloader = DataLoader(reallabels, batch_size=batch_size,shuffle=True)

# Define Net
print('Creating Models')
D = Discriminator()
G = FBPConvNet()

torch.cuda.empty_cache()
num_epochs = 500

GPU = torch.cuda.is_available()
if GPU:
    D = D.cuda()
    G = G.cuda()

print('Beginning Training')
d_losses, g_losses = train_GANs(G,D,faketrainloader,realtrainloader,num_epochs=num_epochs,GPU=GPU)
