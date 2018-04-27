import numpy as np
import matplotlib.pyplot as plt

import h5py
import argparse

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader


# ## Import Model
from FBPConvNet import FBPConvNet


# ## Import train_net
import sys
sys.path.append('../')
from net_utils import train_net

def main(mode,batch_size):
    # # Load Data
    # set path to data
    pathtodata = '../data/LineEllipses20.hdf5'
    #batch_size = 4

    f = h5py.File(pathtodata,'r')
    trainingset = f['training']
    training_data = f['training/training_data']
    training_labels = f['training/training_labels']
    if mode == 'train':
        trainset = TensorDataset(torch.Tensor(training_data[:5000]).unsqueeze(1),
                                torch.Tensor(training_labels[:5000]).unsqueeze(1))
        trainloader = DataLoader(trainset, batch_size=batch_size,shuffle=True)

    testset = TensorDataset(torch.Tensor(training_data[5000:8000]).unsqueeze(1),
                            torch.Tensor(training_labels[5000:8000]).unsqueeze(1))
    testloader = DataLoader(testset, batch_size=4,shuffle=False)

    f.close()

    GPU = torch.cuda.is_available()
    print('Using GPU:',GPU)
    net = FBPConvNet()
    if GPU:
        net = net.cuda()
    if mode == 'train':
    # # Train Net

        num_epochs = 500

        # save path for trained models
        weightpath = './weights/'

        # save every x epochs
        save_epoch = 50

        train_net(net,trainloader,num_epochs,GPU,weightpath=weightpath,
                 save_epoch=save_epoch,lr=0.01,momentum=0.99,saveweights=True)
    elif mode == 'test':


    # # Test Net

        weightspath = '../epoch_500.weights'
        checkpoint = torch.load(weightspath)
        net.load_state_dict(checkpoint)
    else:
        print('Mode value not train or test')
        exit(1)


    for data in testloader:
        imgs,labels = data
        num = imgs.size(0)
        if GPU:
            est_imgs = net.forward(Variable(imgs.cuda()))
            est_imgs = est_imgs.cpu().data
        else:
            est_imgs = net.forward(Variable(imgs))

        plt.figure()
        for i in range(num):
            mse =torch.mean((imgs[i,0,:,:]-labels[i,0,:,:])**2)
            plt.subplot(num,3,3*i+1)
            plt.imshow(imgs[i,0,:,:].numpy())
            plt.title('Noisy %i, MSE=%.4f' % (i,mse))
            plt.axis('off')
            
            mse =torch.mean((est_imgs[i,0,:,:]-labels[i,0,:,:])**2)
            
            plt.subplot(num,3,3*i+2)
            plt.imshow(est_imgs[i,0,:,:].numpy())
            plt.title('Pred %i, MSE = %.4f' % (i, mse))
            plt.axis('off')
            
            plt.subplot(num,3,3*i+3)
            plt.imshow(labels[i,0,:,:].numpy())
            plt.title('Ground Truth %i' % (i))
            plt.axis('off')
        plt.show()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--mode', default='test', 
            help='train to train a net from scratch. test to load trained net')
    parser.add_argument('-bs','--batch_size',type=int, default=4,
            help='Set batch size for training. Defaults to 4')

    args = parser.parse_args()
    # check args
    main(args.mode,args.batch_size)
