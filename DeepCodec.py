from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

import datetime
import os
import time


#
# DeepCodec Neural Net Model
#
# inits with downsampling factor r
#
class DeepCodec(nn.Module):
    def __init__(self,r):
        super(DeepCodec, self).__init__()
        self.r = r
        self.conv2 = nn.Conv1d(self.r,8,49,padding=24)
        self.conv3 = nn.Conv1d(8,4,49,padding=24)
        self.conv4 = nn.Conv1d(4,1,49,padding=24)
        self.conv5 = nn.Conv1d(1,4,49,padding=24)
        self.conv6 = nn.Conv1d(4,8,49,padding=24)
        self.conv7 = nn.Conv1d(8,self.r,49,padding=24)
        
    def forward(self,x):
        # Assumes x is num x N x 1
        # First convert to num x r x M
        num, N,_ = x.size()
        M = int(N/self.r)
        #print(x.size())
        y = x.view(num,self.r,M)
        #print(y.size())
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.conv5(y)
        y = self.conv6(y)
        y = self.conv7(y)
        z = y.view(num,N, 1)
        return z
    
def get_datetime():
    time = datetime.datetime.today()
    out = str(time.date())+'_'+str(time.hour).zfill(2)
    out = out+'-'+str(time.minute).zfill(2)
    out = out+'-'+str(time.second).zfill(2)
    return out
def rgbtogray(img):
    # assumes CxHxW
    grey = .2989*img[0] + .5870*img[1]+.1140*img[2]
    return grey
    
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if len(npimg.shape) == 2:
        plt.imshow(npimg,cmap='gray')
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def train_net(net, trainloader, num_epochs, GPU=False, 
              weightpath='./weights/',save_epoch=50,
              lr=0.001,momentum=0.9):
    # Create output directory
    weightpath = os.path.join(weightpath,get_datetime())
    os.makedirs(weightpath)
    
    # Define Loss Function/Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(),lr=lr,momentum=momentum)
    trainstart = time.time()
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        epochstart = time.time()
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            # wrap them in Variable
            if GPU:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('\t[%d, %5d] loss: %.3f, %.3f seconds elapsed' %
                      (epoch + 1, i + 1, running_loss / 2000, time.time() - epochstart ))
                running_loss = 0.0
        epochend = time.time()
        print('Epoch %d Training Time: %.3f seconds\nTotal Elapsed Time: %.3f seconds' %
               (epoch+1, epochend-epochstart,epochend-trainstart))
        # Save weights
        if epoch % save_epoch == 0 or epoch == num_epochs-1:
            outpath = os.path.join(weightpath,'epoch_'+str(epoch+1)+'.weights')
            torch.save(net.state_dict(),outpath)

    print('Finished Training')
