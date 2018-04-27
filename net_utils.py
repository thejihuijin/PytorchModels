from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import datetime
import os
import time

def get_datetime():
    time = datetime.datetime.today()
    out = str(time.date())+'_'+str(time.hour).zfill(2)
    out = out+'-'+str(time.minute).zfill(2)
    out = out+'-'+str(time.second).zfill(2)
    return out

def train_net(net, trainloader, num_epochs, GPU=False, 
              weightpath='./weights/',save_epoch=50,
              lr=0.01,momentum=0.99,saveweights=True):
    # Create output directory
    weightpath = os.path.join(weightpath,get_datetime())
    os.makedirs(weightpath)
    logpath = os.path.join(weightpath,'log.txt')
    with open(logpath, "wt") as text_file:
        print('Epoch\tLoss\tEpoch Time\tTotal Time',file=text_file)

    num_data = len(trainloader)*trainloader.batch_size
    
    # Accumulate Log text
    logtxt = ''
    
    # Determine Minibatch size
    minibatch=max(1,int(len(trainloader)/10))
    
    # Define Loss Function/Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(),lr=lr,momentum=momentum)
    trainstart = time.time()
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        epoch_loss = 0.0
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
            epoch_loss += loss.data[0]
            if i % minibatch == 0:    # print every 2000 mini-batches
                print('\t[%d, %5d] loss: %.3f, %.3f seconds elapsed' %
                      (epoch + 1, i + 1, running_loss / minibatch, time.time() - epochstart ))
                running_loss = 0.0
        epochend = time.time()
        print('Epoch %d Training Time: %.3f seconds\nTotal Elapsed Time: %.3f seconds' %
               (epoch+1, epochend-epochstart,epochend-trainstart))
        
        # write loss to logfile
        #with open(logpath, "at") as text_file:
        #    print('%i\t%f\t%f\t%f\n' % 
        #        (epoch+1,float(epoch_loss)/num_data,epochend-epochstart,epochend-trainstart)
        #         ,file=text_file)
        logtxt += '%i\t%f\t%f\t%f\n' % (epoch+1,float(epoch_loss)/num_data,epochend-epochstart,epochend-trainstart)
        epoch_loss=0.0

        
        # Save weights
        if (epoch % save_epoch == 0 or epoch == num_epochs-1):
            if saveweights:
                outpath = os.path.join(weightpath,'epoch_'+str(epoch+1)+'.weights')
                net = net.cpu()
                torch.save(net.state_dict(),outpath)
                if GPU:
                    net = net.cuda()
            
            # write loss to logfile
            with open(logpath, "at") as text_file:
                print(logtxt[:-2],file=text_file)
                logtxt = ''

    print('Finished Training')

