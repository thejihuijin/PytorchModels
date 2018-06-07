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
import numpy as np

def get_datetime():
    time = datetime.datetime.today()
    out = str(time.date())+'_'+str(time.hour).zfill(2)
    out = out+'-'+str(time.minute).zfill(2)
    out = out+'-'+str(time.second).zfill(2)
    return out
def preprocess(data):
    return torch.Tensor(data).unsqueeze(1)
def target_ones(N,GPU=False):
    if GPU:
        return torch.ones(N,1).cuda()
    else:
        return torch.ones(N,1)
def target_noisy_ones(N,GPU=False):
    # maps between .7 and 1.2
    labels = torch.rand((N,1))*.5+.7
    if GPU:
        return labels.cuda()
    return labels
def target_zeros(N,GPU=False):
    if GPU:
        return torch.zeros(N,1).cuda()
    else:
        return torch.zeros(N,1)  
def target_noisy_zeros(N,GPU=False):
    # maps between .7 and 1.2
    labels = torch.rand((N,1))*.3
    if GPU:
        return labels.cuda()
    return labels
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
            if (i+1) % minibatch == 0:    # print every 2000 mini-batches
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



def train_GANs(G, D, faketrainloader, realtrainloader, num_epochs=500, GPU=False,
              weightpath='./weights/',save_epoch=50,saveweights=True):
    # Create output directory
    weightpath = os.path.join(weightpath,get_datetime())
    os.makedirs(weightpath)
    logpath = os.path.join(weightpath,'log.txt')
    
    with open(logpath, "wt") as text_file:
        print('Epoch\tD Loss\tG Loss\tEpoch Time\tTotal Time',file=text_file)

    num_data = len(realtrainloader)*realtrainloader.batch_size 
    d_losses = np.zeros(num_epochs)
    g_losses = np.zeros(num_epochs)

    # Accumulate log text
    logtxt = ''
    
    # Determine minibatch size
    minibatch = max(1,int(len(realtrainloader))/10)
    
    # Define Loss Function/Optimizer
    bceloss = nn.BCELoss()
    mseloss = nn.MSELoss()

    d_optimizer = optim.Adam(D.parameters(), lr=0.0002)
    g_optimizer = optim.Adam(G.parameters(), lr=0.0002)

    
    G.train()
    trainstart = time.time()
    for epoch in range(num_epochs):
        # Collect loss information
        d_epoch_loss = 0.0
        g_epoch_loss = 0.0
        d_running_loss = 0.0
        g_running_loss = 0.0
        
        epochstart = time.time()

        fakeiter = iter(faketrainloader)
        realiter = iter(realtrainloader)
        Giter = iter(faketrainloader)
        for batch_index in range(len(realtrainloader)):
            ## prepare data
            truelabels = realiter.next()
            fakeinput, _ = fakeiter.next()
            batch_size = truelabels.size(0)


            if GPU:
                truelabels = truelabels.cuda()
                fakeinput = fakeinput.cuda()
    #             fakelabel = fakelabel.cuda()
            d_real_data = Variable(truelabels)
            d_gen_input = Variable(fakeinput)
            d_fake_data = G(d_gen_input).detach() # detach to avoid training G on these labels
            
            ## Train D
            d_optimizer.zero_grad()

            # Train D on real
            d_real_decision = D(d_real_data)
            d_real_error = bceloss(d_real_decision, Variable(target_noisy_ones(batch_size,GPU)))
            d_real_error.backward()

            # Train D on fake

            d_fake_decision = D(d_fake_data)
            d_fake_error = bceloss(d_fake_decision, Variable(target_noisy_zeros(batch_size,GPU))) 
            d_fake_error.backward()
            d_optimizer.step()
            d_loss = d_real_error+d_fake_error
            
            d_running_loss += d_loss.data[0]
            d_losses[epoch] += d_loss.data[0]
            
        
            ## Train G
            g_fake_input, g_fake_label = Giter.next()
            batch_size = g_fake_input.size(0)

            if GPU:
                g_fake_input = g_fake_input.cuda()
                g_fake_label = g_fake_label.cuda()

            gen_input = Variable(g_fake_input)
            g_fake_data = G(gen_input)
  
            g_optimizer.zero_grad()

            dg_fake_decision = D(g_fake_data)
            g_loss = (10**-3)*bceloss(dg_fake_decision, Variable(target_noisy_ones(batch_size,GPU)))
            g_loss +=  mseloss(g_fake_data,Variable(g_fake_label))

            g_loss.backward()
            g_optimizer.step()
            
            g_running_loss += g_loss.data[0]
            g_losses[epoch] += g_loss.data[0]
            
            # print statistics
            if (batch_index+1) % minibatch == 0:
                print('\t[%d, %5d] D loss: %.3f, G loss: %.3f, %.3f seconds elapsed' %
                      (epoch + 1, batch_index + 1, d_running_loss / minibatch, 
                       g_running_loss/minibatch, time.time() - epochstart))
                d_running_loss = 0.0
                g_running_loss = 0.0
        # Record epoch statistics
        epochend = time.time()        
        print('Epoch %d Training Time: %.3f seconds\nTotal Elapsed Time: %.3f seconds' %
               (epoch+1, epochend-epochstart,epochend-trainstart))
        
        # log losses
        d_losses[epoch] /= num_data
        g_losses[epoch] /= num_data
        logtxt += '%i\t%f\t%f\t%f\t%f\n' % (epoch+1,d_losses[epoch], g_losses[epoch],
                                           epochend-epochstart,epochend-trainstart)

        
        # Save weights
        if (epoch % save_epoch == 0 or epoch == num_epochs-1):
            if saveweights:
                d_outpath = os.path.join(weightpath,'D_epoch_'+str(epoch+1)+'.weights')
                g_outpath = os.path.join(weightpath,'G_epoch_'+str(epoch+1)+'.weights')
                D = D.cpu()
                G = G.cpu()
                torch.save(D.state_dict(),d_outpath)
                torch.save(G.state_dict(),g_outpath)

                if GPU:
                    D = D.cuda()
                    G = G.cuda()
            
            # write loss to logfile
            with open(logpath, "at") as text_file:
                print(logtxt[:-2],file=text_file)
                logtxt = ''

    print('Finished Training')
    return d_losses,g_losses
