import numpy as np
import matplotlib

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import os
import datetime
import time

class FBPConvNet(nn.Module):
    def __init__(self):
        super(FBPConvNet, self).__init__()
        # First level, assumed 256x256
        self.conv1_1 = nn.Conv2d(1,64,3,padding=1)
        self.conv1_2 = nn.Conv2d(64,64,3,padding=1)
        self.batch1 = nn.BatchNorm2d(64)

        # Second level, assumed 64 channels of 128x128
        self.conv2_1 = nn.Conv2d(64,128,3,padding=1)
        self.conv2_2 = nn.Conv2d(128,128,3,padding=1)
        self.batch2 = nn.BatchNorm2d(128)
        
        # Third level, assumed 128 channels of 64 x 64 input
        self.conv3_1 = nn.Conv2d(128,256,3,padding=1)
        self.conv3_2 = nn.Conv2d(256,256,3,padding=1)
        self.batch3 = nn.BatchNorm2d(256)
        
        # Fourth level, assumed 256 channels of 32 x 32 input
        self.conv4_1 = nn.Conv2d(256,512,3,padding=1)
        self.conv4_2 = nn.Conv2d(512,512,3,padding=1)
        self.batch4 = nn.BatchNorm2d(512)
        
        #############################################
        # Fifth level, up-conv to 256 channels of 64 x 64
        self.deconv5 = nn.ConvTranspose2d(512,256,3,padding=1,stride=2,output_padding=1)
        self.conv5_1 = nn.Conv2d(512,256,3,padding=1)
        self.conv5_2 = nn.Conv2d(256,256,3,padding=1)
        
        # Sixth level, up-conv to 128 channels of 128x128
        self.deconv6 = nn.ConvTranspose2d(256, 128, 3, padding=1,stride=2,output_padding=1)
        self.conv6_1 = nn.Conv2d(256,128,3,padding=1)
        self.conv6_2 = nn.Conv2d(128,128,3,padding=1)
        
        # Seventh level, up-conv to 64 channels of 256x256
        self.deconv7 = nn.ConvTranspose2d(128,64,3,padding=1,stride=2,output_padding=1)
        self.conv7_1 = nn.Conv2d(128,64,3,padding=1)
        self.conv7_2 = nn.Conv2d(64,64,3,padding=1)
        
        # Eigth level, 1x1 convolution
        self.conv8 = nn.Conv2d(64,1,1)
        
        self.maxpool = nn.MaxPool2d(2)
        self.elu = nn.ELU()
        
        
    def forward(self,x):
        x1_1 = self.batch1(self.elu(self.conv1_1(x)))
        x1_2 = self.batch1(self.elu(self.conv1_2(x1_1)))
        x1_3 = self.batch1(self.elu(self.conv1_2(x1_2)))
        x1 = self.maxpool(x1_3)
        
        x2_1 = self.batch2(self.elu(self.conv2_1(x1)))
        x2_2 = self.batch2(self.elu(self.conv2_2(x2_1)))
        x2 = self.maxpool(x2_2)
        
        x3_1 = self.batch3(self.elu(self.conv3_1(x2)))
        x3_2 = self.batch3(self.elu(self.conv3_2(x3_1)))
        x3 = self.maxpool(x3_2)
        
        x4_1 = self.batch4(self.elu(self.conv4_1(x3)))
        x4_2 = self.batch4(self.elu(self.conv4_2(x4_1)))

        x5_1 = self.deconv5(x4_2)
        x5_2 = torch.cat((x3_2,x5_1),1)
        x5_3 = self.batch3(self.elu(self.conv5_1(x5_2)))
        x5 = self.batch3(self.elu(self.conv5_2(x5_3)))
        
        x6_1 = self.deconv6(x5)
        x6_2 = torch.cat((x2_2,x6_1),1)
        x6_3 = self.batch2(self.elu(self.conv6_1(x6_2)))
        x6 = self.batch2(self.elu(self.conv6_2(x6_3)))
        
        x7_1 = self.deconv7(x6)
        x7_2 = torch.cat((x1_3,x7_1),1)
        x7_3 = self.batch1(self.elu(self.conv7_1(x7_2)))
        x7 = self.batch1(self.elu(self.conv7_2(x7_3)))
        
        x8 = self.conv8(x7)
        y = x8 + x
        
        return y

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # First level, assumed 256x256
        self.conv1_1 = nn.Conv2d(1,64,3,padding=1)
        self.batch1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64,64,3,padding=1)
        self.batch1_2 = nn.BatchNorm2d(64)
        self.conv1_3 = nn.Conv2d(64,64,3,padding=1,stride=2)
        self.batch1_3 = nn.BatchNorm2d(64)

        # Second level, assumed 64 channels of 128x128
        self.conv2_1 = nn.Conv2d(64,128,3,padding=1)
        self.batch2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128,128,3,padding=1,stride=2)
        self.batch2_2 = nn.BatchNorm2d(128)
        
        # Third level, assumed 128 channels of 64 x 64 input
        self.conv3_1 = nn.Conv2d(128,256,3,padding=1)
        self.batch3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256,256,3,padding=1,stride=2)
        self.batch3_2 = nn.BatchNorm2d(256)
        
        # Fourth level, assumed 256 channels of 32 x 32 input
        self.conv4_1 = nn.Conv2d(256,512,3,padding=1)
        self.batch4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512,512,3,padding=1)
        self.batch4_2 = nn.BatchNorm2d(512)
        
        #############################################
        # Fifth level, up-conv to 256 channels of 64 x 64
        self.deconv5 = nn.ConvTranspose2d(512,256,3,padding=1,stride=2,output_padding=1)
        self.conv5_1 = nn.Conv2d(512,256,3,padding=1)
        self.batch5_1 = nn.BatchNorm2d(256)
        self.conv5_2 = nn.Conv2d(256,256,3,padding=1)
        self.batch5_2 = nn.BatchNorm2d(256)
        
        # Sixth level, up-conv to 128 channels of 128x128
        self.deconv6 = nn.ConvTranspose2d(256, 128, 3, padding=1,stride=2,output_padding=1)
        self.conv6_1 = nn.Conv2d(256,128,3,padding=1)
        self.batch6_1 = nn.BatchNorm2d(128)
        self.conv6_2 = nn.Conv2d(128,128,3,padding=1)
        self.batch6_2 = nn.BatchNorm2d(128)
        
        # Seventh level, up-conv to 64 channels of 256x256
        self.deconv7 = nn.ConvTranspose2d(128,64,3,padding=1,stride=2,output_padding=1)
        self.conv7_1 = nn.Conv2d(128,64,3,padding=1)
        self.batch7_1 = nn.BatchNorm2d(64)
        self.conv7_2 = nn.Conv2d(64,64,3,padding=1)
        self.batch7_2 = nn.BatchNorm2d(64)
        
        # Eigth level, 1x1 convolution
        self.conv8 = nn.Conv2d(64,1,1)
        
        self.maxpool = nn.MaxPool2d(2)
        self.elu = nn.ELU()
        
        
    def forward(self,x):
        x1_1 = self.batch1_1(self.elu(self.conv1_1(x)))
        x1_2 = self.batch1_2(self.elu(self.conv1_2(x1_1)))
        x1 = self.batch1_3(self.elu(self.conv1_3(x1_2)))
        
        x2_1 = self.batch2_1(self.elu(self.conv2_1(x1)))
        x2 = self.batch2_2(self.elu(self.conv2_2(x2_1)))
        
        x3_1 = self.batch3_1(self.elu(self.conv3_1(x2)))
        x3 = self.batch3_2(self.elu(self.conv3_2(x3_1)))
        
        x4_1 = self.batch4_1(self.elu(self.conv4_1(x3)))
        x4_2 = self.batch4_2(self.elu(self.conv4_2(x4_1)))

        x5_1 = self.deconv5(x4_2)
        x5_2 = torch.cat((x3_1,x5_1),1)
        x5_3 = self.batch5_1(self.elu(self.conv5_1(x5_2)))
        x5 = self.batch5_2(self.elu(self.conv5_2(x5_3)))
        
        x6_1 = self.deconv6(x5)
        x6_2 = torch.cat((x2_1,x6_1),1)
        x6_3 = self.batch6_1(self.elu(self.conv6_1(x6_2)))
        x6 = self.batch6_2(self.elu(self.conv6_2(x6_3)))
        
        x7_1 = self.deconv7(x6)
        x7_2 = torch.cat((x1_2,x7_1),1)
        x7_3 = self.batch7_1(self.elu(self.conv7_1(x7_2)))
        x7 = self.batch7_2(self.elu(self.conv7_2(x7_3)))
        
        x8 = self.conv8(x7)
        y = x8 + x
        
        return y



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Input 1 x 256 x 256 -> 64 x 256 x 256
        self.conv1 = nn.Conv2d(1,64,3,padding=1,stride=1)    
        self.batch1 = nn.BatchNorm2d(64)
        
        # -> 128 x 128 x 128
        self.conv2 = nn.Conv2d(64,128,7,padding=3,stride=2)
        self.batch2 = nn.BatchNorm2d(128)

        # -> 256 x 32 x 32 -> 256 x 22 x 22
        self.conv3 = nn.Conv2d(128,256,5,padding=2,stride=2)
        self.batch3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256,256,7,padding=3,stride=3)
        self.batch4 = nn.BatchNorm2d(256)
        
        # -> 512 x 8 x 8
        self.conv5 = nn.Conv2d(256,512,5,padding=2,stride=3)
        self.batch5 = nn.BatchNorm2d(512)
        
        # -> 1024 x 1 x 1
        self.conv6 = nn.Conv2d(512,1024,5,padding=2,stride=3)
        self.batch6 = nn.BatchNorm2d(1024)
        self.conv7 = nn.Conv2d(1024,1024,3,padding=1,stride=3)
        #self.batch7 = nn.BatchNorm2d(1024)
        
        # Decision layers
        self.conv8 = nn.Conv2d(1024,1024,1)
        self.conv9 = nn.Conv2d(1024,1024,1)
        self.conv10 = nn.Conv2d(1024,1,1)
        
        # Non-Linear Activations
        self.leaky = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv2_drop = nn.Dropout2d(p=.2)
        
        
        
    def forward(self,x):
        # Generate Features
        x = self.leaky(self.batch1(self.conv1(x)))
        x = self.leaky(self.batch2(self.conv2(x)))
        x = self.leaky(self.batch3(self.conv3(x)))
        x = self.leaky(self.batch4(self.conv4(x)))
        x = self.leaky(self.batch5(self.conv5(x)))
        x = self.leaky(self.batch6(self.conv6(x)))
        x = self.leaky(self.conv7(x))
        
        # Decision Layers
        x = self.leaky(self.conv2_drop(self.conv8(x)))
        x = self.leaky(self.conv2_drop(self.conv9(x)))
        x = self.sigmoid(self.conv10(x))
        return x

class Discriminator2(nn.Module):
    def __init__(self):
        super(Discriminator2, self).__init__()
        # Input 1 x 256 x 256 -> 64 x 128 x 128
        self.conv1 = nn.Conv2d(1,64,3,padding=1,stride=1)    
        self.batch1 = nn.BatchNorm2d(64)
        
        # -> 128 x 64 x 64
        self.conv2 = nn.Conv2d(64,128,7,padding=3,stride=2)
        self.batch2 = nn.BatchNorm2d(128)

        # -> 256 x 32 x 32 -> 256 x 16 x 16
        self.conv3 = nn.Conv2d(128,256,5,padding=2,stride=2)
        self.batch3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256,256,7,padding=3,stride=2)
        self.batch4 = nn.BatchNorm2d(256)
        
        # -> 512 x 8 x 8
        self.conv5 = nn.Conv2d(256,512,5,padding=2,stride=2)
        self.batch5 = nn.BatchNorm2d(512)
        
        # -> 1024 x 1 x 1
        self.conv6 = nn.Conv2d(512,1024,5,padding=2,stride=2)
        self.batch6 = nn.BatchNorm2d(1024)
        self.conv7 = nn.Conv2d(1024,1024,3,padding=1,stride=2)
        self.batch7 = nn.BatchNorm2d(1024)
        
        self.conv8 = nn.Conv2d(1024,1024,3,padding=1,stride=2)
        self.batch8 = nn.BatchNorm2d(1024)
        
        self.conv9 = nn.Conv2d(1024,1024,3,padding=1,stride=2)
        
        # Decision layers
        self.d_conv1 = nn.Conv2d(1024,1024,1)
#         self.d_conv2 = nn.Conv2d(1024,1024,1)
        self.d_conv3 = nn.Conv2d(1024,1,1)
        
        # Non-Linear Activations
        self.leaky = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv2_drop = nn.Dropout2d(p=.2)
        
        
        
    def forward(self,x):
        # Generate Features
        x = self.leaky(self.conv1(x))
        x = self.leaky(self.batch2(self.conv2(x)))
        x = self.leaky(self.batch3(self.conv3(x)))
        x = self.leaky(self.batch4(self.conv4(x)))
        x = self.leaky(self.batch5(self.conv5(x)))
        x = self.leaky(self.batch6(self.conv6(x)))
        x = self.leaky(self.batch7(self.conv7(x)))
        x = self.leaky(self.batch8(self.conv8(x)))
        x = self.leaky(self.conv9(x))
        
        # Decision Layers
        x = self.leaky(self.conv2_drop(self.d_conv1(x)))
        x = self.sigmoid(self.d_conv3(x))
        return x
