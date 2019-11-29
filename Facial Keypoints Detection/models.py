## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        self.conv1 = nn.Conv2d(1, 32, 5) # (W-F)/S + 1 = 224-5/1 + 1 =220 , OUTPUT_SIZE = 32,220,220
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.pool1 = nn.MaxPool2d(2,2) # OUTPUT 32,110,110
        
        self.conv2 = nn.Conv2d(32, 64, 3) # (W-F)/S + 1 = 110-3/1 + 1 = 108, OUTPUT_SIZE = 64,108,108
        self.pool2 = nn.MaxPool2d(2,2)     #OUTPUT_SIZE = 64,54,54
        self.conv3 = nn.Conv2d(64, 128,3) #(W-F)/S + 1 = 54-3 / 1 + 1 = 52, OUTPUT_SIZE = 128,52,52
        self.pool3 = nn.MaxPool2d(2,2) #OUTPUT_SIZE = 128,26,26
        self.conv4 = nn.Conv2d(128,256,3) ##(W-F)/S + 1 = 26-3 / 1 + 1 = 24, OUTPUT_SIZE = 256,24,24
        self.pool4 = nn.MaxPool2d(2,2) #OUTPUT_SIZE = 256,12,12
        
        # 128 outputs *  26*26 filtered/pooled map size
        self.fc1 = nn.Linear(256*12*12, 4000)  #36864
        self.fc2 = nn.Linear(4000, 500)
        #self.fc3 = nn.Linear(2000, 1000)
        #self.fc4 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500,136)
        self.drop = nn.Dropout(p=0.4)
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool1(F.selu(self.conv1(x)))
        x = self.drop(self.pool2(F.selu(self.conv2(x))))
        x = self.drop(self.pool3(F.selu(self.conv3(x))))
        x = self.drop(self.pool3(F.selu(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.drop(F.selu(self.fc1(x)))
        x = self.drop(F.selu(self.fc2(x)))
        #x = self.drop(F.relu(self.fc3(x)))
        #x = self.drop(F.relu(self.fc4(x)))
        x = self.fc3(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
