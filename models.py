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
        
        ## Assume all images will be resized to 128x128 pixels
        #self.conv1 = nn.Conv2d(1, 32, kernel_size =5, stride=1, padding = 0)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        self.Pooling = nn.MaxPool2d(kernel_size =2,stride=2)
        
        self.layer1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size =3,stride=1, padding = 0),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size =2,stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(32,64, kernel_size =2, stride =1, padding =0),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer3 = nn.Sequential(nn.Conv2d(64,128, kernel_size =2, stride =1, padding =0),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(nn.Conv2d(128,256, kernel_size =2, stride =1, padding =0),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout2d(p =0.4)
        self.fc= nn.Linear(13*13*256, 2048)
        self.fc4 = nn.Linear(2048,1024)
        self.fc3= nn.Linear(1024,512)
        self.fc2 = nn.Linear(512, 256)
        self.fc1 = nn.Linear(256,136)
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.reshape(x.size(0), -1)
        x = self.drop_out(x)
        
        x = F.relu(self.fc(x))
        x = self.drop_out(x)
        x = F.relu(self.fc4(x))
        x = self.drop_out(x)
        x = F.relu(self.fc3(x))
        x = self.drop_out(x)
        x = F.relu(self.fc2(x))
        x = self.drop_out(x)
        x = F.relu(self.fc1(x))
        #x = F.softmax(x, dim=1)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
