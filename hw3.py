import hw3_utils as utils
#import matplotlib.pyplot as plt
#import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim

# Note: The only code that I wrote myself in hw3_utils is line 35.
# In this file (hw3), function names, function signatures, docstrings, the call to super in line 24 below, and the above lines of code are not written by me, but all other code
# is.


class Block(nn.Module):
    """A basic block used to build ResNet."""
    
    def __init__(self, num_channels):
        """Initialize a building block for ResNet.

        Argument:
            num_channels: the number of channels of the input to Block, and is also
                          the number of channels of conv layers of Block.
        """
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)
        
    def forward(self, x):
        """
        The input will have shape (N, num_channels, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have the same shape as input.
        """
        toReturn = x
        toReturn = self.conv1(toReturn)
        toReturn = self.bn1(toReturn)
        toReturn = self.relu(toReturn)
        toReturn = self.conv2(toReturn)
        toReturn = self.bn2(toReturn)
        toReturn+=x
        
        return (nn.ReLU())(toReturn)

class ResNet(nn.Module):
    """A simplified ResNet."""

    def __init__(self, num_channels, num_classes=10):
        """Initialize a shallow ResNet.

        Arguments:
            num_channels: the number of output channels of the conv layer
                          before the building block, and also 
                          the number of channels of the building block.
            num_classes: the number of output units.
        """
        super(ResNet, self).__init__()
        self.conv = nn.Conv2d(1, num_channels, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.block = Block(num_channels)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.lin = nn.Linear(num_channels, 10)
        
    def forward(self, x):
        """
        The input will have shape (N, 1, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have shape (N, 10).
        """
        toReturn = x
        toReturn = self.conv(toReturn)
        toReturn = self.bn(toReturn)
        toReturn = self.relu(toReturn)
        toReturn = self.maxpool(toReturn)
        toReturn = self.block.forward(toReturn)
        toReturn = self.avgpool(toReturn)
        toReturn = self.lin(torch.reshape(toReturn, (10, -1)))
        return toReturn

##### nearest neighbor problem ###
        
        
def one_nearest_neighbor(X,Y,X_test):
        
    # return labels for X_test as torch tensor
    myhelp = []
    for i in range(0, X_test.shape[0]):
        best_index = 0
        min_dist = -1
        for j in range(0, X.shape[0]):
            minusvec = X_test[i] - X[j]
            cur_dist = torch.dot(minusvec, minusvec).item()
            if (min_dist < 0 or cur_dist < min_dist):
                min_dist = cur_dist
                best_index = j
        myhelp.append(Y[best_index])
    return torch.FloatTensor(myhelp)

X,Y = utils.load_one_nearest_neighbor_data()
utils.voronoi_plot(X, Y)
