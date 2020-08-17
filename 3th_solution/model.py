# 샴넷 -> 수정 예정

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

'''
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*100*100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
'''

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv1 = conv_block(3, 64)
        self.conv2 = conv_block(64, 64)
        self.conv3 = conv_block(64, 128)
        self.conv4 = conv_block(128, 128)
        
        self.logits1 = nn.Sequential(
            nn.Linear(3200, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 500),
            nn.ReLU(inplace=True)
        )
        
    def forward_once(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        
        output = output.view(output.size()[0], -1)
        output = self.logits1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


def conv_block(in_channels: int, out_channels: int) -> nn.Module:
    """Returns a Module that performs 3x3 convolution, ReLu activation, 2x2 max pooling.

    # Arguments
        in_channels:
        out_channels:
    """
    return nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_channels, out_channels, 3),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(out_channels, out_channels, 3),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
