
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResBlock, self).__init__()
   
        # create a skip status
        self.resid = nn.Sequential()
        if stride != 1 or input_channels != output_channels:
            self.resid = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(output_channels)
            )
            # Main path for which happens before the block
        self.begin_block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_channels)
        )
        self.reluactiv = nn.ReLU(inplace=True)
        
    def forward(self, x):
        
        
        out = self.begin_block(x)
        # creating the identity finction and add the x to whole procedure
        added = self.resid(x)
        out += added
        out = self.reluactiv(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.convolut1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.reluactiv = nn.ReLU(inplace=True)
        self.maxpooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # creating the blocks
        self.block1 = self.create_layer(64, 64, stride=1)
        self.block2 = self.create_layer(64, 128, stride=2)
        self.block3 = self.create_layer(128, 256, stride=2)
        self.block4 = self.create_layer(256, 512, stride=2)
        
        self.globalaverragepool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        # transforms the 512 flattended extractted features into two classes linearly
        self.fullyc = nn.Linear(512, 2)
        # since using bcelogits we need logits
        # which is the fc output we don't use sigmooid here that activation of 
        # classification of every single binary label (2 channels 512,2!)
        # sigmoid run in trainer when we want the out put exatly with 
        # 0.5 > divider
        
        
        self.dropout = nn.Dropout(0.5)  # Add dropout layer

    
    def forward(self, x):
        output = self.convolut1(x)
        output = self.batchnorm1(output)
        output = self.reluactiv(output)
        output = self.maxpooling(output)
        
        output = self.block1(output)
        output = self.block2(output)
        output = self.block3(output)
        output = self.block4(output)
    # final aspect of making no spatial through GAP
    # flatten for making it usable for fully connected
    #drop out for regularization    
        output = self.globalaverragepool(output)
        output = self.flatten(output)
        output = self.dropout(output)  # Apply dropout
        output = self.fullyc(output)
        
        return output
    
    def create_layer(self, input_channels, output_channels, stride):
        return nn.Sequential(
            ResBlock(input_channels, output_channels, stride),
            ResBlock(output_channels, output_channels)
        )
