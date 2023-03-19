import torch
import torch.nn as nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    def __init__(self,inchannel,outchannel,stride=1):
        super(ResidualBlock,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannel,outchannel,kernel_size=3,padding=1,stride=stride),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel,outchannel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(outchannel),
        )
        self.conv1x1 = nn.Sequential( 
            nn.Conv2d(inchannel,outchannel,kernel_size=1,stride=stride),
            nn.BatchNorm2d(outchannel)
            )
    def forward(self,X):
        Y = self.conv(X)
        X = self.conv1x1(X)
        out = F.relu(X+Y)
        return out
    
block1 =  ResidualBlock(64,64,1)
block2 =  ResidualBlock(64,64,1)
block3 =  ResidualBlock(64,128,2)
block4 =  ResidualBlock(128,128,1)
block5 =  ResidualBlock(128,256,2)
block6 =  ResidualBlock(256,256,1)
block7 =  ResidualBlock(256,512,2)
block8 =  ResidualBlock(512,512,1)

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(kernel_size=3,in_channels=3,out_channels=64,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            block1,block2,block3,block4,block5,block6,block7,block8
        )
        self.classifier = nn.Sequential(
            #nn.AdaptiveAvgPool2d((1,1)),
            #nn.Flatten(),
            nn.Linear(512,3)
        )
    
    def forward(self,X):
        features = self.features(X)
        out = F.avg_pool2d(features, features.shape[2])
        out = out.squeeze()
        out = self.classifier(out)
        return out