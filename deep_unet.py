import torch
import numpy as np
import torch.nn.functional as F

from torch import nn

#  DeepUNet: A Deep Fully Convolutional Networkfor Pixel-level Sea-Land Segmentation

class DownBlock(nn.Module):
    def __init__(self, channel_in, channel_out, maxpool=True):
        super(DownBlock, self).__init__()
        self.maxpool = maxpool
        self.conv1 = nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channel_out, channel_in, kernel_size=3, padding=1)
    
    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        y = input + x
        if self.maxpool:
            x = nn.functional.interpolate(y, size=input.size()[2] // 2)
        else:
            x = y
        return x, y


class UpBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(UpBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_out, channel_out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channel_out, channel_in, kernel_size=3, padding=1)
    
    def forward(self, x1, x2):
        input = torch.cat([x1, x2], dim=1)
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        x = x1 + x
        x = nn.functional.interpolate(x, size=x1.size()[2] * 2)
        return x


class DeepUNet(nn.Module):
    def __init__(self, num_classes):
        super(DeepUNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=2, padding=1)
        
        self.db1 = DownBlock(32, 64)
        self.db2 = DownBlock(32, 64)
        self.db3 = DownBlock(32, 64)
        self.db4 = DownBlock(32, 64)
        self.db5 = DownBlock(32, 64, maxpool=False)
        
        self.ub1 = UpBlock(32, 64)
        self.ub2 = UpBlock(32, 64)
        self.ub3 = UpBlock(32, 64)
        self.ub4 = UpBlock(32, 64)
        self.ub5 = UpBlock(32, 64)
        
        self.conv_end = nn.Conv2d(32, num_classes, kernel_size=3, padding=1)
        
    def forward(self, x):
        input_size = x.size()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = nn.functional.interpolate(x, size=input_size[2] // 2)
        db1_out, dby1 = self.db1(x)
        db2_out, dby2 = self.db2(db1_out)
        db3_out, dby3 = self.db3(db2_out)
        db4_out, dby4 = self.db4(db3_out)
        db5_out, dby5 = self.db5(db4_out)

        ub1_out = self.ub1(db5_out, dby5)
        ub2_out = self.ub2(ub1_out, dby4)
        ub3_out = self.ub3(ub2_out, dby3)
        ub4_out = self.ub4(ub3_out, dby2)
        ub5_out = self.ub5(ub4_out, dby1)
        
        out = self.conv_end(ub5_out)
        
        return out