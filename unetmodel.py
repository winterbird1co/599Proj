#UNet: https://arxiv.org/abs/1505.04597
import torch
import torch.nn as nn
from torch import Tensor

class UNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation=nn.ReLU):
        super(UNetBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), #3x3 2D Conv.
            activation(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            activation(),
            nn.Dropout2d(0.3),
            nn.BatchNorm2d(out_channels),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

class UpConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation=nn.ReLU):
        super(UpConvBlock, self).__init__()
        self.layers = UNetBlock(in_channels, out_channels, activation)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def center_crop(self, layer, target_size):
        _, _, h, w = layer.size()
        dy = (h - target_size[0]) // 2
        dx = (w - target_size[1]) // 2
        return layer[:, :, dy:(dy+target_size[0]), dx:(dx+target_size[1])]

    def forward(self, x: Tensor, copy: Tensor) -> Tensor:
        up = self.up(x)
        crop = self.center_crop(copy, up.shape[2:])
        cat = torch.cat([up, crop], 1)
        return self.layers(cat)

class UNetAuto(nn.Module):
    def __init__(self, num_channels: int, activation=nn.ReLU) -> None:
        super(UNetAuto, self).__init__()
        f_thickness = 64
        self.activation = activation

        # Per Paper Specification
        self.core_input = UNetBlock(num_channels, f_thickness, self.activation)
        self.down_depth_1 = UNetBlock(f_thickness, f_thickness*2, self.activation)
        self.down_depth_2 = UNetBlock(f_thickness*2, f_thickness*4, self.activation)
        self.down_depth_3 = UNetBlock(f_thickness*4, f_thickness*8, self.activation)

        self.bottom_layer = UNetBlock(f_thickness*8, f_thickness*16, self.activation)
        self.up_depth_3 = UpConvBlock(f_thickness*16, f_thickness*8, self.activation)
        self.up_depth_2 = UpConvBlock(f_thickness*8, f_thickness*4, self.activation)
        self.up_depth_1 = UpConvBlock(f_thickness*4, f_thickness*2, self.activation)
        self.out_depth = UpConvBlock(f_thickness*2, f_thickness, self.activation)

        self.output = nn.Sequential(
            nn.Conv2d(f_thickness, num_channels, kernel_size=1)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        c1 = self.core_input(x)
        x = nn.functional.max_pool2d(c1, 2)
        c2 = self.down_depth_1(x)
        x = nn.functional.max_pool2d(c2, 2)
        c3 = self.down_depth_2(x)
        x = nn.functional.max_pool2d(c3, 2)
        c4 = self.down_depth_3(x)
        x = nn.functional.max_pool2d(c4, 2)
        d1 = self.bottom_layer(x)
        d2 = self.up_depth_3(d1, c4)
        d3 = self.up_depth_2(d2, c3)
        d4 = self.up_depth_1(d3, c2)
        out = self.out_depth(d4, c1)
        return self.output(out)
    
    def yield_forward(self, x:Tensor):
        c1 = self.core_input(x)
        x = nn.functional.max_pool2d(c1, 2)
        c2 = self.down_depth_1(x)
        x = nn.functional.max_pool2d(c2, 2)
        c3 = self.down_depth_2(x)
        x = nn.functional.max_pool2d(c3, 2)
        c4 = self.down_depth_3(x)
        x = nn.functional.max_pool2d(c4, 2)
        d1 = self.bottom_layer(x)
        d2 = self.up_depth_3(d1, c4)
        d3 = self.up_depth_2(d2, c3)
        d4 = self.up_depth_1(d3, c2)
        out = self.out_depth(d4, c1)
        return self.output(out), [c1,c2,c3,c4,d1]

class BranchEncoder(nn.Module):
    def __init__(self, num_channels: int, activation=nn.ReLU) -> None:
        super(BranchEncoder, self).__init__()
        f_thickness = 64
        self.activation = activation
        self.core_input = UNetBlock(num_channels, f_thickness, self.activation)
        self.down_depth_1 = UNetBlock(f_thickness, f_thickness*2, self.activation)
        self.down_depth_2 = UNetBlock(f_thickness*2, f_thickness*4, self.activation)
        self.bottom_layer = UNetBlock(f_thickness*4, f_thickness*8)

    def forward(self, x: Tensor) -> Tensor:
        x = self.core_input(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.down_depth_1(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.down_depth_2(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.bottom_layer(x)
        out = nn.functional.max_pool2d(x, 2)
        return out
    
    def yield_forward(self, x:Tensor):
        c1 = self.core_input(x)
        x = nn.functional.max_pool2d(c1, 2)
        c2 = self.down_depth_1(x)
        x = nn.functional.max_pool2d(c2, 2)
        c3 = self.down_depth_2(x)
        x = nn.functional.max_pool2d(c3, 2)
        d1 = self.bottom_layer(x)
        return [c1,c2,c3,d1]
    
class UNetAutoSmall(nn.Module):
    def __init__(self, num_channels: int, activation=nn.ReLU) -> None:
        super(UNetAutoSmall, self).__init__()
        f_thickness = 64
        self.activation = activation

        # Per Paper Specification
        self.core_input = UNetBlock(num_channels, f_thickness, self.activation)
        self.down_depth_1 = UNetBlock(f_thickness, f_thickness*2, self.activation)
        self.down_depth_2 = UNetBlock(f_thickness*2, f_thickness*4, self.activation)

        self.bottom_layer = UNetBlock(f_thickness*4, f_thickness*8, self.activation)
        self.up_depth_2 = UpConvBlock(f_thickness*8, f_thickness*4, self.activation)
        self.up_depth_1 = UpConvBlock(f_thickness*4, f_thickness*2, self.activation)
        self.out_depth = UpConvBlock(f_thickness*2, f_thickness, self.activation)

        self.output = nn.Sequential(
            nn.Conv2d(f_thickness, num_channels, kernel_size=1)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        c1 = self.core_input(x)
        x = nn.functional.max_pool2d(c1, 2)
        c2 = self.down_depth_1(x)
        x = nn.functional.max_pool2d(c2, 2)
        c3 = self.down_depth_2(x)
        x = nn.functional.max_pool2d(c3, 2)
        d1 = self.bottom_layer(x)
        d2 = self.up_depth_2(d1, c3)
        d3 = self.up_depth_1(d2, c2)
        out = self.out_depth(d3, c1)
        return self.output(out)
    
    def yield_forward(self, x: Tensor):
        c1 = self.core_input(x)
        x = nn.functional.max_pool2d(c1, 2)
        c2 = self.down_depth_1(x)
        x = nn.functional.max_pool2d(c2, 2)
        c3 = self.down_depth_2(x)
        x = nn.functional.max_pool2d(c3, 2)
        d1 = self.bottom_layer(x)
        d2 = self.up_depth_2(d1, c3)
        d3 = self.up_depth_1(d2, c2)
        out = self.out_depth(d3, c1)
        return self.output(out), [c1,c2,c3,d1]

        
# https://github.com/jvanvugt/pytorch-unet 
# https://github.com/gerardrbentley/Pytorch-U-Net-AutoEncoder/blob/master/models.py
