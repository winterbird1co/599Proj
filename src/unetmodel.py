#UNet: https://arxiv.org/abs/1505.04597
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.transforms.v2.functional import center_crop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
f_thickness = 48

class UNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation=nn.ReLU(), batchnorm:bool = True):
        super(UNetBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate', bias=not batchnorm)]
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels, 0.1))
        layers.append(activation)
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate', bias=not batchnorm))
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels, 0.1))
        layers.append(activation)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation=nn.ReLU(inplace=True), pad=0):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=pad)
        self.layers = UNetBlock(in_channels, out_channels, activation)
        #self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        #self.up = nn.Upsample(scale_factor=2)

    def center_crop(self, layer, target_size):
        #_, _, h, w = layer.size()
        #dy = (h - target_size[0]) // 2
        #dx = (w - target_size[1]) // 2
        return center_crop(layer, target_size)
        #return layer[:, :, dy:(dy+target_size[0]), dx:(dx+target_size[1])]

    def forward(self, x: Tensor, copy: Tensor) -> Tensor:
        up = self.up(x)
        crop = center_crop(copy, up.shape[2:])
        cat = torch.cat((up, crop), 1)
        return self.layers(cat)

class UNetAuto(nn.Module):
    def __init__(self, num_channels: int, activation=nn.ReLU) -> None:
        super(UNetAuto, self).__init__()

        # Per Paper Specification
        self.core_input = UNetBlock(num_channels, f_thickness, activation, batchnorm=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.down_depth_1 = UNetBlock(f_thickness, f_thickness*2, activation)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.down_depth_2 = UNetBlock(f_thickness*2, f_thickness*4, activation)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.down_depth_3 = UNetBlock(f_thickness*4, f_thickness*8, activation)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.bottom_layer = UNetBlock(f_thickness*8, f_thickness*16, activation, batchnorm=False)
        self.up_depth_3 = UpBlock(f_thickness*16, f_thickness*8)
        self.up_depth_2 = UpBlock(f_thickness*8, f_thickness*4)
        self.up_depth_1 = UpBlock(f_thickness*4, f_thickness*2)
        self.out_depth = UpBlock(f_thickness*2, f_thickness)

        self.output = nn.Sequential(
            nn.Conv2d(f_thickness, num_channels, kernel_size=1)
        )
    
    def forward(self, x:Tensor):
        c1 = self.core_input(x)
        x = self.pool1(c1)
        c2 = self.down_depth_1(x)
        x = self.pool2(c2)
        c3 = self.down_depth_2(x)
        x = self.pool3(c3)
        c4 = self.down_depth_3(x)
        x = self.pool4(c4)
        d1 = self.bottom_layer(x)
        d2 = self.up_depth_3(d1, c4)
        d3 = self.up_depth_2(d2, c3)
        d4 = self.up_depth_1(d3, c2)
        out = self.out_depth(d4, c1)
        return self.output(out)
    
    def encoder_forward(self, x:Tensor):
        c1 = self.core_input(x)
        x = self.pool1(c1)
        c2 = self.down_depth_1(x)
        x = self.pool2(c2)
        c3 = self.down_depth_2(x)
        x = self.pool3(c3)
        c4 = self.down_depth_3(x)
        x = self.pool4(c4)
        d1 = self.bottom_layer(x)
        return c1, c2, c3, c4, d1
    
class UNetAutoSmall(nn.Module):
    def __init__(self, num_channels: int, activation=nn.ReLU()) -> None:
        super(UNetAutoSmall, self).__init__()

        # Per Paper Specification
        self.core_input = UNetBlock(num_channels, f_thickness, activation, batchnorm=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_depth_1 = UNetBlock(f_thickness, f_thickness*2, activation)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_depth_2 = UNetBlock(f_thickness*2, f_thickness*4, activation)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottom_layer = UNetBlock(f_thickness*4, f_thickness*8, activation, batchnorm=False)
        self.up_depth_2 = UpBlock(f_thickness*8, f_thickness*4, activation)
        self.up_depth_1 = UpBlock(f_thickness*4, f_thickness*2, activation)
        self.out_depth = UpBlock(f_thickness*2, f_thickness, activation)

        self.output = nn.Sequential(
            nn.Conv2d(f_thickness, num_channels, kernel_size=1)
        )
    
    def forward(self, x: Tensor):
        #c1 = self.core_input(x)
        #x = nn.functional.max_pool2d(c1, 2)
        #c2 = self.down_depth_1(x)
        #x = nn.functional.max_pool2d(c2, 2)
        #c3 = self.down_depth_2(x)
        #x = nn.functional.max_pool2d(c3, 2)
        c1 = self.core_input(x)
        x = self.pool1(c1)
        c2 = self.down_depth_1(x)
        x = self.pool2(c2)
        c3 = self.down_depth_2(x)
        x = self.pool3(c3)
        d1 = self.bottom_layer(x)
        d2 = self.up_depth_2(d1, c3)
        d3 = self.up_depth_1(d2, c2)
        out = self.out_depth(d3, c1)
        return self.output(out)
    
    def encoder_forward(self, x:Tensor):
        c1 = self.core_input(x)
        x = self.pool1(c1)
        c2 = self.down_depth_1(x)
        x = self.pool2(c2)
        c3 = self.down_depth_2(x)
        x = self.pool3(c3)
        d1 = self.bottom_layer(x)
        return c1, c2, c3, d1

        
# https://github.com/jvanvugt/pytorch-unet 
# https://github.com/gerardrbentley/Pytorch-U-Net-AutoEncoder/blob/master/models.py


class Encoder(nn.Module):
    def __init__(self, num_channels: int, activation=nn.ReLU) -> None:
        super(Encoder, self).__init__()
        self.core_input = UNetBlock(num_channels, f_thickness, activation, batchnorm=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.down_depth_1 = UNetBlock(f_thickness, f_thickness*2, activation)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.down_depth_2 = UNetBlock(f_thickness*2, f_thickness*4, activation)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.bottom_layer = UNetBlock(f_thickness*4, f_thickness*8, activation, batchnorm=False)
    
    def forward(self, x:Tensor):
        c1 = self.core_input(x)
        x = self.pool1(c1)
        c2 = self.down_depth_1(x)
        x = self.pool2(c2)
        c3 = self.down_depth_2(x)
        x = self.pool3(c3)
        d1 = self.bottom_layer(x)
        return c1, c2, c3, d1