#UNet: https://arxiv.org/abs/1505.04597
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.transforms.v2.functional import center_crop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
f_thickness = 48

class UNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation=nn.ReLU(inplace=True), batchnorm:bool = True, ksp_param=(4,1,0)):
        super(UNetBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=ksp_param[0], stride=ksp_param[1], padding=ksp_param[2], padding_mode='replicate', bias=not batchnorm)]
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels, 0.1))
        layers.append(activation)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation=nn.ReLU(inplace=True), pad=0, ksp_param=(4,1,0)):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=pad)
        self.layers = UNetBlock(in_channels, out_channels, activation, ksp_param=ksp_param)
        #self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        #self.up = nn.Upsample(scale_factor=2)

    def center_crop(self, layer, target_size):
        _, _, h, w = layer.size()
        dy = (h - target_size[0]) // 2
        dx = (w - target_size[1]) // 2
        return layer[:, :, dy:(dy+target_size[0]), dx:(dx+target_size[1])]

    def forward(self, x: Tensor, copy: Tensor) -> Tensor:
        up = self.up(x)
        crop = center_crop(copy, up.shape[2:])
        cat = torch.cat((up, crop), 1)
        return self.layers(cat)

class UNetAuto(nn.Module):
    def __init__(self, num_channels: int, activation=nn.ReLU) -> None:
        super(UNetAuto, self).__init__()

        # Per Paper Specification
        self.encoder = Encoder2(num_channels, activation)

        self.up5 = UpBlock(f_thickness*16, f_thickness*8, activation)
        self.up4 = UpBlock(f_thickness*8, f_thickness*8, activation)
        self.up3 = UpBlock(f_thickness*8, f_thickness*4, activation)
        self.up2 = UpBlock(f_thickness*4, f_thickness*4, activation)
        self.up1 = UpBlock(f_thickness*4, f_thickness*2, activation)
        self.out = UpBlock(f_thickness*2, f_thickness*1, activation)

        self.output = nn.Sequential(
            nn.Conv2d(f_thickness, num_channels, kernel_size=1)
        )
    
    def forward(self, x: Tensor):
        c0, c1, c2, c3, c4, cf, d1 = self.encoder(x)
        x = self.up5(d1, cf)
        x = self.up4(x, c4)
        x = self.up3(x, c3)
        x = self.up2(x, c2)
        x = self.up1(x, c1)
        out = self.out(x, c0)
        return self.output(out)
    
class UNetAutoSmall(nn.Module):
    def __init__(self, num_channels: int, activation=nn.ReLU()) -> None:
        super(UNetAutoSmall, self).__init__()

        # Per Paper Specification
        self.encoder = Encoder(num_channels, activation)

        self.up3 = UpBlock(f_thickness*16, f_thickness*8, activation, ksp_param=(4,1,0))
        self.up2 = UpBlock(f_thickness*8, f_thickness*4, activation, ksp_param=(4,1,0))
        self.up1 = UpBlock(f_thickness*4, f_thickness*2, activation, ksp_param=(4,1,0))
        self.out = UpBlock(f_thickness*2, f_thickness, activation, ksp_param=(4,1,0), pad=1)

        self.output = nn.Sequential(
            nn.Conv2d(f_thickness, num_channels, kernel_size=1)
        )
    
    def forward(self, x: Tensor):
        c0, c1, c2, c3, d1 = self.encoder(x)
        x = self.up3(d1, c3)
        x = self.up2(x, c2)
        x = self.up1(x, c1)
        out = self.out(x, c0)
        return self.output(out)

        
# https://github.com/jvanvugt/pytorch-unet 
# https://github.com/gerardrbentley/Pytorch-U-Net-AutoEncoder/blob/master/models.py


class Encoder(nn.Module):
    def __init__(self, num_channels: int, activation=nn.ReLU) -> None:
        super(Encoder, self).__init__()
        self.core_input = UNetBlock(num_channels, f_thickness, activation, batchnorm=False, ksp_param=(4,1,0))
        self.down_depth_1 = UNetBlock(f_thickness, f_thickness*2, activation, batchnorm=True, ksp_param=(4,2,0))
        self.down_depth_2 = UNetBlock(f_thickness*2, f_thickness*4, activation, batchnorm=True, ksp_param=(4,2,0))
        self.down_depth_3 = UNetBlock(f_thickness*4, f_thickness*8, activation, batchnorm=True, ksp_param=(4,2,0))
        self.bottom_layer = UNetBlock(f_thickness*8, f_thickness*16, activation, batchnorm=False, ksp_param=(4,1,0))
    
    def forward(self, x:Tensor):
        c0 = self.core_input(x)
        c1 = self.down_depth_1(c0)
        c2 = self.down_depth_2(c1)
        c3 = self.down_depth_3(c2)
        d1 = self.bottom_layer(c3)
        return c0, c1, c2, c3, d1
    
class Encoder2(nn.Module):
    def __init__(self, num_channels: int, activation=nn.ReLU) -> None:
        super(Encoder, self).__init__()
        self.core_input = UNetBlock(num_channels, f_thickness, activation, batchnorm=False)
        self.down_depth_1 = UNetBlock(f_thickness, f_thickness*2, activation, batchnorm=True, ksp_param=(4,2,0))
        self.down_depth_2 = UNetBlock(f_thickness*2, f_thickness*4, activation, batchnorm=True, ksp_param=(4,2,0))
        self.down_depth_3 = UNetBlock(f_thickness*4, f_thickness*4, activation, batchnorm=True, ksp_param=(4,2,0))
        self.down_depth_4 = UNetBlock(f_thickness*4, f_thickness*8, activation, batchnorm=True, ksp_param=(4,2,0))
        self.down_depth_5 = UNetBlock(f_thickness*8, f_thickness*8, activation, batchnorm=True, ksp_param=(4,2,0))
        self.bottom_layer = UNetBlock(f_thickness*8, f_thickness*16, activation)
    
    def forward(self, x:Tensor):
        c0 = self.core_input(x)
        c1 = self.down_depth_1(c0)
        c2 = self.down_depth_2(c1)
        c3 = self.down_depth_3(c2)
        c4 = self.down_depth_4(c3)
        cf = self.down_depth_5(c4)
        d1 = self.bottom_layer(cf)
        return c0, c1, c2, c3, c4, cf, d1