import torch
import torch.nn as nn
from torch import Tensor
from unetmodel import Encoder

class CNNModel(nn.Module):
    def __init__(self, channels: int, img_shape, activation=nn.ReLU()) -> None:
        super(CNNModel, self).__init__()
        base = 32

        def layer_block(in_layer, out_layer, batchnorm=True):
            block = [nn.Conv2d(in_layer,out_layer,3,2,1,bias=not batchnorm)]
            if batchnorm:
                block.append(nn.BatchNorm2d(out_layer))
            block.append(activation)
            return block

        self.model = nn.Sequential(
            *layer_block(channels, base, False),
            *layer_block(base, base*2),
            nn.MaxPool2d(kernel_size=2, stride=1),
            *layer_block(base*2, base*2),
            *layer_block(base*2, base*4),
            nn.MaxPool2d(kernel_size=2, stride=1),
            *layer_block(base*4, base*4),
            *layer_block(base*4, base*8),
        )

        self.linear = nn.Sequential(
            nn.LazyLinear(out_features=1)
        )

    def forward(self,x):
        out = self.model(x)
        out = out.view(out.shape[0], -1)
        return self.linear(out)
    
class EC_Discriminator(nn.Module):
    def __init__(self, channels: int, activation=nn.ReLU()) -> None:
        super(EC_Discriminator, self).__init__()
        base = 48

        self.model = Encoder(channels, activation)
        self.translate = nn.Sequential(
            nn.Linear(in_features=11, out_features=1)
        )

    def forward(self,X):
        _,_,_,_,X = self.model(X)
        return self.translate(X)