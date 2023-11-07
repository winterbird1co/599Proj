import torch
import torch.nn as nn
from torch import Tensor

class CNNModel(nn.Module):
    def __init__(self, channels: int, img_shape, activation=nn.ReLU) -> None:
        super(CNNModel, self).__init__()
        base = 16
        size = img_shape

        def layer_block(in_layer, out_layer, batchnorm=True):
            block = [nn.Conv2d(in_layer,out_layer,3,2,1), activation(inplace=True), nn.Dropout2d(0.3)]
            if batchnorm:
                block.append(nn.BatchNorm2d(out_layer,0.1))
            return block

        self.model = nn.Sequential(
            *layer_block(channels, base, False),
            *layer_block(base, base*2),
            *layer_block(base*2,base*4),
            *layer_block(base*4,base*8),
        )

        rd_size = size // 2 ** 4
        self.linear = nn.Sequential(
            nn.Linear(in_features=128 * rd_size ** 2, out_features=1),
            nn.Sigmoid()
        )

    def forward(self,x):
        out = self.model(x)
        out = out.view(out.shape[0], -1)
        return self.linear(out)
    
class AlexModel(nn.Module):
    def __init__(self):
        None