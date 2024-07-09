from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from torchvision.models import resnet18

class ResNet18(nn.Module):
    """ CLIP ResNet visual model.

    Args:
        WEIGHTS (str): which pretrained CLIP backbone to load
    """
    
    def __init__(self, model_cfg) -> None:

        self.model_cfg = model_cfg
        self.backbone = self.model_cfg.get('WEIGHTS', 'RN50')
        self.frozen_bn = self.model_cfg.get('FROZEN_BN', False)
        self.attnpooling = self.model_cfg.get('ATTNPOOLING', False)
        self.out_indices = self.model_cfg.get("OUT_INDICES", [1, 2, 3])

        
        super(ResNet18, self).__init__()

        m = resnet18(pretrained=True)

        self.stem = nn.Sequential(
            m.conv1,
            m.bn1,
            m.relu,
            m.maxpool
        )

        self.layers = nn.ModuleList([
            m.layer1,
            m.layer2,
            m.layer3,
            m.layer4,
        ])

        self.attnpool = m.avgpool

        self._freeze_stages()

    def forward(self, batch_dict):
        x = batch_dict['camera_imgs']
        if len(x.shape) == 5:
            B, N, C, H, W = x.size()
            x = x.view(B * N, C, H, W)
        elif len(x.shape) == 4:
            BN, C, H, W = x.size()
            B = BN // 6
            N = 6
        else:
            raise TypeError(f'Images not in correct format shape={x.shape}')

        # x = self.resize(x)
        # x = F.interpolate(x, size=(256, 704), mode='bilinear')
        # print('x after', x.shape)

        x = x.type(self.stem[0].weight.dtype)
        x = self.stem(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        if self.attnpooling: # only want this when querying a specific location
            x = self.attnpool(x)
            # just attnpool
            outs = x

            return dict(attn_features=x)

        batch_dict['image_features'] = outs
        return batch_dict
    
    def init_weights(self):
        pass

    def train(self, mode=True):
        """Convert into training mode whilst having weights fronzen."""
        super(CLIPResNet, self).train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze everything.
        """
        for n, p in self.named_parameters():
            if not self.frozen_bn and 'norm' in n.lower():
                p.requires_grad = True
            else:
                p.requires_grad = False
