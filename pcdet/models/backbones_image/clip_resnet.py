from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from torchvision.transforms import Resize
import clip

class CLIPResNet(nn.Module):
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

        
        super(CLIPResNet, self).__init__()

        assert self.backbone in [x for x in clip.available_models() if 'RN' in x]

        m, _ = clip.load(self.backbone)
        self.out_dim = m.visual.output_dim

        m = m.to(dtype=torch.float32)


        self.stem = nn.Sequential(
            m.visual.conv1,
            m.visual.bn1,
            m.visual.relu1,

            m.visual.conv2,
            m.visual.bn2,
            m.visual.relu2,
            
            m.visual.conv3,
            m.visual.bn3,
            m.visual.relu3,
            #
            m.visual.avgpool
        )

        self.layers = nn.ModuleList([
            m.visual.layer1,
            m.visual.layer2,
            m.visual.layer3,
            m.visual.layer4,
        ])

        self.attnpool = m.visual.attnpool

        # self.resize = Resize((224, 704))

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
        """Convert into training mode whilst having weights frozen."""
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
