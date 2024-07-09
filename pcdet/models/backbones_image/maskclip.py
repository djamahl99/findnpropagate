from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from torchvision.transforms import Resize
import clip
import os

class DenseAttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

        # text embeddings
        text_metainfo_path = '/home/uqdetche/OpenPCDet/tools/nuscenes_text.pkl'
        if os.path.exists(text_metainfo_path):
            text_metainfo = torch.load(text_metainfo_path)
            self.text_features = text_metainfo['text_features'].to('cuda', dtype=torch.float32)
            self.text_classes, self.text_dim = self.text_features.shape
            self.logit_scale = torch.tensor(text_metainfo['logit_scale'], device='cuda')

            print("Got stored text features", self.text_features.shape)
        else:
            raise Exception("need nuscenes text features! 'nuscenes_text.pkl'")

    def forward(self, x, dense=False):
        if not dense:
            x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
            print('x shape', x.shape, self.positional_embedding[:, None, :].shape)
            x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC

            x, _ = F.multi_head_attention_forward(
                query=x[:1], key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=self.num_heads,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
                in_proj_weight=None,
                in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
                bias_k=None,
                bias_v=None,
                add_zero_attn=False,
                dropout_p=0,
                out_proj_weight=self.c_proj.weight,
                out_proj_bias=self.c_proj.bias,
                use_separate_proj_weight=True,
                training=self.training,
                need_weights=False
            )
            x = x.squeeze(0)
            x = x / x.norm(dim=1, keepdim=True)
            print('x shape', x.shape)
            x = x[..., None, None]
            output = F.conv2d(x, self.text_features[:, :, None, None])

            output = F.softmax(output*self.logit_scale, dim=1)

            return output

        else:
            print('weights', self.c_proj.weight.shape)
            print('bias', self.c_proj.bias.shape)
            # dense mode -> use conv
            
            print('x shape', x.shape)
            q = F.conv2d(input=x, weight=self.q_proj.weight.unsqueeze(2).unsqueeze(3), bias=self.q_proj.bias)
            k = F.conv2d(input=x, weight=self.k_proj.weight.unsqueeze(2).unsqueeze(3), bias=self.k_proj.bias)
            q = torch.flatten(q, start_dim=2).transpose(-2, -1)
            k = torch.flatten(k, start_dim=2).transpose(-2, -1)
            v = F.conv2d(input=x, weight=self.v_proj.weight.unsqueeze(2).unsqueeze(3), bias=self.v_proj.bias)
            feat = F.conv2d(input=v, weight=self.c_proj.weight.unsqueeze(2).unsqueeze(3), bias=self.c_proj.bias)

            feat = feat / feat.norm(dim=1, keepdim=True)
            output = F.conv2d(feat, self.text_features[:, :, None, None])

            output = F.softmax(output*self.logit_scale, dim=1)

            return output
        
    def refine_output(self, output, k):
        if self.pd_thresh > 0:
            N, C, H, W = output.shape
            _output = F.softmax(output*100, dim=1)
            max_cls_conf = _output.view(N, C, -1).max(dim=-1)[0]
            selected_cls = (max_cls_conf < self.pd_thresh)[:, :, None, None].expand(N, C, H, W)
            output[selected_cls] = -100

        if k is not None and self.ks_thresh > 0:
            output = F.softmax(output*100, dim=1)
            N, C, H, W = output.shape
            output = output.view(N, C, -1).transpose(-2, -1)
            # softmax
            # weight = k @ k.transpose(-2, -1)
            # weight = F.softmax(weight, dim=-1)
            # L2 distance
            k = F.normalize(k, p=2)
            weight = k @ k.transpose(-2, -1)

            selected_pos = (output.max(dim=-1, keepdim=True)[0] < self.ks_thresh)
            selected_pos = selected_pos.expand(-1, -1, C)

            weighted_output = weight @ output
            output[selected_pos] = weighted_output[selected_pos]
            output = output.transpose(-2, -1).view(N, C, H, W)

        return output

class MaskCLIP(nn.Module):
    """ CLIP ResNet visual model.

    Args:
        WEIGHTS (str): which pretrained CLIP backbone to load
    """
    
    def __init__(self, model_cfg) -> None:

        self.model_cfg = model_cfg
        self.backbone = self.model_cfg.get('WEIGHTS', 'RN50')
        self.frozen_bn = self.model_cfg.get('FROZEN_BN', False)
        
        super(MaskCLIP, self).__init__()

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
            
            m.visual.avgpool
        )

        self.layers = nn.ModuleList([
            m.visual.layer1,
            m.visual.layer2,
            m.visual.layer3,
            m.visual.layer4,
        ])

        # self.attnpool = m.visual.attnpool
        attnpool = m.visual.attnpool
        spacial_dim = np.sqrt(attnpool.positional_embedding.shape[0] - 1)
        print('pos emb shape', attnpool.positional_embedding.shape)
        print('spacial dim', spacial_dim)
        spacial_dim = int(spacial_dim)
        embed_dim = attnpool.k_proj.in_features
        output_dim = attnpool.c_proj.out_features

        self.attnpool = DenseAttentionPool2d(spacial_dim, embed_dim, attnpool.num_heads, output_dim)
        print('pos emb shape', self.attnpool.positional_embedding.shape)

        self.attnpool.load_state_dict(m.visual.attnpool.state_dict())

        self.resize = Resize((224, 224))

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

        x = self.resize(x)
        # x = F.interpolate(x, size=(256, 704), mode='bilinear')
        # print('x after', x.shape)

        x = x.type(self.stem[0].weight.dtype)
        x = self.stem(x)

        for layer in self.layers:
            x = layer(x)
        
        outs = self.attnpool(x)

        batch_dict['image_features'] = outs
        return batch_dict
    
    def init_weights(self):
        pass

    def train(self, mode=True):
        """Convert into training mode whilst having weights fronzen."""
        super(MaskCLIP, self).train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze everything.
        """
        for n, p in self.named_parameters():
            if not self.frozen_bn and 'norm' in n.lower():
                p.requires_grad = True
            else:
                p.requires_grad = False

                

if __name__ == "__main__":
    model = MaskCLIP(dict()).to('cuda')

    B, N, C, H, W = 1, 1, 3, 224, 224
    batch_dict = dict(camera_imgs=torch.randn((B, N, C, H, W), device='cuda'))

    out = model.forward(batch_dict)
    print('out', out['image_features'].shape)