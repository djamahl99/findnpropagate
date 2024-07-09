import torch
from torch import nn
from torch.nn.functional import max_pool1d 
from math import sqrt
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

class ObjectPointsEncoder(nn.Module):
    def __init__(self, input_dim, dim, depth, heads) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.point_emb = nn.Sequential(
            nn.Linear(input_dim, dim),
            # nn.LayerNorm(dim)
        )

        self.mean_emb = nn.Linear(input_dim, dim)
        self.var_emb = nn.Linear(input_dim, dim)

        self.transformer = Transformer(dim, depth, heads, dim, dim, 0.0)

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, dim)
        # )

    def forward(self, x):
        b, n, _ = x.shape

        x_mean, x_var = torch.mean(x, dim=1, keepdim=True), torch.var(x, dim=1, keepdim=True)

        # encode the mean / var
        mean_emb = self.mean_emb(x_mean)
        var_emb = self.var_emb(x_var)

        # print('x shape', x.shape, x_mean.shape, x_var.shape)

        x = (x - x_mean) / (1e-8 + x_var)

        x = self.point_emb(x)

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.transformer(x)

        # maximum feature
        # return torch.max(x, dim=1)[0]

        # print('x', x.shape)
        # print('x[:, 0]', x[:, 0].shape)

        # class token
        return x[:, [0]] + mean_emb + var_emb
        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        # return self.mlp_head(x)

class WorldEncoder(nn.Module):
    def __init__(self, dim=64, input_dim=3, depth=4, heads=8, num_queries=10):
        super().__init__()

        self.max_super_points = 1000

        self.point_emb = nn.Linear(input_dim, dim)
        self.mean_emb = nn.Linear(input_dim, dim)
        self.var_emb = nn.Linear(input_dim, dim)

        self.point_downsampling = nn.Sequential(
            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, dim, kernel_size=15, padding=7),
            nn.ReLU()
        )

        self.point_downsampling2 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=15, padding=7),
            nn.ReLU()
        )

        self.transformer = Transformer(dim, depth, heads, dim, dim, 0.0)

        self.box_to_query = nn.Linear(4, dim)

        decoder_layer = nn.TransformerDecoderLayer(dim, nhead=8, dim_feedforward=dim, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

        # self.pos_head = nn.Linear(dim, 3)
        # self.rot_head = nn.Linear(dim, 4)
        self.pos_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 3),
        )

        self.image_w = 1242
        self.image_h = 375
        self.max_depth = 50

    def downsample(self, x):
        b, n = x.shape[0:2]

        # print(f"world points = {n}")

        s = n / self.max_super_points
        s1 = sqrt(s)

        # print(f"n={n}, s={s}, s1={s1}, s1_int={max(0, int(s1))}")

        s1 = max(0, int(s1))

        x = self.point_downsampling(x)
        # print("x after ds1", x.shape)
        x = max_pool1d(x, kernel_size=s1)
        # print("x after mp1", x.shape)
        x = self.point_downsampling2(x)
        # print("x after ds2", x.shape)
        x = max_pool1d(x, kernel_size=s1)
        # print("x after mp2", x.shape)

        return rearrange(x, 'n c p -> n p c')

    def forward(self, x, boxes):
        # [0, 1] normalization
        # x[..., 0] /= self.image_w
        # x[..., 1] /= self.image_w
        # x[..., 2] = torch.clamp(x[..., 2], 0, self.max_depth) / self.max_depth
        # boxes[..., [0, 2]] /= self.image_w
        # boxes[..., [1, 3]] /= self.image_h

        # x = torch.nan_to_num(x, 0)
        # boxes = torch.nan_to_num(boxes, 0)

        # cx = (boxes[..., 0] + boxes[..., 2]) / 2
        # cy = (boxes[..., 0] + boxes[..., 2]) / 2

        x = self.point_emb(x)
        x = self.downsample(x)

        emb = self.transformer(x)

        obj_emb = self.decoder(self.box_to_query(boxes), emb)

        xyz = self.pos_head(obj_emb)
        # rot = self.rot_head(obj_emb)

        # return xyz, rot
        return xyz