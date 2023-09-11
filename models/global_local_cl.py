from models.unet_model import Unet, default
from torch import Tensor, nn
import torch
from typing import Optional, List   
from einops.layers.torch import Rearrange


class GlobalCL(Unet):
    def __init__(self, 
                 img_size,
                 dim: int = 64,
                 init_dim: Optional[int] = None,
                 dim_mults: List[int] = [1, 2, 4, 8],
                  **kwargs):
        super().__init__(**kwargs)
        init_dim = default(init_dim, dim)
        # from the paper 
        g_emb= 1024
        g_out = 128
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        mid_dim = dims[-1]
        mid_img_size = img_size
        for _ in range(len(dims)-2):
            mid_img_size = int((mid_img_size -1) / 2) + 1
        self.g1 = nn.Sequential(
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(mid_dim * mid_img_size ** 2, g_emb, bias=False),
            nn.ReLU(),
            nn.Linear(g_emb, g_out, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.init_conv(x)

        t = None

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)

            x = block2(x, t)
            x = attn(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        x = self.g1(x)
        return x


class LocalCL(Unet):
    def __init__(self, 
                 img_size,
                 dim: int = 64,
                 init_dim: Optional[int] = None,
                 dim_mults: List[int] = [1, 2, 4, 8],
                  **kwargs):
        super().__init__(**kwargs)
        init_dim = default(init_dim, dim)
        # from the paper 
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        #g_2 small network with two 1x1 convolutions
        self.l = 2
        mid_dim = dims[-self.l-1]
        self.g2 = nn.Sequential(
            nn.Conv2d(mid_dim, mid_dim, 1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(mid_dim),
            nn.Conv2d(mid_dim, mid_dim, 1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.init_conv(x)
        r = x.clone()

        t = None

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups[:self.l]:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)
        
        x = self.g2(x)
        return x




