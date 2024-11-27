"""Building blocks for TiTok.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 

Reference: 
    https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py
"""

import torch
import torch.nn as nn
from collections import OrderedDict


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model,
            n_head,
            mlp_ratio = 4.0,
            act_layer = nn.GELU,
            norm_layer = nn.LayerNorm
        ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.mlp_ratio = mlp_ratio
        # optionally we can disable the FFN
        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            mlp_width = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))

    def attention(
            self,
            x: torch.Tensor
    ):
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(
            self,
            x: torch.Tensor,
    ):
        attn_output = self.attention(x=self.ln_1(x))
        x = x + attn_output
        if self.mlp_ratio > 0:
            x = x + self.mlp(self.ln_2(x))
        return x




def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)

class TiTokEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config.dataset.preprocessing.crop_size 
        self.patch_size = config.model.vq_model.vit_enc_patch_size
        self.grid_size = self.image_size // self.patch_size  # 例如，256/32 = 8
        self.model_size = config.model.vq_model.vit_enc_model_size  # 例如 'large'
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens  # 例如 32
        self.token_size = config.model.vq_model.token_size  # 例如 12

        # 将宽度调整为128，因为我们将通道分为32份
        self.width = 128  
        self.num_layers = config.model.vq_model.layers
        self.num_heads = {
            "large": 2,
        }[self.model_size]
        
        # 调整patch_embed的输出为 width * num_latent_tokens 通道
        self.patch_embed = nn.Conv2d(
            in_channels=3, 
            out_channels=self.width * self.num_latent_tokens,
            kernel_size=self.patch_size, 
            stride=self.patch_size, 
            bias=True
        )

        # 初始化每个token的嵌入
        scale = self.width ** -0.5
        self.class_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, 1, self.width)
        )
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.grid_size ** 2 + 1, self.width)
        )
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width)
        )

        self.ln_pre = nn.LayerNorm(self.width)

        # 为每个token创建独立的transformer堆叠
        self.transformers = nn.ModuleList([
            nn.ModuleList([
                ResidualAttentionBlock(self.width, self.num_heads, mlp_ratio=4.0)
                for _ in range(self.num_layers)
            ])
            for _ in range(self.num_latent_tokens)
        ])

        self.ln_post = nn.LayerNorm(self.width)
        self.conv_out = nn.Conv2d(self.width, self.token_size, kernel_size=1, bias=True)

    def forward(self, pixel_values, latent_tokens):
        batch_size = pixel_values.shape[0]
        
        # 第1步：将x分成32份
        x = self.patch_embed(pixel_values)
        x = x.view(
            batch_size, 
            self.num_latent_tokens, 
            self.width, 
            self.grid_size, 
            self.grid_size
        )

        prev_tokens = []
        for i in range(self.num_latent_tokens):
            # 单独处理每个token
            x_i = x[:, i, :, :, :]  # 形状: [B, width, 8, 8]
            x_i = x_i.reshape(x_i.shape[0], x_i.shape[1], -1).permute(0, 2, 1)

            # 第2步：使用每个token的class和位置嵌入
            #这一步可能有问题
            class_embedding_i = _expand_token(self.class_embedding[i], batch_size)
            x_i = torch.cat([class_embedding_i.to(x_i.dtype), x_i], dim=1)
            x_i = x_i + self.positional_embedding[i].to(x_i.dtype)


            # 第4步：连接先前生成的tokens
            if i > 0:
                prev_tokens_cat = torch.cat(prev_tokens, dim=1)
                x_i = torch.cat([x_i, prev_tokens_cat], dim=1)

            # 第5步：添加latent token位置嵌入
            latent_token_i = _expand_token(latent_tokens[i], batch_size).to(x_i.dtype)
            latent_token_i = latent_token_i + self.latent_token_positional_embedding[i].to(x_i.dtype)
            x_i = torch.cat([x_i, latent_token_i], dim=1)



            x_i = self.ln_pre(x_i)
            x_i = x_i.permute(1, 0, 2)

            # 使用独立的transformer层
            for layer in self.transformers[i]:
                x_i = layer(x_i)

            x_i = x_i.permute(1, 0, 2)

            # 第4步：提取最后位置的token
            token_i = x_i[:, -1, :]
            prev_tokens.append(token_i.unsqueeze(1))

        # 第6步：连接所有tokens
        tokens = torch.cat(prev_tokens, dim=1)
        tokens = self.ln_post(tokens)
        tokens = tokens.permute(0, 2, 1).unsqueeze(-1)
        tokens = self.conv_out(tokens)
        tokens = tokens.reshape(batch_size, self.token_size, 1, self.num_latent_tokens)

        return tokens


    

class TiTokDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config.dataset.preprocessing.crop_size
        self.patch_size = config.model.vq_model.vit_dec_patch_size
        self.grid_size = self.image_size // self.patch_size
        self.model_size = config.model.vq_model.vit_dec_model_size
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        self.token_size = config.model.vq_model.token_size
        self.width = {
                "small": 512,
                "base": 768,
                "large": 1024,
            }[self.model_size]
        self.num_layers = {
                "small": 8,
                "base": 12,
                "large": 30,
            }[self.model_size]
        self.num_heads = {
                "small": 8,
                "base": 12,
                "large": 16,
            }[self.model_size]

        self.decoder_embed = nn.Linear(
            self.token_size, self.width, bias=True)
        scale = self.width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size ** 2 + 1, self.width))
        # add mask token and query pos embed
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, self.width))
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width))
        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0
            ))
        self.ln_post = nn.LayerNorm(self.width)

        self.ffn = nn.Sequential(
            nn.Conv2d(self.width, 2 * self.width, 1, padding=0, bias=True),
            nn.Tanh(),
            nn.Conv2d(2 * self.width, 1024, 1, padding=0, bias=True),
        )
        self.conv_out = nn.Identity()
    
    def forward(self, z_quantized):
        N, C, H, W = z_quantized.shape
        x = z_quantized.reshape(N, C*H, W).permute(0, 2, 1) # NLD
        x = self.decoder_embed(x)

        batchsize, seq_len, _ = x.shape

        mask_tokens = self.mask_token.repeat(batchsize, self.grid_size**2, 1).to(x.dtype)
        mask_tokens = torch.cat([_expand_token(self.class_embedding, mask_tokens.shape[0]).to(mask_tokens.dtype),
                                    mask_tokens], dim=1)
        mask_tokens = mask_tokens + self.positional_embedding.to(mask_tokens.dtype)
        x = x + self.latent_token_positional_embedding[:seq_len]
        x = torch.cat([mask_tokens, x], dim=1)
        
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.transformer[i](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, 1:1+self.grid_size**2] # remove cls embed
        x = self.ln_post(x)
        # N L D -> N D H W
        x = x.permute(0, 2, 1).reshape(batchsize, self.width, self.grid_size, self.grid_size)
        x = self.ffn(x.contiguous())
        x = self.conv_out(x)
        return x
    
