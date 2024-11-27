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
        self.image_size = config.dataset.preprocessing.crop_size  # 图像尺寸
        self.patch_size = config.model.vq_model.vit_enc_patch_size  # Patch 尺寸
        self.grid_size = self.image_size // self.patch_size  # 计算 patch 网格大小，例如 256/32 = 8
        self.model_size = config.model.vq_model.vit_enc_model_size  # 模型大小，例如 'large'
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens  # 隐变量 token 的数量，例如 32
        self.token_size = config.model.vq_model.token_size  # token 尺寸，例如 12

        # 定义通道宽度和嵌入维度
        self.width = config.model.vq_model.width  
        self.embedding_width = config.model.vq_model.embedding_width
        self.num_layers_x = config.model.vq_model.layers_x  # 初步建模的 transformer 层数
        self.num_layers_token = config.model.vq_model.layers_token  # 每个 token 的 transformer 层数

        self.num_heads = {
            "large": 4,  # 设置多头注意力的头数
        }[self.model_size]
        
        # 定义 patch_embed 层，将输入映射为嵌入维度大小
        self.patch_embed = nn.Conv2d(
            in_channels=3, 
            out_channels=self.embedding_width,
            kernel_size=self.patch_size, 
            stride=self.patch_size, 
            bias=True
        )

        # 初始化类别嵌入和位置嵌入
        scale = self.embedding_width ** -0.5  # 初始化缩放因子
        num_patches = self.grid_size ** 2  # patch 的总数
        self.class_embedding = nn.Parameter(
            scale * torch.randn(1, 1, self.embedding_width)  # 类别嵌入
        )
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(1, num_patches + 1, self.embedding_width)  # 位置嵌入
        )

        self.ln_pre = nn.LayerNorm(self.width)  # 输入的归一化层
        self.ln_post = nn.LayerNorm(self.width)  # 输出的归一化层

        # 定义初步建模的 transformer 堆叠
        self.transformers_Pixel = nn.Sequential(*[
            ResidualAttentionBlock(self.embedding_width, self.num_heads, mlp_ratio=4.0)
            for _ in range(self.num_layers_x)
        ])

        # 定义 FFN 层将嵌入维度降到 self.width
        self.reduce_ffn = nn.Linear(self.embedding_width, self.width)

        # 为每个 token 定义独立的 transformer 堆叠
        self.transformers_tokens = nn.ModuleList([
            nn.Sequential(*[
                ResidualAttentionBlock(self.width, self.num_heads, mlp_ratio=4.0)
                for _ in range(self.num_layers_token)
            ])
            for _ in range(self.num_latent_tokens)
        ])

        self.conv_out = nn.Conv2d(self.width, self.token_size, kernel_size=1, bias=True)  # 输出的卷积层

    def forward(self, pixel_values, latent_tokens):
        batch_size = pixel_values.shape[0]

        # 第一步：对图像进行 patch embedding 并进行初步建模
        x = self.patch_embed(pixel_values)  # 形状: [batch_size, embedding_width, grid_size, grid_size]
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1) # 展平空间维度: [batch_size, num_patches, embedding_width]

        # 添加类别嵌入和位置嵌入
        class_embedding = self.class_embedding.expand(batch_size, -1, -1)  # 类别嵌入: [batch_size, 1, embedding_width]
        x = torch.cat([class_embedding, x], dim=1)  # 拼接类别嵌入: [batch_size, num_patches + 1, embedding_width]
        x = x + self.positional_embedding.to(x.dtype)  # 添加位置嵌入

        # 通过初步建模的 transformer 堆叠
        x = x.permute(1, 0, 2)  # 转置维度以适配 transformer 输入: [num_patches + 1, batch_size, embedding_width]
        x = self.transformers_Pixel(x)
        x = x.permute(1, 0, 2)  # 转回原始形状: [batch_size, num_patches + 1, embedding_width]

        # 第二步：降维到 self.width，并初始化 x_info
        x = self.reduce_ffn(x)  # 降维: [batch_size, num_patches + 1, self.width]
        x_info = x  # 初始 x_info 固定

        prev_tokens = []

        # 第三步：逐步生成 tokens
        for i in range(self.num_latent_tokens):
            # 获取当前的 latent token
            latent_token_i = latent_tokens[:, i, :].unsqueeze(1)  # 当前 token: [batch_size, 1, self.width]

            # 将当前 token 拼接到 x_info
            x = torch.cat([x_info, latent_token_i], dim=1)  # 拼接后: [batch_size, seq_length + 1, self.width]

            x = self.ln_pre(x)  # 归一化
            x = x.permute(1, 0, 2)  # 转置: [seq_length + 1, batch_size, self.width]

            # 通过对应的 transformer 堆叠
            x = self.transformers_tokens[i](x)

            x = x.permute(1, 0, 2)  # 转回原始形状: [batch_size, seq_length + 1, self.width]

            # 提取最后一个位置的 token
            token_i = x[:, -1, :]  # [batch_size, self.width]
            prev_tokens.append(token_i.unsqueeze(1))  # 存储当前 token

            # 更新 x_info，将当前 token 拼接到 x_info
            x_info = torch.cat([x_info, token_i.unsqueeze(1)], dim=1)

        # 拼接所有生成的 tokens
        tokens = torch.cat(prev_tokens, dim=1)  # [batch_size, num_latent_tokens, self.width]
        tokens = self.ln_post(tokens)  # 归一化
        tokens = tokens.permute(0, 2, 1).unsqueeze(-1)  # 调整形状: [batch_size, self.width, num_latent_tokens, 1]
        tokens = self.conv_out(tokens)  # 卷积层输出
        tokens = tokens.reshape(batch_size, self.token_size, 1, self.num_latent_tokens)  # 调整到最终输出形状

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
        self.embedding_width = config.model.vq_model.embedding_width
        self.width=config.model.vq_model.width
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
    
