import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, head_num):
        super().__init__()

        self.head_num = head_num
        self.dk = (embedding_dim // head_num) ** (1 / 2)

        self.qkv_layer = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        self.out_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x, mask=None):
        qkv = self.qkv_layer(x)

        query, key, value = tuple(rearrange(qkv, 'b t (d k h ) -> k b h t d ', k=3, h=self.head_num))
        energy = torch.einsum("... i d , ... j d -> ... i j", query, key) * self.dk

        if mask is not None:
            energy = energy.masked_fill(mask, -np.inf)

        attention = torch.softmax(energy, dim=-1)

        x = torch.einsum("... i j , ... j d -> ... i d", attention, value)

        x = rearrange(x, "b h t d -> b t (h d)")
        x = self.out_attention(x)

        return x


class MLP(nn.Module):
    def __init__(self, embedding_dim, mlp_dim):
        super().__init__()

        self.mlp_layers = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = self.mlp_layers(x)

        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(embedding_dim, head_num)
        self.mlp = MLP(embedding_dim, mlp_dim)

        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.mlp_norm = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.multi_head_attention(x)
        #_x = self.dropout(_x)
        x = x + h
        #x = self.layer_norm1(x)

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + h
        #x = self.mlp_norm(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim, block_num=12):
        super().__init__()

        self.layer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(embedding_dim, head_num, mlp_dim) for _ in range(block_num)])

    def forward(self, x):
        for layer_block in self.layer_blocks:
            x = layer_block(x)

        return x


class ViT(nn.Module):
    def __init__(self, img_dim, in_channels, embedding_dim, head_num, mlp_dim,
                 block_num, patch_dim, classification=False, num_classes=1):
        super().__init__()

        self.patch_dim = patch_dim
        self.classification = classification
        self.num_tokens = (img_dim // patch_dim) ** 2#196
        self.token_dim = in_channels * (patch_dim ** 2)#768

        #self.projection = nn.Linear(self.token_dim, embedding_dim)
        self.embedding = nn.Parameter(torch.rand(self.num_tokens + 1, embedding_dim))

        #self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.dropout = nn.Dropout(0.1)

        self.transformer = TransformerEncoder(embedding_dim, head_num, mlp_dim, block_num)

        self.patch_embeddings = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim, kernel_size=(1, 1), stride=(1, 1))

        self.tokens = (img_dim//patch_dim) * (img_dim//patch_dim)

        if self.classification:
            self.mlp_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        #img_patches = rearrange(x, 'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)', patch_x=self.patch_dim, patch_y=self.patch_dim)
        img_patches = self.patch_embeddings(x)
        img_patches = img_patches.flatten(2)
        patches = img_patches.transpose(-1, -2)  # (B, n_patches, hidden)
        #print('图像分块:'+str(patches.shape))
        #batch_size, tokens, _ = img_patches.shape

        #project = self.projection(img_patches)
        #patches = project
        #print('变换通道数:'+str(patches.shape))
        #token = repeat(self.cls_token, 'b ... -> (b batch_size) ...', batch_size=batch_size)

        #patches = torch.cat([token, project], dim=1)
        #print(patches.shape)
        #print(self.tokens)
        patches += self.embedding[:self.tokens, :]
        #print('位置编码:' + str(patches.shape))
        x = self.dropout(patches)
        x = self.transformer(x)
        #print('Transformer编码:' + str(patches.shape))
        #x = self.mlp_head(x[:, 0, :]) if self.classification else x[:, 1:, :]

        return x

'''
if __name__ == '__main__':
    vit = ViT(img_dim=128,
              in_channels=3,
              patch_dim=16,
              embedding_dim=512,
              block_num=6,
              head_num=4,
              mlp_dim=1024)
    print(sum(p.numel() for p in vit.parameters()))
    print(vit(torch.rand(1, 3, 128, 128)).shape)
'''

