import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PatchPartition(nn.Module):
    def __init__(self, patch_height, patch_width):
        super(PatchPartition, self).__init__()
        self.partition = Rearrange(
            'b c (h p1) (w p2) -> b h w (p1 p2 c)', p1=patch_height, p2=patch_width)

    def forward(self, img):
        '''Change the dimensions of the picture: b c w h --> b h/4 w/4 16C'''
        x = self.partition(img)
        _, H, W, _ = x.shape
        return x, H, W


class LinearEmbedding(nn.Module):
    def __init__(self, patch_dim, embed_dim=96, norm_layer=None):
        super(LinearEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.embeding = Rearrange('b h w c -> b (h w) c')
        self.Linear = nn.Linear(patch_dim, embed_dim)

    def forward(self, img):
        x = self.embeding(img)
        x = self.Linear(x)
        return x


class ToPatchEmbed(nn.Module):
    def __init__(self, in_ch, patch_height, patch_width):
        super(ToPatchEmbed, self).__init__()
        # in_ch - Number of input channels
        patch_dim = int(patch_height * patch_width * in_ch)
        self.patch_partition = PatchPartition(patch_height, patch_width)
        self.linear_embedding = LinearEmbedding(patch_dim, embed_dim=96)

    def forward(self, img):
        x, H, W = self.patch_partition(img)
        x = self.linear_embedding(x)

        return x, H, W


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(
            [coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - \
            coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]

        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - \
            1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv.unbind(0)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Restore each window to a feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def window_partition(x, window_size: int):
    """
    Divide the feature map into windows without overlapping according to window_size
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous(
    ).view(-1, window_size, window_size, C)
    return windows


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(
                self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        # Give the feature map to the pad to an integer multiple of the window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size

        # The first two last dimensions are expanded to 0 rows and 0 columns,
        # and the dimensions are expanded forward accordingly
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        # [nW*B, Mh*Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        # merge windows
        # [nW*B, Mh, Mw, C]
        attn_windows = attn_windows.view(-1,
                                         self.window_size, self.window_size, C)
        shifted_x = window_reverse(
            attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(
                self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            # Remove the data from the previous pad
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(
                    drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # Ensure that Hp and Wp are integer multiples of window_size
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size

        # Have the same channel arrangement order as the feature map,
        # which is convenient for subsequent window_partition
        img_mask = torch.zeros(
            (1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(
            img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        # [nW, Mh*Mw]
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size)
        # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):
        attn_mask = self.create_mask(x, H, W)  # [nW, Mh*Mw, Mh*Mw]
        for blk in self.blocks:
            blk.H, blk.W = H, W

            # I don't know what it does here
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2

        return x, H, W


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        # If the H and W of the input feature map are not integer multiples of 2, padding is required
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # Note that the Tensor channel here is [B, H, W, C], so it will be somewhat different from the official document
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        '''Calculate the length and width results after passing through this module in advance, 
        and return them to the next module for use'''
        H = H // 2
        W = W // 2
        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x, H, W


class PatchExpanding(nn.Module):
    def __init__(self, dim, scale, flag='normal'):
        super(PatchExpanding, self).__init__()
        self.flag = flag
        # The dimension is expanded twice, and then divided into length and width
        self.outdim = dim * scale
        self.linear = nn.Linear(dim, self.outdim)

    def forward(self, x, H, W):
        x = self.linear(x)
        B, _, _ = x.shape
        # self.partition = Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1=patch_height, p2=patch_width)
        x = rearrange(x, 'b (h w) d -> b h w d ', w=W, h=H)
        if self.flag == 'normal':
            x = x.view(B, H * 2, W * 2, -1)
            H = H * 2
            W = W * 2
        elif self.flag == 'special':
            x = x.view(B, H * 4, W * 4, -1)
            H = H * 4
            W = W * 4
        else:
            raise ValueError("There is no such PatchExpanding like this")
        x = rearrange(x, 'b h w d -> b (h w) d')

        return x, H, W


class SkipConnection(nn.Module):
    '''for skip connections'''

    def __init__(self, in_size, out_size):
        '''in_size refers to the dimension after the concat operation, 
        and out_size is restored to the original dimension'''
        super(SkipConnection, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2, H, W):
        inputs1 = rearrange(inputs1, 'b (h w) d -> b d h w ', h=H, w=W)
        inputs2 = rearrange(inputs2, 'b (h w) d -> b d h w ', h=H, w=W)
        outputs = torch.cat([inputs1, inputs2], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        outputs = rearrange(outputs, 'b d h w  -> b (h w) d',
                            h=H, w=W)  # Dimension reduction

        return outputs, H, W


class LinearProjection(nn.Module):
    def __init__(self, in_size, class_num):
        super(LinearProjection, self).__init__()
        self.Linear = nn.Linear(in_size, class_num)
        self.norm = nn.LayerNorm(in_size)

    def forward(self, x, H, W):
        x = self.norm(x)
        x = self.Linear(x)
        x = rearrange(x, 'b (h w) d -> b d h w', h=H, w=W)
        return x


class SwinUNet(nn.Module):
    def __init__(self, in_ch=3, n_classes=2, embed_dim=96, patch_height=4, patch_width=4):
        super(SwinUNet, self).__init__()
        self.to_patch_embed = ToPatchEmbed(in_ch=in_ch,
                                           patch_height=patch_height, patch_width=patch_width)

        # for i_layer in range(self.num_layers):
        #     # print(depths[:i_layer],depths[:i_layer + 1])
        #     # print(sum(depths[:i_layer]), sum(depths[:i_layer + 1]))
        #     # print('BasicLayerinputdrp:', dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])])
        #
        #     # Note that there are some differences between the stage constructed here and the paper diagram
        #     # The stage here does not contain the patch_merging layer of the stage, but contains the next stage
        #
        #     layers = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
        #                         depth=depths[i_layer],
        #                         num_heads=num_heads[i_layer],
        #                         window_size=window_size,
        #                         mlp_ratio=self.mlp_ratio,
        #                         qkv_bias=qkv_bias,
        #                         drop=drop_rate,
        #                         attn_drop=attn_drop_rate,
        #                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
        #                         norm_layer=norm_layer,
        #                         downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
        #                         use_checkpoint=use_checkpoint)
        #     self.layers.append(layers)
        '''swintransformer'''
        self.basic_layer_0 = BasicLayer(dim=embed_dim,
                                        depth=2,
                                        num_heads=3,
                                        window_size=7,
                                        mlp_ratio=4,
                                        qkv_bias=True,
                                        drop=0,
                                        attn_drop=0,
                                        drop_path=0.1,
                                        norm_layer=nn.LayerNorm,
                                        downsample=None,
                                        use_checkpoint=False)
        self.basic_layer_1 = BasicLayer(dim=2 * embed_dim,
                                        depth=2,
                                        num_heads=6,
                                        window_size=7,
                                        mlp_ratio=4,
                                        qkv_bias=True,
                                        drop=0,
                                        attn_drop=0,
                                        drop_path=0.1,
                                        norm_layer=nn.LayerNorm,
                                        downsample=None,
                                        use_checkpoint=False)
        self.basic_layer_2 = BasicLayer(dim=4 * embed_dim,
                                        depth=6,
                                        num_heads=12,
                                        window_size=7,
                                        mlp_ratio=4,
                                        qkv_bias=True,
                                        drop=0,
                                        attn_drop=0,
                                        drop_path=0.1,
                                        norm_layer=nn.LayerNorm,
                                        downsample=None,
                                        use_checkpoint=False)
        self.basic_layer_single = BasicLayer(dim=8 * embed_dim,
                                             depth=2,
                                             num_heads=24,
                                             window_size=7,
                                             mlp_ratio=4,
                                             qkv_bias=True,
                                             drop=0,
                                             attn_drop=0,
                                             drop_path=0.1,
                                             norm_layer=nn.LayerNorm,
                                             downsample=None,
                                             use_checkpoint=False)
        '''merging'''
        self.patch_merging_0 = PatchMerging(dim=embed_dim)
        self.patch_merging_1 = PatchMerging(dim=2 * embed_dim)
        self.patch_merging_2 = PatchMerging(dim=4 * embed_dim)

        '''upsample'''
        self.patch_expanding_0 = PatchExpanding(dim=8 * embed_dim, scale=2)
        self.patch_expanding_1 = PatchExpanding(dim=4 * embed_dim, scale=2)
        self.patch_expanding_2 = PatchExpanding(dim=2 * embed_dim, scale=2)
        self.patch_expanding_3 = PatchExpanding(
            dim=embed_dim, scale=16, flag='special')  # special

        '''skipconnection'''
        self.skip_3 = SkipConnection(
            in_size=2 * 4 * embed_dim, out_size=2 * embed_dim)
        self.skip_2 = SkipConnection(
            in_size=2 * 2 * embed_dim, out_size=2 * embed_dim)
        self.skip_1 = SkipConnection(in_size=2 * embed_dim, out_size=embed_dim)

        '''Linear Projection'''
        self.linear_projection = LinearProjection(embed_dim, n_classes)

    def forward(self, img):
        '''x1 corresponds to y1, x2 corresponds to y2, do skipconnection'''
        # step1
        # H, W The length and width in units of patch
        x, H, W = self.to_patch_embed(img)
        x1, H, W = self.basic_layer_0(x, H, W)
        '''x1=1,3136,96'''

        # step2
        # patchmerging numbered from top to bottom
        x, H, W = self.patch_merging_0(x1, H, W)
        x2, H, W = self.basic_layer_1(x, H, W)

        # step3
        x, H, W = self.patch_merging_1(x2, H, W)
        x3, H, W = self.basic_layer_2(x, H, W)
        x, H, W = self.patch_merging_2(x3, H, W)

        '''BottleNeck'''
        x, H, W = self.basic_layer_single(x, H, W)  # step1
        y, H, W = self.basic_layer_single(x, H, W)  # step2

        '''UpSample'''
        # step1, expanding number from bottom to top
        y, H, W = self.patch_expanding_0(y, H, W)
        y3, H, W = self.basic_layer_2(y, H, W)
        y, H, W = self.skip_3(y3, x3, H, W)  # skipconnection

        # step2
        y, H, W = self.patch_expanding_1(y3, H, W)
        y2, H, W = self.basic_layer_1(y, H, W)
        y, H, W = self.skip_2(y2, x2, H, W)  # skipconnection

        # step3
        y, H, W = self.patch_expanding_2(y2, H, W)
        y1, H, W = self.basic_layer_0(y, H, W)
        y, H, W = self.skip_1(y1, x1, H, W)  # skipconnection

        '''segment'''
        y, H, W = self.patch_expanding_3(y, H, W)
        y = self.linear_projection(y, H, W)

        return y
