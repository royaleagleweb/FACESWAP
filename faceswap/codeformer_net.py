"""CodeFormer network used for face restoration.

Adapted from sczhou/CodeFormer (the v0.1.0 inference graph). Only the
generator path is included. Weights are the official ``codeformer.pth``.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = 32 if channels % 32 == 0 else 1
    return nn.GroupNorm(num_groups=groups, num_channels=channels, eps=1e-6, affine=True)


def _swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, emb_dim: int, beta: float) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.emb_dim = emb_dim
        self.beta = beta
        self.embedding = nn.Embedding(codebook_size, emb_dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

    def get_codebook_feat(self, indices: Tensor, shape) -> Tensor:
        indices = indices.reshape(-1, 1)
        min_encodings = torch.zeros(indices.shape[0], self.codebook_size, device=indices.device)
        min_encodings.scatter_(1, indices, 1)
        z_q = torch.matmul(min_encodings, self.embedding.weight)
        if shape is not None:
            z_q = z_q.view(shape).permute(0, 3, 1, 2).contiguous()
        return z_q


class Downsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: Optional[int] = None) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = _group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = _group_norm(self.out_channels)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_out = None
        if self.in_channels != self.out_channels:
            self.conv_out = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in: Tensor) -> Tensor:
        x = _swish(self.norm1(x_in))
        x = self.conv1(x)
        x = _swish(self.norm2(x))
        x = self.conv2(x)
        if self.conv_out is not None:
            x_in = self.conv_out(x_in)
        return x + x_in


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.norm = _group_norm(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        weights = torch.bmm(q, k) * (int(c) ** -0.5)
        weights = F.softmax(weights, dim=2).permute(0, 2, 1)
        v = v.reshape(b, c, h * w)
        h_ = torch.bmm(v, weights).reshape(b, c, h, w)
        return x + self.proj_out(h_)


class Encoder(nn.Module):
    def __init__(self, in_channels, nf, emb_dim, ch_mult, num_res_blocks, resolution, attn_resolutions) -> None:
        super().__init__()
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        blocks: list[nn.Module] = [nn.Conv2d(in_channels, nf, kernel_size=3, stride=1, padding=1)]
        for i in range(len(ch_mult)):
            block_in = nf * in_ch_mult[i]
            block_out = nf * ch_mult[i]
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(block_in, block_out))
                block_in = block_out
            if curr_res in attn_resolutions:
                blocks.append(AttnBlock(block_in))
            if i != len(ch_mult) - 1:
                blocks.append(Downsample(block_in))
                curr_res = curr_res // 2
        blocks.append(ResBlock(block_in, block_in))
        blocks.append(AttnBlock(block_in))
        blocks.append(ResBlock(block_in, block_in))
        blocks.append(_group_norm(block_in))
        blocks.append(nn.Conv2d(block_in, emb_dim, kernel_size=3, stride=1, padding=1))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class Generator(nn.Module):
    def __init__(self, nf, emb_dim, ch_mult, res_blocks, img_size, attn_resolutions) -> None:
        super().__init__()
        num_resolutions = len(ch_mult)
        block_in = nf * ch_mult[-1]
        curr_res = img_size // 2 ** (num_resolutions - 1)
        blocks: list[nn.Module] = [nn.Conv2d(emb_dim, block_in, kernel_size=3, stride=1, padding=1)]
        blocks.append(ResBlock(block_in, block_in))
        blocks.append(AttnBlock(block_in))
        blocks.append(ResBlock(block_in, block_in))
        for i in reversed(range(num_resolutions)):
            block_out = nf * ch_mult[i]
            for _ in range(res_blocks):
                blocks.append(ResBlock(block_in, block_out))
                block_in = block_out
            if curr_res in attn_resolutions:
                blocks.append(AttnBlock(block_in))
            if i != 0:
                blocks.append(Upsample(block_in))
                curr_res = curr_res * 2
        blocks.append(_group_norm(block_in))
        blocks.append(nn.Conv2d(block_in, 3, kernel_size=3, stride=1, padding=1))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class VQAutoEncoder(nn.Module):
    def __init__(self, img_size, nf, ch_mult, quantizer="nearest", res_blocks=2, attn_resolutions=None, codebook_size=1024, emb_dim=256, beta=0.25) -> None:
        super().__init__()
        if attn_resolutions is None:
            attn_resolutions = [16]
        self.encoder = Encoder(3, nf, emb_dim, ch_mult, res_blocks, img_size, attn_resolutions)
        self.quantize = VectorQuantizer(codebook_size, emb_dim, beta)
        self.generator = Generator(nf, emb_dim, ch_mult, res_blocks, img_size, attn_resolutions)


def _calc_mean_std(feat: Tensor, eps: float = 1e-5):
    b, c = feat.shape[:2]
    var = feat.view(b, c, -1).var(dim=2) + eps
    std = var.sqrt().view(b, c, 1, 1)
    mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return mean, std


def _adain(content: Tensor, style: Tensor) -> Tensor:
    style_mean, style_std = _calc_mean_std(style)
    content_mean, content_std = _calc_mean_std(content)
    normalized = (content - content_mean) / content_std
    return normalized * style_std + style_mean


class TransformerSALayer(nn.Module):
    def __init__(self, embed_dim: int, nhead: int = 8, dim_mlp: int = 2048) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=0.0)
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, tgt: Tensor, query_pos: Optional[Tensor] = None) -> Tensor:
        tgt2 = self.norm1(tgt)
        q = k = tgt2 if query_pos is None else tgt2 + query_pos
        tgt2 = self.self_attn(q, k, value=tgt2)[0]
        tgt = tgt + tgt2
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(F.gelu(self.linear1(tgt2)))
        return tgt + tgt2


class FuseSftBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.encode_enc = ResBlock(2 * in_ch, out_ch)
        self.scale = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )
        self.shift = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )

    def forward(self, enc_feat: Tensor, dec_feat: Tensor, w: float = 1) -> Tensor:
        enc_feat = self.encode_enc(torch.cat([enc_feat, dec_feat], dim=1))
        scale = self.scale(enc_feat)
        shift = self.shift(enc_feat)
        return dec_feat + w * (dec_feat * scale + shift)


class CodeFormer(VQAutoEncoder):
    def __init__(
        self,
        dim_embd: int = 512,
        n_head: int = 8,
        n_layers: int = 9,
        codebook_size: int = 1024,
        latent_size: int = 256,
        connect_list=None,
    ) -> None:
        super().__init__(512, 64, [1, 2, 2, 4, 4, 8], "nearest", 2, [16], codebook_size)
        if connect_list is None:
            connect_list = ["32", "64", "128", "256"]
        self.connect_list = list(connect_list)
        self.position_emb = nn.Parameter(torch.zeros(latent_size, dim_embd))
        self.feat_emb = nn.Linear(256, dim_embd)
        self.ft_layers = nn.Sequential(
            *[TransformerSALayer(embed_dim=dim_embd, nhead=n_head, dim_mlp=dim_embd * 2) for _ in range(n_layers)]
        )
        self.idx_pred_layer = nn.Sequential(nn.LayerNorm(dim_embd), nn.Linear(dim_embd, codebook_size, bias=False))
        channels = {"32": 256, "64": 256, "128": 128, "256": 128}
        self.fuse_encoder_block = {"256": 5, "128": 8, "64": 11, "32": 14}
        self.fuse_generator_block = {"32": 9, "64": 12, "128": 15, "256": 18}
        self.fuse_convs_dict = nn.ModuleDict(
            {size: FuseSftBlock(channels[size], channels[size]) for size in self.connect_list}
        )

    def forward(self, x: Tensor, w: float = 0, adain: bool = True):
        enc_feat = {}
        out_list = [self.fuse_encoder_block[size] for size in self.connect_list]
        for index, block in enumerate(self.encoder.blocks):
            x = block(x)
            if index in out_list:
                enc_feat[str(x.shape[-1])] = x.clone()
        lq_feat = x
        pos_emb = self.position_emb.unsqueeze(1).repeat(1, x.shape[0], 1)
        query = self.feat_emb(lq_feat.flatten(2).permute(2, 0, 1))
        for layer in self.ft_layers:
            query = layer(query, query_pos=pos_emb)
        logits = self.idx_pred_layer(query).permute(1, 0, 2)
        soft = F.softmax(logits, dim=2)
        _, top_idx = torch.topk(soft, 1, dim=2)
        quant = self.quantize.get_codebook_feat(top_idx, shape=[x.shape[0], 16, 16, 256]).detach()
        if adain:
            quant = _adain(quant, lq_feat)
        x = quant
        fuse_list = [self.fuse_generator_block[size] for size in self.connect_list]
        for index, block in enumerate(self.generator.blocks):
            x = block(x)
            if index in fuse_list and w > 0:
                size = str(x.shape[-1])
                x = self.fuse_convs_dict[size](enc_feat[size].detach(), x, w)
        return x, logits
