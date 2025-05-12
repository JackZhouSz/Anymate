F"""
Adapted from: https://github.com/openai/openai/blob/55363aa496049423c37124b440e9e30366db3ed6/orc/orc/diffusion/vit.py
"""

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn

from einops import repeat
from Anymate.utils.diffusion_utils import *
from ThirdParty.michelangelo.models.modules.transformer_blocks import Transformer, ResidualCrossAttentionBlock

from diffusers import DDPMScheduler, DDIMScheduler
from sklearn.cluster import DBSCAN

def init_linear(l, stddev):
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)

class projection_transformer(nn.Module):
    def __init__(self, num_latents=16, width = 16, heads=8, dtype = torch.float32):
        super().__init__()
        self.num_latents = num_latents
        self.query = nn.Parameter(torch.randn((num_latents, width), dtype=dtype) * 0.02)

        self.cross_attn = ResidualCrossAttentionBlock(
            device= 'cuda',
            dtype=dtype,
            width=width,
            heads=heads,
            init_scale=0.25,
            qkv_bias=True,
            flash=False,
        )
        self.output_proj = nn.Linear(width, width,dtype=dtype)
        
    def forward(self, latents):
        bs = latents.shape[0]
        query = repeat(self.query, "m c -> b m c", b=bs)
        embed = self.cross_attn(query, latents)
        logits = self.output_proj(embed)

        return logits

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].to(timesteps.dtype) * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, dtype=dtype)
        self.c_proj = nn.Linear(width, width, dtype=dtype)
        self.attention = QKVMultiheadAttention(dtype=dtype, heads=heads, n_ctx=n_ctx)
        init_linear(self.c_qkv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        x = self.c_qkv(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x

class MLP(nn.Module):
    def __init__(self, *, dtype: torch.dtype, width: int, init_scale: float):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4,  dtype=dtype)
        self.c_proj = nn.Linear(width * 4, width, dtype=dtype)
        self.gelu = nn.GELU()
        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class QKVMultiheadAttention(nn.Module):
    def __init__(self, *, dtype: torch.dtype, heads: int, n_ctx: int):
        super().__init__()
        self.dtype = dtype
        self.heads = heads
        self.n_ctx = n_ctx

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)
        weight = torch.einsum(
            "bthc,bshc->bhts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        wdtype = weight.dtype
        weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
        return torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)

class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float = 1.0,
    ):
        super().__init__()

        self.attn = MultiheadAttention(
            dtype=dtype,
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale,
        )
        self.ln_1 = nn.LayerNorm(width, dtype=dtype)
        self.mlp = MLP(dtype=dtype, width=width, init_scale=init_scale)
        self.ln_2 = nn.LayerNorm(width, dtype=dtype)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        init_scale: float = 0.25,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        init_scale = init_scale * math.sqrt(1.0 / width)
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    dtype=dtype,
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    init_scale=init_scale,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x

class PointDiffusionTransformer(nn.Module):
    def __init__(
        self,
        *,
        dtype: torch.dtype,
        input_channels: int = 3,
        output_channels: int = 3,
        n_ctx: int = 1024,
        width: int = 768,
        layers: int = 12,
        heads: int = 8,
        init_scale: float = 0.25,
        time_token_cond: bool = True,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_ctx = n_ctx
        self.time_token_cond = time_token_cond
        self.time_embed = MLP(
            dtype=dtype, width=width, init_scale=init_scale * math.sqrt(1.0 / width)
        )
        self.ln_pre = nn.LayerNorm(width, dtype=dtype)
        self.backbone = Transformer(
            dtype=dtype,
            n_ctx=n_ctx + int(time_token_cond),
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
        )
        self.ln_post = nn.LayerNorm(width,dtype=dtype)
        self.input_proj = nn.Linear(input_channels, width, dtype=dtype)
        self.output_proj = nn.Linear(width, output_channels,dtype=dtype)
        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.bias.zero_()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        :param x: an [N x C x T] tensor.
        :param t: an [N] tensor.
        :return: an [N x C' x T] tensor.
        """
        assert x.shape[-1] == self.n_ctx
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        return self._forward_with_cond(x, [(t_embed, self.time_token_cond)])

    def _forward_with_cond(
        self, x: torch.Tensor, cond_as_token: List[Tuple[torch.Tensor, bool]]
    ) -> torch.Tensor:
        h = self.input_proj(x.permute(0, 2, 1))  # NCL -> NLC
        for emb, as_token in cond_as_token:
            if not as_token:
                h = h + emb[:, None]
        extra_tokens = [
            (emb[:, None] if len(emb.shape) == 2 else emb)
            for emb, as_token in cond_as_token
            if as_token
        ]
        if len(extra_tokens):
            h = torch.cat(extra_tokens + [h], dim=1)

        h = self.ln_pre(h)
        h = self.backbone(h)
        h = self.ln_post(h)
        if len(extra_tokens):
            h = h[:, sum(h.shape[1] for h in extra_tokens) :]
        h = self.output_proj(h)
        return h.permute(0, 2, 1)

class Pointe_Diffusion(PointDiffusionTransformer):
    '''
    input: data: data dict
            x: [N x C x T] tensor
            t: [N] tensor
    init:
            n_ctx: int = 1024: context length
    '''
    def __init__(
        self,
        *,
        device = 'cuda',
        dtype = torch.float32,
        encoder = 'miche',
        n_ctx: int = 1024,
        token_cond: bool = True,
        cond_drop_prob: float = 0.1,
        fix_emb: bool = False,
        
        **kwargs,
    ):
        super().__init__(dtype=dtype, n_ctx=n_ctx + int(token_cond), **kwargs)
        self.n_ctx = n_ctx
        self.token_cond = token_cond
        # self.proj_transformer = projection_transformer(**kwargs)
        self.encoder_name = encoder
        self.cond_drop_prob = cond_drop_prob
        self.fix_emb = fix_emb
        self.dtype = dtype
        self.inference = False
    def cached_model_kwargs(self, batch_size: int, model_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        with torch.no_grad():
            return dict(embeddings=self.clip(batch_size, **model_kwargs))
        
    def inference_mode(self,eps=0.03):
        self.inference = True
        
    def forward_func(
        self,
        latent: torch.Tensor,
        data,
        device='cuda', 
        downsample = False,
        **kwargs,
    ):
        t = kwargs['timesteps'].to(latent.device)
        x = kwargs['noisy_joints'].to(latent.device)
        assert x.shape[-1] == self.n_ctx, f"x shape: {x.shape}, n_ctx: {self.n_ctx}"
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))

        if self.training:
            mask = torch.rand(size=[len(x)]) >= self.cond_drop_prob
            latent = latent * mask[:,None,None].to(latent.device)

        latent = [(latent, self.token_cond), (t_embed, self.time_token_cond)]
        return self._forward_with_cond(x, latent)

    def forward(self, latent, data, device='cuda', downsample = False, **kwargs):
        if self.inference == False:
            return self.forward_func(latent, data, device, downsample, **kwargs)
        else:
            generator=torch.Generator(device='cpu')
            scheduler = DDIMScheduler(100)
            scheduler.set_timesteps(100)
            points_shape = [1, self.n_ctx, 3]

            points_noise = randn_tensor(points_shape, generator=generator)
            points = points_noise.permute(0, 2, 1).to(latent.device)
            for t in scheduler.timesteps:
                with torch.no_grad():
                    time_steps = torch.ones(1, 1, dtype=torch.long) * t
                    model_output = self.forward_func(latent, data, noisy_joints=points, timesteps = time_steps)

                    points = scheduler.step(model_output, t, points, generator=generator).prev_sample
            points = points.permute(0, 2, 1).cpu()
            assert points.shape[0] == 1, "Inference mode only supports batch size 1"
            joints = points[0].detach().cpu().numpy()
            clustering = DBSCAN(eps=0.05, min_samples=1).fit(joints)
            cluster_centers = []
            for cluster in set(clustering.labels_):
                cluster_centers.append(joints[clustering.labels_ == cluster].mean(axis=0))
            return cluster_centers

class Cross_Attention_Diffusion(nn.Module):
    def __init__(self,
                 input_channels=3, output_channels=3,
                 num_z=16, num_x=1024, z_dim=768, x_dim=512, 
                 num_blocks=6, num_compute_layers=4, num_heads=8, 
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,num_latents=16,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 use_projection = True,):
        super().__init__()
        self.use_projection = use_projection
        self.device = device
        self.num_z = num_z
        self.num_x = num_x
        self.z_dim = z_dim
        if use_projection:
            self.proj_transformer = projection_transformer(num_latents=num_latents, width=z_dim, heads=num_heads)
        self.prev_latent = nn.Parameter(torch.zeros(1, self.num_z + num_latents + 1, z_dim))
        self.inference = False

        self.input_proj = nn.Linear(input_channels, x_dim)
        self.ln_pre = nn.LayerNorm(x_dim)
        self.z_init = nn.Parameter(torch.zeros(1, num_z, z_dim))

        mlp_hidden_dim = int(z_dim * mlp_ratio)
        self.time_embed = Mlp(in_features=z_dim, hidden_features=mlp_hidden_dim)

        self.latent_mlp = Mlp(in_features=z_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ln_latent = nn.LayerNorm(z_dim)
        self.blocks = nn.ModuleList([
            RCW_Block(z_dim, x_dim, num_compute_layers=num_compute_layers, 
                      num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                      drop=drop, attn_drop=attn_drop, drop_path=drop_path, 
                      act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(num_blocks)
        ])

        # output blocks
        self.ln_post = nn.LayerNorm(x_dim)
        self.output_proj = nn.Linear(x_dim, output_channels)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.z_init, std=.02)
        
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

        nn.init.constant_(self.ln_latent.weight, 0)
        nn.init.constant_(self.ln_latent.bias, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def inference_mode(self,eps=0.03):
        self.inference = True
        
    def forward_func(self, latent, data, device='cuda', downsample = False, **kwargs):
        """
        Forward pass of the model.

        Parameters:
        x: [B, num_x, C_in]
        t: [B]
        cond: [B, num_cond, C_latent]
        prev_latent: [B, num_z + num_cond + 1, C_latent]

        Returns:
        x_denoised: [B, num_x, C_out]
        z: [B, num_z + num_cond + 1, C_latent]
        """
        t = kwargs['timesteps'].to(latent.device)
        x = kwargs['noisy_joints'].to(latent.device)
        x = x.permute(0, 2, 1)
        B, num_x, _ = x.shape
        if self.use_projection:
            latent = self.proj_transformer(latent)
        assert num_x == self.num_x, f"x shape: {x.shape}, num_x: {self.num_x}"
        # if prev_latent is not None:
        #     _, num_z, _ = prev_latent.shape
        #     assert num_z == self.num_z + num_cond + 1
        # else:
        #     prev_latent = torch.zeros(B, self.num_z + num_cond + 1, self.z_dim).to(x.device)
        
        # timestep embedding, [B, 1, z_dim]
        t_embed = self.time_embed(timestep_embedding(t, self.z_dim))
        if t_embed.dim() == 2:
            t_embed = t_embed.unsqueeze(1)

        # project x -> [B, num_x, C_x]
        x = self.input_proj(x)
        x = self.ln_pre(x)

        # latent self-conditioning
        z = self.z_init.repeat(B, 1, 1) # [B, num_z, z_dim
        z = torch.cat([z, latent, t_embed], dim=1) # [B, num_z + num_cond + 1, z_dim]
        prev_latent = self.prev_latent + self.latent_mlp(self.prev_latent.detach())
        z = z + (self.ln_latent(prev_latent))

        # compute
        for blk in self.blocks:
            z, x = blk(z, x)
        
        # output proj
        x = self.ln_post(x)
        x_denoised = self.output_proj(x)
        return x_denoised.permute(0, 2, 1)

    def forward(self, latent, data, device='cuda', downsample = False, **kwargs):
        if self.inference == False:
            return self.forward_func(latent, data, device, downsample, **kwargs)
        else:
            generator=torch.Generator(device='cpu')
            scheduler = DDIMScheduler(100)
            scheduler.set_timesteps(100)
            points_shape = [1, self.num_x, 3]

            points_noise = randn_tensor(points_shape, generator=generator)
            points = points_noise.permute(0, 2, 1).to(latent.device)
            for t in scheduler.timesteps:
                with torch.no_grad():
                    time_steps = torch.ones(1, 1, dtype=torch.long) * t
                    time_steps = time_steps.to(latent.device)
                    model_output = self.forward_func(latent, data, noisy_joints=points, timesteps = time_steps)

                    points = scheduler.step(model_output, t, points, generator=generator).prev_sample
            points = points.permute(0, 2, 1).cpu()
            assert points.shape[0] == 1, "Inference mode only supports batch size 1"
            joints = points[0].detach().cpu().numpy()
            clustering = DBSCAN(eps=0.05, min_samples=1).fit(joints)
            cluster_centers = []
            for cluster in set(clustering.labels_):
                cluster_centers.append(joints[clustering.labels_ == cluster].mean(axis=0))
            return cluster_centers
    