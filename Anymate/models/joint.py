import torch
import torch.nn as nn
from ThirdParty.michelangelo.models.modules.embedder import FourierEmbedder
from ThirdParty.michelangelo.models.modules.transformer_blocks import ResidualCrossAttentionBlock
from ThirdParty.eg3d.training.networks_stylegan2 import Generator as StyleGAN2Backbone
from ThirdParty.eg3d.training.networks_stylegan2 import FullyConnectedLayer
from Anymate.utils.vol_utils import get_co, sample_from_planes, generate_planes
from einops import repeat
from sklearn.cluster import DBSCAN
from Anymate.utils.vol_utils import extract_keypoints

class TransformerDecoder(nn.Module):
    def __init__(self, 
                 num_latents = 96,
                 num_kv_latents = 257,
                 out_channels = 3,
                 width = 768,
                 layers = 7,
                 device = 'cuda',
                 dtype = torch.float32,
                 heads = 12,
                 init_scale: float = 0.25,
                 flash = False,
                 use_checkpoint = False,
                 qkv_bias = False):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.num_latents = num_latents
        self.inference = False
        self.eps = 0.03

        self.query = nn.Parameter(torch.randn((num_latents, width), device=device, dtype=dtype) * 0.02)

        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            device=device,
            dtype=dtype,
            n_data=num_kv_latents,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash
        )

        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, out_channels, device=device, dtype=dtype)
        
    def inference_mode(self, eps=0.03, min_samples=1):
        self.inference = True
        self.eps = eps
        self.min_samples = min_samples
        
    def forward(self, latents, data=None, device='cuda', downsample=False, dtype=torch.float32):
        
        bs = latents.shape[0]
        query = repeat(self.query, "m c -> b m c", b=bs)
        logits = self.cross_attn_decoder(query, latents)
        logits = self.ln_post(logits)
        logits = self.output_proj(logits)
        if self.inference:
            assert logits.shape[0] == 1, "Inference mode only supports batch size 1"
            joints = logits[0].detach().cpu().numpy()
            clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(joints)
            cluster_centers = []
            for cluster in set(clustering.labels_):
                cluster_centers.append(joints[clustering.labels_ == cluster].mean(axis=0))
            return cluster_centers
        return logits
    

class ImplicitTransformerDecoder(nn.Module):

    def __init__(self, *,
                 device = 'cuda',
                 dtype = torch.float32,
                 num_latents = 257,
                 out_channels = 1,
                 width = 768,
                 heads = 12,
                 num_freqs: int = 8,
                 include_pi: bool = True,
                 init_scale: float = 0.25,
                 qkv_bias: bool = False,
                 flash: bool = False,
                 use_checkpoint: bool = False):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)
        self.inference = False

        self.query_proj = nn.Linear(self.fourier_embedder.out_dim, width, device=device, dtype=dtype)

        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            device=device,
            dtype=dtype,
            n_data=num_latents,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash
        )

        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, out_channels, device=device, dtype=dtype)

        # self.queries = get_vol().to(device)
        
    def inference_mode(self):
        self.inference = True
        
    def forward(self, latents: torch.FloatTensor, data=None, device='cuda', downsample=False):
        bs = latents.shape[0]
        # queries = repeat(self.queries, "m c -> b m c", b=bs)
        out = []
        for b in range(bs):
            queries = get_co(data['vox'][b]).to(device).unsqueeze(0)
            if downsample and data['vox'][b].shape[0] > 50000:
                # random sample
                idx = torch.randperm(data['vox'][b].shape[0])[:50000]
                queries = queries[:, idx]
            queries = self.query_proj(self.fourier_embedder(queries))
            x = self.cross_attn_decoder(queries, latents[b:b+1])
            x = self.ln_post(x)
            x = self.output_proj(x)
            if downsample and data['vox'][b].shape[0] > 50000:
                out.append((x.squeeze(0), idx))
            else:
                out.append(x.squeeze(0))
        if self.inference:
            assert len(out) == 1, "Inference mode only supports batch size 1"
            return extract_keypoints(out[0], data['vox'][0])
            
        return out
    

class TriPlaneDecoder(torch.nn.Module):
    def __init__(self,
        z_dim = 768,                      # Input latent (Z) dimensionality.
        c_dim = 0,                      # Conditioning label (C) dimensionality.
        w_dim = 768,                      # Intermediate latent (W) dimensionality.
        # img_resolution,             # Output resolution.
        # img_channels,               # Number of output color channels.
        # sr_num_fp16_res     = 0,
        mapping_kwargs      = {'num_layers': 2},   # Arguments for MappingNetwork.
        # rendering_kwargs    = {},
        # sr_kwargs = {},
        synthesis_kwargs    = {'num_fp16_res': 0, 'conv_clamp': None, 'fused_modconv_default': 'inference_only'},         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.w_dim=w_dim
        # self.img_resolution=img_resolution
        # self.img_channels=img_channels
        # self.renderer = ImportanceRenderer()
        # self.ray_sampler = RaySampler()
        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32*3, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        # self.superresolution = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        self.decoder = OSGDecoder(32, {'decoder_output_dim': 0})
        self.inference = False
        # self.neural_rendering_resolution = 64
        # self.rendering_kwargs = rendering_kwargs
    
        self._last_planes = None
        self.plane_axes = generate_planes()
    
    def mapping(self, z, c=None, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        # if self.rendering_kwargs['c_gen_conditioning_zero']:
        #         c = torch.zeros_like(c)
        # return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.backbone.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def synthesis(self, ws, c=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # cam2world_matrix = c[:, :16].view(-1, 4, 4)
        # intrinsics = c[:, 16:25].view(-1, 3, 3)

        # if neural_rendering_resolution is None:
        #     neural_rendering_resolution = self.neural_rendering_resolution
        # else:
        #     self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        # ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        # N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three 32-channel planes
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return planes

        # Perform volume rendering
        feature_samples, depth_samples, weights_samples = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image}
    
    def sample(self, coordinates, directions, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def sample_mixed(self, coordinates, directions, ws, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        planes = self.backbone.synthesis(ws, update_emas = update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def inference_mode(self):
        self.inference = True
        
    def forward(self, z, data=None, device='cuda', downsample=False, c=None, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        assert z.shape[-1] == self.z_dim
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        planes = self.synthesis(ws, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)
        bs = planes.shape[0]
        logits = []
        for b in range(bs):
            queries = get_co(data['vox'][b]).to(device).unsqueeze(0)
            if downsample and data['vox'][b].shape[0] > 50000:
                # random sample
                idx = torch.randperm(data['vox'][b].shape[0])[:50000]
                queries = queries[:, idx]
            out = sample_from_planes(self.plane_axes.to(device), planes[b:b+1], queries)
            out = self.decoder(out)
            if downsample and data['vox'][b].shape[0] > 50000:
                logits.append((out.squeeze(0), idx))
            else:
                logits.append(out.squeeze(0))
        if self.inference:
            assert len(logits) == 1, "Inference mode only supports batch size 1"
            return extract_keypoints(logits[0], data['vox'][0])
        return logits


class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'])
        )
        
    def forward(self, sampled_features, ray_directions=None):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        return x
        rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}