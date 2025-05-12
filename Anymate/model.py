import torch
import torch.nn as nn
from ThirdParty.michelangelo.utils.misc import get_config_from_file, instantiate_from_config
from ThirdParty.PointLLM.pointllm.model.pointllm import PointLLMLlamaForCausalLM
from ThirdParty.michelangelo.models.modules.distributions import DiagonalGaussianDistribution
from ThirdParty.michelangelo.models.modules.embedder import components_from_spherical_harmonics
from Anymate.utils.diffusion_encoder import TransformerEncoder
from Anymate.models.joint import TransformerDecoder, ImplicitTransformerDecoder, TriPlaneDecoder
from Anymate.models.conn import AttendjointsDecoder_con_combine, AttendjointsDecoder_con_token
from Anymate.models.skin import AttendjointsDecoder_combine, AttendjointsDecoder_multi
from Anymate.models.diffusion import Pointe_Diffusion, Cross_Attention_Diffusion

class Encoder(nn.Module):
    def __init__(self, 
                 only_embed = True,
                 config_path = './ThirdParty/michelangelo/configs/aligned_shape_latents/shapevae-256.yaml',
                 ckpt_path = './ThirdParty/michelangelo/checkpoints/aligned_shape_latents/shapevae-256.ckpt',
                 num_latents = 257,
                 device = 'cuda'):

        super().__init__()

        model_config = get_config_from_file(config_path)
        if hasattr(model_config, "model"):
            model_config = model_config.model

        if ckpt_path is not None:
            model = instantiate_from_config(model_config, ckpt_path=ckpt_path)
        else:
            model = instantiate_from_config(model_config)
            model.model.shape_model.encoder.num_latents = num_latents
            model.model.shape_model.encoder.query = nn.Parameter(torch.randn((num_latents, 768), device=device, dtype=torch.float32) * 0.02)

        self.shape_projection = model.model.shape_projection
        self.encoder = model.model.shape_model.encoder
        self.normal_embedder = components_from_spherical_harmonics
        old_linear_proj = self.encoder.input_proj
        self.encoder.input_proj = nn.Linear(old_linear_proj.in_features + 25, old_linear_proj.out_features)
        self.encoder.input_proj.weight.data[:, :old_linear_proj.in_features] = old_linear_proj.weight.data[:, :old_linear_proj.in_features].clone()
        self.encoder.input_proj.bias.data = old_linear_proj.bias.data.clone()
        if not only_embed:
            self.embed_dim = model.model.shape_model.embed_dim
            self.pre_kl = model.model.shape_model.pre_kl
            self.post_kl = model.model.shape_model.post_kl
            self.transformer = model.model.shape_model.transformer
        

    def encode_latents(self,
                       pc: torch.FloatTensor,
                       feats = None):

        feats_embed = self.normal_embedder(feats)
        feats = torch.cat([feats, feats_embed], dim=-1)

        x, _ = self.encoder(pc, feats)

        shape_embed = x[:, 0]
        latents = x[:, 1:]

        return shape_embed, latents


    def encode_shape_embed(self, surface, return_latents: bool = False):
        """

        Args:
            surface (torch.FloatTensor): [bs, n, 3 + c]
            return_latents (bool):

        Returns:
            x (torch.FloatTensor): [bs, projection_dim]
            shape_latents (torch.FloatTensor): [bs, m, d]
        """

        pc = surface[..., 0:3]
        feats = surface[..., 3:]

        shape_embed, shape_latents = self.encode_latents(pc, feats)
        x = shape_embed @ self.shape_projection

        if return_latents:
            return x, shape_latents
        else:
            return x


    def encode_kl_embed(self, latents: torch.FloatTensor, sample_posterior: bool = True):
        posterior = None
        if self.embed_dim > 0:
            moments = self.pre_kl(latents)
            posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)

            if sample_posterior:
                kl_embed = posterior.sample()
            else:
                kl_embed = posterior.mode()
        else:
            kl_embed = latents

        return kl_embed, posterior


    def decode(self, latents: torch.FloatTensor):
        latents = self.post_kl(latents)
        return self.transformer(latents)


class EncoderDecoder(nn.Module):
    def __init__(self, 
                 decoder = 'mlp',
                 encoder = 'miche',
                 config_path = './ThirdParty/michelangelo/configs/aligned_shape_latents/shapevae-256.yaml',
                 ckpt_path = './ThirdParty/michelangelo/checkpoints/aligned_shape_latents/shapevae-256.ckpt',
                 load_encoder = '',
                 load_bert = True,
                 num_joints = 96,
                 out_channels = 3,
                 width = 768,
                 device = 'cuda',
                 dtype = torch.float32,
                 heads = 12,
                 init_scale: float = 0.25,
                 flash = False,
                 use_checkpoint = False,
                 qkv_bias = False,
                 separate = False,
                 **kwargs):

        super().__init__()
        self.decoder_name = decoder
        self.encoder_name = encoder
        self.dtype = dtype
        self.load_encoder = load_encoder
        self.load_bert = load_bert
        if decoder == 'transformer_latent':
            self.only_embed = False
            self.return_latents = True
            self.decoder = TransformerDecoder(
                num_latents = num_joints,
                out_channels = out_channels,
                width = width,
                device = device,
                dtype = dtype,
                heads = heads,
                init_scale = init_scale,
                flash = flash,
                use_checkpoint = use_checkpoint,
                qkv_bias = qkv_bias
            )
        elif decoder == 'implicit_transformer':
            self.only_embed = False
            self.return_latents = True
            self.decoder = ImplicitTransformerDecoder(
                device = device,
                dtype = dtype,
                num_latents = 257,
                out_channels = 1,
                width = width,
                heads = heads,
                init_scale = init_scale,
                flash = flash,
                use_checkpoint = use_checkpoint,
                qkv_bias = qkv_bias
            )
        elif decoder == 'triplane': #consider add these parameters to config
            self.only_embed = True
            self.return_latents = False
            self.decoder = TriPlaneDecoder(
                z_dim = 768,
                c_dim = 0,
                w_dim = 768,
                mapping_kwargs = {'num_layers': 2},
                synthesis_kwargs = {'num_fp16_res': 0, 'conv_clamp': None, 'fused_modconv_default': 'inference_only'}
            )
            
        elif decoder == 'Pointe_Diffusion':
            self.only_embed = False
            self.return_latents = True
            self.decoder = Pointe_Diffusion(**kwargs)
            
        elif decoder == 'Cross_Attention_Diffusion':
            self.only_embed = False
            self.return_latents = True
            self.decoder = Cross_Attention_Diffusion(**kwargs)
            
        elif decoder == 'attendjoints_combine':
            self.only_embed = False
            self.return_latents = True
            self.decoder = AttendjointsDecoder_combine(
                width = width,
                device = device,
                dtype = dtype,
                heads = heads,
                init_scale = init_scale,
                flash = flash,
                use_checkpoint = use_checkpoint,
                separate = separate,
                qkv_bias = qkv_bias
            )
        elif decoder == 'attendjoints_multi':
            self.only_embed = False
            self.return_latents = True
            self.decoder = AttendjointsDecoder_multi(
                width = width,
                device = device,
                dtype = dtype,
                heads = heads,
                init_scale = init_scale,
                flash = flash,
                use_checkpoint = use_checkpoint,
                qkv_bias = qkv_bias,
                separate=separate
            )   
        elif decoder == 'attendjoints_con_combine':
            self.only_embed = False
            self.return_latents = True
            self.decoder = AttendjointsDecoder_con_combine(
                width = width,
                device = device,
                dtype = dtype,
                heads = heads,
                init_scale = init_scale,
                flash = flash,
                use_checkpoint = use_checkpoint,
                qkv_bias = qkv_bias
            )
        elif decoder == 'attendjoints_con_token':
            self.only_embed = False
            self.return_latents = True
            self.decoder = AttendjointsDecoder_con_token(
                width = width,
                device = device,
                dtype = dtype,
                heads = heads,
                init_scale = init_scale,
                flash = flash,
                use_checkpoint = use_checkpoint,
                qkv_bias = qkv_bias,
                separate = separate
            )
        
        if encoder == 'miche':
            if not self.load_encoder:
                self.encoder = Encoder(only_embed=self.only_embed, config_path=config_path, ckpt_path=ckpt_path, device=device)
            else:
                self.encoder = Encoder(only_embed=self.only_embed, config_path=config_path, ckpt_path=None, device=device)
                try:
                    print("=> loading encoder checkpoint '{}'".format(self.load_encoder))
                    checkpoint = torch.load(self.load_encoder, map_location='cpu')
                    state_dict = {k[8:]: v for k, v in checkpoint['state_dict'].items() if k.startswith('encoder')}
                    self.encoder.load_state_dict(state_dict)
                    print("=> loaded encoder checkpoint '{}'".format(self.load_encoder))
                except:
                    print("=> no encoder checkpoint found at '{}'".format(self.load_encoder))
                if self.load_encoder:
                    self.point_proj = nn.Sequential(
                        nn.Linear(768, 768, dtype=dtype),
                        nn.GELU(),
                        nn.Linear(768, 768, dtype=dtype),
                    )
        
        if encoder == 'bert':
            if self.load_bert == 'checkpoint':
                model_name = 'RunsenXu/PointLLM_7B_v1.2'
                model = PointLLMLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=False, use_cache=True, torch_dtype=dtype)
                self.encoder = model.model.point_backbone.to(device)
                model = None
            else:
                from ThirdParty.PointLLM.pointllm.model import PointTransformer
                from ThirdParty.PointLLM.pointllm.utils import cfg_from_yaml_file
                import os
                # address of config file, in the same dir of this file
                point_bert_config_name = "PointTransformer_8192point_2layer" # * default for v1.2, v1.1 uses PointTransformer_base_8192point.yaml
                point_bert_config_addr = os.path.join("./ThirdParty/PointLLM/pointllm/model/pointbert/PointTransformer_8192point_2layer.yaml")
                print(f"Loading PointBERT config from {point_bert_config_addr}.")
                point_bert_config = cfg_from_yaml_file(point_bert_config_addr)
                point_bert_config.model.point_dims = 6
                use_max_pool = getattr(point_bert_config.model, "use_max_pool", False) # * default is false
                self.encoder = PointTransformer(point_bert_config.model, use_max_pool=use_max_pool).to(device)
                
            if self.return_latents:
                self.point_proj = nn.Sequential(
                    nn.Linear(384, 512, dtype=dtype),
                    nn.GELU(),
                    nn.Linear(512, 512, dtype=dtype),
                    nn.GELU(),
                    nn.Linear(512, 768, dtype=dtype)
                )
            else:
                self.point_proj = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(384, 512, dtype=dtype),
                        nn.GELU(),
                        nn.Linear(512, 512, dtype=dtype),
                        nn.GELU(),
                        nn.Linear(512, 768, dtype=dtype)
                    ),
                    nn.Linear(513, 1, dtype=dtype)
                ])
        if encoder == 'transformer':
            self.points_cloud_embed = nn.Linear(
            768, 768, device=device, dtype=dtype
        )
            self.encoder = TransformerEncoder(device=device,dtype=dtype, num_latents=kwargs['num_latents'])
            


    def encode(self, data, device='cuda'):
        assert self.encoder_name in ['miche', 'bert', 'transformer'], f'Encoder {self.encoder_name} not supported'
        if self.encoder_name == 'miche':
            surface = data['points_cloud'].to(self.dtype).to(device)

            # encoding
            shape_embed, shape_latents = self.encoder.encode_shape_embed(surface, return_latents=True)  # ShapeAsLatentPerceiver.encode_latents(): encoder

            if self.only_embed:
                if self.return_latents:
                    if self.load_encoder:
                        return self.point_proj(torch.cat([shape_embed.unsqueeze(1), shape_latents], dim=1))
                    return torch.cat([shape_embed.unsqueeze(1), shape_latents], dim=1)  # torch.Size([bs, 257, 768]
                return shape_embed  # shape_embed: torch.Size([bs, 768])
            
            shape_zq, posterior = self.encoder.encode_kl_embed(shape_latents)  # ShapeAsLatentPerceiver.encode_kl_embed(): pre_kl + DiagonalGaussianDistribution()
            # shape_zq, posterior = self.encoder.encode_kl_embed(shape_latents, sample_posterior=False)  # not sample
            # pretrained weight has 0 +- 0.7 mean and 0.5 +- 0.5 std
            # trained weight has 0 +- 1.8 mean and 0.1 +- 0.1 std
            # generally okay

            latents = self.encoder.decode(shape_zq)  # ShapeAsLatentPerceiver.decode(): post_kl + transformer

            if not self.return_latents:
                latents = torch.cat([shape_latents, latents], dim=1)  # torch.Size([bs, 512, 768])

            if self.load_encoder:
                return self.point_proj(torch.cat([shape_embed.unsqueeze(1), latents], dim=1))
            return torch.cat([shape_embed.unsqueeze(1), latents], dim=1)  # torch.Size([bs, 257 / 513, 768])
        
        if self.encoder_name == 'bert':
            points = data['points_cloud'].to(self.dtype).to(device)
            points = points[:, :, :3] / 2
            points = torch.cat([points, torch.zeros_like(points)], dim=-1)
            points = self.encoder(points)

            if self.return_latents:
                points = self.point_proj(points)
            else:
                points = self.point_proj[0](points)
                points = self.point_proj[1](points.permute(0, 2, 1)).squeeze(-1)
            return points
        
        if self.encoder_name == 'transformer':
            points = data['points_cloud'].to(self.dtype).to(device)
            cond = self.encoder.encode_pc(points)
            cond = self.points_cloud_embed(cond)
            return cond
    
    def forward(self, data, device='cuda', downsample=False, **kwargs):
        latents = self.encode(data, device)
        # print('latents shape', latents.shape)
        
        logits = self.decoder(latents, data, device=device, downsample=downsample,**kwargs)

        return logits