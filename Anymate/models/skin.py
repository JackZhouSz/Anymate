import torch
import torch.nn as nn
from ThirdParty.michelangelo.models.modules.transformer_blocks import ResidualCrossAttentionBlock, Transformer
from ThirdParty.michelangelo.models.modules.embedder import components_from_spherical_harmonics, FourierEmbedder
from einops import repeat, rearrange

class AttendjointsDecoder_combine(nn.Module):
    def __init__(self, 
                 width = 768,
                 layers = 2,
                 device = 'cuda',
                 dtype = torch.float32,
                 heads = 12,
                 init_scale: float = 0.25,
                 flash = False,
                 use_checkpoint = False,
                 qkv_bias = False,
                 num_freqs: int = 8,
                 include_pi: bool = True,
                 separate = False,
                 use_mask = True,
                 use_bone = True,
                 inference= False):

        super().__init__()
        self.inference = inference
        self.use_checkpoint = use_checkpoint
        self.separate = separate
        self.use_mask = use_mask
        # self.num_latents = num_latents

        # self.query = nn.Parameter(torch.randn((num_latents, width), device=device, dtype=dtype) * 0.02)

        self.normal_embedder = components_from_spherical_harmonics
        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)
        self.bone_proj = None if not use_bone else nn.Linear(self.fourier_embedder.out_dim * 2, width, device=device, dtype=dtype)
        self.use_bone = use_bone
        
        if not self.separate:
            self.co_proj = nn.Linear(self.fourier_embedder.out_dim, width, device=device, dtype=dtype)
            self.normal_proj = nn.Linear(25, width, device=device, dtype=dtype)
        else:
            self.pc_proj = nn.Linear(self.fourier_embedder.out_dim + 25, width, device=device, dtype=dtype)


        # self.proj_attn = nn.Linear(width, width, device=device, dtype=dtype)

        self.cross_attn = nn.ModuleList([ResidualCrossAttentionBlock(
            device=device,
            dtype=dtype,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
        ) for _ in range(layers)])

        self.cross_attn_joint = nn.ModuleList([ResidualCrossAttentionBlock(
            device=device,
            dtype=dtype,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
        ) for _ in range(layers)])

        # self.joint_embed_proj = nn.ModuleList([nn.Linear(width, width, device=device, dtype=dtype) for _ in range(layers)])


        self.q_proj = nn.Linear(width, width, device=device, dtype=dtype)
        self.k_proj = nn.Linear(width, width, device=device, dtype=dtype)
        self.ln_1 = nn.LayerNorm(width, device=device, dtype=dtype)
        self.ln_2 = nn.LayerNorm(width, device=device, dtype=dtype)

        # self.last_cross_attn = ResidualCrossAttentionBlock(
        #     device=device,
        #     dtype=dtype,
        #     width=width,
        #     heads=heads,
        #     init_scale=init_scale,
        #     qkv_bias=qkv_bias,
        #     flash=flash,
        # )
        # self.mlp = MLP(device=device, dtype=dtype, width=width, init_scale=init_scale)
        # self.output_proj = nn.Linear(width, 1, device=device, dtype=dtype)
    
    def forward(self, latents, data=None, device='cuda', downsample=None, dtype=torch.float32):
        joints = data['bones'].to(device) if self.use_bone else data['joints'].to(device)
        max_joints = max(data['bones_num']) if self.use_bone else max(data['joints_num'])
        mask = data['bones_mask'].to(device) if self.use_bone else data['joints_mask']
        
        pc = data['vertices'][..., 0:3].to(device) if self.inference else data['points_cloud'][..., 0:3].to(device)
        feats = data['vertices'][..., 3:].to(device) if self.inference else data['points_cloud'][..., 3:].to(device)
        
        if downsample and not self.inference:
            # random sample
            idx = torch.randperm(pc.shape[1])[:downsample].to(device)
            pc = pc[:, idx]
            feats = feats[:, idx]

        # Embed the input data
        co_embeds = self.fourier_embedder(pc)
        if not self.separate:
            co_embeds = self.co_proj(co_embeds)

        if self.use_bone:
            # joints_fourier = torch.cat((self.fourier_embedder(joints[:,:max_joints*2:2, :3]), self.fourier_embedder(joints[:,1:max_joints*2:2, :3])), dim=-1)
            joints_fourier = torch.cat((self.fourier_embedder(joints[:,:max_joints,:3]), self.fourier_embedder(joints[:,:max_joints, 3:])), dim=-1)
        else:
            joints_fourier = self.fourier_embedder(joints[:,:max_joints, :3])
            
        if not self.separate:
            joints_embeds = self.co_proj(joints_fourier) if not self.use_bone else self.bone_proj(joints_fourier)

        normal_embeds = self.normal_proj(self.normal_embedder(feats)) if not self.separate else self.normal_embedder(feats)

        if not self.separate:
            pc_embeds = co_embeds + normal_embeds
        else:
            joints_embeds = self.co_proj(joints_fourier.to(dtype)) if not self.use_bone else self.bone_proj(joints_fourier.to(dtype))
            pc_embeds = self.pc_proj(torch.cat([co_embeds.to(dtype), normal_embeds.to(dtype)], dim=-1))

        pc_num = pc_embeds.shape[-2]
        joints_num = joints_embeds.shape[-2]
        x = torch.cat([pc_embeds, joints_embeds], dim=-2)
        for i, layer in enumerate(self.cross_attn):

            x = layer(x, latents)
            if self.use_mask:
                x = self.cross_attn_joint[i](x, x[:, pc_num:], mask=mask.to(device))
            else:
                x = self.cross_attn_joint[i](x, x[:, pc_num:])
        pc_embeds, joints_embeds = x.split([pc_num, joints_num], dim=1)

        logits = torch.einsum('bnc,bmc->bnm', self.k_proj(self.ln_1(pc_embeds)), self.q_proj(self.ln_2(joints_embeds)))  # (b, n, m)

        if self.use_mask:
            logits = logits.masked_fill(mask.unsqueeze(1) == 0, -1e8)

        if downsample and not self.inference:
            return logits, idx

        return logits

class AttendjointsDecoder_multi(nn.Module):
    def __init__(self, 
                #  num_latents = 64,
                #  num_kv_latents = 257,
                #  out_channels = 3,
                 width = 768,
                 layers = 4,
                 device = 'cuda',
                 dtype = torch.float32,
                 heads = 12,
                 init_scale: float = 0.25,
                 flash = False,
                 use_checkpoint = False,
                 qkv_bias = False,
                 num_freqs: int = 8,
                 concat_num: int = 512,
                 include_pi: bool = True,
                separate = False,
                use_mask = True,
                inference_with_repeat=False,
                use_bone = True,
                inference = False):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.use_mask = use_mask
        self.inference_with_repeat = inference_with_repeat
        self.inference = inference

        self.self_attn = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=-1,
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_checkpoint=False,

        )
        self.concat_number = concat_num
        self.separate = separate
        self.normal_embedder = components_from_spherical_harmonics
        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)
        self.bone_proj = None if not use_bone else nn.Linear(self.fourier_embedder.out_dim * 2, width, device=device, dtype=dtype)
        self.use_bone = use_bone
        if not self.separate:
            self.co_proj = nn.Linear(self.fourier_embedder.out_dim, width, device=device, dtype=dtype)
            self.normal_proj = nn.Linear(25, width, device=device, dtype=dtype)
        else:
            self.pc_proj = nn.Linear(self.fourier_embedder.out_dim + 25, width, device=device, dtype=dtype)

        # self.proj_attn = nn.Linear(width, width, device=device, dtype=dtype)

        # self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        self.output_proj_joints = nn.Linear(width, width, device=device, dtype=dtype)
        self.output_proj_points = nn.Linear(width, width, device=device, dtype=dtype)
        self.layer_norm = nn.LayerNorm(width)
        
    # def inference(self, latents, data=None,device='cuda', dtype='float32', use_mask=False):
    def inference_mode(self):
        self.inference = True
    
    def forward(self, latents, data=None,device='cuda', downsample=None, dtype='float32'):
        joints = data['bones'].to(device) if self.use_bone else data['joints'].to(device)
        max_joints = max(data['bones_num']) if self.use_bone else max(data['joints_num'])
        
        pc = data['points_cloud'][..., 0:3].to(device)
        feats = data['points_cloud'][..., 3:].to(device)
        
        if downsample:
            # random sample
            idx = torch.randperm(pc.shape[1])[:downsample].to(device)
            pc = pc[:, idx]
            feats = feats[:, idx]

        bs = pc.shape[1]//self.concat_number

        # Embed the input data
        if self.use_bone:
            # joints_fourier = torch.cat((self.fourier_embedder(joints[:,:max_joints*2:2, :3]), self.fourier_embedder(joints[:,1:max_joints*2:2, :3])), dim=-1)
            joints_fourier = torch.cat((self.fourier_embedder(joints[:,:max_joints,:3]), self.fourier_embedder(joints[:,:max_joints, 3:])), dim=-1)
        else:
            joints_fourier = self.fourier_embedder(joints[:,:max_joints, :3])
        
        if self.separate:
            joints_embeds = self.co_proj(joints_fourier.to(dtype)) if not self.use_bone else self.bone_proj(joints_fourier.to(dtype))
            points_embeds = self.fourier_embedder(pc)
            normal_embeds = self.normal_embedder(feats)
            points = self.pc_proj(torch.cat([points_embeds, normal_embeds], dim=-1))
        else:
            joints_embeds = self.co_proj(joints_fourier) if not self.use_bone else self.bone_proj(joints_fourier)
            co_embeds = self.fourier_embedder(pc)
            co_embeds = self.co_proj(co_embeds)
            # Embed the normals
            normal_embeds = self.normal_embedder(feats)
            normal_embeds = self.normal_proj(normal_embeds)  # (b, n, c)
            points = (co_embeds + normal_embeds)

        repeated_latents = repeat(latents, "b m c -> b n m c", n=bs)
        repeated_joints = repeat(joints_embeds, "b m c -> b n m c", n=bs)
        points = points.reshape( latents.shape[0], bs, self.concat_number, -1)

        # Concatenate embeddings
        x = torch.cat([repeated_joints, points, repeated_latents], dim=-2) # (b, bs, concat_number+latent_num+joints_num, c)

        # Pass through self-attention
        if self.use_mask:
            mask = data['bones_mask'].to(device)
            append_size = x.shape[2]-mask.shape[1] # the zero needs to append after mask
            batch_size = mask.shape[0]
            mask_extend = torch.ones((batch_size,append_size)).to(device)
            mask = torch.cat([mask,mask_extend],dim=-1).repeat(bs,1).to(device)
            x = rearrange(x, "b n m c -> (b n) m c")
            x = self.self_attn(x,mask)
        else:
            x = rearrange(x, "b n m c -> (b n) m c")
            x = self.self_attn(x)
        joints, points, _ = x.split([joints_embeds.shape[1],self.concat_number, latents.shape[1]], dim=1) 
        joints = self.output_proj_joints(self.layer_norm(joints))
        points = self.output_proj_points(self.layer_norm(points))
        
        logits = torch.einsum('bik,bjk->bij', points, joints)
        logits = rearrange(logits, '(b n) m c -> b (n m) c', b=pc.shape[0],n=bs) # (b, n, c)

        if self.use_mask:
            mask = data['bones_mask'].to(device)
            logits = logits.masked_fill(mask.unsqueeze(1) == 0, -1e8)
        
        if self.inference:
            vertices = data['vertice']
            points_cloud = data['points_cloud'][0,..., 0:3].to(device)
            vertices_exp = vertices[0,...,:3]  # (batch_size, num_vertices, 1, 3) 
            logits = compute_nearest_points(vertices_exp, points_cloud, logits[0], device)

        if downsample:
            return logits, idx

        return logits
    
def compute_nearest_points(vertices, points, logits, device, batch_size=1024):
    # vertices: [N, 3]
    # points: [M, 3]
    # logits: [M, K]  (K is the number of skinning weights)
    
    num_vertices = vertices.shape[0]
    # Initialize the output tensor for skinning weights
    skin_predict = torch.zeros((num_vertices, logits.shape[1]), device=device)
    
    # Split vertices into batches
    for i in range(0, num_vertices, batch_size):

        batch_vertices = vertices[i:i+batch_size]  # [batch_size, 3]
        vertices_exp = batch_vertices.unsqueeze(1)  # [batch_size, 1, 3]
        points_exp = points.unsqueeze(0)  # [1, num_points, 3]
        distances = torch.sum((vertices_exp - points_exp) ** 2, dim=-1)  # [batch_size, num_points]
        nearest_idx = torch.argmin(distances, dim=-1)  # [batch_size]
        skin_predict_batch = logits[nearest_idx]  # [batch_size, K]
        skin_predict[i:i+batch_size] = skin_predict_batch
    
    return skin_predict