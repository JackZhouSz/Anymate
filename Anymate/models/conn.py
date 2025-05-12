import torch
import torch.nn as nn
from ThirdParty.michelangelo.models.modules.transformer_blocks import ResidualCrossAttentionBlock, ResidualAttentionBlock, Transformer
from ThirdParty.michelangelo.models.modules.embedder import FourierEmbedder, components_from_spherical_harmonics

class AttendjointsDecoder_con_combine(nn.Module):
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
                 use_mask = True):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.separate = separate
        self.use_mask = use_mask
        # self.num_latents = num_latents

        # self.query = nn.Parameter(torch.randn((num_latents, width), device=device, dtype=dtype) * 0.02)

        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)
        self.co_proj = nn.Linear(self.fourier_embedder.out_dim, width, device=device, dtype=dtype)

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

        self.self_attn = nn.ModuleList([ResidualAttentionBlock(
            device=device,
            dtype=dtype,
            n_ctx=-1,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
        ) for _ in range(layers * 2)])

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

        joints = data['joints'].to(device)
        max_joints = max(data['joints_num'])
        joints = joints[:, :max_joints, :3]

        joints_embeds = self.fourier_embedder(joints)
        joints_embeds = self.co_proj(joints_embeds)  

        joints_num = joints_embeds.shape[-2]

        x = [joints_embeds, joints_embeds.clone()]

        for i in range(2):
            for j, layer in enumerate(self.cross_attn):
                
                x[i] = layer(x[i], latents)

                if self.use_mask:
                    x[i] = self.self_attn[2*i+j](x[i], mask=data['joints_mask'].to(device))
                else:
                    x[i] = self.self_attn[2*i+j](x[i])

        # Dot Product between points and joints
        logits = torch.einsum('bnc,bmc->bnm', self.k_proj(self.ln_1(x[0])), self.q_proj(self.ln_2(x[1])))  # (b, n, m)

        if self.use_mask:
            mask = data['joints_mask'].to(device)
            logits = logits.masked_fill(mask.unsqueeze(1) == 0, -1e8)

        return logits
    
class AttendjointsDecoder_con_token(nn.Module):
    def __init__(self, 
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
                 include_pi: bool = True,
                 head_token_length =128,
                separate = False,
                use_mask = True):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.use_mask = use_mask
        self.layer_norm = nn.LayerNorm(width)
        self.head_token = nn.Parameter(torch.randn((1, 1, head_token_length), device=device, dtype=dtype) * 0.02)
        self.tail_token = nn.Parameter(torch.randn((1, 1, head_token_length), device=device, dtype=dtype) * 0.02)
        self.head_mlp = nn.ModuleList([
            nn.Linear(width + head_token_length, 512, device=device, dtype=dtype),
            nn.Linear(512, 512, device=device, dtype=dtype),
            nn.Linear(512, width, device=device, dtype=dtype),
            nn.LayerNorm(width)
            
        ])
        self.tail_mlp = nn.ModuleList([
            nn.Linear(width + head_token_length, 512, device=device, dtype=dtype),
            nn.Linear(512, 512, device=device, dtype=dtype),
            nn.Linear(512, width, device=device, dtype=dtype),
            nn.LayerNorm(width)
        ])

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
        self.separate = separate
        self.normal_embedder = components_from_spherical_harmonics
        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)
        self.joints_proj = nn.Linear(self.fourier_embedder.out_dim, width, device=device, dtype=dtype)
        self.output_proj_joints = nn.Linear(width, width, device=device, dtype=dtype)
        
    def forward(self, latents, data=None,device='cuda', downsample=None, dtype='float32'):
        joints = data['joints'].to(device)
        max_joints = max(data['joints_num'])
        joints = joints[:, :max_joints, :3]
        joints_embeds_fourier = self.fourier_embedder(joints)
        joints_embeds = self.joints_proj(joints_embeds_fourier)  
        # Concatenate embeddings
        x = torch.cat([joints_embeds, latents], dim=-2) # (b, max_joint+token_num, c)
        # Pass through self-attention
        if self.use_mask:
            mask = data['mask'].to(device)
            append_size = x.shape[1]-mask.shape[1] # the zero needs to append after mask
            batch_size = mask.shape[0]
            
            mask_extend = torch.ones((batch_size,append_size)).to(device)
            mask = torch.cat([mask,mask_extend],dim=-1).to(device)
            
            x = self.self_attn(x,mask)
        else:
            x = self.self_attn(x)
        joints, _= x.split([joints_embeds.shape[1], latents.shape[1]], dim=1) 
        joints = self.output_proj_joints(self.layer_norm(joints))
        joints_head = torch.concat([joints, self.head_token.repeat(joints.shape[0],joints.shape[1],1)], dim=-1)
        joints_tail = torch.concat([joints, self.tail_token.repeat(joints.shape[0],joints.shape[1],1)], dim=-1)
        for layer in self.head_mlp:
            joints_head = layer(joints_head)
        for layer in self.tail_mlp:
            joints_tail = layer(joints_tail)
        logits = torch.einsum('bik,bjk->bij', joints_head, joints_tail)

        return logits