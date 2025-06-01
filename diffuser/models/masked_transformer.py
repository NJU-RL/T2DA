import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import Mlp
from timm.models.layers import trunc_normal_
import math
from torch.distributions import Bernoulli

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # self.rel_pos_bias = RelativePositionBias(window_size=[int(num_patches**0.5), int(num_patches**0.5)], num_heads=num_heads)

    def get_masked_rel_bias(self, B, ids_keep):
        # get masked rel_pos_bias
        rel_pos_bias = self.rel_pos_bias()
        rel_pos_bias = rel_pos_bias.unsqueeze(dim=0).repeat(B, 1, 1, 1)

        rel_pos_bias_masked = torch.gather(
            rel_pos_bias, dim=2, index=ids_keep.unsqueeze(dim=1).unsqueeze(dim=-1).repeat(1, rel_pos_bias.shape[1], 1, rel_pos_bias.shape[-1]))
        rel_pos_bias_masked = torch.gather(
            rel_pos_bias_masked, dim=3, index=ids_keep.unsqueeze(dim=1).unsqueeze(dim=2).repeat(1, rel_pos_bias.shape[1], ids_keep.shape[1], 1))
        return rel_pos_bias_masked

    def forward(self, x, ids_keep=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        '''
        if ids_keep is not None:
            rp_bias = self.get_masked_rel_bias(B, ids_keep)
        else:
            rp_bias = self.rel_pos_bias()
        attn += rp_bias
        '''

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RelativePositionBias(nn.Module):
    # https://github.com/microsoft/unilm/blob/master/beit/modeling_finetune.py
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(torch.zeros(self.num_relative_distance, num_heads))

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() 
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = torch.zeros(size=(window_size[0] * window_size[1],) * 2, dtype=relative_coords.dtype)
        relative_position_index = relative_coords.sum(-1)

        self.register_buffer("relative_position_index", relative_position_index)

        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], 
            -1
        )  # Wh*Ww,Wh*Ww,nH
        # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.permute(2, 0, 1).contiguous()



#################################################################################
#                         Embedding Layers for Timesteps                        #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0,
                                                 end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                                 Core MDT Model                                #
#################################################################################

class MDTBlock(nn.Module):
    """
    A MDT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, skip=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        def approx_gelu(): return nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.skip_linear = nn.Linear(2 * hidden_size, hidden_size) if skip else None

    def forward(self, x, c, skip=None, ids_keep=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), ids_keep=ids_keep)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of MDT.
    """

    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class MaskedDiT(nn.Module):
    """
    Masked Diffusion Transformer v2.
    """

    def __init__(
        self,
        trajectory_length=8,
        transition_dim=4,
        hidden_size=128,
        depth=9,
        decode_layer=3,
        num_heads=4,
        mlp_ratio=4.0,
        returns_condition=False,
        condition_dropout=0.1,
        mask_ratio=None,
        calc_energy=False,
    ):
        super().__init__()
        self.trajectory_length = trajectory_length
        self.transition_dim = transition_dim
        self.num_heads = num_heads
        self.returns_condition = returns_condition
        self.condition_dropout = condition_dropout
        self.calc_energy = calc_energy
        self.decode_layer = int(decode_layer)
        self.depth = depth

        self.t_embedder = TimestepEmbedder(hidden_size)

        if self.returns_condition:
            self.returns_mlp = nn.Sequential(
                        nn.Linear(1, hidden_size),
                        nn.SiLU(),
                        nn.Linear(hidden_size, hidden_size * 4),
                        nn.SiLU(),
                        nn.Linear(hidden_size * 4, hidden_size),
                    )
            self.mask_dist = Bernoulli(probs=1-self.condition_dropout)

            embed_dim = 2 * hidden_size
        else:
            embed_dim = hidden_size

        self.x_embedder = nn.Linear(transition_dim, embed_dim, bias=True)
        
        # Will use learnbale sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, trajectory_length, embed_dim), requires_grad=True)

        half_depth = (depth - self.decode_layer)//2
        self.half_depth = half_depth
        
        self.en_inblocks = nn.ModuleList([
            MDTBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(half_depth)
        ])
        self.en_outblocks = nn.ModuleList([
            MDTBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, skip=True) for _ in range(half_depth)
        ])
        self.de_blocks = nn.ModuleList([
            MDTBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, skip=True) for i in range(self.decode_layer)
        ])
        self.sideblocks = nn.ModuleList([
            MDTBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(1)
        ])
        self.final_layer = FinalLayer(embed_dim, transition_dim)

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, trajectory_length, embed_dim), requires_grad=True)

        if mask_ratio is not None:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.mask_ratio = float(mask_ratio)
        else:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=False)
            self.mask_ratio = None
        print(f"mask ratio: {self.mask_ratio}, decode_layer: {self.decode_layer}")
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed_from_grid(
            self.pos_embed.shape[-1], 
            np.arange(int(self.trajectory_length), dtype=np.float32)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed_from_grid(
            self.decoder_pos_embed.shape[-1], 
            np.arange(int(self.trajectory_length), dtype=np.float32)
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.en_inblocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        for block in self.en_outblocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        for block in self.de_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        for block in self.sideblocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        if self.mask_ratio is not None:
            torch.nn.init.normal_(self.mask_token, std=.02)


    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep
    

    def forward_side_interpolater(self, x, c, mask, ids_restore):
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        
        # add pos embed
        x = x + self.decoder_pos_embed

        # pass to the basic block
        x_before = x
        for sideblock in self.sideblocks:
            x = sideblock(x, c, ids_keep=None)
        
        # masked shortcut
        mask = mask.unsqueeze(dim=-1)
        x = x * mask + (1-mask) * x_before

        return x
    

    def forward(self, x, cond, time, returns=None, prompts=None, enable_mask=False, use_dropout=True, force_dropout=False):
        """
        Forward pass of MDT.
        x:       (N, T, D) tensor of a trajectory 
        time:    (N,) tensor of diffusion timesteps
        returns: (N, T)
        enable_mask: Use mask latent modeling
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D)
        t = self.t_embedder(time)                # (N, D)

        if self.returns_condition:
            assert returns is not None
            returns_embed = self.returns_mlp(returns)
            if use_dropout:
                mask = self.mask_dist.sample(sample_shape=(returns_embed.size(0), 1)).to(returns_embed.device)
                returns_embed = mask * returns_embed
            if force_dropout:
                returns_embed = 0 * returns_embed

            t = torch.cat([t, returns_embed], dim=-1)

        input_skip = x

        masked_stage = False
        skips = []
        # masking op for training

        if self.mask_ratio is not None and enable_mask:
            # masking: length -> length * mask_ratio
            rand_mask_ratio = torch.rand(1, device=x.device)  # noise in [0, 1]
            rand_mask_ratio = rand_mask_ratio * 0. + self.mask_ratio # mask_ratio, mask_ratio + 0.2 
            # print(rand_mask_ratio)

            # print(f'before masking, x shape: {x.shape}')
            x, mask, ids_restore, ids_keep = self.random_masking(x, rand_mask_ratio)
            masked_stage = True
            # print(f'after masking, x shape: {x.shape}')

        for block in self.en_inblocks:
            if masked_stage:
                x = block(x, t, ids_keep=ids_keep)
            else:
                x = block(x, t, ids_keep=None)
            skips.append(x)

        for block in self.en_outblocks:
            if masked_stage:
                x = block(x, t, skip=skips.pop(), ids_keep=ids_keep)
            else:
                x = block(x, t, skip=skips.pop(), ids_keep=None)

        if self.mask_ratio is not None and enable_mask:
            # print(f'before side interpolater, x shape: {x.shape}')
            x = self.forward_side_interpolater(x, t, mask, ids_restore)
            # print(f'after side interpolater, x shape: {x.shape}')
            masked_stage = False
        else:
            # add pos embed
            x = x + self.decoder_pos_embed

        for i in range(len(self.de_blocks)):
            block = self.de_blocks[i]
            this_skip = input_skip

            x = block(x, t, skip=this_skip, ids_keep=None)

        x = self.final_layer(x, t)  # (N, T, transition_dim)
        return x



#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


if __name__ == '__main__':

    batch_size = 32
    trajectory_length = 8
    transition_dim = 4
    diffusion_steps = 20
    mask_ratio = 0.3
    device = 'cuda:0'

    traj = torch.rand(batch_size, trajectory_length, transition_dim).to(device)     #(N, T, D)
    t = torch.randint(0, diffusion_steps, (batch_size,)).to(device)
    returns = torch.rand(batch_size, trajectory_length).to(device)
    model = MaskedDiT(
        trajectory_length=trajectory_length,
        transition_dim=transition_dim,
        hidden_size=128,
        depth=9,
        num_heads=4,
        mlp_ratio=4.0,
        mask_ratio=mask_ratio,
    ).to(device)

    output = model(traj, None, t, returns=returns, enable_mask=False)
    
    print(output.shape)



