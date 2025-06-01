import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import Attention, Mlp
from torch.distributions import Bernoulli


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#                       Embedding Layers for Timesteps                          #
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
        :param          t: a 1-D Tensor of N indices, one per batch element. These may be fractional.
        :param        dim: the dimension of the output.
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
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        def approx_gelu(): return nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size,
                       hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c).chunk(6, dim=1)
        x = x + \
            gate_msa.unsqueeze(
                1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + \
            gate_mlp.unsqueeze(
                1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)

        self.linear = nn.Linear(hidden_size, output_size, bias=True)
        # nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
        # if self.linear.bias is not None:
        #     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
        #         self.linear.weight)
        #     bound = 1 / math.sqrt(fan_in)
        #     nn.init.uniform_(self.linear.bias, -bound, bound)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)

        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        env_name,
        task_max_length,
        trajectory_length=8,
        transition_dim=4,
        hidden_size=128,
        depth=3,
        num_heads=4,
        mlp_ratio=4.0,
        include_returns=False,
        condition_dropout=0.1,
        prompts_type=None,
        prompt_oracle_dim=2,
        prompt_embed_dim=16,
        observation_dim=2,
        max_path_length=20,
        train_device='cuda',
    ):
        super().__init__()
        self.env_name = env_name
        self.task_max_length = task_max_length
        self.trajectory_length = trajectory_length
        self.transition_dim = transition_dim
        self.num_heads = num_heads
        self.include_returns = include_returns
        self.condition_dropout = condition_dropout
        self.calc_energy = False
        self.prompts_type = prompts_type
        self.prompt_oracle_dim = prompt_oracle_dim
        self.prompt_embed_dim = prompt_embed_dim
        self.max_path_length = max_path_length
        self.device = train_device
        self.observation_dim = observation_dim
        self.action_dim = transition_dim - observation_dim

        if self.include_returns:
            self.returns_mlp = nn.Sequential(
                nn.Linear(1, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size * 4),
                nn.SiLU(),
                nn.Linear(hidden_size * 4, hidden_size),
            )

            if self.prompts_type is None:
                embed_dim = 2 * hidden_size
            else:
                if self.prompts_type == 'clip':
                    embed_dim = 2 * hidden_size + 512

                elif self.prompts_type in ['aligned_clip', 'aligned_clip_wo_wm']:
                    embed_dim = 2 * hidden_size + 256

                elif self.prompts_type == 't5':
                    embed_dim = 2 * hidden_size + 256 * 6

                elif self.prompts_type == 'aligned_t5':
                    embed_dim = 2 * hidden_size + 256 * 6

                else:
                    raise NotImplementedError

            self.mask_dist = Bernoulli(probs=1-self.condition_dropout)

        else:
            embed_dim = hidden_size

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.x_embedder = nn.Linear(transition_dim, embed_dim, bias=True)
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, trajectory_length, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(embed_dim, transition_dim)
        self.initialize_weights()

    def get_state_dict(self):
        return {key: value for key, value in self.state_dict().items() if key.split('.')[0] not in ['t5_model', 'clip_model', 'tokenizer', 'text_projection_layer']}

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_1d_sincos_pos_embed_from_grid(
            self.pos_embed.shape[-1],
            np.arange(int(self.trajectory_length), dtype=np.float32)
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    # def forward(self, x, t):
    def forward(self, x, cond, time, returns=None, prompts=None, use_dropout=True, force_dropout=False):
        """
        Forward pass of DiT.
        x:       (N, T, D) tensor of a trajectory
        time:    (N,) tensor of diffusion timesteps
        returns: (N, T)
        prompts: for clip, a list of N
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D)
        t = self.t_embedder(time)                # (N, D)

        if self.include_returns:
            assert returns is not None
            returns_embed = self.returns_mlp(returns)

            if self.prompts_type is not None:
                assert prompts is not None

                if self.prompts_type == 'oracle':
                    # prompts size: (batch_size, prompt_oracle_dim)
                    prompts_embed = self.prompts_mlp(prompts)

                elif self.prompts_type == 'trajectory':
                    # prompts: a dict
                    # prompts['observations']:    batch_size * prompt_traj_len * observations_dim
                    # prompts['actions']:         batch_size * prompt_traj_len * actions_dim
                    # prompts['rewards']:         batch_size * prompt_traj_len * 1
                    # prompts['timesteps']:       batch_size * prompt_traj_len
                    # prompts['masks']:           batch_size * prompt_traj_len
                    state_embeddings = self.prompt_embed_state(
                        prompts['observations'])
                    action_embeddings = self.prompt_embed_action(
                        prompts['actions'])
                    rewards_embeddings = self.prompt_embed_reward(
                        prompts['rewards'])
                    time_embeddings = self.prompt_embed_timestep(
                        prompts['timesteps'].long())

                    # time embeddings are treated similar to positional embeddings
                    state_embeddings = state_embeddings + time_embeddings
                    action_embeddings = action_embeddings + time_embeddings
                    rewards_embeddings = rewards_embeddings + time_embeddings

                    # batch_size * prompt_traj_len * (3 * prompt_embed_dim)
                    concat_embeddings = torch.cat(
                        [state_embeddings, action_embeddings, rewards_embeddings], dim=-1)
                    encoder_in = F.relu(
                        self.prompt_embed_in(concat_embeddings))
                    encoder_out = self.prompt_encoder(
                        encoder_in.permute(1, 0, 2))[-1]

                    # batch_size * dim
                    prompts_embed = self.prompt_embed_out(encoder_out)
                    # prompts_embed

                elif self.prompts_type in ['clip', 'aligned_clip', 'aligned_clip_wo_wm']:
                    prompts_embed = prompts

                elif self.prompts_type == 't5':
                    self.t5_model.to(self.device)
                    inputs = self.tokenizer(prompts,
                                            padding='max_length', truncation=True, max_length=self.task_max_length, return_tensors="pt").input_ids.to(self.device)
                    prompts_embed = self.t5_model(
                        input_ids=inputs).last_hidden_state.to(self.device)
                    prompts_embed = prompts_embed.reshape(
                        prompts_embed.shape[0], -1)
                elif self.prompts_type == 'aligned_t5':
                    self.t5_model.to(self.device)
                    inputs = self.tokenizer(prompts,
                                            padding='max_length', truncation=True, max_length=self.task_max_length, return_tensors="pt").input_ids.to(self.device)
                    prompts_embed = self.t5_model(
                        input_ids=inputs).last_hidden_state.to(self.device)
                    prompts_embed = prompts_embed.reshape(
                        prompts_embed.shape[0], -1)
                else:
                    raise NotImplementedError

                returns_embed = torch.cat(
                    [returns_embed, prompts_embed], dim=-1)

            if use_dropout:
                mask = self.mask_dist.sample(sample_shape=(
                    returns_embed.size(0), 1)).to(returns_embed.device)
                returns_embed = mask * returns_embed
            if force_dropout:
                returns_embed = 0 * returns_embed

            t = torch.cat([t, returns_embed], dim=-1)

        for block in self.blocks:
            x = block(x, t)                      # (N, T, D)
        x = self.final_layer(x, t)               # (N, T, transition_dim)
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