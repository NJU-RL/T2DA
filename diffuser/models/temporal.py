import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
from einops import rearrange
import pdb
from torch.distributions import Bernoulli
import torch.nn.functional as F

from .helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.InstanceNorm2d(dim, affine = True)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 128):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class GlobalMixing(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 128):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5, mish=True):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size, mish),
            Conv1dBlock(out_channels, out_channels, kernel_size, mish),
        ])

        if mish:
            act_fn = nn.Mish()
        else:
            act_fn = nn.SiLU()

        self.time_mlp = nn.Sequential(
            act_fn,
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)

        return out + self.residual_conv(x)


class TemporalUnet(nn.Module):
    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=128,
        dim_mults=(1, 2, 4, 8),
        include_returns=False,
        condition_dropout=0.1,
        kernel_size=5,
        calc_energy=False,
        ### hyperparameters for prompt design
        prompts_type=None,
        prompt_oracle_dim=2,
        prompt_embed_dim=16,
        observation_dim=2,
        max_path_length=20,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        if calc_energy:
            mish = False
            act_fn = nn.SiLU()
        else:
            mish = True
            act_fn = nn.Mish()

        self.time_dim = dim
        self.returns_dim = dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            act_fn,
            nn.Linear(dim * 4, dim),
        )

        self.include_returns = include_returns
        self.prompts_type = prompts_type
        self.prompt_oracle_dim = prompt_oracle_dim
        self.prompt_embed_dim = prompt_embed_dim
        self.max_path_length = max_path_length
        self.observation_dim = observation_dim

        self.condition_dropout = condition_dropout
        self.calc_energy = calc_energy

        if self.include_returns:
            self.returns_mlp = nn.Sequential(
                nn.Linear(1, dim),
                act_fn,
                nn.Linear(dim, dim * 4),
                act_fn,
                nn.Linear(dim * 4, dim),
            )
            self.mask_dist = Bernoulli(probs=1-self.condition_dropout)

            if self.prompts_type is None:
                embed_dim = 2 * dim 

            else:
                if self.prompts_type == 'oracle':
                    self.prompts_mlp = nn.Sequential(
                            nn.Linear(self.prompt_oracle_dim, dim),
                            act_fn,
                            nn.Linear(dim, dim * 4),
                            act_fn,
                            nn.Linear(dim * 4, dim),
                        )

                elif self.prompts_type == 'trajectory':
                    self.prompt_embed_timestep = nn.Embedding(max_path_length, prompt_embed_dim)
                    self.prompt_embed_reward = torch.nn.Linear(1, prompt_embed_dim)
                    self.prompt_embed_state = torch.nn.Linear(observation_dim, prompt_embed_dim)
                    self.prompt_embed_action = torch.nn.Linear(transition_dim-observation_dim, prompt_embed_dim)

                    d_model = 512; nhead = 8
                    self.prompt_embed_in = nn.Linear(3*prompt_embed_dim, d_model)
                    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
                    self.prompt_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
                    self.prompt_embed_out = nn.Linear(d_model, dim)

                else:
                    assert NotImplementedError

                embed_dim = 3 * dim
                
        else:
            embed_dim = dim

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=kernel_size, mish=mish),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x, cond, time, returns=None, prompts=None, use_dropout=True, force_dropout=False):
        '''
            x : [ batch x horizon x transition ]
            returns : [batch x horizon]
        '''
        if self.calc_energy:
            x_inp = x

        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)

        if self.include_returns:
            assert returns is not None
            returns_embed = self.returns_mlp(returns)

            if self.prompts_type is not None:
                assert prompts is not None 

                if self.prompts_type == 'oracle':
                    ### prompts size: (batch_size, prompt_oracle_dim)
                    prompts_embed = self.prompts_mlp(prompts)
                    
                elif self.prompts_type == 'trajectory':
                    ### prompts: a dict 
                    ### prompts['observations']:    batch_size * prompt_traj_len * observations_dim
                    ### prompts['actions']:         batch_size * prompt_traj_len * actions_dim
                    ### prompts['rewards']:         batch_size * prompt_traj_len * 1
                    ### prompts['timesteps']:       batch_size * prompt_traj_len
                    ### prompts['masks']:           batch_size * prompt_traj_len
                    state_embeddings = self.prompt_embed_state(prompts['observations'])
                    action_embeddings = self.prompt_embed_action(prompts['actions'])
                    rewards_embeddings = self.prompt_embed_reward(prompts['rewards'])
                    time_embeddings = self.prompt_embed_timestep(prompts['timesteps'].long())

                    # time embeddings are treated similar to positional embeddings
                    state_embeddings = state_embeddings + time_embeddings
                    action_embeddings = action_embeddings + time_embeddings
                    rewards_embeddings = rewards_embeddings + time_embeddings

                    ### batch_size * prompt_traj_len * (3 * prompt_embed_dim)
                    concat_embeddings = torch.cat([state_embeddings, action_embeddings, rewards_embeddings], dim=-1)
                    encoder_in = F.relu(self.prompt_embed_in(concat_embeddings))
                    encoder_out = self.prompt_encoder(encoder_in.permute(1, 0, 2))[-1]

                    ### batch_size * dim
                    prompts_embed = self.prompt_embed_out(encoder_out)
                
                else:
                    raise NotImplementedError

                returns_embed = torch.cat([returns_embed, prompts_embed], dim=-1)

            if use_dropout:
                mask = self.mask_dist.sample(sample_shape=(returns_embed.size(0), 1)).to(returns_embed.device)
                returns_embed = mask * returns_embed
            if force_dropout:
                returns_embed = 0 * returns_embed

            t = torch.cat([t, returns_embed], dim=-1)

        h = []

        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        # import pdb; pdb.set_trace()

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')

        if self.calc_energy:
            # Energy function
            energy = ((x - x_inp)**2).mean()
            grad = torch.autograd.grad(outputs=energy, inputs=x_inp, create_graph=True)
            return grad[0]
        else:
            return x

