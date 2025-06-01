import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class TrajTransfomerEncoder(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, hidden_dim=256):
        super(TrajTransfomerEncoder, self).__init__()
        # self.horizon = config['context_horizon']
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        n_head = 8
        dropout = 0.1
        # self.device = config['device']

        # networks
        encoder_layer = nn.TransformerEncoderLayer(self.hidden_dim, n_head,
                                                   self.hidden_dim * 4, dropout, batch_first=True, norm_first=True)  # TODO norm_first
        self.state_emb = nn.Linear(state_dim, self.hidden_dim)
        self.action_emb = nn.Linear(action_dim, self.hidden_dim)
        self.reward_emb = nn.Linear(1, self.hidden_dim)
        self.positional_encoding = nn.Embedding(
            200, self.hidden_dim)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 6)
        self.fc_latent = nn.Linear(
            self.hidden_dim, self.latent_dim)

        self.weight_init_()

    def weight_init_(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, traj):
        '''
        The shape of states, actions and rewards: [batch_size, seq_len, dim]. Use the mean of outputs corresponding to rewards.
        '''
        states, actions, rewards, _ = traj
        batch_size, seq_len, _ = states.shape
        # timesteps = self.positional_encoding(timesteps)
        states = self.state_emb(states)
        actions = self.action_emb(actions)
        rewards = self.reward_emb(rewards)
        x = torch.stack([states, actions, rewards], dim=1).permute(
            0, 2, 1, 3).reshape(batch_size, 3 * seq_len, -1)
        x: Tensor = self.transformer(x)  # [batch_size, 3 * seq_len, dim]
        x = x.reshape(batch_size, seq_len, 3, -1).permute(0,
                                                          2, 1, 3)  # [batch_size, 3, seq_len, dim]
        return self.fc_latent(x[:, -1].mean(dim=1))  # [batch_size, latent_dim]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class RewardDecoder(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, hidden_dim=256):
        super(RewardDecoder, self).__init__()

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, latent_dim), nn.ReLU())
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, latent_dim), nn.ReLU())
        self.next_state_encoder = nn.Sequential(
            nn.Linear(state_dim, latent_dim), nn.ReLU())

        self.linear1 = nn.Linear(latent_dim * 4, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action, next_state, traj_embed):

        # Merge batch and num_transitions dimensions for encoding
        batch_size, num_transitions = state.shape[:2]

        # Expand traj_embed to match (batch_size, num_transitions, latent_dim)
        # (batch_size, num_transitions, latent_dim)
        traj_embed = traj_embed.unsqueeze(1).repeat(1, num_transitions, 1)

        # Flatten the first two dimensions for processing
        # (batch_size * num_transitions, state_dim)
        state = state.view(batch_size * num_transitions, -1)
        # (batch_size * num_transitions, action_dim)
        action = action.view(batch_size * num_transitions, -1)
        # (batch_size * num_transitions, state_dim)
        next_state = next_state.view(batch_size * num_transitions, -1)
        # (batch_size * num_transitions, latent_dim)
        traj_embed = traj_embed.view(batch_size * num_transitions, -1)

        # extract features for states, actions
        hs = self.state_encoder(state)
        ha = self.action_encoder(action)
        hs_next = self.next_state_encoder(next_state)
        h = torch.cat((hs, ha, hs_next, traj_embed), dim=-1)

        h = F.relu(self.linear1(h))
        h = F.relu(self.linear2(h))
        reward_predict = self.linear3(h)

        return reward_predict


class StateDecoder(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, hidden_dim=256):
        super(StateDecoder, self).__init__()

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, latent_dim), nn.ReLU())
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, latent_dim), nn.ReLU())
        # self.reward_encoder = nn.Sequential(
        #     nn.Linear(1, latent_dim), nn.ReLU)

        self.linear1 = nn.Linear(latent_dim * 3, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, state_dim)

        self.apply(weights_init_)

    def forward(self, state, action, traj_embed):
        # Merge batch and num_transitions dimensions for encoding
        batch_size, num_transitions = state.shape[:2]

        # Expand traj_embed to match (batch_size, num_transitions, latent_dim)
        # (batch_size, num_transitions, latent_dim)
        traj_embed = traj_embed.unsqueeze(1).repeat(1, num_transitions, 1)

        # Flatten the first two dimensions for processing
        # (batch_size * num_transitions, state_dim)
        state = state.view(batch_size * num_transitions, -1)
        # (batch_size * num_transitions, action_dim)
        action = action.view(batch_size * num_transitions, -1)
        # (batch_size * num_transitions, latent_dim)
        traj_embed = traj_embed.view(batch_size * num_transitions, -1)
        
        # extract features for states, actions
        hs = self.state_encoder(state)
        ha = self.action_encoder(action)
        # hr = self.reward_encoder(reward)
        h = torch.cat((hs, ha, traj_embed), dim=-1)

        h = F.relu(self.linear1(h))
        h = F.relu(self.linear2(h))
        state_predict = self.linear3(h)

        return state_predict
