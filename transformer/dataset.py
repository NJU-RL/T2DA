from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import torch
from collections import OrderedDict
from transformers import (
    CLIPTextModel, AutoTokenizer,
    T5EncoderModel, BertModel
)
from peft import LoraConfig, get_peft_model, PeftModel
from utils import freeze
# Assume CLIPProjection and TrajEncoder are imported or defined elsewhere


class TextProjection(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.projection = nn.Parameter(torch.empty(input_dim, latent_dim))
        nn.init.normal_(self.projection, mean=0.0, std=input_dim ** -0.5)

    def forward(self, text_embed):
        return text_embed @ self.projection


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


class DT_Dataset(Dataset):
    def __init__(
        self, env_name, trajectories, horizon, max_episode_steps, return_scale, device,
        task_describe, observation_dim, action_dim,
        prompt_type=None,
        prompt_demos=None, prompt_len=4, use_com_prompt=False,
        use_context=False, context_horizon=4, world_model=None
    ):
        self.env_name = env_name
        self.trajectories = trajectories
        self.horizon = horizon
        self.max_episode_steps = max_episode_steps
        self.return_scale = return_scale
        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu')

        self.prompt_demos = prompt_demos
        self.prompt_len = prompt_len
        self.use_com_prompt = use_com_prompt
        self.prompt_type = prompt_type
        self.task_describe = task_describe
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.use_context = use_context
        self.context_horizon = context_horizon
        self.world_model = world_model

        lengths = [len(s.split()) for s in task_describe]
        self.task_max_length = max(lengths)

        states, traj_lens, returns = [], [], []
        for path in trajectories:
            states.append(path['observations'])
            traj_lens.append(len(path['observations']))
            returns.append(path['rewards'].sum())
        traj_lens, returns = np.array(traj_lens), np.array(returns)

        # used for input normalization
        states = np.concatenate(states, axis=0)
        self.state_mean = np.mean(states, axis=0)
        self.state_std = np.std(states, axis=0) + 1e-6

        self.return_min = np.min(returns)
        self.return_max = np.max(returns)
        self.return_avg = np.average(returns)
        print(
            f'Dataset info: {len(trajectories)} trajectories, {sum(traj_lens)} transitions, returns [{returns.min()}, {returns.max()}]'
        )

        print('Preparing the training data for MetaDT...')
        self.build_index()
        print(f'Size of training data: {self.__len__()}')

        # Initialize and move models to device once
        self.prompt_cache = {}
        if self.prompt_type == 'clip':
            self.clip_model = CLIPTextModel.from_pretrained(
                "pre_trained/clip-vit-base-patch32", local_files_only=True
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                "pre_trained/clip-vit-base-patch32"
            )
            self.clip_model.eval()
            freeze(self.clip_model)
            # Precompute prompts
            self._precompute_clip_prompts()

        elif self.prompt_type == 'aligned_clip':
            self.clip_model = CLIPTextModel.from_pretrained(
                "pre_trained/clip-vit-base-patch32", local_files_only=True
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                "pre_trained/clip-vit-base-patch32")
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj"]
            )
            load_path = f'saves_align/{self.env_name}/clip/'
            self.clip_model = PeftModel.from_pretrained(
                self.clip_model, load_path + "lora_adapter").to(self.device)
            self.text_projection_layer = TextProjection(
                512, 256).to(self.device)
            proj_load_path = load_path + "text_proj/"
            self.text_projection_layer.load_state_dict(
                torch.load(proj_load_path+'state.pt', map_location=self.device))
            self.clip_model.eval()
            self.text_projection_layer.eval()
            freeze(self.clip_model)
            freeze(self.text_projection_layer)
            # Precompute prompts
            self._precompute_clip_prompts()

        elif self.prompt_type == 'aligned_clip_wo_wm':
            self.clip_model = CLIPTextModel.from_pretrained(
                "pre_trained/clip-vit-base-patch32", local_files_only=True
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                "pre_trained/clip-vit-base-patch32")
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj"]
            )
            load_path = f'saves_align_wo_world_model/{self.env_name}/clip/'
            self.clip_model = PeftModel.from_pretrained(
                self.clip_model, load_path + "lora_adapter").to(self.device)
            self.text_projection_layer = TextProjection(
                512, 256).to(self.device)
            proj_load_path = load_path + "text_proj/"
            self.text_projection_layer.load_state_dict(
                torch.load(proj_load_path+'state.pt', map_location=self.device))
            self.clip_model.eval()
            self.text_projection_layer.eval()
            freeze(self.clip_model)
            freeze(self.text_projection_layer)
            # Precompute prompts
            self._precompute_clip_prompts()

        elif self.prompt_type == 'full_tune':
            load_path = f'saves_align_full_params/{self.env_name}/clip/'
            self.clip_model = CLIPTextModel.from_pretrained(
                f"saves_align_full_params/{self.env_name}/clip/clip_model/", local_files_only=True
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                "pre_trained/clip-vit-base-patch32")
            self.text_projection_layer = TextProjection(
                512, 256).to(self.device)
            proj_load_path = load_path + "text_proj/"
            self.text_projection_layer.load_state_dict(
                torch.load(proj_load_path+'state.pt', map_location=self.device))
            # Precompute prompts
            self._precompute_clip_prompts()

        elif self.prompt_type == 'aligned_bert':
            self.bert_model = BertModel.from_pretrained(
                "pre_trained/bert-base-uncased", local_files_only=True).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                "pre_trained/bert-base-uncased")
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["query", "key", "value"]  # target modules
            )
            load_path = f'saves_align/{self.env_name}/bert/'
            self.bert_model = PeftModel.from_pretrained(
                self.bert_model, load_path + "lora_adapter").to(self.device)
            self.text_projection_layer = TextProjection(
                768, 256).to(self.device)
            proj_load_path = load_path + "text_proj/"
            self.text_projection_layer.load_state_dict(
                torch.load(proj_load_path+'state.pt', map_location=self.device))
            self.bert_model.eval()
            self.text_projection_layer.eval()
            freeze(self.bert_model)
            freeze(self.text_projection_layer)
            # Precompute prompts
            self._precompute_bert_prompts()

        elif self.prompt_type == 'aligned_t5':
            self.t5_model = T5EncoderModel.from_pretrained(
                "pre_trained/t5-efficient-small", local_files_only=True).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                "pre_trained/t5-efficient-small")
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q", "k", "v"]  # target modules
            )
            load_path = f'saves_align/{self.env_name}/t5/'
            self.t5_model = PeftModel.from_pretrained(
                self.t5_model, load_path + "lora_adapter").to(self.device)
            self.text_projection_layer = TextProjection(
                512, 256).to(self.device)
            proj_load_path = load_path + "text_proj/"
            self.text_projection_layer.load_state_dict(
                torch.load(proj_load_path+'state.pt', map_location=self.device))
            self.t5_model.eval()
            self.text_projection_layer.eval()
            freeze(self.t5_model)
            freeze(self.text_projection_layer)
            # Precompute prompts
            self._precompute_t5_prompts()

    def _precompute_clip_prompts(self):
        """
        Precompute and cache CLIP prompts for each task_id.
        """
        print("Precomputing CLIP prompts...")
        self.clip_model.eval()
        self.clip_model.to(self.device)
        self.prompt_cache = {}
        for task_id, description in enumerate(self.task_describe):
            inputs = self.tokenizer(
                description, return_tensors="pt", padding=True)
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
            prompt = outputs.pooler_output.to(self.device)
            if self.prompt_type == 'aligned_clip' or self.prompt_type == 'aligned_clip_wo_wm' or self.prompt_type == 'full_tune':
                # prompt = self.clip_projection.text_projection(prompt)
                prompt = self.text_projection_layer(prompt)
            self.prompt_cache[task_id] = prompt.detach()

    def _precompute_bert_prompts(self):
        """
        Precompute and cache Bert prompts for each task_id.
        """
        print("Precomputing Bert prompts...")
        self.bert_model.eval()
        self.bert_model.to(self.device)
        self.prompt_cache = {}
        for task_id, description in enumerate(self.task_describe):
            inputs = self.tokenizer(
                description, return_tensors="pt", padding=True)
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            prompt = outputs.pooler_output.to(self.device)
            if self.prompt_type == 'aligned_bert':
                prompt = self.text_projection_layer(prompt)
            self.prompt_cache[task_id] = prompt.detach()

    def _precompute_t5_prompts(self):
        """
        Precompute and cache T5 prompts for each task_id.
        """
        print("Precomputing T5 prompts...")
        self.t5_model.eval()
        self.t5_model.to(self.device)
        self.prompt_cache = {}
        for task_id, description in enumerate(self.task_describe):
            inputs = self.tokenizer(
                description, return_tensors="pt", padding=True)
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = self.t5_model(**inputs)
            prompt = outputs.last_hidden_state.mean(dim=1).to(self.device)
            if self.prompt_type == 'aligned_t5':
                prompt = self.text_projection_layer(prompt)
            self.prompt_cache[task_id] = prompt.detach()

    def build_index(self):
        self.trajectory_indices = []
        for traj_idx, traj in enumerate(self.trajectories):
            # Precompute normalized observations and rtgs for each trajectory
            traj['norm_observations'] = (
                traj['observations'] - self.state_mean) / self.state_std
            for si in range(1, len(traj['observations'])):
                self.trajectory_indices.append((traj_idx, si))

    def __getitem__(self, index):
        traj_idx, si = self.trajectory_indices[index]
        traj = self.trajectories[traj_idx]

        start_ind = max(0, si - self.horizon + 1)
        state_seg = traj['norm_observations'][start_ind: si+1]
        # state_seg = traj['observations'][start_ind: si+1]
        action_seg = traj['actions'][start_ind: si+1]
        reward_seg = traj['rewards'][start_ind: si+1].reshape(-1, 1)
        done_seg = traj['terminals'][start_ind: si+1].reshape(-1)
        rtg_seg = traj['rtg'][start_ind: si+1].reshape(-1, 1)
        timestep_seg = np.arange(start_ind, si+1).reshape(-1)
        task_id = traj['task_id']

        # padding cutoff
        timestep_seg[timestep_seg >=
                     self.max_episode_steps] = self.max_episode_steps - 1

        # padding and normalization already handled for states
        tlen = state_seg.shape[0]
        pad_len = self.horizon - tlen

        if pad_len > 0:
            state_pad = np.zeros((pad_len, state_seg.shape[1]))
            state_seg = np.concatenate([state_pad, state_seg], axis=0)

            action_pad = np.zeros((pad_len, action_seg.shape[1]))
            action_seg = np.concatenate([action_pad, action_seg], axis=0)

            reward_pad = np.zeros((pad_len, 1))
            reward_seg = np.concatenate([reward_pad, reward_seg], axis=0)

            done_pad = np.ones((pad_len)) * 2  # arbitrary value for padding
            done_seg = np.concatenate([done_pad, done_seg], axis=0)

            rtg_pad = np.zeros((pad_len, 1))
            rtg_seg = np.concatenate([rtg_pad, rtg_seg], axis=0)

            timestep_pad = np.zeros((pad_len))
            timestep_seg = np.concatenate([timestep_pad, timestep_seg], axis=0)

            mask_seg = np.concatenate(
                [np.zeros(pad_len), np.ones(tlen)], axis=0)
        else:
            mask_seg = np.ones(self.horizon)

        # Convert to tensors
        states = torch.tensor(
            state_seg, dtype=torch.float32, device=self.device)
        actions = torch.tensor(
            action_seg, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(
            reward_seg, dtype=torch.float32, device=self.device)
        dones = torch.tensor(done_seg, dtype=torch.long, device=self.device)
        rtg = torch.tensor(rtg_seg, dtype=torch.float32,
                           device=self.device) / self.return_scale
        timesteps = torch.tensor(
            timestep_seg, dtype=torch.long, device=self.device)
        masks = torch.tensor(mask_seg, dtype=torch.long, device=self.device)

        input_seq = (
            states,
            actions,
            rewards,
            dones,
            rtg,
            timesteps,
            masks,
        )

        if self.prompt_type is None:
            return input_seq
        else:
            prompt = self.prompt_cache[task_id]
            return input_seq, prompt

    def __len__(self):
        return len(self.trajectory_indices)

    def get_prompt(self, task_id):
        return self.prompt_cache.get(task_id, None)

    def append_error_to_trajectory(self, traj):
        context_encoder, dynamics_decoder = self.world_model
        context_encoder.eval()
        dynamics_decoder.eval()
        context_encoder.to(self.device)
        dynamics_decoder.to(self.device)

        states = traj['observations']
        actions = traj['actions']
        rewards = traj['rewards'].reshape(-1, 1)

        states_segment, actions_segment, rewards_segment = [], [], []
        for ind in range(states.shape[0]):
            start_ind = max(0, ind-self.context_horizon)

            if ind == 0:
                state_seg = np.zeros((self.context_horizon, states.shape[1]))
                action_seg = np.zeros((self.context_horizon, actions.shape[1]))
                reward_seg = np.zeros((self.context_horizon, rewards.shape[1]))
            else:
                state_seg = states[start_ind: ind]
                action_seg = actions[start_ind: ind]
                reward_seg = rewards[start_ind: ind]

            tlen = state_seg.shape[0]
            state_seg = np.concatenate(
                [np.zeros((self.context_horizon-tlen, state_seg.shape[1])), state_seg], axis=0)
            action_seg = np.concatenate(
                [np.zeros((self.context_horizon-tlen, action_seg.shape[1])), action_seg], axis=0)
            reward_seg = np.concatenate(
                [np.zeros((self.context_horizon-tlen, reward_seg.shape[1])), reward_seg], axis=0)

            states_segment.append(state_seg)
            actions_segment.append(action_seg)
            rewards_segment.append(reward_seg)

        # size: (seq_len, context_horizon, dim)
        states_segment = torch.from_numpy(
            np.stack(states_segment, axis=0)).float().to(self.device)
        actions_segment = torch.from_numpy(
            np.stack(actions_segment, axis=0)).float().to(self.device)
        rewards_segment = torch.from_numpy(
            np.stack(rewards_segment, axis=0)).float().to(self.device)

        # size: (seq_len, dim)
        states = torch.from_numpy(traj['observations']).float().to(self.device)
        actions = torch.from_numpy(traj['actions']).float().to(self.device)
        next_states = torch.from_numpy(
            traj['next_observations']).float().to(self.device)

        with torch.no_grad():
            contexts = context_encoder(states_segment.transpose(
                0, 1), actions_segment.transpose(0, 1), rewards_segment.transpose(0, 1))
            reward_predict = dynamics_decoder(
                states, actions, next_states, contexts).detach().cpu().numpy()

        traj['errors'] = abs(reward_predict - traj['rewards'].reshape(-1, 1))

        return traj

#################################################################
####### Some functions to help dataset construction #############
#################################################################


def convert_data_to_trajectories(data, max_path_length, task_id=None):
    trajectories = []
    start_ind = 0
    current_step = 0
    for ind, terminal in enumerate(data['terminals']):
        current_step += 1
        if terminal or current_step >= max_path_length:
            traj = OrderedDict()
            for key, value in data.items():
                traj[key] = value[start_ind: ind+1]
            traj['rtg'] = discount_cumsum(traj['rewards'], gamma=1.)

            if task_id is not None:
                traj['task_id'] = task_id

            trajectories.append(traj)
            start_ind = ind + 1
            current_step = 0

    print(f'Convert {ind} transitions to {(len(trajectories))} trajectories.')
    return trajectories


def append_context_to_trajectories(trajectories, context_encoder, horizon=4, device='cpu'):
    context_encoder.eval()
    context_encoder = context_encoder.to(device)

    for traj in trajectories:
        states = traj['observations']
        actions = traj['actions']
        rewards = traj['rewards'].reshape(-1, 1)

        states_segment, actions_segment, rewards_segment = [], [], []
        for ind in range(states.shape[0]):
            start_ind = max(0, ind-horizon)

            if ind == 0:
                state_seg = np.zeros((horizon, states.shape[1]))
                action_seg = np.zeros((horizon, actions.shape[1]))
                reward_seg = np.zeros((horizon, rewards.shape[1]))
            else:
                state_seg = states[start_ind: ind]
                action_seg = actions[start_ind: ind]
                reward_seg = rewards[start_ind: ind]

            tlen = state_seg.shape[0]
            state_seg = np.concatenate(
                [np.zeros((horizon-tlen, state_seg.shape[1])), state_seg], axis=0)
            action_seg = np.concatenate(
                [np.zeros((horizon-tlen, action_seg.shape[1])), action_seg], axis=0)
            reward_seg = np.concatenate(
                [np.zeros((horizon-tlen, reward_seg.shape[1])), reward_seg], axis=0)

            states_segment.append(state_seg)
            actions_segment.append(action_seg)
            rewards_segment.append(reward_seg)

        # size: (num_samples, seq_len, dim)
        states_segment = torch.from_numpy(
            np.stack(states_segment, axis=0)).float().to(device)
        actions_segment = torch.from_numpy(
            np.stack(actions_segment, axis=0)).float().to(device)
        rewards_segment = torch.from_numpy(
            np.stack(rewards_segment, axis=0)).float().to(device)

        with torch.no_grad():
            contexts = context_encoder(states_segment.transpose(
                0, 1), actions_segment.transpose(0, 1), rewards_segment.transpose(0, 1))

        traj['contexts'] = contexts.detach().cpu().numpy()

    return trajectories
