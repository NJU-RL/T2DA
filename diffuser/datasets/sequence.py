from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn

from transformers import AutoTokenizer, CLIPTextModel
from peft import LoraConfig, PeftModel

from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer
from utils import freeze

PromptRewardBatch = namedtuple(
    'Batch', 'trajectories conditions returns prompts')
RewardBatch = namedtuple('Batch', 'trajectories conditions returns')
AlignBatch = namedtuple('Batch', 'trajectories descriptions task_ids')
Batch = namedtuple('Batch', 'trajectories conditions')


class TextProjection(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.projection = nn.Parameter(torch.empty(input_dim, latent_dim))
        nn.init.normal_(self.projection, mean=0.0, std=input_dim ** -0.5)

    def forward(self, text_embed):
        return text_embed @ self.projection


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        env='hopper-medium-replay',
        horizon=64,
        normalizer='GuassianNormalizer',
        preprocess_fns=[],
        max_path_length=1000,
        max_n_episodes=150000,
        termination_penalty=0,
        use_padding=True,
        discount=0.99,
        returns_scale=1000,
        include_returns=False,
        # hyperparameters for the prompt design
        prompts_type=None,
        num_demos=5,
        prompt_traj_len=4,
        align_traj_len=200,
        align_num_demos=400,
        env_name='hopper-medium-replay',
        model_device='cpu',
    ):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.returns_scale = returns_scale
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.discount = discount
        self.discounts = self.discount ** np.arange(
            self.max_path_length)[:, None]
        self.use_padding = use_padding
        self.include_returns = include_returns
        itr = sequence_dataset(env, self.preprocess_fn)

        self.prompts_type = prompts_type
        self.num_demos = num_demos
        self.prompt_traj_len = prompt_traj_len
        self.align_traj_len = align_traj_len

        if prompts_type == 'clip':
            self.clip_model = CLIPTextModel.from_pretrained("pre_trained/clip-vit-base-patch32", local_files_only=True).to(model_device)
            self.tokenizer = AutoTokenizer.from_pretrained("pre_trained/clip-vit-base-patch32")
            self.clip_model.eval()
            freeze(self.clip_model)

            self.task_describle = env.task_describe
            with torch.no_grad():
                inputs = self.tokenizer(self.task_describle, padding=True, return_tensors="pt").to(model_device)
                self.prompt_cache = self.clip_model(**inputs).pooler_output.cpu()

        elif prompts_type == 'aligned_clip':
            self.clip_model = CLIPTextModel.from_pretrained("pre_trained/clip-vit-base-patch32", local_files_only=True).to(model_device)
            self.tokenizer = AutoTokenizer.from_pretrained("pre_trained/clip-vit-base-patch32")
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj"]
            )
            load_path = f'saves_align/{env_name}/clip/'
            self.clip_model = PeftModel.from_pretrained(self.clip_model, load_path + "lora_adapter").to(model_device)
            self.text_projection_layer = TextProjection(512, 256).to(model_device)
            proj_load_path = load_path + "text_proj/"
            self.text_projection_layer.load_state_dict(torch.load(proj_load_path + 'state.pt', map_location=model_device))
            self.clip_model.eval()
            self.text_projection_layer.eval()
            freeze(self.clip_model)
            freeze(self.text_projection_layer)

            self.task_describle = env.task_describe
            with torch.no_grad():
                inputs = self.tokenizer(self.task_describle, padding=True, return_tensors="pt").to(model_device)
                outputs = self.clip_model(**inputs).pooler_output
                self.prompt_cache = self.text_projection_layer(outputs).cpu()

        elif prompts_type == 'aligned_clip_wo_wm':
            self.clip_model = CLIPTextModel.from_pretrained("pre_trained/clip-vit-base-patch32", local_files_only=True).to(model_device)
            self.tokenizer = AutoTokenizer.from_pretrained("pre_trained/clip-vit-base-patch32")
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj"]
            )
            load_path = f'saves_align_wo_world_model/{env_name}/clip/'
            self.clip_model = PeftModel.from_pretrained(self.clip_model, load_path + "lora_adapter").to(model_device)
            self.text_projection_layer = TextProjection(512, 256).to(model_device)
            proj_load_path = load_path + "text_proj/"
            self.text_projection_layer.load_state_dict(torch.load(proj_load_path + 'state.pt', map_location=model_device))
            self.clip_model.eval()
            self.text_projection_layer.eval()
            freeze(self.clip_model)
            freeze(self.text_projection_layer)

            self.task_describle = env.task_describe
            with torch.no_grad():
                inputs = self.tokenizer(self.task_describle, padding=True, return_tensors="pt").to(model_device)
                outputs = self.clip_model(**inputs).pooler_output
                self.prompt_cache = self.text_projection_layer(outputs).cpu()

        fields = ReplayBuffer(
            max_n_episodes, max_path_length, termination_penalty)
        epi_returns = []
        for i, episode in enumerate(itr):
            fields.add_path(episode)
            discounts = self.discounts[:len(episode['rewards'])].reshape(-1)
            epi_returns.append(
                (discounts * episode['rewards'].reshape(-1)).sum())
        fields.finalize()
        # print(f'Episode return: [{min(epi_returns)}, {max(epi_returns)}]')

        self.normalizer = DatasetNormalizer(
            fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions']):
    # def normalize(self, keys=['observations']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(
                self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(
                self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        # actions = self.fields.normed_actions[path_ind, start:end]
        actions = self.fields.actions[path_ind, start:end]

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)

        if self.include_returns:
            rewards = self.fields.rewards[path_ind, start:]
            discounts = self.discounts[:len(rewards)]
            returns = (discounts * rewards).sum()
            returns = np.array([returns/self.returns_scale], dtype=np.float32)

            if self.prompts_type is not None and self.prompts_type != 'align':
                if self.prompts_type in ['clip', 'aligned_clip', 'aligned_clip_wo_wm']:
                    task_id = int(self.fields.task_id[path_ind][0])
                    prompts = self.prompt_cache[task_id]
                elif self.prompts_type == 't5' or self.prompts_type == 'aligned_t5':
                    task_id = int(self.fields.task_id[path_ind][0])
                    # prompts = self.task_describle[task_id]
                    prompts = self.prompt_cache[task_id]
                else:
                    assert NotImplementedError

                batch = PromptRewardBatch(
                    trajectories, conditions, returns, prompts)
            elif self.prompts_type == 'align':
                task_id = int(self.fields.task_id[path_ind][0])
                traj = self.get_traj_align(task_id)
                description = self.task_describle[task_id]
                batch = AlignBatch(traj, description, task_id)

            else:
                batch = RewardBatch(trajectories, conditions, returns)
        else:
            batch = Batch(trajectories, conditions)
        return batch

    def get_traj_prompts(self, task_id):
        demo = self.prompt_trajectories[task_id][np.random.randint(
            len(self.prompt_trajectories[task_id]))]

        seq_len = demo['observations'].shape[0]

        #### Default: select the tail segment from the demo trajectory ####
        si = max(0, seq_len-self.prompt_traj_len)

        prompt_obs = demo['observations'][si: si+self.prompt_traj_len]
        prompt_actions = demo['actions'][si: si+self.prompt_traj_len]
        prompt_rewards = demo['rewards'][si: si +
                                         self.prompt_traj_len].reshape(-1, 1)
        prompt_timesteps = np.arange(si, si+prompt_obs.shape[0]).reshape(-1)
        # padding cutoff
        if hasattr(self.env, 'max_episode_steps'):
            prompt_timesteps[prompt_timesteps >=
                             self.env.max_episode_steps] = self.env.max_episode_steps - 1
        else:
            prompt_timesteps[prompt_timesteps >= 200] = 199

        prompts = {}
        tlen = prompt_obs.shape[0]
        prompts['observations'] = np.concatenate([np.zeros(
            (self.prompt_traj_len - tlen, prompt_obs.shape[1])), prompt_obs], axis=0).astype(np.float32)
        prompts['actions'] = np.concatenate([np.ones(
            (self.prompt_traj_len - tlen, prompt_actions.shape[1])) * -10., prompt_actions], axis=0).astype(np.float32)
        prompts['rewards'] = np.concatenate([np.zeros(
            (self.prompt_traj_len - tlen, 1)), prompt_rewards], axis=0).astype(np.float32)
        prompts['timesteps'] = np.concatenate([np.zeros(
            (self.prompt_traj_len - tlen)), prompt_timesteps], axis=0).astype(np.int32)
        prompts['masks'] = np.concatenate(
            [np.zeros((self.prompt_traj_len - tlen)), np.ones((tlen))], axis=0).astype(np.int32)

        return prompts

    def get_traj_align(self, task_id):
        demo = self.prompt_trajectories[task_id][np.random.randint(
            len(self.prompt_trajectories[task_id]))]

        seq_len = demo['observations'].shape[0]

        #### Default: select the tail segment from the demo trajectory ####
        # si = max(0, seq_len-self.align_traj_len)
        end_idx = seq_len-self.align_traj_len
        # end_idx = end_idx if end_idx > self.align_traj_len else self.align_traj_len
        # si = np.random.randint(
        #     0, end_idx) if self.align_traj_len != 200 else 0
        # si = max(0, end_idx)
        si = np.random.randint(0, end_idx) if end_idx > 0 else 0
        # print(si)

        prompt_obs = demo['observations'][si: si+self.align_traj_len]
        prompt_actions = demo['actions'][si: si+self.align_traj_len]
        prompt_rewards = demo['rewards'][si: si +
                                         self.align_traj_len].reshape(-1, 1)
        prompt_timesteps = np.arange(si, si+prompt_obs.shape[0]).reshape(-1)
        # padding cutoff
        if hasattr(self.env, 'max_episode_steps'):
            prompt_timesteps[prompt_timesteps >=
                             self.env.max_episode_steps] = self.env.max_episode_steps - 1
        else:
            prompt_timesteps[prompt_timesteps >= 200] = 199

        prompts = {}
        tlen = prompt_obs.shape[0]
        prompts['observations'] = np.concatenate([np.zeros(
            (self.align_traj_len - tlen, prompt_obs.shape[1])), prompt_obs], axis=0).astype(np.float32)
        prompts['actions'] = np.concatenate([np.ones(
            (self.align_traj_len - tlen, prompt_actions.shape[1])) * -10., prompt_actions], axis=0).astype(np.float32)
        prompts['rewards'] = np.concatenate([np.zeros(
            (self.align_traj_len - tlen, 1)), prompt_rewards], axis=0).astype(np.float32)
        prompts['timesteps'] = np.concatenate([np.zeros(
            (self.align_traj_len - tlen)), prompt_timesteps], axis=0).astype(np.int32)
        prompts['masks'] = np.concatenate(
            [np.zeros((self.align_traj_len - tlen)), np.ones((tlen))], axis=0).astype(np.int32)

        return prompts
