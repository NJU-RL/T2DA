import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
import os
# import gymnasium as gym
import gym
from collections import OrderedDict
import pickle
import tqdm
import time

from configs import args_metaworld, args_half_cheetah_vel, args_ant_dir, args_point_robot
from wrapped_metaworld.envs import get_cl_env
from world_model.model import RewardDecoder, StateDecoder, TrajTransfomerEncoder
from world_model.dataset import ContextDataset
# from context.dataset import ContextDataset
import diffuser.utils as utils
from wrapped_metaworld.envs import get_cl_env


parser = argparse.ArgumentParser()
parser.add_argument('--env', default='point-robot',
                    choices=['point-robot', 'ant-dir', 'cheetah-vel', 'metaworld'])
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--context_horizon', type=int, default=20)  # 200 for cheetah-vel, ant-dir and metaworld
parser.add_argument('--context_lr', type=float, default=5e-4)
parser.add_argument('--context_train_epochs', type=int, default=50)
parser.add_argument('--latent_dim', type=int, default=256)
parser.add_argument('--save_context_model_every', type=int, default=1)
parser.add_argument('--context_batch_size', type=int, default=128)
parser.add_argument('--decoder_type', default='reward')
args, rest_args = parser.parse_known_args()
context_horizon = args.context_horizon
args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')


if args.env == 'cheetah-vel':
    half_cheetah_vel_args = args_half_cheetah_vel.get_args(rest_args)
    args_dict = vars(args)
    half_cheetah_vel_args_dict = vars(half_cheetah_vel_args)
    args_dict.update(half_cheetah_vel_args_dict)
    args = argparse.Namespace(**args_dict)
    from register_mujoco import register_mujoco
    register_mujoco()
    train_tasks = [i for i in range(args.num_tasks) if i not in args.eval_tasks]
    with open('datasets/HalfCheetahVel-v0/task_goals.pkl', 'rb') as fp:
        tasks = pickle.load(fp)
    env = gym.make(args.env_name, tasks=tasks, train_task_ids=train_tasks)

elif args.env == 'point-robot':
    point_robot_args = args_point_robot.get_args(rest_args)
    args_dict = vars(args)
    point_robot_args_dict = vars(point_robot_args)
    args_dict.update(point_robot_args_dict)
    args = argparse.Namespace(**args_dict)
    from register_mujoco import register_mujoco
    register_mujoco()
    train_tasks = [i for i in range(args.num_tasks) if i not in args.eval_tasks]
    with open('datasets/PointRobot-v0/task_goals.pkl', 'rb') as fp:
        tasks = pickle.load(fp)
    env = gym.make(args.env_name, tasks=tasks, train_task_ids=train_tasks)

elif args.env == 'ant-dir':
    ant_dir_args = args_ant_dir.get_args(rest_args)
    args_dict = vars(args)
    ant_dir_args_dict = vars(ant_dir_args)
    args_dict.update(ant_dir_args_dict)
    args = argparse.Namespace(**args_dict)
    from register_mujoco import register_mujoco
    register_mujoco()
    train_tasks = [i for i in range(args.num_tasks) if i not in args.eval_tasks]
    with open('datasets/AntDir-v0/task_goals.pkl', 'rb') as fp:
        tasks = pickle.load(fp)
    env = gym.make(args.env_name, tasks=tasks, train_task_ids=train_tasks)

elif args.env == 'metaworld':
    metaworld_args = args_metaworld.get_args(rest_args)
    args_dict = vars(args)
    metaworld_args_dict = vars(metaworld_args)
    args_dict.update(metaworld_args_dict)
    args = argparse.Namespace(**args_dict)
    tasks = ['faucet-close-v2', 'faucet-open-v2', 'door-lock-v2', 'door-unlock-v2',
             'drawer-close-v2', 'window-close-v2', 'window-open-v2', 'coffee-button-v2',
             'drawer-open-v2', 'door-open-v2',
             'button-press-v2',  'button-press-topdown-v2', 'button-press-topdown-wall-v2', 'button-press-wall-v2', 'reach-wall-v2',
             'handle-press-side-v2', 'handle-press-v2', 'handle-pull-v2',
             'plate-slide-back-v2', 'plate-slide-side-v2', 'plate-slide-v2', 'plate-slide-back-side-v2']
    env = get_cl_env(tasks, int(1e7) + 1)
    env.task_describe = ['rotate faucet clockwise', 'rotate faucet counter-clockwise', 'lock door by rotate clockwise', 'unlock door by rotate counter-clockwise',
                         'push and close drawer', 'push and close window', 'push and open window', 'push button on coffee machine',
                         'open drawer', 'open door with revolving joint',
                         'press button', 'press button from top', 'bypass wall and press button from top', 'bypass wall and press button', 'bypass wall and reach goal',
                         'press handle down sideways', 'press handle down', 'pull handle up',
                         'slide plate back', 'slide plate side', 'slide plate', 'slide plate back side']

env.eval_tasks = args.eval_tasks
torch.manual_seed(args.seed)
np.random.seed(args.seed)
np.set_printoptions(precision=4, suppress=True)

observation_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
if args.env != 'metaworld':
    env.seed(args.seed)
    env.action_space.seed(args.seed)

# load dataset
data_path = f'datasets/{args.env_name}'
keys = ['observations', 'actions', 'rewards',
        'next_observations', 'terminals']

train_data, test_data = OrderedDict(), OrderedDict()
for key in keys:
    train_data[key] = []
    test_data[key] = []

train_data['task_ids'] = []
test_data['task_ids'] = []

# ============= load dataset =================
if args.env in ['point-robot', 'cheetah-vel', 'ant-dir']:
    for task_id in range(args.num_tasks):
        with open(f'{data_path}/dataset_task_{task_id}.pkl', "rb") as f:
            data = pickle.load(f)

        for ind, terminal in enumerate(data['terminals']):
            if (ind + 1) % args.max_episode_steps == 0:
                data['terminals'][ind] = True

        # Add task_id information
        num_samples = len(data['observations'])
        task_ids = np.full((num_samples, 1), task_id, dtype=np.int32)

        for key, values in data.items():
            if key in keys:
                if task_id not in args.eval_tasks:
                    train_data[key].append(values)
                else:
                    test_data[key].append(values)

        if task_id not in args.eval_tasks:
            train_data['task_ids'].append(task_ids)
        else:
            test_data['task_ids'].append(task_ids)

# if env is metaworld, select expert and medium data
elif args.env == 'metaworld':
    for task_id in range(args.num_tasks):
            with open(f'{data_path}/dataset_task_{task_id}.pkl', "rb") as f:
                data = pickle.load(f)

            for ind, terminal in enumerate(data['terminals']):
                if (ind + 1) % 200 == 0:
                    data['terminals'][ind] = True

            # Add task_id information
            num_samples = len(data['observations'])
            task_ids = np.full((num_samples, 1), task_id, dtype=np.int32)

            for key, values in data.items():
                if key in keys:
                    if task_id not in args.eval_tasks:
                        train_data[key].append(values)
                    else:
                        test_data[key].append(values)

            if task_id not in args.eval_tasks:
                train_data['task_ids'].append(task_ids)
            else:
                test_data['task_ids'].append(task_ids)


for key, values in train_data.items():
    train_data[key] = np.concatenate(train_data[key], axis=0)
    test_data[key] = np.concatenate(test_data[key], axis=0)

train_dataset = ContextDataset(
    train_data, horizon=args.context_horizon, device=args.device)
train_dataloader = DataLoader(
    train_dataset, batch_size=args.context_batch_size, shuffle=True)

test_dataset = ContextDataset(
    test_data, horizon=args.context_horizon, device=args.device)
test_dataloader = DataLoader(
    test_dataset, batch_size=args.context_batch_size, shuffle=True)

save_path = f'saves_world_model/{args.env}/{args.decoder_type}/horizon{args.context_horizon}'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Tesnorboard
writer = SummaryWriter(save_path)


traj_encoder = TrajTransfomerEncoder(state_dim=observation_dim,
                                     action_dim=action_dim, latent_dim=args.latent_dim).to(args.device)

reward_decoder = RewardDecoder(
    state_dim=observation_dim, action_dim=action_dim, latent_dim=args.latent_dim).to(args.device)

state_decoder = StateDecoder(
    state_dim=observation_dim, action_dim=action_dim, latent_dim=args.latent_dim).to(args.device)

optimizer = torch.optim.Adam(
    [*traj_encoder.parameters(), *reward_decoder.parameters()], lr=args.context_lr)

save_model_path = save_path + '/checkpoints/'
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)

global_step = 0
best_loss = float('inf')

for epoch in range(args.context_train_epochs):
    print(f'\n========== Epoch {epoch+1} ==========')

    traj_encoder.train()
    reward_decoder.train()
    for step, (segment, transition) in enumerate(train_dataloader):
        state, action, reward, next_state, _, _ = transition

        traj_embed = traj_encoder(segment)
        if args.decoder_type == 'state':
            next_state_predict = state_decoder(state, action, traj_embed)
            loss = F.mse_loss(next_state_predict, next_state)

        else:
            reward_predict = reward_decoder(
                state, action, next_state, traj_embed)
            # num_transitions = reward.shape[1]
            reward = reward.view(-1, 1)
            loss = F.mse_loss(reward_predict, reward)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [*traj_encoder.parameters(), *reward_decoder.parameters()], 1.0)
        optimizer.step()

        global_step += 1
        writer.add_scalar('loss/train', loss.item(), global_step)

        # if global_step % 50 == 0:
        #     print(f'loss: {loss:.5f}')

        # if global_step % 200 == 0:
    with torch.no_grad():
        traj_encoder.eval()
        reward_decoder.eval()
        segment, transition = next(iter(test_dataloader))
        state, action, reward, next_state, _, _ = transition

        traj_embed = traj_encoder(segment)
        if args.decoder_type == 'state':
            next_state_predict = state_decoder(
                state, action, traj_embed)
            loss = F.mse_loss(next_state_predict, next_state)
            print(
                f'Predicted state: {next_state_predict.detach().cpu().numpy()[:1].reshape(-1)}')
            print(
                f'   Real state  : {next_state.detach().cpu().numpy()[:1].reshape(-1)}')

        else:
            reward_predict = reward_decoder(
                state, action, next_state, traj_embed)
            reward = reward.view(-1, 1)
            loss = F.mse_loss(reward_predict, reward)
            print(
                f'Predicted rewards: {reward_predict.detach().cpu().numpy()[:8].reshape(-1)}')
            print(
                f'   Real rewards  : {reward.detach().cpu().numpy()[:8].reshape(-1)}')

        writer.add_scalar('reward loss/test', loss, global_step)
        print(
            f'Model Evaluation at step{global_step}, test loss: {loss}')
        # print(state.mean())
    traj_encoder.train()
    reward_decoder.train()

    if (epoch + 1) % args.save_context_model_every == 0:
        torch.save(traj_encoder.state_dict(),
                   save_model_path + f'state_{epoch}.pt')
