import argparse
import gym
import json
import pickle
import torch
import numpy as np

from glob import glob
from configs import args_point_robot, args_half_cheetah_vel, args_ant_dir
from collections import OrderedDict
from data_collection.sac import SAC
from data_collection.replay_memory import ReplayMemory
from pathlib import Path
import re
from typing import List

from register_mujoco import register_mujoco
register_mujoco()


def set_seed(seed, env):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)

def roll_out(env: gym.Env, replaybuffer: ReplayMemory, returns: List[float]):
    env.reset_task(task_id)
    total_timestep = 0
    while total_timestep < local_args.capacity:
        episode_return = 0.
        state = env.reset()
        for step in range(args['max_episode_steps']):
            action = agent.select_action(state, False)
            action = np.clip(action, action_min, action_max)
            next_state, reward, done, _ = env.step(action)
            mask = True if (step == args['max_episode_steps'] - 1) else (not done)
            replaybuffer.push(state, action, reward, next_state, done, mask)
            
            state = next_state
            total_timestep += 1
            episode_return += reward
            
            if done:
                break
        
        returns.append(episode_return)

    return returns, replaybuffer


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='point-robot', help='environment', choices=['point-robot', 'cheetah-vel', 'ant-dir'])
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=123456)
parser.add_argument('--task_id_start', type=int, default=0)
parser.add_argument('--task_id_end', type=int, default=5, help='collect dataset of tasks from task_id_start to task_id_end-1')
parser.add_argument('--capacity', type=int, default=80, help='total timesteps of one checkpoint')
local_args, rest_args = parser.parse_known_args()

if local_args.env == 'point-robot':
    args = vars(args_point_robot.get_args(rest_args))
    with open('./datasets/PointRobot-v0/task_goals.pkl', 'rb') as fp:
        tasks = pickle.load(fp)
elif local_args.env == 'cheetah-vel':
    args = vars(args_half_cheetah_vel.get_args(rest_args))
    with open('./datasets/HalfCheetahVel-v0/task_goals.pkl', 'rb') as fp:
        tasks = pickle.load(fp)
elif local_args.env == 'ant-dir':
    args = vars(args_ant_dir.get_args(rest_args))
    with open('./datasets/AntDir-v0/task_goals.pkl', 'rb') as fp:
        tasks = pickle.load(fp)
else:
    raise NotImplementedError(f"Unsupported environment: {local_args.env}")

train_tasks = [i for i in range(args['num_tasks']) if i not in args['eval_tasks']]
env = gym.make(args['env_name'], tasks=tasks, train_task_ids=train_tasks)
args['device'] = torch.device(local_args.device) if torch.cuda.is_available() else torch.device('cpu')
args['save_path'] = Path(f"./datasets/{args['env_name']}")
args['save_path'].mkdir(parents=True, exist_ok=True)

# environment
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_min, action_max = env.action_space.low, env.action_space.high
action_abs_min = min(np.abs(action_min).min(), np.abs(action_max).min())

# set seed
set_seed(local_args.seed, env)

return_scale_info = OrderedDict()
agent = SAC(env, args['hidden_dim'], args['alpha'], args['lr'], args['gamma'], args['tau'], args['device'])
for task_id in range(local_args.task_id_start, local_args.task_id_end):
    model_paths = glob(f"./datasets/{args['env_name']}/checkpoints/task_{task_id}/*.pt")
    model_paths = sorted(model_paths, key=lambda x: int(re.findall(r'\d+', Path(x).name)[0]))
    replaybuffer = ReplayMemory(local_args.capacity * len(model_paths), local_args.seed)
    episode_returns = []
    
    for model_path in model_paths:
        agent.load(model_path)
        episode_returns, replaybuffer = roll_out(env, replaybuffer, episode_returns)
    
    replaybuffer.save_buffer(args['save_path'] / f"dataset_task_{task_id}.pkl")
    return_scale_info[f"task_{task_id}"] = [min(episode_returns), max(episode_returns)]

with open(args['save_path'] / f"return_scale_info_{local_args.task_id_start}.json", 'w') as fp:
    json.dump(return_scale_info, fp, indent=4)