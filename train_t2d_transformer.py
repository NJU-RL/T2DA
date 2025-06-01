import diffuser.environments
import pickle
from wrapped_metaworld.envs import get_cl_env
from transformer.evaluation import evaluate_episode_rtg
from transformer.dataset import DT_Dataset, append_context_to_trajectories, convert_data_to_trajectories
from transformer.model import DecisionTransformer
from transformer.trainer import DT_Trainer
from configs import args_metaworld, args_point_robot, args_half_cheetah_vel, args_ant_dir
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import argparse
import gym
import numpy as np
import os
import torch
import time

from utils import copy_files
start_time = time.time()


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='point-robot')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=str, default='cuda:0')
# parser.add_argument('--eval_tasks', default=[45, 46, 47, 48, 49])
parser.add_argument('--prompts_type', type=str, default=None, choices=['clip', 'aligned_clip', 'aligned_clip_wo_wm', 'aligned_t5', 'aligned_bert', None])
args, rest_args = parser.parse_known_args()
env_type = args.env
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

if env_type == 'metaworld':
    metaworld_args = args_metaworld.get_args(rest_args)
    args_dict = vars(args)
    metaworld_args_dict = vars(metaworld_args)
    args_dict.update(metaworld_args_dict)
    args = argparse.Namespace(**args_dict)
elif env_type == 'ant-dir':
    ant_dir_args = args_ant_dir.get_args(rest_args)
    args_dict = vars(args)
    ant_dir_args_dict = vars(ant_dir_args)
    args_dict.update(ant_dir_args_dict)
    args = argparse.Namespace(**args_dict)
elif env_type == 'point-robot':
    point_robot_args = args_point_robot.get_args(rest_args)
    args_dict = vars(args)
    point_robot_args_dict = vars(point_robot_args)
    args_dict.update(point_robot_args_dict)
    args = argparse.Namespace(**args_dict)
elif env_type == 'cheetah-vel' :
    half_cheetah_vel_args = args_half_cheetah_vel.get_args(rest_args)
    args_dict = vars(args)
    half_cheetah_vel_args_dict = vars(half_cheetah_vel_args)
    args_dict.update(half_cheetah_vel_args_dict)
    args = argparse.Namespace(**args_dict)
else:
    raise NotImplementedError

torch.manual_seed(args.seed)
np.random.seed(args.seed)
np.set_printoptions(precision=3, suppress=True)

if env_type == 'metaworld':
    # make env, multi-task setting
    tasks = ['faucet-close-v2', 'faucet-open-v2', 'door-lock-v2', 'door-unlock-v2',
             'drawer-close-v2', 'window-close-v2', 'window-open-v2', 'coffee-button-v2',
             'drawer-open-v2', 'door-open-v2',
             'button-press-v2',  'button-press-topdown-v2', 'button-press-topdown-wall-v2', 'button-press-wall-v2', 'reach-wall-v2',
             'handle-press-side-v2', 'handle-press-v2', 'handle-pull-v2',
             'plate-slide-back-v2', 'plate-slide-side-v2', 'plate-slide-v2', 'plate-slide-back-side-v2']
    env = get_cl_env(tasks, int(1e7) + 1)
    env.task_describe = ['rotate faucet clockwise', 'rotate faucet counter-clockwise', 'lock door by rotating clockwise', 'unlock door by rotating counter-clockwise',
                         'push and close drawer', 'push and close window', 'push and open window', 'push button on coffee machine',
                         'open drawer', 'open door with revolving joint',
                         'press button', 'press button from top', 'bypass wall and press button from top', 'bypass wall and press button', 'bypass wall and reach goal',
                         'press handle down sideways', 'press handle down', 'pull handle up',
                         'slide plate back', 'slide plate side', 'slide plate', 'slide plate back side']

elif env_type == 'ant-dir':
    from register_mujoco import register_mujoco
    register_mujoco()
    train_tasks = [i for i in range(args.num_tasks) if i not in args.eval_tasks]
    with open('datasets/AntDir-v0/task_goals.pkl', 'rb') as fp:
        tasks = pickle.load(fp)
    env = gym.make(args.env_name, tasks=tasks, train_task_ids=train_tasks)
    env.task_describe = [f"Direction {round(number=task['goal'], ndigits=2)}" for task in tasks]

elif env_type == 'cheetah-vel':
    from register_mujoco import register_mujoco
    register_mujoco()
    train_tasks = [i for i in range(args.num_tasks) if i not in args.eval_tasks]
    with open('datasets/HalfCheetahVel-v0/task_goals.pkl', 'rb') as fp:
        tasks = pickle.load(fp)
    env = gym.make(args.env_name, tasks=tasks, train_task_ids=train_tasks)
    env.task_describe = [f"{task['velocity']:.4f}" for task in tasks]

elif env_type == 'point-robot':
    from register_mujoco import register_mujoco
    register_mujoco()
    train_tasks = [i for i in range(args.num_tasks) if i not in args.eval_tasks]
    with open('datasets/PointRobot-v0/task_goals.pkl', 'rb') as fp:
        tasks = pickle.load(fp)
    env = gym.make(args.env_name, tasks=tasks, train_task_ids=train_tasks)
    env.task_describe = [f"Navigate to position ({round(number=task[0], ndigits=2)}, {round(number=task[1], ndigits=2)})" for task in tasks]

env.eval_tasks = args.eval_tasks
args.bucket = f'saves_dt/{args.env_name}/{args.seed}/prompt_{args.prompts_type}'
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# prepare data from training tasks
# train_task_ids = np.arange(args.num_train_tasks)
# eval_train_task_ids = np.arange(5)
eval_train_task_ids = [3, 5, 6, 10]

train_trajectories, all_demos = [], []
for task_id in np.arange(args.num_tasks):
    if task_id not in args.eval_tasks:
        with open(f'datasets/{args.env_name}/dataset_task_{task_id}.pkl', "rb") as f:#TODO
            data = pickle.load(f)

        print(f'\n========== Processing data of Task {task_id} ==========')
        if env_type == 'point-robot':
            trajectories = convert_data_to_trajectories(data, max_path_length=args.max_episode_steps, task_id=task_id)
            for traj in trajectories:
                train_trajectories.append(traj)
        elif env_type == 'cheetah-vel':
            trajectories = convert_data_to_trajectories(data, max_path_length=args.max_episode_steps, task_id=task_id)
            for traj in trajectories:
                train_trajectories.append(traj)
        elif env_type == 'ant-dir':
            trajectories = convert_data_to_trajectories(data, max_path_length=args.max_episode_steps, task_id=task_id)
            for traj in trajectories:
                train_trajectories.append(traj)
        else:
            trajectories = convert_data_to_trajectories(data, max_path_length=args.max_episode_steps, task_id=task_id)
            returns = [traj['rewards'].sum() for traj in trajectories]
            sorted_inds = sorted(range(len(returns)),
                                 key=lambda x: returns[x], reverse=True)
            quality_ind = 0
            selected_trajs = [trajectories[sorted_inds[i]]
                              for i in range(quality_ind, quality_ind+200)]
            for traj in selected_trajs:
                train_trajectories.append(traj)

print(f'\nProcessed {len(train_trajectories)} trajectories from {args.num_train_tasks} training tasks...')

train_dataset = DT_Dataset(
    args.env,
    train_trajectories,
    args.dt_horizon,
    args.max_episode_steps,
    args.dt_return_scale,
    device,
    # task_describe=env.task_describe,
    env.task_describe,
    state_dim,
    action_dim,

    prompt_type=args.prompts_type,
    prompt_demos=all_demos,
    prompt_len=args.prompt_len,
)
state_mean, state_std = train_dataset.state_mean, train_dataset.state_std

train_dataloader = DataLoader(train_dataset, batch_size=args.transformer_batch_size, shuffle=True)

# save the arguments for debugging
results_dir = args.bucket
writer_dir = results_dir + '/tensorboard'
if not os.path.exists(writer_dir):
    os.makedirs(writer_dir)
writer = SummaryWriter(writer_dir)
copy_files(writer_dir, folders=['config', 'configs', 'diffuser', 'transformer', 'wrapped_metaworld'], files=['train_t2d_transformer.py', 'train_t2d_diffuser.py'])

variant = vars(args)
variant.update(version=f"Decision Transformer")
variant.update(state_dim=state_dim)
variant.update(action_dim=action_dim)
variant.update(return_min=float(train_dataset.return_min))
variant.update(return_max=float(train_dataset.return_max))
variant.update(return_avg=float(train_dataset.return_avg))

model = DecisionTransformer(
    state_dim=state_dim,
    act_dim=action_dim,
    max_length=args.dt_horizon,
    max_ep_len=args.max_episode_steps,
    hidden_size=args.dt_embed_dim,
    n_layer=args.dt_n_layer,
    n_head=args.dt_n_head,
    n_inner=4*args.dt_embed_dim,
    activation_function=args.dt_activation_function,
    n_positions=1024,
    resid_pdrop=args.dt_dropout,
    attn_pdrop=args.dt_dropout,
    prompt_type=args.prompts_type,
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.dt_lr,
    weight_decay=args.dt_weight_decay,
)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lambda steps: min((steps+1)/args.transformer_warmup_steps, 1)
)

agent = DT_Trainer(model, optimizer)

global_step = 0
while global_step <= args.transformer_num_iters:
    for step, batch in enumerate(train_dataloader):
        # print(batch)
        # print(step)
        if args.prompts_type != None:
            states, actions, rewards, dones, rtg, timesteps, masks = batch[0]
            train_loss = agent.train_step(
                states, actions, rewards, dones, rtg, timesteps, masks, prompt=batch[1])
        else:
            states, actions, rewards, dones, rtg, timesteps, masks = batch
            train_loss = agent.train_step(
                states, actions, rewards, dones, rtg, timesteps, masks)
        scheduler.step()

        global_step += 1
        writer.add_scalar('train/loss', train_loss, global_step)

        if global_step % args.transformer_eval_iters == 0:
            print(
                f'\n===== Evaluate at {global_step} training iterations =====')
            # evaluate on five tranining tasks
            model.eval()
            avg_epi_return = 0.0
            avg_max_return_offline = 0.0
            print(f'\n---------- Evaluate on five training tasks ----------')
            for task_id in eval_train_task_ids:
                env.set_task_idx(
                    task_id) if env_type == 'metaworld' else env.reset_task(task_id)
                # target_ret = task_info[f'task {task_id}']['return_scale'][1]
                target_ret = args.dt_target_return

                epi_return, epi_length, trajectory = evaluate_episode_rtg(
                    env,
                    state_dim,
                    action_dim,
                    model,
                    max_episode_steps=args.max_episode_steps,
                    scale=args.dt_return_scale,
                    state_mean=state_mean,
                    state_std=state_std,
                    device=device,
                    target_return=target_ret,
                    num_eval_episodes=args.num_eval_episodes,
                    dataset=train_dataset,
                    task_id=task_id,
                    prompt_type=args.prompts_type
                )
                avg_epi_return += epi_return
                avg_max_return_offline += target_ret
                print(
                    f'Evaluate on the {task_id}-th training task, target return {target_ret:.2f}, received return {epi_return:.2f}')

            avg_epi_return /= len(eval_train_task_ids)
            avg_max_return_offline /= len(eval_train_task_ids)
            writer.add_scalar(f'return/train tasks',
                              avg_epi_return, global_step)

            print(
                f'\nAverage performance on five training tasks, received return {avg_epi_return:.2f}, average max return from offline dataset {avg_max_return_offline:.2f}')

            ### for debugging, print the evaluation trajctory ###
            # print(f'Print the example evaluation trajectory of last evaluation task')
            # env.print_task()
            # for transition in trajectory: print(transition)

            # evaluate on five test tasks
            avg_epi_return = 0.0
            avg_max_return_offline = 0.0
            print(f'\n---------- Evaluate on five test tasks ----------')
            for task_id in args.eval_tasks:
                env.set_task_idx(
                    task_id) if env_type == 'metaworld' else env.reset_task(task_id)
                # target_ret = task_info[f'task {task_id}']['return_scale'][1]
                target_ret = args.dt_target_return

                epi_return, epi_length, trajectory = evaluate_episode_rtg(
                    env,
                    state_dim,
                    action_dim,
                    model,
                    max_episode_steps=args.max_episode_steps,
                    scale=args.dt_return_scale,
                    state_mean=state_mean,
                    state_std=state_std,
                    device=device,
                    target_return=target_ret,
                    num_eval_episodes=args.num_eval_episodes,
                    dataset=train_dataset,
                    task_id=task_id,
                    prompt_type=args.prompts_type,
                )
                avg_epi_return += epi_return
                avg_max_return_offline += target_ret
                print(
                    f'Evaluate on the {task_id}-th test task, target return {target_ret:.2f}, received return {epi_return:.2f}')
                writer.add_scalar(
                    f'return/test task{task_id}', epi_return, global_step)

            avg_epi_return /= len(args.eval_tasks)
            avg_max_return_offline /= len(args.eval_tasks)
            writer.add_scalar(f'return/test tasks',
                              avg_epi_return, global_step)

            print(
                f'\nAverage performance on five test tasks, received return {avg_epi_return:.2f}, average max return from offline dataset {avg_max_return_offline:.2f}')

            ### for debugging, print the evaluation trajctory ###
            # print(f'Print the example evaluation trajectory of last evaluation task')
            # env.print_task()
            # for transition in trajectory: print(transition)

            print(
                f'\nElapsed time: {(time.time()-start_time)/60.:.2f} minutes')
