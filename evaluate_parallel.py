from wrapped_metaworld.envs import get_cl_env
import diffuser.utils as utils
from ml_logger import logger
import torch
from copy import deepcopy
import numpy as np
import pickle
import gym
from diffuser.utils.arrays import to_torch, to_np, to_device
import argparse
import json
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
import time

from utils import copy_files
start_time = time.time()


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--env', type=str, default='point-robot',
                    choices=['metaworld', 'cheetah-vel', 'point-robot', 'ant-dir'])
parser.add_argument('--model', type=str, default='transformer',
                    choices=['unet', 'transformer'])
parser.add_argument('--prompts_type', type=str,
                    default=None, choices=['clip', 'aligned_clip', 'aligned_clip_wo_wm', None])
args = parser.parse_args()

if args.env == 'metaworld':
    from config.metaworld_config import Config
    Config.prompts_type = args.prompts_type
    Config.bucket += f'/{args.model}/prompt_{Config.prompts_type}'
    num_eval = 4

elif args.env in ['point-robot', 'cheetah-vel', 'ant-dir']:
    from register_mujoco import register_mujoco
    register_mujoco()
    if args.env == 'point-robot':
        from config.pointrobot_meta_config import Config
    elif args.env == 'cheetah-vel':
        from config.half_cheetah_vel_meta_config import Config
    elif args.env == 'ant-dir':
        from config.ant_dir_config import Config
    Config.prompts_type = args.prompts_type
    Config.bucket += f'/{args.model}/prompt_{Config.prompts_type}'
    num_eval = 5

else:
    assert NotImplementedError

results = OrderedDict()

# goals = np.load(f'data_collection/PointRobot-v0/task_goals.npy')
# if Config.dataset == 'PointRobot-v0':
#     task_id = 0     # single task learning
# elif Config.dataset == 'PointRobotMT-v0':
#     task_id = 45    # test tasks id: [45, 46, 47, 48, 49]
# elif Config.dataset == 'HalfCheetahVel-v0':
#     goals = np.load(
#         f'data_collection/HalfCheetahVel-v0/task_goals.npy', allow_pickle=True)
#     task_id = 19
# elif Config.dataset == 'metaworld':
#     task_id = args.task_id
# elif Config.dataset == 'AntCliff-v0':
#     task_id = args.task_id
#     # Config.hidden_dim = 128

# tb_log = f'{Config.bucket}/tensorboard/evaluation_task{task_id}'
tb_log = f'{Config.bucket}/tensorboard/evaluation'
writer = SummaryWriter(tb_log)
copy_files(tb_log, folders=['config', 'configs', 'diffuser', 'meta_dt', 'wrapped_metaworld'], files=['train_t2d_transformer.py', 'train_t2d_diffuser.py', 'evaluate_parallel.py'])

env_list = []
for task_id in Config.eval_tasks:
    if args.env == 'metaworld':
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
        env.set_task_idx(task_id)

    elif args.env == 'point-robot':
        train_tasks = [i for i in range(Config.num_tasks) if i not in Config.eval_tasks]
        with open('datasets/PointRobot-v0/task_goals.pkl', 'rb') as fp:
            tasks = pickle.load(fp)
        env = gym.make('PointRobot-v0', tasks=tasks, train_task_ids=train_tasks)
        env.task_describe = [f"Navigate to position ({round(number=task[0], ndigits=2)}, {round(number=task[1], ndigits=2)})" for task in tasks]
        env.set_task_idx(task_id)

    elif args.env == 'cheetah-vel':
        train_tasks = [i for i in range(Config.num_tasks) if i not in Config.eval_tasks]
        with open('datasets/HalfCheetahVel-v0/task_goals.pkl', 'rb') as fp:
            tasks = pickle.load(fp)
        env = gym.make('HalfCheetahVel-v0', tasks=tasks, train_task_ids=train_tasks)
        env.task_describe = [f"{task['velocity']:.4f}" for task in tasks]
        env.set_task_idx(task_id)

    elif args.env == 'ant-dir':
        train_tasks = [i for i in range(Config.num_tasks) if i not in Config.eval_tasks]
        with open('datasets/AntDir-v0/task_goals.pkl', 'rb') as fp:
            tasks = pickle.load(fp)
        env = gym.make('AntDir-v0', tasks=tasks, train_task_ids=train_tasks)
        env.task_describe = [f"Direction {round(number=task['goal'], ndigits=2)}" for task in tasks]
        env.set_task_idx(task_id)
    
    env_list.append(env)

lengths = [len(s.split()) for s in env.task_describe]
task_max_length = max(lengths)


# choose the model
if args.model == 'unet':
    Config.model = 'models.TemporalUnet'
elif args.model == 'transformer':
    Config.model = 'models.DiT'
else:
    assert NotImplementedError


# Load configs
torch.backends.cudnn.benchmark = True
utils.set_seed(Config.seed)
device = args.device if torch.cuda.is_available() else "cpu"

# tb_log = f'{Config.bucket}/tensorboard/evaluation'
# writer = SummaryWriter(tb_log)


dataset_config = utils.Config(
    Config.loader,
    env=env,
    horizon=Config.horizon,
    normalizer=Config.normalizer,
    preprocess_fns=Config.preprocess_fns,
    use_padding=Config.use_padding,
    max_path_length=Config.max_path_length,
    include_returns=Config.include_returns,
    returns_scale=Config.returns_scale,
    discount=Config.discount,
    termination_penalty=Config.termination_penalty,
    # hyperparameters for the prompt design
    prompts_type=Config.prompts_type,
    num_demos=Config.num_demos,
    prompt_traj_len=Config.prompt_traj_len,
    env_name=args.env,
    model_device=device,
)

dataset = dataset_config()
observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

# -----------------------------------------------------------------------------#
# ------------------------------ model & trainer ------------------------------#
# -----------------------------------------------------------------------------#
if Config.model == 'models.TemporalUnet':
    model_config = utils.Config(
        Config.model,
        horizon=Config.horizon,
        transition_dim=observation_dim + action_dim,
        cond_dim=observation_dim,
        dim_mults=Config.dim_mults,
        include_returns=Config.include_returns,
        prompts_type=Config.prompts_type,
        prompt_oracle_dim=Config.prompt_oracle_dim,
        prompt_embed_dim=Config.prompt_embed_dim,
        observation_dim=observation_dim,
        max_path_length=Config.max_path_length,
        dim=Config.hidden_dim,
        condition_dropout=Config.condition_dropout,
        device=device,
    )

elif Config.model == 'models.DiT':
    # Diffusion Transformer
    model_config = utils.Config(
        Config.model,
        # device='cuda:3',
        env_name=args.env,
        task_max_length=task_max_length,
        trajectory_length=Config.horizon,
        transition_dim=observation_dim+action_dim,
        hidden_size=Config.hidden_dim,
        depth=Config.depth,
        num_heads=Config.num_heads,
        mlp_ratio=Config.mlp_ratio,
        include_returns=Config.include_returns,
        condition_dropout=Config.condition_dropout,
        # hyperparameters for prompt design
        prompts_type=Config.prompts_type,
        prompt_oracle_dim=Config.prompt_oracle_dim,
        prompt_embed_dim=Config.prompt_embed_dim,
        observation_dim=observation_dim,
        max_path_length=Config.max_path_length,
        train_device=device,
    )

else:
    assert NotImplementedError


diffusion_config = utils.Config(
    Config.diffusion,
    horizon=Config.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=Config.n_diffusion_steps,
    loss_type=Config.loss_type,
    clip_denoised=Config.clip_denoised,
    predict_epsilon=Config.predict_epsilon,
    # loss weighting
    action_weight=Config.action_weight,
    loss_weights=Config.loss_weights,
    loss_discount=Config.loss_discount,
    include_returns=Config.include_returns,
    prompts_type=Config.prompts_type,
    condition_guidance_w=Config.condition_guidance_w,
    device=device,
)

trainer_config = utils.Config(
    utils.Trainer,
    train_batch_size=Config.batch_size,
    train_lr=Config.learning_rate,
    gradient_accumulate_every=Config.gradient_accumulate_every,
    ema_decay=Config.ema_decay,
    save_freq=Config.save_freq,
    log_freq=Config.log_freq,
    bucket=Config.bucket,
    train_device=device,
    is_evaluate=True,
)

model = model_config()
diffusion = diffusion_config(model)
trainer = trainer_config(diffusion, dataset)
logger.print(utils.report_parameters(model), color='green')

for ind, step in enumerate(range(Config.save_freq, Config.n_train_steps+Config.save_freq, Config.save_freq)):
    print(f'\n========== Evaluate at step {step} ==========')
    loadpath = f'{Config.bucket}/checkpoint/state_{step}.pt'
    state_dict = torch.load(loadpath, map_location=device)
    print(f'Load model from {loadpath}...')

    trainer.step = state_dict['step']
    trainer.model.load_state_dict(state_dict['model'], strict=False)
    # trainer.ema_model.load_state_dict(state_dict['ema'], strict=False)

    dones = [0 for _ in range(num_eval)]
    steps = [0 for _ in range(num_eval)]
    episode_rewards = [0 for _ in range(num_eval)]

    assert trainer.model.condition_guidance_w == Config.condition_guidance_w
    returns = to_device(Config.test_ret * torch.ones(num_eval, 1), device)

    if Config.dataset == 'PointRobot-v0':
        if Config.prompts_type in ['clip', 'aligned_clip', 'aligned_clip_wo_wm']:
            # prompts = [env.task_describe[task_id] for task_id in Config.eval_tasks]
            eval_tasks = np.array(Config.eval_tasks)
            prompts = dataset.prompt_cache[eval_tasks].to(device)
        elif Config.prompts_type is None:
            prompts = None
    elif Config.dataset == 'HalfCheetahVel-v0':
        if Config.prompts_type in ['clip', 'aligned_clip', 'aligned_clip_wo_wm']:
            # prompts = [env.task_describe[task_id] for task_id in Config.eval_tasks]
            eval_tasks = np.array(Config.eval_tasks)
            prompts = dataset.prompt_cache[eval_tasks].to(device)
        elif Config.prompts_type is None:
            prompts = None
    elif Config.dataset == 'AntDir-v0':
        if Config.prompts_type in ['clip', 'aligned_clip', 'aligned_clip_wo_wm']:
            # prompts = [env.task_describe[task_id] for task_id in Config.eval_tasks]
            eval_tasks = np.array(Config.eval_tasks)
            prompts = dataset.prompt_cache[eval_tasks].to(device)
        elif Config.prompts_type is None:
            prompts = None
    elif Config.dataset == 'metaworld':
        if Config.prompts_type in ['clip', 'aligned_clip', 'aligned_clip_wo_wm']:
            eval_tasks = np.array(Config.eval_tasks)
            prompts = dataset.prompt_cache[eval_tasks].to(device)
        elif Config.prompts_type is None:
            prompts = None
    else:
        prompts = None

    t = 0
    obs_list = [env.reset()[None] for env in env_list]
    obs = np.concatenate(obs_list, axis=0)
    recorded_obs = [deepcopy(obs[:, None])]

    while sum(dones) < num_eval:
        obs = dataset.normalizer.normalize(obs, 'observations')
        conditions = {0: to_torch(obs, device=device)}
        samples = trainer.model.conditional_sample(  # ema_model
            conditions, returns=returns, prompts=prompts)

        if Config.diffusion == 'models.GaussianInvDynDiffusion':
            obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
            obs_comb = obs_comb.reshape(-1, 2*observation_dim)
            action = trainer.model.inv_model(obs_comb)  # ema_model
        else:
            action = samples[:, 0, :action_dim]

        samples = to_np(samples)
        action = to_np(action)

        # action = dataset.normalizer.unnormalize(action, 'actions')

        if t == 0:
            if Config.diffusion == 'models.GaussianInvDynDiffusion':
                normed_observations = samples[:, :, :]
            else:
                normed_observations = samples[:, :, action_dim:]
            observations = dataset.normalizer.unnormalize(
                normed_observations, 'observations')

        obs_list = []
        for i in range(num_eval):
            this_obs, this_reward, this_done, _ = env_list[i].step(action[i])
            obs_list.append(this_obs[None])
            if this_done:
                if dones[i] == 1:
                    pass
                else:
                    dones[i] = 1
                    episode_rewards[i] += this_reward
                    logger.print(
                        f"Episode ({i}): {episode_rewards[i]}", color='green')
            else:
                if dones[i] == 1:
                    pass
                else:
                    episode_rewards[i] += this_reward
                    returns[i] -= this_reward / Config.returns_scale

        obs = np.concatenate(obs_list, axis=0)
        recorded_obs.append(deepcopy(obs[:, None]))
        t += 1

    episode_rewards = np.array(episode_rewards)
    # normalized_score = 100 * \
    #     max(0, (np.mean(episode_rewards) - REF_MIN_SCORE)) / \
    #     (REF_MAX_SCORE - REF_MIN_SCORE)
    normalized_score = np.mean(episode_rewards)

    # logger.print(f"average_ep_reward: {np.mean(episode_rewards)}, std_ep_reward: {np.std(episode_rewards)}, normalized score: {normalized_score}", color='green')
    logger.log_metrics_summary({'average_ep_reward': np.mean(episode_rewards),
                                'std_ep_reward': np.std(episode_rewards),
                                'normalized score': normalized_score})

    for i, task_id in enumerate(Config.eval_tasks):
        writer.add_scalar(f'returns/task_{task_id}', episode_rewards[i], ind)
    writer.add_scalar('diffusion/evaluation return mean',
                      np.mean(episode_rewards), ind)
    writer.add_scalar('diffusion/evaluation return std',
                      np.std(episode_rewards), ind)
    # writer.add_scalar('diffusion/evaluation episode_rewards',
    #                   np.mean(episode_rewards))
    writer.add_scalar('diffusion/evaluation score', normalized_score, ind)

    results[step] = [np.mean(episode_rewards), np.std(episode_rewards)]
    with open(f'{tb_log}/eval_results.json', 'w') as f:
        f.write(json.dumps(results, indent=4))
    f.close()

    # print(results)
    print(f'\nElapsed time: {(time.time()-start_time)/60.:.2f} minutes')


def get_normalized_score(episode_rewards, task_id):
    return_scale = np.load(
        f'saves/{Config.dataset}/data_collecion/return_scale.npy', allow_pickle=True)

    return 100 * (max(0, episode_rewards - return_scale[task_id][0])) / (return_scale[task_id][1] - return_scale[task_id][0])
