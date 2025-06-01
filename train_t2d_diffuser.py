import pickle
import diffuser.utils as utils
import torch
import argparse
from ml_logger import logger, RUN
import gym

from wrapped_metaworld.envs import get_cl_env
from tqdm import trange

from utils import copy_files


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--env', type=str, default='point-robot',
                    choices=['metaworld', 'cheetah-vel', 'point-robot', 'ant-dir'])
parser.add_argument('--model', type=str, default='transformer',
                    choices=['unet', 'transformer'])
parser.add_argument('--prompts_type', type=str,
                    default=None, choices=['clip', 'aligned_clip', 'aligned_clip_wo_wm', None])
# parser.add_argument('--eval_tasks', default={45, 46, 47, 48, 49})
args = parser.parse_args()

if args.env == 'metaworld':
    from config.metaworld_config import Config
    Config.prompts_type = args.prompts_type
    Config.bucket += f'/{args.model}/prompt_{Config.prompts_type}'
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
    env.eval_tasks = Config.eval_tasks

elif args.env in ['point-robot', 'cheetah-vel', 'ant-dir']:
    # 50 tasks
    from register_mujoco import register_mujoco
    register_mujoco()
    if args.env == 'point-robot':
        from config.pointrobot_meta_config import Config
        train_tasks = [i for i in range(Config.num_tasks) if i not in Config.eval_tasks]
        with open('datasets/PointRobot-v0/task_goals.pkl', 'rb') as fp:
            tasks = pickle.load(fp)
        env = gym.make('PointRobot-v0', tasks=tasks, train_task_ids=train_tasks)
        env.task_describe = [f"Navigate to position ({round(number=task[0], ndigits=2)}, {round(number=task[1], ndigits=2)})" for task in tasks]
    elif args.env == 'cheetah-vel':
        from config.half_cheetah_vel_meta_config import Config
        train_tasks = [i for i in range(Config.num_tasks) if i not in Config.eval_tasks]
        with open('datasets/HalfCheetahVel-v0/task_goals.pkl', 'rb') as fp:
            tasks = pickle.load(fp)
        env = gym.make('HalfCheetahVel-v0', tasks=tasks, train_task_ids=train_tasks)
        env.task_describe = [f"{task['velocity']:.4f}" for task in tasks]
    elif args.env == 'ant-dir':
        from config.ant_dir_config import Config
        train_tasks = [i for i in range(Config.num_tasks) if i not in Config.eval_tasks]
        with open('datasets/AntDir-v0/task_goals.pkl', 'rb') as fp:
            tasks = pickle.load(fp)
        env = gym.make('AntDir-v0', tasks=tasks, train_task_ids=train_tasks)
        env.task_describe = [f"Direction {round(number=task['goal'], ndigits=2)}" for task in tasks]
    env.eval_tasks = Config.eval_tasks
    Config.prompts_type = args.prompts_type
    Config.bucket += f'/{args.model}/prompt_{Config.prompts_type}'

else:
    assert NotImplementedError

lengths = [len(s.split()) for s in env.task_describe]
task_max_length = max(lengths)

# choose the model
if args.model == 'unet':
    Config.model = 'models.TemporalUnet'
elif args.model == 'transformer':
    Config.model = 'models.DiT'
else:
    assert NotImplementedError


torch.backends.cudnn.benchmark = True
utils.set_seed(Config.seed)
device = torch.device(args.device if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------#
# ---------------------------------- dataset ----------------------------------#
# -----------------------------------------------------------------------------#

dataset_config = utils.Config(
    Config.loader,
    # env=env if args.env == 'metaworld' else Config.dataset,
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
)

# -----------------------------------------------------------------------------#
# -------------------------------- instantiate --------------------------------#
# -----------------------------------------------------------------------------#

model = model_config()

diffusion = diffusion_config(model)

trainer = trainer_config(diffusion, dataset)


# -----------------------------------------------------------------------------#
# ------------------------ test forward & backward pass -----------------------#
# -----------------------------------------------------------------------------#
utils.report_parameters(model)
# logger.print('Testing forward...', end=' ', flush=True)
# batch = utils.batchify(dataset[0], device)
# loss, _ = diffusion.loss(*batch)
# loss.backward()
# logger.print('✓')
copy_files(f'{Config.bucket}/tensorboard/train', folders=['config', 'configs', 'diffuser', 'meta_dt', 'world_model', 'wrapped_metaworld'], files=['train_t2d_transformer.py', 'train_t2d_diffuser.py', 'evaluate_parallel.py'])

# -----------------------------------------------------------------------------#
# --------------------------------- main loop ---------------------------------#
# -----------------------------------------------------------------------------#
n_epochs = int(Config.n_train_steps // Config.n_steps_per_epoch)

for i in trange(n_epochs, ncols=50):
    logger.print(f'Epoch {i} / {n_epochs} | {Config.bucket}')
    trainer.train(n_train_steps=Config.n_steps_per_epoch)
