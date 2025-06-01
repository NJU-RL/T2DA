import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel, BertModel, T5EncoderModel
from torch.utils.data import DataLoader
import torch.nn.functional as F
import gym
from peft import LoraConfig, get_peft_model
import pickle
from collections import OrderedDict
from utils import copy_files
from wrapped_metaworld.envs import get_cl_env
import diffuser.utils as utils
from world_model.model import TrajTransfomerEncoder
from world_model.dataset import ContextDataset
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np
import os
from configs import args_metaworld, args_half_cheetah_vel, args_ant_dir, args_point_robot
from utils import freeze


def print_model_memory_analysis(model):
    trainable_mem = sum([param.nelement() * param.element_size()
                        for param in model.parameters() if param.requires_grad])

    total_mem = sum([param.nelement() * param.element_size()
                    for param in model.parameters()])

    buffer_mem = sum([buf.nelement() * buf.element_size()
                     for buf in model.buffers()])

    trainable_mem_mb = trainable_mem / (1024**2)
    total_mem_mb = total_mem / (1024**2)
    buffer_mem_mb = buffer_mem / (1024**2)

    print("\nModel Memory Analysis:")
    print(f"Trainable parameters memory: {trainable_mem_mb:.2f} MB")
    print(f"Total parameters memory: {total_mem_mb:.2f} MB")
    print(f"Buffer memory: {buffer_mem_mb:.2f} MB")
    print(f"Total model memory: {(total_mem_mb + buffer_mem_mb):.2f} MB")
    print(
        f"Percentage of trainable memory: {100 * trainable_mem / total_mem:.2f}%")


class TextProjection(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.projection = nn.Parameter(torch.empty(input_dim, latent_dim))
        nn.init.normal_(self.projection, mean=0.0, std=input_dim ** -0.5)

    def forward(self, text_embed):
        return text_embed @ self.projection


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--text_encoder', type=str, default='clip',
                        choices=['clip', 'bert', 't5'])
    parser.add_argument('--env', type=str, default='point-robot',
                        choices=['metaworld', 'ant-dir', 'point-robot', 'cheetah-vel'])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42,
                        metavar='N', help='random seed')
    parser.add_argument('--context_batch_size', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--align_traj_len', type=int, default=100)
    parser.add_argument('--align_num_demos', type=int, default=400)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--context_horizon', type=int, default=20)  # 200 for cheetah-vel, ant-dir and metaworld
    parser.add_argument('--with_world_model', type=bool, default=True)
    parser.add_argument('--context_train_epochs', type=int, default=50)
    # tau for contrastive learning
    parser.add_argument('--temperature', type=float, default=0.02)
    # increase weight for positive samples
    parser.add_argument('--positive_weight', type=float, default=1)

    args, rest_args = parser.parse_known_args()

    # Config.prompt_traj_len = args.prompt_traj_len
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    utils.set_seed(args.seed)

    if args.env == 'metaworld':
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
    elif args.env == 'cheetah-vel':
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
        env.task_describe = [f"{task['velocity']:.4f}" for task in tasks]
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
        env.task_describe = [f"Direction {round(number=task['goal'], ndigits=2)}" for task in tasks]
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
        env.task_describe = [f"Navigate to position ({round(number=task[0], ndigits=2)}, {round(number=task[1], ndigits=2)})" for task in tasks]

    else:
        raise NotImplementedError

    env.eval_tasks = args.eval_tasks
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if args.text_encoder == 'clip':
        text_encoder_model = CLIPTextModel.from_pretrained(
            "pre_trained/clip-vit-base-patch32", local_files_only=True).to(args.device)
        tokenizer = AutoTokenizer.from_pretrained("pre_trained/clip-vit-base-patch32")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj"]  # target modules
        )
        output_dim = 512
    elif args.text_encoder == 'bert':
        text_encoder_model = BertModel.from_pretrained(
            "pre_trained/bert-base-uncased", local_files_only=True).to(args.device)
        tokenizer = AutoTokenizer.from_pretrained("pre_trained/bert-base-uncased")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "key", "value"]  # target modules
        )
        output_dim = 768
    elif args.text_encoder == 't5':
        text_encoder_model = T5EncoderModel.from_pretrained(
            "pre_trained/t5-efficient-small", local_files_only=True).to(args.device)
        tokenizer = AutoTokenizer.from_pretrained("pre_trained/t5-efficient-small")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q", "k", "v"]  # target modules
        )
        output_dim = 512
    else:
        raise NotImplementedError

    text_encoder_model.add_adapter(lora_config)
    text_encoder_model.enable_adapters()

    # print trainable parameters in model
    for name, param in text_encoder_model.named_parameters():
        if param.requires_grad:
            print(f"Parameter name: {name}, Shape: {param.shape}")

    trainable_params = sum(param.numel()
                           for param in text_encoder_model.parameters() if param.requires_grad)
    total_params = sum(param.numel() for param in text_encoder_model.parameters())

    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params}")
    print(
        f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")

    # print_model_memory_analysis(text_encoder_model)

    # ============== load dataset =================
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

    # if dataset contains only best trajectories
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

    # metaworld
    else:
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

    # ================== load pretrained traj encoder===========================
    traj_encoder = TrajTransfomerEncoder(state_dim=observation_dim,
                                         action_dim=action_dim, latent_dim=args.latent_dim).to(args.device)
    text_projection_layer = TextProjection(output_dim, args.latent_dim).to(args.device)
    temperature = torch.nn.Parameter(torch.tensor(args.temperature))

    if args.with_world_model:
        load_path = f'saves_world_model/{args.env}/reward/horizon{args.context_horizon}/checkpoints/state_49.pt'
        traj_encoder.load_state_dict(torch.load(
            load_path, map_location=args.device))
        traj_encoder.eval()
        freeze(traj_encoder)
        save_path = f'saves_align/{args.env}/{args.text_encoder}/'
        optimizer = torch.optim.Adam(
            [*text_encoder_model.parameters(), *text_projection_layer.parameters(), temperature], lr=args.lr
        )
    else:
        traj_encoder.train()
        save_path = f'saves_align_wo_world_model/{args.env}/{args.text_encoder}/'
        optimizer = torch.optim.Adam(
            [*text_encoder_model.parameters(), *text_projection_layer.parameters(), temperature, *traj_encoder.parameters()], lr=args.lr
        )

    save_weight = traj_encoder.state_emb.weight

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    writer = SummaryWriter(save_path + f'/tensorboard_{args.temperature}')

    copy_files(save_path, files=['train_align.py'])

    # optimizer = torch.optim.Adam(
    #     [*text_encoder_model.parameters(), *text_projection_layer.parameters(), temperature], lr=1e-4
    # )

    last_text_embed = 0
    train_dataloader_iter = iter(train_dataloader)
    test_dataloader_iter = iter(test_dataloader)

    global_step = 0
    best_loss = float('inf')
    for epoch in range(args.context_train_epochs):
        for step, (segment, transition) in enumerate(train_dataloader):
            global_step += 1
            task_ids = segment[3][:, 0, :]
            text = [env.task_describe[i.item()] for i in task_ids]

            text_inputs = tokenizer(
                text, padding=True, return_tensors="pt").to(args.device)
            outputs = text_encoder_model(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"]
            )
            if args.text_encoder == 't5':
                outputs = outputs.last_hidden_state.mean(dim=1)
            else:
                outputs = outputs.pooler_output

            text_embed = text_projection_layer(
                outputs)

            # prepare text embedding
            traj_embed = traj_encoder(segment)

            # compute consine similarity loss
            cosine_loss = 1 - \
                F.cosine_similarity(traj_embed, text_embed).mean()

            # ===== compute contrastive loss =====
            ordered_text_inputs = tokenizer(env.task_describe, padding=True,
                                            return_tensors="pt").to(args.device)
            ordered_outputs = text_encoder_model(
                input_ids=ordered_text_inputs["input_ids"],
                attention_mask=ordered_text_inputs["attention_mask"]
            )
            if args.text_encoder == 't5':
                ordered_outputs = ordered_outputs.last_hidden_state.mean(dim=1)
            else:
                ordered_outputs = ordered_outputs.pooler_output

            ordered_text_embed = F.normalize(
                text_projection_layer(ordered_outputs), p=2, dim=-1)
            traj_embed = F.normalize(traj_embed, p=2, dim=-1)

            similarity_matrix = torch.matmul(
                traj_embed, ordered_text_embed.T) / temperature
            task_ids = task_ids.squeeze(dim=-1)
            assert task_ids.max().item() < ordered_text_embed.size(
                0), f"task_ids contains invalid indices: {task_ids.max().item()}"
            text_labels = task_ids

            weights = torch.ones(similarity_matrix.size(1)).to(
                similarity_matrix.device)
            # increase weight for positive samples
            weights.index_fill_(0, text_labels, args.positive_weight)

            contrastive_loss = F.cross_entropy(
                similarity_matrix, text_labels, weight=weights)
            # =====================================

            loss = contrastive_loss
            # loss = cosine_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [*text_encoder_model.parameters(), *text_projection_layer.parameters(), temperature], 1.0)
            # print(loss)

            optimizer.step()
            optimizer.zero_grad()

            writer.add_scalar('train/loss', loss, global_step)
            writer.add_scalar('train/cosine loss', cosine_loss, global_step)

        with torch.no_grad():
            text_encoder_model.eval()
            text_projection_layer.eval()
            segment, transition = next(iter(test_dataloader))
            task_ids = segment[3][:, 0, :]
            text = [env.task_describe[i.item()] for i in task_ids]

            text_inputs = tokenizer(
                text, padding=True, return_tensors="pt").to(args.device)
            outputs = text_encoder_model(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"]
            )
            if args.text_encoder == 't5':
                outputs = outputs.last_hidden_state.mean(dim=1)
            else:
                outputs = outputs.pooler_output

            text_embed = text_projection_layer(
                outputs)

            # prepare text embedding
            traj_embed = traj_encoder(segment)
            # print(f'traj embed: {traj_embed[0]}')
            cos_loss = 1 - \
                F.cosine_similarity(traj_embed, text_embed).mean()
            # eval_loss = F.mse_loss(traj_embed, text_embed)
            print(f'eval cos loss: {cos_loss}')
            writer.add_scalar('evaluation/cosine loss',
                              cos_loss, global_step)

            ordered_text_inputs = tokenizer(
                env.task_describe, padding=True, return_tensors="pt").to(args.device)
            ordered_outputs = text_encoder_model(
                input_ids=ordered_text_inputs["input_ids"],
                attention_mask=ordered_text_inputs["attention_mask"]
            )
            if args.text_encoder == 't5':
                ordered_outputs = ordered_outputs.last_hidden_state.mean(dim=1)
            else:
                ordered_outputs = ordered_outputs.pooler_output

            ordered_text_embed = text_projection_layer(
                ordered_outputs)

            traj_embed = F.normalize(traj_embed, p=2, dim=-1)
            ordered_text_embed = F.normalize(
                ordered_text_embed, p=2, dim=-1)

            similarity_matrix = torch.matmul(
                traj_embed, ordered_text_embed.T) / args.temperature

            task_ids = task_ids.squeeze(dim=-1)
            assert task_ids.max().item() < ordered_text_embed.size(
                0), "task_ids contains invalid indices"
            text_labels = task_ids

            weights = torch.ones(similarity_matrix.size(1)).to(
                similarity_matrix.device)
            # increase weight for positive samples
            weights.index_fill_(0, text_labels, args.positive_weight)

            contrastive_loss = F.cross_entropy(
                similarity_matrix, text_labels, weight=weights)
            print(f'eval contrastive loss: {contrastive_loss}')
            writer.add_scalar('evaluation/constrastive loss',
                              contrastive_loss, global_step)

            if epoch > args.context_train_epochs / 2 and contrastive_loss < best_loss:
                best_loss = contrastive_loss
                print(f'best loss: {best_loss}')
                text_encoder_model.save_pretrained(save_path + "/lora_adapter")
                proj_save_path = save_path + "/text_proj/"
                if not os.path.exists(proj_save_path):
                    os.makedirs(proj_save_path)
                torch.save(text_projection_layer.state_dict(),
                           proj_save_path + 'state.pt')

            text_encoder_model.train()
            text_projection_layer.train()

    # text_encoder_model.save_pretrained(save_path + "/lora_adapter")
    # proj_save_path = save_path + "/text_proj/"
    # if not os.path.exists(proj_save_path):
    #     os.makedirs(proj_save_path)
    # torch.save(text_projection_layer.state_dict(),
    #            proj_save_path + 'state.pt')
