import numpy as np
import torch
from collections import OrderedDict


def evaluate_episode_rtg(
        env,
        state_dim,
        action_dim,
        model,
        max_episode_steps=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        num_eval_episodes=10,
        context_encoder=None,
        context_dim=16,
        context_horizon=4,
        dataset=None,
        task_id=0,
        com_prompt=False,
        prompt_type=None,
        ):

    model.eval()
    model.to(device=device)
    # if context_encoder is not None:
    #     context_encoder.eval()
    #     context_encoder.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    avg_epi_return = 0.
    avg_epi_len = 0

    # demos = []

    for _ in range(num_eval_episodes):

        traj = OrderedDict()
        traj['observations'] = np.zeros((0, state_dim))
        traj['actions'] = np.zeros((0, action_dim))
        traj['rewards'] = np.zeros(0)
        traj['next_observations'] = np.zeros((0, state_dim))
        traj['rtg'] = np.zeros(0)
        traj['terminals'] = np.zeros(0).astype(bool)

        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        if prompt_type is not None:
            prompt = dataset.get_prompt(task_id)
            # if com_prompt:
            #     prompt = dataset.get_complementary_prompt(task_id, unsqueeze=True)
            # else:
            #     prompt = dataset.get_prompt(task_id, unsqueeze=True)
        else:
            prompt=None

        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
        actions = torch.zeros((0, action_dim), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)

        states_traj = np.zeros((200, env.observation_space.shape[0]))
        actions_traj = np.zeros((200, env.action_space.shape[0]))
        rewards_traj = np.zeros((200, 1))

        target_returns = torch.tensor(target_return / scale, device=device, dtype=torch.float32).reshape(1, 1)
        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

        if context_encoder is not None:
            contexts = torch.zeros((1, context_dim), device=device, dtype=torch.float32)
        else:
            contexts = None

        for t in range(max_episode_steps):

            # add padding
            actions = torch.cat([actions, torch.zeros((1, action_dim), device=device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])

            action = model.get_action(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_returns.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
                contexts=contexts,
                prompt=prompt,
            )
            actions[-1] = action
            action = action.detach().cpu().numpy()

            step_result = env.step(action)
            next_state = step_result[0]
            reward = step_result[1]
            done = step_result[2]            

            ####################################################
            # compute the current context
            states_traj[t] = np.copy(states[-1].detach().cpu().numpy().reshape(-1))
            actions_traj[t] = np.copy(action)
            rewards_traj[t] = np.copy(reward)

            if context_encoder is not None:
                state_seg = states_traj[t+1-context_horizon : t+1]
                action_seg = actions_traj[t+1-context_horizon : t+1]
                reward_seg = rewards_traj[t+1-context_horizon : t+1]

                tlen = state_seg.shape[0] 
                state_seg = np.concatenate([np.zeros((context_horizon - tlen, state_seg.shape[1])), state_seg], axis=0)
                action_seg = np.concatenate([np.ones((context_horizon - tlen, action_seg.shape[1])) * -10., action_seg], axis=0)
                reward_seg = np.concatenate([np.zeros((context_horizon - tlen, 1)), reward_seg], axis=0)

                state_seg = torch.FloatTensor(state_seg).to(device).unsqueeze(1)
                action_seg = torch.FloatTensor(action_seg).to(device).unsqueeze(1)
                reward_seg = torch.FloatTensor(reward_seg).to(device).unsqueeze(1)

                cur_context = context_encoder(state_seg, action_seg, reward_seg).detach().reshape(1, -1)
                contexts = torch.cat([contexts, cur_context], dim=0).to(dtype=torch.float32)
            ####################################################
                
            ####################################################
            # record the trajectory to construct the prompt            
            state = states[-1].detach().cpu().numpy().reshape(1, -1)
            rtg = target_returns[:, -1].reshape(-1).detach().cpu().numpy() * scale
            traj['observations'] = np.concatenate([traj['observations'], state], axis=0)
            traj['actions'] = np.concatenate([traj['actions'], action.reshape(1, -1)], axis=0)
            traj['rewards'] = np.concatenate([traj['rewards'], np.ones(1)*reward], axis=0)
            traj['next_observations'] = np.concatenate([traj['next_observations'], next_state.reshape(1, -1)], axis=0)
            traj['rtg'] = np.concatenate([traj['rtg'], np.ones(1)*rtg], axis=0)
            traj['terminals'] = np.concatenate([traj['terminals'], np.ones(1)*done], axis=0).astype(bool)
            ####################################################
                
            cur_state = torch.from_numpy(next_state).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)

            rewards[-1] = reward
            pred_return = target_returns[0,-1] - (reward/scale)
            target_returns = torch.cat([target_returns, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

            avg_epi_return += reward
            avg_epi_len += 1

            if done:
                break

        # demos.append(traj)

        trajectory = torch.cat((states[:-1], actions, rewards.reshape(-1,1)), dim=-1).detach().cpu().numpy()

    return avg_epi_return/num_eval_episodes, avg_epi_len/num_eval_episodes, trajectory


















