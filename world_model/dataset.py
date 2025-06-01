import os
import pickle
import random
import numpy as np
from collections import OrderedDict
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


from collections import OrderedDict
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class ContextDataset(Dataset):
    def __init__(self, data, horizon=4, num_transitions=10, device='cpu'):
        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu')
        self.data = data

        # size: (num_samples * dim)
        self.states = torch.from_numpy(
            data['observations']).float().to(self.device)
        self.actions = torch.from_numpy(
            data['actions']).float().to(self.device)
        self.next_states = torch.from_numpy(
            data['next_observations']).float().to(self.device)
        self.rewards = torch.from_numpy(
            data['rewards']).view(-1, 1).float().to(self.device)
        self.terminals = torch.from_numpy(
            data['terminals']).view(-1, 1).long().to(self.device)
        self.task_ids = torch.from_numpy(
            data['task_ids']).view(-1, 1).long().to(self.device)

        self.horizon = horizon
        self.num_transitions = num_transitions
        self.trajectories = self.parse_trajectories()
        self.task_to_trajectories = self.group_trajectories_by_task()
        print(f'Prepared dataset: {len(self.trajectories)} trajectories')

    def __getitem__(self, index):
        # Randomly select a trajectory
        trajectory = random.choice(self.trajectories)
        # All steps in trajectory share the same task_id
        task_id = trajectory['task_ids'][0, 0]

        # Sample a trajectory segment
        segment = self._sample_segment(trajectory)

        # Sample `num_transitions` transitions from the same task_id
        task_transitions = self._sample_task_transitions(task_id)

        # Convert task_transitions to tensors with shape (batch_size, num_transitions, *)
        task_transitions_tuple = self._convert_transitions_to_tuple(
            task_transitions)

        return (
            segment,
            task_transitions_tuple,
        )

    def __len__(self):
        return len(self.trajectories)

    def parse_trajectories(self):
        """Parse the data into trajectories based on terminal flags."""
        trajectories = []
        trajectory = {'states': [], 'actions': [],
                      'rewards': [], 'task_ids': []}

        for idx in range(len(self.states)):
            trajectory['states'].append(self.data['observations'][idx])
            trajectory['actions'].append(self.data['actions'][idx])
            trajectory['rewards'].append(self.data['rewards'][idx])
            trajectory['task_ids'].append(self.data['task_ids'][idx])

            if self.data['terminals'][idx]:
                trajectories.append({
                    'states': np.array(trajectory['states']),
                    'actions': np.array(trajectory['actions']),
                    'rewards': np.array(trajectory['rewards']).reshape(-1, 1),
                    'task_ids': np.array(trajectory['task_ids']).reshape(-1, 1),
                })
                trajectory = {'states': [], 'actions': [],
                              'rewards': [], 'task_ids': []}

        # Handle the last trajectory if it didn't end with a terminal
        if len(trajectory['states']) > 0:
            trajectories.append({
                'states': np.array(trajectory['states']),
                'actions': np.array(trajectory['actions']),
                'rewards': np.array(trajectory['rewards']).reshape(-1, 1),
                'task_ids': np.array(trajectory['task_ids']).reshape(-1, 1),
            })

        return trajectories

    def group_trajectories_by_task(self):
        """Group trajectories by task_id."""
        task_to_trajectories = OrderedDict()
        for trajectory in self.trajectories:
            # All steps in a trajectory have the same task_id
            task_id = trajectory['task_ids'][0, 0]
            if task_id not in task_to_trajectories:
                task_to_trajectories[task_id] = []
            task_to_trajectories[task_id].append(trajectory)
        return task_to_trajectories

    def _sample_segment(self, trajectory):
        """Randomly sample a segment of `horizon` steps from a trajectory."""
        traj_length = len(trajectory['states'])
        start_idx = random.randint(0, max(0, traj_length - self.horizon))
        end_idx = start_idx + self.horizon

        states_segment = trajectory['states'][start_idx:end_idx]
        actions_segment = trajectory['actions'][start_idx:end_idx]
        rewards_segment = trajectory['rewards'][start_idx:end_idx]
        task_ids_segment = trajectory['task_ids'][start_idx:end_idx]
        task_id = task_ids_segment[0]

        # Pad if the segment is shorter than horizon
        pad_length = self.horizon - len(states_segment)
        if pad_length > 0:
            states_segment = np.pad(states_segment, ((pad_length, 0), (0, 0)))
            actions_segment = np.pad(
                actions_segment, ((pad_length, 0), (0, 0)))
            rewards_segment = np.pad(
                rewards_segment, ((pad_length, 0), (0, 0)))
            task_ids_segment = np.pad(task_ids_segment, ((pad_length, 0), (0, 0)),
                                      mode='constant', constant_values=task_id)

        return (
            torch.from_numpy(states_segment).float().to(self.device),
            torch.from_numpy(actions_segment).float().to(self.device),
            torch.from_numpy(rewards_segment).float().to(self.device),
            torch.from_numpy(task_ids_segment).long().to(self.device),
        )

    def _sample_task_transitions(self, task_id):
        """Sample `num_transitions` transitions from trajectories with the same task_id."""
        transitions = []
        candidate_trajectories = self.task_to_trajectories[task_id]

        while len(transitions) < self.num_transitions:
            # Randomly pick a trajectory with the same task_id
            trajectory = random.choice(candidate_trajectories)

            # Randomly pick a step in the trajectory
            traj_length = len(trajectory['states'])
            index = random.randint(0, traj_length - 1)

            # Extract the transition
            transition = {
                'state': torch.from_numpy(trajectory['states'][index]).float().to(self.device),
                'action': torch.from_numpy(trajectory['actions'][index]).float().to(self.device),
                'reward': torch.from_numpy(trajectory['rewards'][index]).float().to(self.device),
                'next_state': torch.from_numpy(trajectory['states'][index + 1] if index + 1 < traj_length else trajectory['states'][index]).float().to(self.device),
                'terminal': torch.tensor(1 if index + 1 == traj_length else 0).long().to(self.device),
                'task_id': torch.tensor(task_id).long().to(self.device),
            }
            transitions.append(transition)

        return transitions

    def _convert_transitions_to_tuple(self, transitions):
        """Convert a list of transitions into a tuple of tensors."""
        states = torch.stack([t['state'] for t in transitions],
                             dim=0)  # (num_transitions, state_dim)
        # (num_transitions, action_dim)
        actions = torch.stack([t['action'] for t in transitions], dim=0)
        # (num_transitions, 1)
        rewards = torch.stack([t['reward'] for t in transitions], dim=0)
        # (num_transitions, state_dim)
        next_states = torch.stack([t['next_state']
                                  for t in transitions], dim=0)
        # (num_transitions, 1)
        terminals = torch.stack([t['terminal'] for t in transitions], dim=0)
        # (num_transitions, 1)
        task_ids = torch.stack([t['task_id'] for t in transitions], dim=0)

        return states, actions, rewards, next_states, terminals, task_ids
