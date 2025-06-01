import numpy as np
import torch
from gym import spaces
from gym import Env
import json
import pickle
from collections import OrderedDict
from typing import List, Optional


class PointEnv(Env):
    """
    point robot on a 2-D plane with position control
    tasks (aka goals) are positions on the plane
     - tasks sampled from unit square
     - reward is L2 distance
    """

    def __init__(
            self,
            max_episode_steps=20,
            num_tasks=50,
            num_train_tasks=45,
        ):

        self._max_episode_steps = max_episode_steps
        self.step_count = 0
        self.num_tasks = num_tasks
        self.num_train_tasks = num_train_tasks
        self.goals = np.array([[np.random.uniform(-1., 1.), np.random.uniform(-1., 1.)] for _ in range(num_tasks)])

        self.reset_task(0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))

        self._goal = np.array([1.0, 1.0])

    def reset_task(self, idx):
        ''' reset goal AND reset the agent '''
        if idx is not None:
            self._goal = np.array(self.goals[idx])
        self.reset()

    def print_task(self):
        print(f'Task information: Goal position {self._goal}')

    def set_goal(self, goal):
        self._goal = np.asarray(goal)

    def load_all_tasks(self, goals):
        assert self.num_tasks == len(goals)
        self.goals = np.array([g for g in goals])
        self.reset_task(0)

    def reset_model(self):
        self._state = np.zeros(2)
        return self._get_obs()

    def reset(self):
        self.step_count = 0
        return self.reset_model()

    def _get_obs(self):
        return np.copy(self._state)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self._state = self._state + action

        x = self._state[0] - self._goal[0]
        y = self._state[1] - self._goal[1]

        reward = - (x ** 2 + y ** 2) ** 0.5

        done = (abs(x) < 0.01) and (abs(y) < 0.01)
        self.step_count += 1
        if self.step_count == self._max_episode_steps:
            done = True 

        ob = self._get_obs()
        return ob, reward, done, dict()

    def reward(self, state, action=None):
        return - ((state[0] - self._goal[0]) ** 2 + (state[1] - self._goal[1]) ** 2) ** 0.5

    def render(self):
        print('current state:', self._state)

    def get_dataset(self):
        with open(f'datasets/PointRobot-v0/dataset_task_0.pkl', "rb") as f:
            dataset = pickle.load(f)
        f.close()

        return dataset
    
    def get_normalized_score(self, values):
        return values


class PointMTEnv(Env):
    """
    point robot on a 2-D plane with position control
    tasks (aka goals) are positions on the plane
     - tasks sampled from unit square
     - reward is L2 distance
    """

    def __init__(
            self,
            tasks: np.ndarray,
            train_task_ids: Optional[List[int]] = None,
            max_episode_steps=20,
        ):

        self._max_episode_steps = max_episode_steps
        self.step_count = 0
        self.goals = tasks.copy()
        self.num_tasks = len(tasks)
        self.train_task_ids = train_task_ids if train_task_ids is not None else list(range(self.num_tasks - 5))
        self.num_train_tasks = len(self.train_task_ids)

        self.reset_task(0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))

        self._goal = np.array([1.0, 1.0])

    def reset_task(self, idx):
        ''' reset goal AND reset the agent '''
        if idx is not None:
            self._goal = np.array(self.goals[idx])
        self.reset()

    def set_task_idx(self, idx: int):
        return self.reset_task(idx)

    def print_task(self):
        print(f'Task information: Goal position {self._goal}')

    def set_goal(self, goal):
        self._goal = np.asarray(goal)

    # def load_all_tasks(self, goals):
    #     assert self.num_tasks == len(goals)
    #     self.goals = np.array([g for g in goals])
    #     self.reset_task(0)

    def reset_model(self):
        self._state = np.zeros(2)
        return self._get_obs()

    def reset(self):
        self.step_count = 0
        return self.reset_model()

    def _get_obs(self):
        return np.copy(self._state)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high) * 0.1

        self._state = self._state + action

        x = self._state[0] - self._goal[0]
        y = self._state[1] - self._goal[1]

        reward = - (x ** 2 + y ** 2) ** 0.5

        done = (abs(x) < 0.01) and (abs(y) < 0.01)
        self.step_count += 1
        if self.step_count == self._max_episode_steps:
            done = True 

        ob = self._get_obs()
        return ob, reward, done, dict()

    def reward(self, state, action=None):
        return - ((state[0] - self._goal[0]) ** 2 + (state[1] - self._goal[1]) ** 2) ** 0.5

    def render(self):
        print('current state:', self._state)

    def get_dataset(self):
        train_dataset = OrderedDict()
        with open(f"datasets/PointRobot-v0/dataset_task_{self.train_task_ids[0]}.pkl", "rb") as f:
            dataset = pickle.load(f)
        for key, value in dataset.items():
            train_dataset[key] = [value]
        train_dataset['task_id'] = [(np.ones(dataset['observations'].shape[0]) * self.train_task_ids[0]).astype(np.int32)]

        for task_id in self.train_task_ids[1:]:
            with open(f"datasets/PointRobot-v0/dataset_task_{task_id}.pkl", "rb") as f:
                dataset = pickle.load(f)

            for key, value in dataset.items():
                train_dataset[key].append(value)
            train_dataset['task_id'].append((np.ones(dataset['observations'].shape[0]) * task_id).astype(np.int32))

        for key, value in train_dataset.items():
            train_dataset[key] = np.concatenate(value, axis=0)

        num_samples = train_dataset['observations'].shape[0]

        print('================================================')
        print(f'Successfully constructed the training dataset from {self.num_train_tasks} tasks...')
        print(f'Number of training samples: {num_samples}')
        print('================================================')

        return train_dataset

    def get_dataset_by_task_ids(self, task_ids: List[int]):
        dataset = OrderedDict()
        with open(f"datasets/PointRobot-v0/dataset_task_{task_ids[0]}.pkl", "rb") as f:
            datas = pickle.load(f)
        for key, value in datas.items():
            dataset[key] = [value]
        dataset['task_id'] = [(np.ones(datas['observations'].shape[0]) * task_ids[0]).astype(np.int32)]

        for task_id in task_ids[1:]:
            with open(f"datasets/PointRobot-v0/dataset_task_{task_id}.pkl", "rb") as f:
                datas = pickle.load(f)

            for key, value in datas.items():
                dataset[key].append(value)
            dataset['task_id'].append((np.ones(datas['observations'].shape[0]) * task_id).astype(np.int32))

        for key, value in dataset.items():
            dataset[key] = np.concatenate(value, axis=0)

        num_samples = dataset['observations'].shape[0]

        print('================================================')
        print(f'Successfully constructed the training dataset from {len(task_ids)} tasks...')
        print(f'Number of training samples: {num_samples}')
        print('================================================')

        return dataset

    def get_prompt_trajectories(self, num_demos=5):
        prompt_trajectories = []
        
        for task_id in range(self.num_tasks):
            with open(f'datasets/PointRobot-v0/dataset_task_{task_id}.pkl', "rb") as f:
                dataset = pickle.load(f)
            f.close()
            trajectories = convert_data_to_trajectories(dataset)

            returns = [traj['rewards'].sum() for traj in trajectories]
            sorted_inds = sorted(range(len(returns)), key=lambda x:returns[x], reverse=True)

            demos = [trajectories[sorted_inds[i]] for i in range(num_demos)]
            prompt_trajectories.append(demos)

        return prompt_trajectories
    
    def get_normalized_score(self, values):
        return values


def convert_data_to_trajectories(data):
    trajectories = []
    start_ind = 0
    for ind, terminal in enumerate(data['terminals']):
        timeout = data['timeouts'][ind]
        if terminal or timeout:
            traj = OrderedDict()
            for key, value in data.items():
                traj[key] = value[start_ind : ind+1]

            trajectories.append(traj)
            start_ind = ind + 1
    
    print(f'Convert {ind} transitions to {(len(trajectories))} trajectories.')
    return trajectories

