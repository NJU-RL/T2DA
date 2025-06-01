import numpy as np
import random
from collections import OrderedDict
import pickle
from .ant import AntFullObsEnv


def convert_data_to_trajectories(data):
    trajectories = []
    start_ind = 0
    for ind, terminal in enumerate(data['terminals']):
        timeout = data['timeouts'][ind]
        if terminal or timeout:
            traj = OrderedDict()
            for key, value in data.items():
                traj[key] = value[start_ind: ind+1]

            trajectories.append(traj)
            start_ind = ind + 1

    print(f'Convert {ind} transitions to {(len(trajectories))} trajectories.')
    return trajectories


class AntCliffEnv(AntFullObsEnv):
    """
    Ant Cliff Environment where the agent must navigate a winding path to reach a goal.
    """

    def __init__(self, max_episode_steps=200):
        self._max_episode_steps = max_episode_steps
        self.task_dim = 2  # Goal position (x, y)
        self.step_count = 0
        self.goal_position = np.array([10.0, 10.0])  # Example goal position
        # _, self.goal_position = self._generate_path_tiles
        self.path_direction_sequence = self._generate_path_sequence(
            0)
        self.path_tiles = self._generate_path_tiles()
        self.task_describe = ['north north north', 'east north east north east north', 'east east east', 'south east south east south east', 'south south south', 'west south west south west south', 'west west west', 'north west north west north west',
                              'north north', 'east east north north', 'east east', 'south south east east', 'south south', 'west west south south', 'west west', 'north north west west']
        self.eval_tasks = {}
        super(AntCliffEnv, self).__init__()

    def reset_task(self, idx):
        self.path_direction_sequence = self._generate_path_sequence(idx)
        self.path_tiles = self._generate_path_tiles()
        self.reset()
        print(f'goal: {self.goal_position}')

    def step(self, action):
        self.step_count += 1
        torso_xyz_before = np.array(self.get_body_com("torso"))

        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = np.array(self.get_body_com("torso"))
        torso_velocity = torso_xyz_after - torso_xyz_before
        direct = (np.cos(self.goal_position[0]), np.sin(self.goal_position[1]))
        progress_reward = np.dot((torso_velocity[:2]/self.dt), direct)

        goal_reward = 0
        fall_penalty = 1.0
        survive_reward = 1.0
        # ctrl_cost = 0.5 * np.square(action).sum()
        ctrl_cost = 0
        # contact_cost = (
        #     0.5 * 1e-3 *
        #     np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        # )
        contact_cost = 0

        # Calculate distance to goal
        distance_to_goal = np.linalg.norm(
            torso_xyz_after[:2] - self.goal_position)
        if distance_to_goal < 1.0:  # Threshold for reaching the goal
            goal_reward = 200.0  # Large reward for reaching the goal

        if not self._is_on_path(torso_xyz_after[:2]):
            fall_penalty = 0  # Penalty for falling off the path

        reward = progress_reward + goal_reward - ctrl_cost - \
            contact_cost - fall_penalty + survive_reward

        state = self.state_vector()
        notdone = np.isfinite(state).all(
        ) and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone or goal_reward > 0 or (
            self.step_count == self._max_episode_steps - 1)

        ob = self._get_obs()

        return ob, reward, done, dict(
            reward_progress=progress_reward,
            reward_goal=goal_reward,
            reward_fall=fall_penalty,
            torso_velocity=torso_velocity,
            goal_position=self.goal_position
        )

    def _generate_path_sequence(self, idx):
        # Defines the winding path directions
        if idx == 0:
            return ['north', 'north', 'north']
        elif idx == 1:
            return ['east', 'north', 'east', 'north', 'east', 'north']
        elif idx == 2:
            return ['east', 'east', 'east']
        elif idx == 3:
            return ['south', 'east', 'south', 'east', 'south', 'east']
        elif idx == 4:
            return ['south', 'south', 'south']
        elif idx == 5:
            return ['west', 'south', 'west', 'south', 'west', 'south']
        elif idx == 6:
            return ['west', 'west', 'west']
        elif idx == 7:
            return ['north', 'west', 'north', 'west', 'north', 'west']
        elif idx == 8:
            return ['north', 'north']
        elif idx == 9:
            return ['east', 'east', 'north', 'north']
        elif idx == 10:
            return ['east', 'east']
        elif idx == 11:
            return ['south', 'south', 'east', 'east']
        elif idx == 12:
            return ['south', 'south']
        elif idx == 13:
            return ['west', 'west', 'south', 'south']
        elif idx == 14:
            return ['west', 'west']
        elif idx == 15:
            return ['north', 'north', 'west', 'west']
        else:
            raise NotImplementedError

    def _generate_path_tiles(self):
        # Generates a list of path tiles based on direction sequence
        path_tiles = [(0., 0.)]
        current_position = np.array([0., 0.])
        step_size = 1.0  # Distance between each tile

        for direction in self.path_direction_sequence:
            if direction == 'north':
                current_position += np.array([0., step_size])
            elif direction == 'east':
                current_position += np.array([step_size, 0.])
            elif direction == 'south':
                current_position += np.array([0., -step_size])
            elif direction == 'west':
                current_position += np.array([-step_size, 0.])
            path_tiles.append(tuple(current_position))
        self.goal_position = current_position
        return path_tiles

    def _is_on_path(self, position):
        # Check if the agent's position is close to any tile in the path
        for tile in self.path_tiles:
            # Tolerance for being on path
            if np.linalg.norm(position - np.array(tile)) < 1:
                return True
        return False

    def reset(self):
        self.step_count = 0
        # self.goal_position = np.array([10.0, 10.0])
        # Reset other necessary states
        return super(AntCliffEnv, self).reset()

    def sample_tasks(self, n_tasks):
        # Sample different goal positions
        return [np.random.uniform(low=5.0, high=15.0, size=(2,)) for _ in range(n_tasks)]

    def set_task(self, task):
        self.goal_position = task

    def get_task(self):
        return self.goal_position

    def get_dataset(self):
        train_dataset = OrderedDict()
        with open(f'data_collection/AntCliff-v0/replay/dataset_task_0.pkl', "rb") as f:
            dataset = pickle.load(f)
        f.close()
        for key, value in dataset.items():
            train_dataset[key] = [value]
        train_dataset['task_id'] = [
            (np.ones(dataset['observations'].shape[0]) * 0).astype(np.int32)]

        # for task_id in range(1, self.num_train_tasks):
        for task_id in range(1, len(self.task_describe)):
            if task_id not in self.eval_tasks:
                with open(f'data_collection/AntCliff-v0/replay/dataset_task_{task_id}.pkl', "rb") as f:
                    dataset = pickle.load(f)
                f.close()

                for key, value in dataset.items():
                    train_dataset[key].append(value)
                train_dataset['task_id'].append(
                    (np.ones(dataset['observations'].shape[0]) * task_id).astype(np.int32))

        for key, value in train_dataset.items():
            train_dataset[key] = np.concatenate(value, axis=0)

        num_samples = train_dataset['observations'].shape[0]

        print('================================================')
        print(
            f'Successfully constructed the training dataset from {len(self.eval_tasks)} tasks...')
        print(f'Number of training samples: {num_samples}')
        print('================================================')

        return train_dataset

    def get_align_trajectories(self, num_demos=400, num_medium=200):
        prompt_trajectories = []

        for task_id in range(len(self.task_describe)):
            with open(f'data_collection/AntCliff-v0/replay/dataset_task_{task_id}.pkl', "rb") as f:
                dataset = pickle.load(f)
            f.close()
            trajectories = convert_data_to_trajectories(dataset)

            returns = [traj['rewards'].sum() for traj in trajectories]
            sorted_inds = sorted(range(len(returns)),
                                 key=lambda x: returns[x], reverse=True)

            demos = [trajectories[sorted_inds[i]] for i in range(num_demos)]
            # medium_demos = [trajectories[sorted_inds[i]]
            #                 for i in range(2500, 2500 + num_medium)]
            prompt_trajectories.append(demos)
            # prompt_trajectories.append(medium_demos)

        return prompt_trajectories
