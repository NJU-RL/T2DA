import pickle
import numpy as np

from collections import OrderedDict
from typing import Dict, List, Optional

from .mujoco_env import MujocoEnv


class AntEnv(MujocoEnv):
    def __init__(self, use_low_gear_ratio=False):
        # self.init_serialization(locals())
        if use_low_gear_ratio:
            xml_path = 'low_gear_ratio_ant.xml'
        else:
            xml_path = 'ant_dir.xml'
        super().__init__(
            xml_path,
            frame_skip=5,
            automatically_set_obs_and_action_space=True,
        )

    def step(self, a):
        torso_xyz_before = self.get_body_com("torso")
        self.do_simulation(a, self.frame_skip)
        torso_xyz_after = self.get_body_com("torso")
        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_reward = torso_velocity[0]/self.dt
        ctrl_cost = 0. # .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0. # 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
                  and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            torso_velocity=torso_velocity,
        )

    def _get_obs(self):
        # this is gym ant obs, should use rllab?
        # if position is needed, override this in subclasses
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


class MultitaskAntEnv(AntEnv):
    def __init__(self, task={}, n_tasks=2, **kwargs):
        self._task = task
        self.tasks = self.sample_tasks(n_tasks)
        self._goal = self.tasks[0]['goal']
        super().__init__(**kwargs)

    """
    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = -1.0 * abs(forward_vel - self._goal_vel)
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward,
                     reward_ctrl=-ctrl_cost, task=self._task)
        return (observation, reward, done, infos)
    """


    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['goal'] # assume parameterization of task by single vector
        self.reset()


class AntDirEnv_(MultitaskAntEnv):

    def __init__(self, task={}, n_tasks=2, forward_backward=False, randomize_tasks=True, **kwargs):
        self.forward_backward = forward_backward
        super(AntDirEnv_, self).__init__(task, n_tasks, **kwargs)

    def step(self, action):
        torso_xyz_before = np.array(self.get_body_com("torso"))

        direct = (np.cos(self._goal), np.sin(self._goal))

        self.do_simulation(action, self.frame_skip)

        goal_marker_idx = self.sim.model.site_name2id('goal')

        self.data.site_xpos[goal_marker_idx,:2] = 5 * np.array([np.cos(self._goal), np.sin(self._goal)])
        self.data.site_xpos[goal_marker_idx,-1] = 1

        torso_xyz_after = np.array(self.get_body_com("torso"))
        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_reward = np.dot((torso_velocity[:2]/self.dt), direct)

        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        # notdone = np.isfinite(state).all() \
        #           and state[2] >= 0.2 and state[2] <= 1.0
        # done = not notdone
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            torso_velocity=torso_velocity,
        )

    def sample_tasks(self, num_tasks):
        if self.forward_backward:
            assert num_tasks == 2
            velocities = np.array([0., np.pi])
        else:
            velocities = np.random.uniform(0., 2.0 * np.pi, size=(num_tasks,))
        tasks = [{'goal': velocity} for velocity in velocities]
        return tasks


class AntDirEnv(AntDirEnv_):
    def __init__(self, tasks: Optional[List[Dict]] = None, n_tasks: Optional[int] = None, include_goal: bool = False, train_task_ids: Optional[List[int]] = None):
        self.include_goal = include_goal
        super(AntDirEnv, self).__init__(forward_backward=n_tasks == 2)
        if tasks is None:
            assert n_tasks is not None, "Either tasks or n_tasks must be non-None"
            tasks = self.sample_tasks(n_tasks)
        self.tasks = tasks
        self.n_tasks = len(self.tasks)
        self.train_task_ids = train_task_ids if train_task_ids is not None else list(range(self.n_tasks - 5))
        self.num_train_tasks = len(self.train_task_ids)
        self.set_task_idx(0)
        self._max_episode_steps = 200

    def _get_obs(self):
        if self.include_goal:
            idx = 0
            try:
                idx = self.tasks.index(self._task)
            except:
                pass
            one_hot = np.zeros(50, dtype=np.float32)
            one_hot[idx] = 1.0
            obs = super()._get_obs()
            obs = np.concatenate([obs, one_hot])
        else:
            obs = super()._get_obs()
        return obs

    def set_task(self, task):
        self._task = task
        self._goal = task['goal']
        self.reset()

    def set_task_idx(self, idx):
        self.set_task(self.tasks[idx])

    def get_dataset(self):
        train_dataset = OrderedDict()
        with open(f"datasets/AntDir-v0/dataset_task_{self.train_task_ids[0]}.pkl", "rb") as f:
            dataset = pickle.load(f)
        for key, value in dataset.items():
            train_dataset[key] = [value]
        train_dataset['task_id'] = [(np.ones(dataset['observations'].shape[0]) * self.train_task_ids[0]).astype(np.int32)]

        for task_id in self.train_task_ids[1:]:
            with open(f"datasets/AntDir-v0/dataset_task_{task_id}.pkl", "rb") as f:
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