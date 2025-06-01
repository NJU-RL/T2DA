from gym.envs.registration import register

def register_mujoco():
    register(
        id='PointRobot-v0',
        entry_point='diffuser.environments.point_robot:PointMTEnv',
        max_episode_steps=20,
    )

    register(
        id='HalfCheetahVel-v0',
        entry_point='diffuser.environments.half_cheetah_vel:HalfCheetahVelEnv',
        max_episode_steps=200,
    )

    register(
        id='AntDir-v0',
        entry_point='diffuser.environments.ant_dir:AntDirEnv',
        max_episode_steps=200,
    )