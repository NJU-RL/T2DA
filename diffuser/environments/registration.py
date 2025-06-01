import gym

# ENVIRONMENT_SPECS = (
#     {
#         'id': 'Walker2dCliff-v0',
#         'entry_point': ('diffuser.environments.walker2d_cliff:Walker2dCliffEnv')
#     },
#     {
#         'id': 'AntCliff-v0',
#         'entry_point': ('diffuser.environments.ant_cliff:AntCliffEnv')
#     },
#     {
#         'id': 'Ant-v0',
#         'entry_point': ('diffuser.environments.ant:AntFullObsEnv'),
#     },
#     {
#         'id': 'PointRobot-v0',
#         'entry_point': ('diffuser.environments.point_robot:PointEnv'),
#     },
#     {
#         'id': 'PointRobotMT-v0',
#         'entry_point': ('diffuser.environments.point_robot:PointMTEnv'),
#     },
#     {
#         'id': 'HalfCheetah-v0',
#         'entry_point': ('diffuser.environments.half_cheetah:HalfCheetahEnv')
#     },
#     {
#         'id': 'HalfCheetahVel-v0',
#         'entry_point': ('diffuser.environments.half_cheetah_vel:HalfCheetahVelEnv')
#     },
# )


# def register_environments():
#     try:
#         for environment in ENVIRONMENT_SPECS:
#             gym.register(**environment)

#         gym_ids = tuple(
#             environment_spec['id']
#             for environment_spec in ENVIRONMENT_SPECS)

#         return gym_ids
#     except:
#         print(
#             '[ diffuser/environments/registration ] WARNING: not registering diffuser environments')
#         return tuple()
