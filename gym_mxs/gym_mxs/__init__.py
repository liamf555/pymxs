# from gym.envs.registration import register

# register(
#     id="gym_mxs/MXS-v0",
#     entry_point="gym_mxs.envs:MxsEnv",
# )
try:
    from gymnasium.envs.registration import register
except:
    from gym.envs.registration import register

register(
    id="gym_mxs/MXSBox2D-v0",
    entry_point="gym_mxs.envs:MxsEnvBox2D",
    max_episode_steps=2000,
)

register(
    id="gym_mxs/MXSBox2DLidar-v0",
    entry_point="gym_mxs.envs:MxsEnvBox2DLidar",
    max_episode_steps=2000,
)

register(
    id="gym_mxs/MXSBox2DLand-v0",
    entry_point="gym_mxs.envs:MxsEnvBox2DLand",
    max_episode_steps=2000,
)

