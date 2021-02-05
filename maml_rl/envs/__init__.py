from gym.envs.registration import register

register(
    'THzVR-v0',
    entry_point='maml_rl.envs.THzVR:THzVREnv',
    max_episode_steps=60
)

register(
    'DualTHzVR-v0',
    entry_point='maml_rl.envs.DualTHzVR:THzVREnv',
    max_episode_steps=30
)



