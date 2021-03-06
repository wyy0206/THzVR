import multiprocessing as mp

import gym
import tensorflow as tf

from maml_rl.envs.subproc_vec_env import SubprocVecEnv
from maml_rl.episode import BatchEpisodes


"""
The code is taken from and perhaps will be changed in the future
https://github.com/tristandeleu/pytorch-maml-rl/blob/master/maml_rl/sampler.py
"""


def make_env(env_name):
    def _make_env():
        return gym.make(env_name)
    return _make_env


class BatchSampler(object):
    def __init__(self, env_name, batch_size, num_workers=mp.cpu_count() - 1):
        self.env_name = env_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.queue = mp.Queue()
        self.envs = SubprocVecEnv([make_env(env_name) for _ in range(num_workers)],
                                  queue=self.queue)
        self._env = gym.make(env_name)

    def sample(self, policy, params=None, gamma=0.9):
        episodes = BatchEpisodes(batch_size=self.batch_size, gamma=gamma)
        for i in range(self.batch_size):
            self.queue.put(i)
        for _ in range(self.num_workers):
            self.queue.put(None)
        observations, batch_ids = self.envs.reset()
        dones = [False]
        while (not all(dones)) or (not self.queue.empty()):
            observations_tensor = observations
            # 气死 observations和action要一样的维度 垃圾
            # observations_tensor = observations.reshape(observations.shape[0], -1)
            actions_tensor = policy(observations_tensor, params=params).sample()
            # /CPU:0
            with tf.device('/CPU:0'):
                actions = actions_tensor.numpy()
            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            episodes.append(observations, actions, rewards, batch_ids)
            observations, batch_ids = new_observations, new_batch_ids

        return episodes

    def reset_task(self, task):
        tasks = [task for _ in range(self.num_workers)]
        reset = self.envs.reset_task(tasks)
        return all(reset)

    def sample_tasks(self, num_tasks):
        tasks = self._env.unwrapped.sample_tasks(num_tasks)
        return tasks
