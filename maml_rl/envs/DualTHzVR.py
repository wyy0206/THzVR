import gym
import numpy as np
import random
import copy
import itertools
from gym import spaces
from gym.utils import seeding
from ..hungarian import Hungarian

class THzVREnv(gym.Env):
    def __init__(self, task={}):
        super(THzVREnv, self).__init__()

        # 6m * 6m 的房间，5个用户, 均匀分布9个VAP和9个SBSs
        # shape=(4, USER_NUM)：[0]服务状态；[1]：是否为当前时刻新服务用户；[3][4]：用户位置x y坐标

        self.USER_NUM = 12
        self.VAP_NUM = 7
        self.VAP_LOCs = [(1, 2), (1, 4), (3, 1), (3, 3), (3, 5), (5, 2), (5, 4)]
        self.SBS_NUM = 7
        self.SBS_LOCs = [(1, 2), (1, 4), (3, 1), (3, 3), (3, 5), (5, 2), (5, 4)]

        high = []
        high.append([1 for i in range(2*self.USER_NUM)])
        high.append([6 for i in range(2*self.USER_NUM)])
        high = np.array(high)

        self.observation_space = spaces.Box(
            low=np.zeros(4 * self.USER_NUM,dtype=int),
            high= high.flatten(), dtype=np.float32)

        # 选3个灯
        self.VAP = range(self.VAP_NUM)
        self.VAP_com = list(itertools.combinations(self.VAP, 3))

        # action space
        self.action_num = len(self.VAP_com)
        self.action_space = spaces.Discrete(self.action_num)


        self._task = task
        # 用户位置转移概率
        self._transition = task.get('transition', [6,1,1,1])
        # 初始状态
        self._state = np.zeros(4*self.USER_NUM, dtype=np.float32)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample_tasks(self, num_tasks):
        # transitions = self.np_random.randint(0, 10, size=(num_tasks, 4))
        transition = np.array([6, 1, 1, 1])
        zero_mat = np.zeros((num_tasks, 4))
        transitions = transition + zero_mat
        tasks = [{'transition': transition} for transition in transitions]
        return tasks

    def reset_task(self, task):
        self._task = task
        self._transition = task['transition']

    def reset(self, env=True):
        self._state = np.zeros(4 * self.USER_NUM, dtype=np.float32)
        return self._state

    def step(self, action):
        assert self.action_space.contains(action), f"Action {action} not in the action space"
        self._state = self._state.reshape(4, self.USER_NUM)

        # selected VAP:[1,2,3]
        self.LVAP = self.VAP_com[action]

        # 对每个用户，前后左右随机选择一个方向 index=0，1，2，3
        for u in range(self.USER_NUM):
            start = 0
            index = 0
            randnum = random.randint(1, sum(self._transition))
            for index, scope in enumerate(self._transition):
                start += scope
                if randnum <= start:
                    break
            # user position
            user_loc = self._state[2:,u]
            if index == 0:
                user_loc[0] = (user_loc[0] + 0.5) % 6
            elif index == 1:
                user_loc[0] = (user_loc[0] - 0.5) % 6
            elif index == 2:
                user_loc[1] = (user_loc[1] + 0.5) % 6
            else:
                user_loc[1] = (user_loc[1] - 0.5) % 6
            self._state[2:, u] = user_loc


        # 定位状态
        state_p = np.ones(self.USER_NUM,dtype=int)
        for k in self.LVAP:
            loc_k = self.VAP_LOCs[k]
            for u in range(self.USER_NUM):
                loc_u = self._state[2:, u]
                for m in range(self.USER_NUM):
                    if m != u:
                        loc_m = self._state[2:, m]
                        # state_p1 VAP k, 用户m, 用户u 共线
                        state_p1= (loc_m[1] - loc_k[1]) * (loc_u[0] - loc_k[0]) - (loc_u[1] - loc_k[1]) * (loc_m[0] - loc_k[0]) == 0
                        # state_p2 用户m 在 VAP k 和 用户u 之间
                        state_p2= np.sqrt(np.sum((loc_k - loc_m) ** 2)) < np.sqrt(np.sum((loc_k - loc_u) ** 2))
                        # np.linalg.norm(loc_k - loc_m) <= np.linalg.norm(loc_k - loc_u)
                        if state_p1 and state_p2:
                            state_p[u]=0
                if np.sqrt(np.sum((loc_k - loc_u) ** 2)) > 3:
                    state_p[u] = 0

        # 定位到的用户位置 [x,y, userid]
        user_loceds = []
        for u in range(self.USER_NUM):
            if state_p[u]==1 and self._state[0,u]==0:
                user_loceds.append(np.array([self._state[2,u], self._state[3,u],u]))
        np.array(user_loceds)


        # 服务状态
        state_h = np.zeros(self.USER_NUM,dtype=int)
        # 匈牙利算法求解user association
        if len(user_loceds)!=0:
            profit_matrix = np.ones((len(user_loceds), self.SBS_NUM))
            for k in range(self.SBS_NUM):
                loc_k = self.SBS_LOCs[k]
                for u in range(len(user_loceds)):
                    loc_u = np.array([user_loceds[u][0], user_loceds[u][1]])
                    for m in range(len(user_loceds)):
                        if m != u:
                            loc_m = np.array([user_loceds[m][0], user_loceds[m][1]])
                            state_h1 = (loc_m[1] - loc_k[1]) * (loc_u[0] - loc_k[0]) - (loc_u[1] - loc_k[1]) * (
                                        loc_m[0] - loc_k[0]) == 0
                            state_h2 = np.sqrt(np.sum((loc_k - loc_m) ** 2)) <= np.sqrt(np.sum((loc_k - loc_u) ** 2))
                            if state_h1 and state_h2:
                                profit_matrix[u][k] = 0
                    if np.sqrt(np.sum((loc_k - loc_u) ** 2)) > 1.5:
                        profit_matrix[u][k] = 0
            hungarian = Hungarian()
            hungarian.calculate(profit_matrix, is_profit_matrix=True)
            # 匹配结果hungarian.get_results()
            # [(0, 0), (1, 2), (2, 1)]
            self.asso = hungarian.get_results()
            for i in range(len(self.asso)):
                u_loced = self.asso[i][0]
                k = self.asso[i][1]
                u = int(user_loceds[u_loced][2])
                state_h[u] = profit_matrix[u_loced][k]

        # 服务状态
        state_wt = state_p * state_h
        state_w = copy.deepcopy(self._state[0,:]).astype(int)
        self._state[0, :] = state_wt | state_w
        self._state[1, :] = state_w ^ (self._state[0, :].astype(int))


        reward = sum(self._state[1, :])
        done = (sum(self._state[0, :]) == self.USER_NUM)
        self._state = self._state.reshape(-1)

        return self._state, reward, done, self._task

    # """随机变量的概率函数"""
    # 传入数组为概率分布列表例如[10, 90]，返回值为下标索引，返回值返回0的概率为10%，返回1的概率为90%
    # def random_index(self,rate):
    #     # 参数rate为list<int>
    #     start = 0
    #     index = 0
    #     randnum = random.randint(1, sum(rate))
    #     for index, scope in enumerate(rate):
    #         start += scope
    #         if randnum <= start:
    #             break
    #     return index