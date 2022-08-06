"""16.3节A3C算法实现。"""

import ray
from torch import nn
import torch.nn.functional as F
import torch
import gym
import numpy as np


class QNet(nn.Module):
    """QNet.
    Input: feature
    Output: num_act of values
    """

    def __init__(self, dim_state, dim_action):
        super().__init__()
        self.fc1 = nn.Linear(dim_state + dim_action, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], -1)  # check?
        x = F.relu(self.fc1(sa))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PolicyNet(nn.Module):
    def __init__(self, dim_state, dim_action):
        super().__init__()
        self.fc1 = nn.Linear(dim_state, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, dim_action)

    def forward(self, state):
        """输出action的取值范围是[-1, 1]。"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


class Rollout:
    def __init__(self):
        self.state_lst = []
        self.action_lst = []
        self.reward_lst = []
        self.done_lst = []
        self.next_state_lst = []

    def put(self, state, action, reward, done, next_state):
        self.state_lst.append(state)
        self.action_lst.append(action)
        self.reward_lst.append(reward)
        self.done_lst.append(done)
        self.next_state_lst.append(next_state)

    def numpy(self):
        state = np.array(self.state_lst).float()
        action = np.array(self.action_lst).float()
        reward = np.array(self.reward_lst).float()
        done = np.array(self.done_lst).float()
        next_state = np.array(self.next_state_lst).float()
        return state, action, reward, done, next_state

    def torch(self):
        state, action, reward, done, next_state = self.numpy()
        state = torch.from_numpy(state)
        action = torch.from_numpy(action)
        reward = torch.from_numpy(reward)
        done = torch.from_numpy(done)
        next_state = torch.from_numpy(next_state)
        return state, action, reward, done, next_state


@ray.remote(num_cpus=2)
class A3C:
    def __init__(self, dim_state, dim_action):
        self.pi = PolicyNet(dim_state, dim_action)
        self.Q = QNet(dim_state, dim_action)
        self.Q_target = QNet(dim_state, dim_action)
        self.Q_target.load_state_dict(self.Q.state_dict())

        self.env = gym.make("Pendulum-v1")

    def play_one_rollout(self):
        rollout = Rollout()
        s = self.env.reset()
        while True:
            a = self.pi(s)
            a = a.cpu().numpy() * args.max_action

            ns, d, r, _ = self.env.step(a)
            rollout.put(s, a, d, r, ns)

            if d is True:
                break
        return rollout

    def compute_gradient(self, args, pi_state_dict, Q_state_dict):
        """计算策略网络和值网络的梯度。"""
        # 更新策略网络，值网络，目标值网络参数。
        self.pi.load_state_dict(pi_state_dict)
        self.Q.load_state_dict(Q_state_dict)
        self.soft_update()

        # 与环境进行一个完整回合的游戏。
        rollout = self.play_one_rollout()

        # 计算网络参数梯度。
        bs, ba, br, bd, bns = rollout.torch()

        self.Q_target()

    def pull_weight(self):
        """从主节点处拉取网络权重。"""

    def soft_update(self, tau=0.01):
        def soft_update_(target, source, tau_=0.01):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau_) + param.data * tau_
                )

        soft_update_(self.Q_target, self.Q, tau)


worker_lst = [A3C.remote() for _ in range(4)]

gradient_lst = [worker.compute_gradient.remote() for worker in worker_lst]

gradient_sum = 0
while True:
    gradient, rest_gradient_lst = ray.wait(gradient_lst)
    gradient_sum += gradient
    gradient_lst = rest_gradient_lst

gradient = gradient_sum / 4
