"""15.3节MAC-A2C算法实现，采用中心化训练+中心化决策方案。"""
import argparse
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from pettingzoo.mpe import simple_spread_v2
import time
from collections import defaultdict

from torch import Tensor
from typing import Dict


class ValueNet(nn.Module):
    def __init__(self, dim_state):
        super().__init__()
        self.fc1 = nn.Linear(dim_state, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PolicyNet(nn.Module):
    def __init__(self, dim_state, num_action):
        super().__init__()
        self.fc1 = nn.Linear(dim_state, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_action)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        prob = F.softmax(x, dim=-1)
        return prob

    def policy(self, observation):
        x = F.relu(self.fc1(observation))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        log_prob_action = F.log_softmax(x, dim=-1)
        return log_prob_action


class MAC:
    def __init__(
        self,
        num_agents=2,
        num_states=24,
        num_actions=5,
        gamma=0.9,
        tau=0.01,
    ):

        self.num_agents = num_agents
        self.num_states = num_states
        self.num_actions = num_actions

        self.gamma = gamma
        self.tau = tau

        self.agent2policy = {}
        for i in range(num_agents):
            self.agent2policy[f"agent_{i}"] = PolicyNet(num_states, num_actions)

        self.value_net = ValueNet(num_states)
        self.target_value_net = deepcopy(self.value_net)

    def policy(self, observation, agent):
        # 参考https://pytorch.org/docs/stable/distributions.html#score-function。
        log_prob_action = self.agent2policy[agent].policy(observation)
        m = Categorical(logits=log_prob_action)
        action = m.sample()
        log_prob_a = m.log_prob(action)
        return action.item(), log_prob_a

    def value(self, observation):
        value_ = self.value_net(observation)
        return value_

    def target_value(self, observation):
        target_value_ = self.target_value_net(observation)
        return target_value_

    # def compute_policy_loss(self, last_observation, current_observation, reward, log_prob_action_dct):
    #     with torch.no_grad():
    #         td_target = reward + self.gamma * self.target_value(current_observation)
    #         value = self.value(last_observation)
    #         advantage = value - td_target

    #     policy_loss = 0
    #     for i in range(self.num_agents):
    #         policy_loss += -log_prob_action_dct[f"agent_{i}"] * advantage
    #     return policy_loss

    # (s0, a0, r0, d0, s1)

    # V(s_{t+1}) = r_t + self.gamma * V(s_t) * (1 - done)

    # V(s_t) = \sum_{i=t}^T = gamma^{i-t} * r_i.

    def compute_policy_loss(self, last_observation: Tensor, current_observation: Tensor, reward: Tensor, log_prob_action_dct, dones):

        with torch.no_grad():
            target_value = [0]
            for i in reversed(range(len(reward))):
                value = target_value[-1] * self.gamma + reward[i]
                target_value.append(value)

            target_value = target_value[1:]
            target_value = torch.tensor(target_value[::-1])

            value = self.value(last_observation)
            advantage = value - target_value

        # with torch.no_grad():
        #     current_observation_value = self.target_value(current_observation).squeeze()
        #     td_target = reward + self.gamma * current_observation_value * (1 - dones)
        #     value = self.value(last_observation).squeeze()
        #     advantage = value - td_target

        policy_loss = 0
        for i in range(self.num_agents):
            policy_loss += -log_prob_action_dct[f"agent_{i}"] * advantage
        policy_loss = policy_loss.mean()
        return policy_loss

    # def compute_value_loss(self, last_observation, current_observation, reward):
    #     # 1. 输入agent0和agent1的联合observation。last_observation是当前动作之前的观察，current_observation是当前动作之后的观察，reward是当前动作之后的奖励。
    #     # 2. 使用target value计算TD目标。target_value(current_observation) + gamma * reward。
    #     # 3. 使用value计算状态值value(last_observatoin)。
    #     with torch.no_grad():
    #         td_target = reward * self.gamma + central_controller.target_value(current_observation)

    #     value = central_controller.value(last_observation)
    #     value_loss = F.mse_loss(value, td_target)
    #     return value_loss

    def compute_value_loss(self, last_observation, current_observation, reward):
        with torch.no_grad():
            target_value = [0]
            for i in reversed(range(len(reward))):
                value = target_value[-1] * self.gamma + reward[i]
                target_value.append(value)

            target_value = target_value[1:]
            target_value = torch.tensor(target_value[::-1])

        value = self.value(last_observation)
        value_loss = F.mse_loss(value, target_value)
        return value_loss

    def update_target_value(self):
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)


class Rollout:
    def __init__(self):
        self.last_observation_lst = []
        self.current_observation_lst = []
        self.log_prob_action_lst_dct = defaultdict(list)
        self.reward_lst = []
        self.done_lst = []

    def put(self, last_observation, current_observation, log_prob_action_dct, reward, done):
        self.last_observation_lst.append(last_observation)
        self.current_observation_lst.append(current_observation)
        for k, v in log_prob_action_dct.items():
            self.log_prob_action_lst_dct[k].append(v)
        self.reward_lst.append(reward)
        self.done_lst.append(done)

    def tensor(self):
        batch_last_observation = torch.stack(self.last_observation_lst)
        batch_current_observation = torch.stack(self.current_observation_lst)
        batch_log_prob_action_dct = {k: torch.stack(v) for k, v in self.log_prob_action_lst_dct.items()}
        batch_reward = torch.as_tensor(self.reward_lst, dtype=torch.float32)
        batch_done = torch.as_tensor(self.done_lst, dtype=torch.float32)
        return batch_last_observation, batch_current_observation, batch_log_prob_action_dct, batch_reward, batch_done


env = simple_spread_v2.env(N=2, local_ratio=0.5, max_cycles=25, continuous_actions=False)


num_agents = len(env.possible_agents)
num_actions = env.action_space(env.possible_agents[0]).n
observation_size = env.observation_space(env.possible_agents[0]).shape

print(f"{num_agents} agents")
for i in range(num_agents):
    num_actions = env.action_space(env.possible_agents[i]).n
    observation_size = env.observation_space(env.possible_agents[i]).shape
    print(i, env.possible_agents[i], "num_actions:", num_actions, "observation_size:", observation_size)


central_controller = MAC()


policy_params = []
for i in range(num_agents):
    policy_params += list(central_controller.agent2policy[f"agent_{i}"].parameters())
policy_optimizer = Adam(policy_params, lr=1e-3)
value_optimizer = Adam(central_controller.value_net.parameters(), lr=1e-3)


num_episode = 1000
episode_reward_lst = []
for episode in range(num_episode):
    env.reset()
    log_prob_action_dct = {}
    observation_lst = [env.observe(f"agent_{x}") for x in range(num_agents)]
    current_observation = np.concatenate(observation_lst)
    current_observation = torch.as_tensor(current_observation, dtype=torch.float32)

    episode_reward = 0

    rollout = Rollout()

    for i, agent in enumerate(env.agent_iter()):

        # # 如果环境运行结束，则打印当前episode的信息，并断开环境。
        # _, _, termination, _ = env.last()
        # if termination is True:

        #     # batch_last_observation, batch_current_observation, batch_log_prob_action_dct, batch_reward, batch_done = rollout.tensor()

        #     # print(batch_last_observation.shape)
        #     # print(batch_current_observation.shape)
        #     # print(batch_reward.reshape)
        #     # print(batch_done.shape)

        #     episode_reward_lst.append(episode_reward)
        #     if episode % 20 == 0:
        #         print(f"episode={episode}, episode_reward mean={np.mean(episode_reward_lst[-20:])}")

        #     if episode % 1000 == 0:
        #         x_axis = np.arange(len(episode_reward_lst))
        #         plt.plot(x_axis, episode_reward_lst)
        #         plt.xlabel("episode")
        #         plt.ylabel("reward")
        #         plt.savefig("simple_spread.png", bbox_inches="tight")
        #         plt.close()
        #     break

        action, log_prob_action = central_controller.policy(current_observation, agent)
        log_prob_action_dct[agent] = log_prob_action
        env.step(action)

        # 每当下一个执行动作的agent变成0号agent时，重新收集所有agent的observation，用于action选择。
        if env.agent_selection == "agent_0":
            # 收集所有agent的observation。
            last_observation = current_observation
            observation_lst = [env.observe(f"agent_{x}") for x in range(num_agents)]
            current_observation = np.concatenate(observation_lst)
            current_observation = torch.as_tensor(current_observation, dtype=torch.float32)
            reward = env.rewards["agent_0"]  # 所有agent的奖励是一样的。
            termination = env.terminations["agent_0"] or env.truncations["agent_0"]

            rollout.put(last_observation, current_observation, log_prob_action_dct, reward, termination)
            episode_reward += reward

            # 如果运行到环境终点，打印过程信息，并训练模型。
            if termination is True:
                # 打印信息和画图。
                episode_reward_lst.append(episode_reward)
                if episode % 20 == 0:
                    print(f"episode={episode}, episode_reward mean={np.mean(episode_reward_lst[-20:])}")
                if episode % 1000 == 0:
                    x_axis = np.arange(len(episode_reward_lst))
                    plt.plot(x_axis, episode_reward_lst)
                    plt.xlabel("episode")
                    plt.ylabel("reward")
                    plt.savefig("simple_spread.png", bbox_inches="tight")
                    plt.close()

                # 检查训练素材。
                batch_last_observation, batch_current_observation, batch_log_prob_action_dct, batch_reward, batch_done = rollout.tensor()
                # print(batch_last_observation.shape)
                # print(batch_current_observation.shape)
                # print(batch_reward.shape)
                # print(batch_done.shape)
                # print(batch_log_prob_action_dct["agent_0"].shape, batch_log_prob_action_dct["agent_1"].shape)

                # 训练模型。
                policy_loss = central_controller.compute_policy_loss(batch_last_observation, batch_current_observation, batch_reward, batch_log_prob_action_dct, batch_done)
                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()

                value_loss = central_controller.compute_value_loss(batch_last_observation, batch_current_observation, batch_reward)
                value_optimizer.zero_grad()
                value_loss.backward()
                value_optimizer.step()

                # print(f"policy loss={policy_loss.item()}, value loss={value_loss.item()}")

                # 运行到环境重点，终止交互。
                break

            # # 训练模型。
            # policy_loss = central_controller.compute_policy_loss(last_observation, current_observation, reward, log_prob_action_dct)
            # policy_optimizer.zero_grad()
            # policy_loss.backward()
            # policy_optimizer.step()

            # value_loss = central_controller.compute_value_loss(last_observation, current_observation, reward)
            # value_optimizer.zero_grad()
            # value_loss.backward()
            # value_optimizer.step()

            # central_controller.update_target_value()

            # print(f">> {i}, policy_loss={policy_loss.item()}, value_loss={value_loss.item()}")
