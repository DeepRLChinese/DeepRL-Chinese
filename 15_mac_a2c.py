"""15.3节MAC-A2C算法实现，采用中心化训练+中心化决策方案。"""
import argparse
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
import os


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


class MAC(nn.Module):
    def __init__(
        self,
        num_agents=1,
        num_states=6,
        num_actions=5,
        gamma=0.95,
        tau=0.01,
    ):
        super().__init__()

        self.num_agents = num_agents
        self.num_states = num_states
        self.num_actions = num_actions

        self.gamma = gamma
        self.tau = tau

        self.agent2policy = {}
        for i in range(num_agents):
            self.agent2policy[f"agent_{i}"] = PolicyNet(num_states, num_actions)

        self.value_net = ValueNet(num_states)
        self.target_value_net = ValueNet(num_states)
        self.target_value_net.load_state_dict(self.value_net.state_dict())

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

    def compute_policy_loss(self, bs, br, bd, bns, logp_action_dict):

        with torch.no_grad():
            # td_value = self.target_value(bns).squeeze()
            # td_value = br + self.gamma * td_value * (1 - bd)
            predicted_value = self.value(bs).squeeze()
            # advantage = predicted_value - td_value

            # compute_value_loss使用br作为td目标。计算advantage时，同样使用br作为baseline。
            advantage = predicted_value - br

        policy_loss = 0
        for i in range(self.num_agents):
            policy_loss += logp_action_dict[f"agent_{i}"] * advantage
        policy_loss = policy_loss.mean()
        return policy_loss

    def compute_value_loss(self, bs, br, bd, bns, blopg_action_dict):
        # 注意到simple_spread_v2中，reward是根据当前状态到目标位置的距离而计算的奖励。因此，直接使用reward作为td目标值更合适。
        # with torch.no_grad():
        #     td_value = self.target_value(bns).squeeze()
        #     td_value = br + self.gamma * td_value * (1 - bd)
        td_value = br

        predicted_value = self.value(bs).squeeze()
        value_loss = F.mse_loss(predicted_value, td_value)
        return value_loss

    def update_target_value(self):
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)


class Rollout:
    def __init__(self):
        self.state_list = []
        self.reward_list = []
        self.done_list = []
        self.next_state_list = []
        self.logp_actions_dict = defaultdict(list)

    def put(self, state, reward, done, next_state, logp_action_dict):
        self.state_list.append(state)
        self.reward_list.append(reward)
        self.done_list.append(done)
        self.next_state_list.append(next_state)
        for k, v in logp_action_dict.items():
            self.logp_actions_dict[k].append(v)

    def tensor(self):
        bs = torch.tensor(np.asarray(self.state_list)).float()
        br = torch.tensor(np.asarray(self.reward_list)).float()
        bd = torch.tensor(np.asarray(self.done_list)).float()
        bns = torch.tensor(np.asarray(self.next_state_list)).float()
        blogp_action_dict = {k: torch.stack(v) for k, v in self.logp_actions_dict.items()}
        return bs, br, bd, bns, blogp_action_dict


def train(args, env, central_controller: MAC):
    # 训练初始化。
    policy_params = []
    for i in range(num_agents):
        policy_params += list(central_controller.agent2policy[f"agent_{i}"].parameters())
    policy_optimizer = Adam(policy_params, lr=args.lr_policy)
    value_optimizer = Adam(central_controller.value_net.parameters(), lr=args.lr_value)

    max_reward = 0
    episode_reward_lst = []
    log = defaultdict(list)

    for episode in range(args.num_episode):
        env.reset()
        state = [env.observe(f"agent_{x}") for x in range(num_agents)]
        state = np.concatenate(state)
        logp_action_dict = {}
        episode_reward = 0
        rollout = Rollout()

        for i, agent in enumerate(env.agent_iter()):
            action, logp_action = central_controller.policy(torch.as_tensor(state).float(), agent)
            logp_action_dict[agent] = logp_action
            env.step(action)

            # 当下一个执行动作的agent变成0号agent时，表示所有agent完成了动作选择，此时重新收集所有agent的state。
            if env.agent_selection == "agent_0":
                # 收集所有agent的observation。
                next_state = [env.observe(f"agent_{x}") for x in range(num_agents)]
                next_state = np.concatenate(next_state)
                reward = env.rewards["agent_0"]  # 所有agent的奖励是一样的。
                done = env.terminations["agent_0"] or env.truncations["agent_0"]

                rollout.put(state, reward, done, next_state, logp_action_dict)
                state = next_state

                episode_reward += reward
                # 如果运行到环境终点，训练模型。
                if done is True:
                    episode_reward_lst.append(episode_reward)

                    # if episode_reward >= max(episode_reward_lst):
                    if episode % 1000 == 0:

                        agent2policynet = {}
                        for agent, policynet in central_controller.agent2policy.items():
                            agent2policynet[agent] = policynet.state_dict()
                        torch.save(agent2policynet, os.path.join(args.output_dir, "model.pt"))

                    if episode % 1000 == 0:
                        x_axis = np.arange(len(episode_reward_lst))
                        plt.plot(x_axis, episode_reward_lst)
                        plt.xlabel("episode")
                        plt.ylabel("reward")
                        plt.savefig("simple_spread.png", bbox_inches="tight")
                        plt.close()

                    # 检查训练素材。
                    bs, br, bd, bns, blogp_action_dict = rollout.tensor()

                    # 训练模型。
                    value_loss = central_controller.compute_value_loss(bs, br, bd, bns, blogp_action_dict)
                    value_optimizer.zero_grad()
                    value_loss.backward()
                    value_optimizer.step()

                    policy_loss = central_controller.compute_policy_loss(bs, br, bd, bns, blogp_action_dict)
                    policy_optimizer.zero_grad()
                    policy_loss.backward()
                    policy_optimizer.step()

                    central_controller.update_target_value()

                    log["value_loss"].append(value_loss.item())
                    log["policy_loss"].append(policy_loss.item())
                    if episode % 20 == 0:
                        avg_value_loss = np.mean(log["value_loss"][-20:])
                        avg_policy_loss = np.mean(log["policy_loss"][-20:])
                        avg_reward = np.mean(episode_reward_lst[-20:])
                        print(f"episode={episode}, moving reward={avg_reward:.2f}, value loss={avg_value_loss:.4f}, policy loss={avg_policy_loss:.4f}")

                    break


def eval(args):
    env = simple_spread_v2.env(N=args.num_agents, local_ratio=0.5, max_cycles=25, continuous_actions=False, render_mode="human")
    central_controller = MAC(num_agents=args.num_agents, num_states=args.num_states, num_actions=args.num_actions)

    agent2policynet = torch.load(os.path.join(args.output_dir, "model.pt"))
    for agent, state_dict in agent2policynet.items():
        central_controller.agent2policy[agent].load_state_dict(state_dict)

    central_controller.eval()

    episode_reward_lst = []
    for episode in range(10):
        episode_reward = 0

        env.reset()
        for i, agent in enumerate(env.agent_iter()):
            state = [env.observe(f"agent_{x}") for x in range(num_agents)]
            state = np.concatenate(state)

            action, _ = central_controller.policy(torch.as_tensor(state).float(), agent)
            env.step(action)

            if env.agent_selection == "agent_0":
                next_state = [env.observe(f"agent_{x}") for x in range(num_agents)]
                next_state = np.concatenate(next_state)
                reward = env.rewards["agent_0"]
                done = env.terminations["agent_0"] or env.truncations["agent_0"]
                state = next_state

                episode_reward += reward

                time.sleep(0.1)

                if done is True:
                    episode_reward_lst.append(episode_reward)
                    avg_reward = np.mean(episode_reward_lst[-20:])
                    print(f"episode={episode}, episode reward={episode_reward}, moving reward={avg_reward:.2f}")
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合作型游戏。")
    parser.add_argument("--num_agents", default=2, type=int)
    parser.add_argument("--num_states", default=24, type=int)
    parser.add_argument("--num_actions", default=5, type=int)
    parser.add_argument("--num_episode", default=20000, type=int)
    parser.add_argument("--lr_policy", default=1e-3, type=float)  # 1e-3
    parser.add_argument("--lr_value", default=1e-3, type=float)  # 1e-2
    parser.add_argument("--output_dir", default="output", type=str)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    env = simple_spread_v2.env(N=args.num_agents, local_ratio=0.5, max_cycles=25, continuous_actions=False)
    central_controller = MAC(num_agents=args.num_agents, num_states=args.num_states, num_actions=args.num_actions)

    num_agents = len(env.possible_agents)
    num_actions = env.action_space(env.possible_agents[0]).n
    observation_size = env.observation_space(env.possible_agents[0]).shape

    print(f"{num_agents} agents")
    for i in range(num_agents):
        num_actions = env.action_space(env.possible_agents[i]).n
        observation_size = env.observation_space(env.possible_agents[i]).shape
        print(i, env.possible_agents[i], "num_actions:", num_actions, "observation_size:", observation_size)

    if args.do_train:
        train(args, env, central_controller)

    if args.do_eval:
        eval(args)
