"""10.4节TD3算法实现。
"""
import argparse
from collections import defaultdict
import os
import random
from dataclasses import dataclass, field
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

import matplotlib.pyplot as plt


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
        sa = torch.cat([state, action], -1)
        x = F.relu(self.fc1(sa))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PolicyNet(nn.Module):
    def __init__(self, dim_state, dim_action, max_action=2.0):
        super().__init__()
        self.max_action = max_action
        self.fc1 = nn.Linear(dim_state, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, dim_action)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.max_action * torch.tanh(self.fc3(x))
        return x


class TD3:
    def __init__(self, dim_state, dim_action, max_action):
        super().__init__()

        self.max_action = max_action
        self.Q1 = QNet(dim_state, dim_action)
        self.Q2 = QNet(dim_state, dim_action)
        self.Mu = PolicyNet(dim_state, dim_action, max_action)
        self.target_Q1 = QNet(dim_state, dim_action)
        self.target_Q2 = QNet(dim_state, dim_action)
        self.target_Mu = PolicyNet(dim_state, dim_action, max_action)
        self.target_Q1.load_state_dict(self.Q1.state_dict())
        self.target_Q2.load_state_dict(self.Q2.state_dict())
        self.target_Mu.load_state_dict(self.Mu.state_dict())

    def get_action(self, state):
        action = self.Mu(state)
        return action

    def compute_value_loss(self, args, s_batch, a_batch, r_batch, d_batch, next_s_batch):
        with torch.no_grad():
            # 让目标策略网络做预测。
            a = self.target_Mu(next_s_batch)
            noise = torch.clamp(
                torch.randn_like(a) * args.policy_noise,
                -args.noise_clip,
                args.noise_clip,
            )
            a = torch.clamp(a + noise, min=-self.max_action, max=self.max_action)

            # 让两个目标价值网络做预测。
            q1 = self.target_Q1(next_s_batch, a).squeeze()
            q2 = self.target_Q2(next_s_batch, a).squeeze()

            # 计算 TD 目标。
            y = r_batch + args.gamma * torch.min(q1, q2) * (1 - d_batch)

        # 让两个价值网络做预测。
        qvals1 = self.Q1(s_batch, a_batch).squeeze()
        qvals2 = self.Q2(s_batch, a_batch).squeeze()
        value_loss1 = F.mse_loss(y, qvals1)
        value_loss2 = F.mse_loss(y, qvals2)
        return value_loss1, value_loss2

    def compute_policy_loss(self, s_batch):
        a = self.Mu(s_batch)
        policy_loss = -self.Q1(s_batch, a).mean()
        return policy_loss

    def soft_update(self, tau=0.01):
        def soft_update_(target, source, tau_=0.01):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau_) + param.data * tau_)

        soft_update_(self.target_Q1, self.Q1, tau)
        soft_update_(self.target_Q2, self.Q2, tau)
        soft_update_(self.target_Mu, self.Mu, tau)


@dataclass
class ReplayBuffer:
    maxsize: int
    size: int = 0
    state: list = field(default_factory=list)
    action: list = field(default_factory=list)
    reward: list = field(default_factory=list)
    done: list = field(default_factory=list)
    next_state: list = field(default_factory=list)

    def push(self, state, action, reward, done, next_state):
        if self.size < self.maxsize:
            self.state.append(state)
            self.action.append(action)
            self.reward.append(reward)
            self.done.append(done)
            self.next_state.append(next_state)
        else:
            position = self.size % self.maxsize
            self.state[position] = state
            self.action[position] = action
            self.reward[position] = reward
            self.done[position] = done
            self.next_state[position] = next_state
        self.size += 1

    def sample(self, n):
        total_number = self.size if self.size < self.maxsize else self.maxsize
        indices = np.random.randint(total_number, size=n)
        state = [self.state[i] for i in indices]
        action = [self.action[i] for i in indices]
        reward = [self.reward[i] for i in indices]
        done = [self.done[i] for i in indices]
        next_state = [self.next_state[i] for i in indices]
        return state, action, reward, done, next_state


class INFO:
    def __init__(self):
        self.log = defaultdict(list)
        self.episode_length = 0
        self.episode_reward = 0
        self.max_episode_reward = -float("inf")

    def put(self, done, reward):
        if done is True:
            self.episode_length += 1
            self.episode_reward += reward
            self.log["episode_length"].append(self.episode_length)
            self.log["episode_reward"].append(self.episode_reward)

            if self.episode_reward > self.max_episode_reward:
                self.max_episode_reward = self.episode_reward

            self.episode_length = 0
            self.episode_reward = 0
        else:
            self.episode_length += 1
            self.episode_reward += reward


def train(args, env, agent: TD3):
    Q1_optimizer = torch.optim.Adam(agent.Q1.parameters(), lr=args.lr)
    Q2_optimizer = torch.optim.Adam(agent.Q2.parameters(), lr=args.lr)
    Mu_optimizer = torch.optim.Adam(agent.Mu.parameters(), lr=args.lr)
    replay_buffer = ReplayBuffer(maxsize=100_000)
    info = INFO()

    state, _ = env.reset(seed=args.seed)
    for step in range(args.max_steps):
        if step < args.warmup_steps:
            action = env.action_space.sample()
        else:
            action = agent.get_action(torch.from_numpy(state))
            action = action.cpu().data.numpy()
            action_noise = np.clip(np.random.randn(args.dim_action), -args.max_action, args.max_action)
            action = np.clip(action + action_noise, -args.max_action, args.max_action)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.push(state, action, reward, done, next_state)
        state = next_state
        info.put(done, reward)

        if done is True:
            # 打印信息。
            episode_reward = info.log["episode_reward"][-1]
            episode_length = info.log["episode_length"][-1]
            value_loss = info.log["value_loss1"][-1] if len(info.log["value_loss1"]) > 0 else 0
            print(f"step={step}, reward={episode_reward:.0f}, length={episode_length}, max_reward={info.max_episode_reward}, value_loss={value_loss:.2f}")

            # 如果得分更高，保存模型。
            if episode_reward == info.max_episode_reward:
                save_path = os.path.join(args.output_dir, "model.bin")
                torch.save(agent.Mu.state_dict(), save_path)

            state, _ = env.reset()

        if step > args.warmup_steps:
            s_batch, a_batch, r_batch, d_batch, ns_batch = replay_buffer.sample(n=args.batch_size)

            s_batch = np.array(s_batch)
            a_batch = np.array(a_batch)
            r_batch = np.array(r_batch)
            d_batch = np.array(d_batch)
            ns_batch = np.array(ns_batch)

            s_batch = torch.tensor(s_batch, dtype=torch.float32)
            a_batch = torch.tensor(a_batch, dtype=torch.float32)
            r_batch = torch.tensor(r_batch, dtype=torch.float32)
            d_batch = torch.tensor(d_batch, dtype=torch.float32)
            ns_batch = torch.tensor(ns_batch, dtype=torch.float32)

            value_loss1, value_loss2 = agent.compute_value_loss(args, s_batch, a_batch, r_batch, d_batch, ns_batch)

            Q1_optimizer.zero_grad()
            value_loss1.backward(retain_graph=True)
            Q1_optimizer.step()

            Q2_optimizer.zero_grad()
            value_loss2.backward()
            Q2_optimizer.step()

            info.log["value_loss1"].append(value_loss1.item())
            info.log["value_loss2"].append(value_loss2.item())

            if step % args.K == 0:
                policy_loss = agent.compute_policy_loss(s_batch)
                Mu_optimizer.zero_grad()
                policy_loss.backward()
                Mu_optimizer.step()
                agent.soft_update()

                info.log["policy_loss"].append(policy_loss.item())

            if step % 10000 == 0:
                # 画图。
                plt.plot(info.log["value_loss1"], label="loss1")
                plt.plot(info.log["value_loss2"], label="loss2")
                plt.legend()
                plt.savefig(f"{args.output_dir}/value_loss.png", bbox_inches="tight")
                plt.close()

                plt.plot(info.log["episode_reward"])
                plt.savefig(f"{args.output_dir}/episode_reward.png", bbox_inches="tight")
                plt.close()


def eval(args, env, agent):
    agent = TD3(args.dim_state, args.dim_action, args.max_action)
    model_path = os.path.join(args.output_dir, "model.bin")
    agent.Mu.load_state_dict(torch.load(model_path))

    episode_length = 0
    episode_reward = 0
    state, _ = env.reset()
    for i in range(5000):
        episode_length += 1
        action = agent.get_action(torch.from_numpy(state)).cpu().data.numpy()
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
        episode_reward += reward

        state = next_state
        if done is True:
            print(f"episode reward={episode_reward}, length={episode_length}")
            state, _ = env.reset()
            episode_length = 0
            episode_reward = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Pendulum-v1", type=str, help="Environment name.")
    parser.add_argument("--dim_state", default=3, type=int, help="Dimension of observation.")
    parser.add_argument("--dim_action", default=1, type=int, help="Number of actions.")
    parser.add_argument("--max_action", default=2.0, type=float, help="Action scale, [-max, max].")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discount coefficient.")

    parser.add_argument("--max_steps", default=100_000, type=int, help="Maximum steps for interaction.")
    parser.add_argument("--warmup_steps", default=10_000, type=int, help="Warmup steps without training.")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--K", default=2, type=int, help="Delay K steps to update policy and target network.")
    parser.add_argument("--policy_noise", default=0.2, type=float, help="Policy noise.")
    parser.add_argument("--noise_clip", default=0.5, type=float, help="Policy noise.")

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--output_dir", default="output", type=str, help="Output directory.")
    parser.add_argument("--do_train", action="store_true", help="Train policy.")
    parser.add_argument("--do_eval", action="store_true", help="Evaluate policy.")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # 初始化环境。
    env = gym.make(args.env)

    agent = TD3(dim_state=args.dim_state, dim_action=args.dim_action, max_action=args.max_action)

    if args.do_train:
        train(args, env, agent)

    if args.do_eval:
        eval(args, env, agent)


if __name__ == "__main__":
    main()
