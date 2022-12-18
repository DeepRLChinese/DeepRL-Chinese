"""5.3节多步SARSA算法实现。
"""
import argparse
import os
import random
from collections import defaultdict
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):
    def __init__(self, dim_state, num_action):
        super().__init__()
        self.fc1 = nn.Linear(dim_state, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_action)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SARSA:
    def __init__(self, dim_state, num_action, gamma=0.99):
        self.gamma = gamma
        self.Q = QNet(dim_state, num_action)
        self.target_Q = QNet(dim_state, num_action)
        self.target_Q.load_state_dict(self.Q.state_dict())

    def get_action(self, state):
        qvals = self.Q(state)
        return qvals.argmax(dim=-1)

    def compute_loss(self, args, s_list, a_list, r_list):
        batch_s = np.array(s_list)
        batch_s = torch.tensor(batch_s, dtype=torch.float32)
        batch_a = np.array(a_list)
        batch_a = torch.tensor(batch_a, dtype=torch.long)

        # 2. 对于所有的t=1，...，n-m，计算$\hat{q}_t = q(s_t, a_t; w_{now}$。
        num = len(r_list)
        state = batch_s[: num - args.m + 1, :]
        action = batch_a[: num - args.m + 1]
        qvals = self.Q(state).gather(1, action.unsqueeze(1)).squeeze()

        # s0, a0, r0, s1, a1, r1, s2, a2, r2, s3, a3, r3, ..., s6, a6, r6

        # s0, s1, s2

        # s5, s6,

        # 3. 对于所有的 t = 1, ..., n − m，计算多步 TD 目标和 TD 误差。
        R = 0
        for i in reversed(range(num - args.m, num)):
            R = args.gamma * R + r_list[i]

        rewards = [R]
        for i in reversed(range(num - args.m)):
            R -= args.gamma ** (args.m - 1) * r_list[i + args.m]
            R = args.gamma * R + r_list[i]
            rewards.append(R)
        rewards.reverse()
        rewards = torch.tensor(rewards, dtype=torch.float32)

        with torch.no_grad():
            state = batch_s[args.m :, :]
            action = batch_a[args.m :]
            m_step_qvals = self.target_Q(state).gather(1, action.unsqueeze(1)).squeeze()

            target_values = args.gamma**args.m * m_step_qvals
            target_values = torch.cat([target_values, torch.tensor([0.0])])
            target_values = target_values + rewards

        td_delta = qvals - target_values

        # 计算loss。
        value_loss = td_delta.square().mean()
        return value_loss

    def soft_update(self, tau=0.01):
        for target_param, param in zip(self.target_Q.parameters(), self.Q.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda:
        torch.cuda.manual_seed(args.seed)


def train(args, env, agent):
    optimizer = torch.optim.Adam(agent.Q.parameters(), lr=args.lr)

    epsilon = 1
    epsilon_max = 1
    epsilon_min = 0.1
    max_episode_reward = -float("inf")
    episode_reward = 0
    episode_length = 0
    log = defaultdict(list)
    s_list = []
    a_list = []
    r_list = []

    state, _ = env.reset(seed=args.seed)
    for step in range(args.max_steps):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = agent.get_action(torch.from_numpy(state))
            action = action.item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        s_list.append(state)
        a_list.append(action)
        r_list.append(reward)

        episode_reward += reward
        episode_length += 1

        state = next_state

        # 一个episode结束后，进行训练。
        if done is True:
            log["episode_reward"].append(episode_reward)
            log["episode_length"].append(episode_length)

            if episode_reward > max_episode_reward:
                save_path = os.path.join(args.output_dir, "model.bin")
                torch.save(agent.Q.state_dict(), save_path)
                max_episode_reward = episode_reward

            # 训练模型。
            for _ in range(2):
                loss = agent.compute_loss(args, s_list, a_list, r_list)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            agent.soft_update()

            print(f"step={step}, reward={episode_reward}, length={episode_length}, max_reward={max_episode_reward}, epsilon={epsilon:.2f}, loss={loss.item():.5f}")

            log["loss"].append(loss.item())

            epsilon = max(epsilon - (epsilon_max - epsilon_min) * args.epsilon_decay, 1e-1)
            episode_reward = 0
            episode_length = 0
            state, _ = env.reset()

    # 3. 画图。
    plt.plot(log["loss"])
    plt.yscale("log")
    plt.savefig(f"{args.output_dir}/loss.png", bbox_inches="tight")
    plt.close()

    plt.plot(np.cumsum(log["episode_length"]), log["episode_reward"])
    plt.savefig(f"{args.output_dir}/episode_reward.png", bbox_inches="tight")
    plt.close()


def eval(args, env, agent):
    agent = SARSA(args.dim_state, args.num_action)
    model_path = os.path.join(args.output_dir, "model.bin")
    agent.Q.load_state_dict(torch.load(model_path))

    episode_length = 0
    episode_reward = 0
    state = env.reset()
    for i in range(5000):
        episode_length += 1
        action = agent.get_action(torch.from_numpy(state)).item()
        next_state, reward, done, _ = env.step(action)
        env.render()
        episode_reward += reward

        state = next_state
        if done is True:
            print(f"episode reward={episode_reward}, episode length={episode_length}")
            state = env.reset()
            episode_length = 0
            episode_reward = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1", type=str, help="Environment name.")
    parser.add_argument("--dim_state", default=4, type=int, help="Dimension of observation.")
    parser.add_argument("--num_action", default=2, type=int, help="Number of actions.")
    parser.add_argument("--m", default=5, type=int, help="Multi-step TD target.")

    parser.add_argument("--gamma", default=0.99, type=float, help="Discount coefficient.")
    parser.add_argument("--max_steps", default=100_000, type=int, help="Maximum steps for interaction.")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate.")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")

    parser.add_argument("--epsilon_decay", default=1 / 2000, type=float, help="Epsilon-greedy algorithm decay coefficient.")
    parser.add_argument("--output_dir", default="output", type=str, help="Output directory.")
    parser.add_argument("--do_train", action="store_true", help="Train policy.")
    parser.add_argument("--do_eval", action="store_true", help="Evaluate policy.")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    env = gym.make(args.env)
    set_seed(args)
    agent = SARSA(dim_state=args.dim_state, num_action=args.num_action, gamma=args.gamma)
    agent.Q.to(args.device)

    if args.do_train:
        train(args, env, agent)

    if args.do_eval:
        eval(args, env, agent)


if __name__ == "__main__":
    main()
