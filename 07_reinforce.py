"""8.3节带基线的REINFORCE算法实现。"""
import argparse
import os
from collections import defaultdict
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


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


class REINFORCE_with_Baseline:
    def __init__(self, args):
        self.args = args
        self.V = ValueNet(args.dim_state)
        self.V_target = ValueNet(args.dim_state)
        self.pi = PolicyNet(args.dim_state, args.num_action)
        self.V_target.load_state_dict(self.V.state_dict())

    def get_action(self, state):
        probs = self.pi(state)
        m = Categorical(probs)
        action = m.sample()
        logp_action = m.log_prob(action)
        return action, logp_action

    def compute_value_loss(self, bs, blogp_a, br, bd, bns):
        # 累积奖励。
        r_lst = []
        R = 0
        for i in reversed(range(len(br))):
            R = self.args.discount * R + br[i]
            r_lst.append(R)
        r_lst.reverse()
        batch_r = torch.tensor(r_lst)

        # 计算value loss。
        value_loss = F.mse_loss(self.V(bs).squeeze(), batch_r)
        return value_loss

    def compute_policy_loss(self, bs, blogp_a, br, bd, bns):
        # 累积奖励。
        r_lst = []
        R = 0
        for i in reversed(range(len(br))):
            R = self.args.discount * R + br[i]
            r_lst.append(R)
        r_lst.reverse()
        batch_r = torch.tensor(r_lst)

        policy_loss = 0
        for i, logp_a in enumerate(blogp_a):
            policy_loss += -logp_a * batch_r[i]
        policy_loss = policy_loss.mean()
        return policy_loss

    def soft_update(self, tau=0.01):
        def soft_update_(target, source, tau_=0.01):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau_) + param.data * tau_)

        soft_update_(self.V_target, self.V, tau)


class Rollout:
    def __init__(self):
        self.state_lst = []
        self.action_lst = []
        self.logp_action_lst = []
        self.reward_lst = []
        self.done_lst = []
        self.next_state_lst = []

    def put(self, state, action, logp_action, reward, done, next_state):
        self.state_lst.append(state)
        self.action_lst.append(action)
        self.logp_action_lst.append(logp_action)
        self.reward_lst.append(reward)
        self.done_lst.append(done)
        self.next_state_lst.append(next_state)

    def tensor(self):
        bs = torch.as_tensor(self.state_lst).float()
        ba = torch.as_tensor(self.action_lst).float()
        blogp_a = self.logp_action_lst
        br = self.reward_lst
        bd = torch.as_tensor(self.done_lst)
        bns = torch.as_tensor(self.next_state_lst).float()
        return bs, ba, blogp_a, br, bd, bns


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


def train(args, env, agent: REINFORCE_with_Baseline):
    V_optimizer = torch.optim.Adam(agent.V.parameters(), lr=args.lr)
    pi_optimizer = torch.optim.Adam(agent.pi.parameters(), lr=args.lr)
    info = INFO()

    rollout = Rollout()
    state, _ = env.reset()
    for step in range(args.max_steps):
        action, logp_action = agent.get_action(torch.tensor(state).float())
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        info.put(done, reward)

        rollout.put(
            state,
            action,
            logp_action,
            reward,
            done,
            next_state,
        )
        state = next_state

        if done is True:
            # 模型训练。
            bs, ba, blogp_a, br, bd, bns = rollout.tensor()

            value_loss = agent.compute_value_loss(bs, blogp_a, br, bd, bns)
            V_optimizer.zero_grad()
            value_loss.backward(retain_graph=True)
            V_optimizer.step()

            policy_loss = agent.compute_policy_loss(bs, blogp_a, br, bd, bns)
            pi_optimizer.zero_grad()
            policy_loss.backward()
            pi_optimizer.step()

            agent.soft_update()

            # 打印信息。
            info.log["value_loss"].append(value_loss.item())
            info.log["policy_loss"].append(policy_loss.item())

            episode_reward = info.log["episode_reward"][-1]
            episode_length = info.log["episode_length"][-1]
            value_loss = info.log["value_loss"][-1]
            print(f"step={step}, reward={episode_reward:.0f}, length={episode_length}, max_reward={info.max_episode_reward}, value_loss={value_loss:.1e}")

            # 重置环境。
            state, _ = env.reset()
            rollout = Rollout()

            # 保存模型。
            if episode_reward == info.max_episode_reward:
                save_path = os.path.join(args.output_dir, "model.bin")
                torch.save(agent.pi.state_dict(), save_path)

        if step % 10000 == 0:
            plt.plot(info.log["value_loss"], label="value loss")
            plt.legend()
            plt.savefig(f"{args.output_dir}/value_loss.png", bbox_inches="tight")
            plt.close()

            plt.plot(info.log["episode_reward"])
            plt.savefig(f"{args.output_dir}/episode_reward.png", bbox_inches="tight")
            plt.close()


def eval(args, env, agent):
    agent = REINFORCE_with_Baseline(args)
    model_path = os.path.join(args.output_dir, "model.bin")
    agent.pi.load_state_dict(torch.load(model_path))

    episode_length = 0
    episode_reward = 0
    state, _ = env.reset()
    for i in range(5000):
        episode_length += 1
        action, _ = agent.get_action(torch.from_numpy(state))
        next_state, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        episode_reward += reward

        state = next_state
        if done is True:
            print(f"episode reward={episode_reward}, episode length={episode_length}")
            state, _ = env.reset()
            episode_length = 0
            episode_reward = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1", type=str, help="Environment name.")
    parser.add_argument("--dim_state", default=4, type=int, help="Dimension of state.")
    parser.add_argument("--num_action", default=2, type=int, help="Number of action.")
    parser.add_argument("--output_dir", default="output", type=str, help="Output directory.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")

    parser.add_argument("--max_steps", default=100_000, type=int, help="Maximum steps for interaction.")
    parser.add_argument("--discount", default=0.99, type=float, help="Discount coefficient.")
    parser.add_argument("--lr", default=3e-3, type=float, help="Learning rate.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    parser.add_argument("--do_train", action="store_true", help="Train policy.")
    parser.add_argument("--do_eval", action="store_true", help="Evaluate policy.")
    args = parser.parse_args()

    env = gym.make(args.env)
    agent = REINFORCE_with_Baseline(args)

    if args.do_train:
        train(args, env, agent)

    if args.do_eval:
        eval(args, env, agent)
