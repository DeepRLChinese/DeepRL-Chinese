"""10.2节DDPG算法实现。"""
import argparse
import gym
from torch import nn
import torch
import numpy as np
import random
import torch.nn.functional as F
from collections import defaultdict, deque
from torch.optim import Adam
import matplotlib.pyplot as plt
import os


def fanin_init(size, fanin=None):
    """weight initializer known from https://arxiv.org/abs/1502.01852"""
    fanin = fanin or size[0]
    v = 1.0 / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, h1=400, h2=300, eps=0.03):
        """
        :param action_lim: Used to limit action space in [-action_lim,action_lim]
        :return:
        """
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, action_dim)

    def forward(self, state):
        """
        return actor policy function Pi(s)
        :param state: state [n, state_dim]
        :return: action [n, action_dim]
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = F.tanh(self.fc3(x))
        return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, h1=200, h2=300, eps=0.03):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, h1)
        self.fc2 = nn.Linear(action_dim, h1)
        self.fc3 = nn.Linear(h1 + h1, h2)
        self.fc4 = nn.Linear(h2, 1)

    def forward(self, state, action):
        """return critic Q(s,a)
        :param state: state [n, state_dim] (n is batch_size)
        :param action: action [n, action_dim]
        :return: Q(s,a) [n, 1]
        """
        x = F.relu(self.fc1(state))
        y = F.relu(self.fc2(action))
        h = torch.cat((x, y), dim=1)
        h = F.relu(self.fc3(h))
        h = self.fc4(h)
        return h


class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=123):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = []
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.pop(0)
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        s_batch = torch.tensor(s_batch, dtype=torch.float32)
        a_batch = torch.tensor(a_batch, dtype=torch.float32)
        r_batch = torch.tensor(r_batch, dtype=torch.float32)
        t_batch = torch.tensor(t_batch, dtype=torch.float32)
        s2_batch = torch.tensor(s2_batch, dtype=torch.float32)
        return s_batch, a_batch, r_batch, t_batch, s2_batch


class OUNoise:
    """docstring for OUNoise"""

    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class DDPG:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float,
    ):
        self.policy = Actor(state_dim, action_dim)
        self.value = Critic(state_dim, action_dim)
        self.target_policy = Actor(state_dim, action_dim)
        self.target_value = Critic(state_dim, action_dim)
        self.target_policy.load_state_dict(self.policy.state_dict())
        self.target_value.load_state_dict(self.value.state_dict())
        self.gamma = gamma

    def get_action(self, state):
        action = self.policy(state)
        return action * 2

    def compute_policy_loss(self, bs, ba, br, bd, bns):
        predicted_action = self.get_action(bs)
        loss = -self.value(bs, predicted_action).mean()
        return loss

    def compute_value_loss(self, bs, ba, br, bd, bns):
        with torch.no_grad():
            predicted_bna = self.target_policy(bns) * 2
            target_value = self.gamma * self.target_value(bns, predicted_bna).squeeze() * (1 - bd) + br

        value = self.value(bs, ba).squeeze()
        loss = F.mse_loss(value, target_value)
        return loss

    def soft_update(self, tau=0.01):
        """
        update target by target = tau * source + (1 - tau) * target.
        """
        for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(self.target_policy.parameters(), self.policy.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def train(args, env, agent: DDPG):
    policy_optimizer = Adam(agent.policy.parameters(), lr=args.lr_policy)
    value_optimizer = Adam(agent.value.parameters(), lr=args.lr_value)

    replay_buffer = ReplayBuffer(buffer_size=args.buffer_size)

    log = defaultdict(list)

    episode_reward = 0
    episode_length = 0
    max_episode_reward = -float("inf")
    value_loss_list = [0]
    policy_loss_list = [0]

    state, _ = env.reset()
    for i in range(args.max_steps):

        action = agent.get_action(torch.tensor(state))
        action = action.detach().numpy()
        action = (action + np.random.normal(0, 0.1, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        replay_buffer.add(state, action, reward, done, next_state)

        state = next_state

        episode_reward += reward
        episode_length += 1

        if done:
            if episode_reward > max_episode_reward:
                max_episode_reward = episode_reward
                torch.save({"policy": agent.policy.state_dict(), "value": agent.value.state_dict()}, os.path.join(args.output_dir, "model.bin"))

            print(f"i={i}, episode reward={episode_reward:.2f}, max episode reward={max_episode_reward:.2f}, value loss={np.mean(value_loss_list):.2f}, policy loss={np.mean(policy_loss_list):.2f}")

            log["episode_reward"].append(episode_reward)

            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            value_loss_list = [0]
            policy_loss_list = [0]

            if i > args.warm_steps:

                for _ in range(20):
                    bs, ba, br, bd, bns = replay_buffer.sample_batch(args.batch_size)

                    value_loss = agent.compute_value_loss(bs, ba, br, bd, bns)
                    value_optimizer.zero_grad()
                    value_loss.backward()
                    value_optimizer.step()

                    value_loss_list.append(value_loss.item())

                    policy_loss = agent.compute_policy_loss(bs, ba, br, bd, bns)
                    policy_optimizer.zero_grad()
                    policy_loss.backward()
                    policy_optimizer.step()

                    policy_loss_list.append(policy_loss.item())

                    agent.soft_update()

    # 画图。
    moving_avg = []
    d = deque(maxlen=10)
    for x in log["episode_reward"]:
        d.append(x)
        moving_avg.append(np.mean(d))

    plt.plot(moving_avg)
    plt.xlabel("episode")
    plt.ylabel("episode reward")
    plt.savefig(f"{args.output_dir}/episode_reward.png", bbox_inches="tight")


def eval(args, agent: DDPG):
    state_dict = torch.load(os.path.join(args.output_dir, "model.bin"))
    agent.policy.load_state_dict(state_dict["policy"])
    agent.value.load_state_dict(state_dict["value"])

    episode_reward = 0

    env = gym.make(args.env, render_mode="human")
    state, _ = env.reset()
    for i in range(1000):

        action = agent.get_action(torch.tensor(state))
        action = action.detach().numpy()
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        episode_reward += reward

        if done:
            print(f"episode reward={episode_reward}")
            state, _ = env.reset()
            episode_reward = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Pendulum-v1", type=str, help="gym environment")
    parser.add_argument("--max_steps", default=100_000, type=int)
    parser.add_argument("--warm_steps", default=1_000, type=int)

    parser.add_argument("--gamma", default=0.95, type=float, help="discount")
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--lr_policy", default=1e-4, type=float)
    parser.add_argument("--lr_value", default=1e-3, type=float)

    parser.add_argument("--tau", default=0.001, type=float, help="target network update coefficient")
    parser.add_argument("--buffer_size", default=100_000, type=int, help="replay buff size")
    parser.add_argument("--output_dir", default="output", type=str, help="result output dir")
    parser.add_argument("--model_path", type=str, help="if test mode, import the model")

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    env = gym.make(args.env)
    agent = DDPG(env.observation_space.shape[0], env.action_space.shape[0], args.gamma)

    if args.do_train:
        train(args, env, agent)

    if args.do_eval:
        eval(args, agent)
