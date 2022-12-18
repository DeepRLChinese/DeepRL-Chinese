import argparse
import os
import random
from dataclasses import dataclass, field
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):
    """QNet.
    Input: feature
    Output: num_act of values
    """

    def __init__(self, dim_obs, num_act):
        super().__init__()
        self.fc1 = nn.Linear(dim_obs, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_act)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DoubleDQN:
    def __init__(self, dim_obs=None, num_act=None, discount=0.9):
        self.discount = discount
        self.model = QNet(dim_obs, num_act)
        self.target_model = QNet(dim_obs, num_act)
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, obs):
        qvals = self.model(obs)
        return qvals.argmax()

    def compute_loss(self, s_batch, a_batch, r_batch, d_batch, next_s_batch):
        # Compute current Q value based on current states and actions.
        qvals = self.model(s_batch).gather(1, a_batch.unsqueeze(1)).squeeze()
        # next state的value不参与导数计算，避免不收敛。
        next_qvals, _ = self.target_model(next_s_batch).detach().max(dim=1)
        loss = F.mse_loss(r_batch + self.discount * next_qvals * (1 - d_batch), qvals)
        return loss


@dataclass
class ReplayBuffer:
    maxsize: int
    size: int = 0
    state: list = field(default_factory=list)
    action: list = field(default_factory=list)
    next_state: list = field(default_factory=list)
    reward: list = field(default_factory=list)
    done: list = field(default_factory=list)

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


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda:
        torch.cuda.manual_seed(args.seed)


def train(args, env, agent):
    replay_buffer = ReplayBuffer(100_000)
    optimizer = torch.optim.Adam(agent.model.parameters(), lr=args.lr)
    optimizer.zero_grad()

    epsilon = 1
    episode_reward = 0
    episode_length = 0
    max_episode_reward = -float("inf")
    log_ep_length = []
    log_ep_rewards = []
    log_losses = [0]

    agent.model.train()
    agent.target_model.train()
    agent.model.zero_grad()
    agent.target_model.zero_grad()
    state, _ = env.reset()
    for i in range(args.max_steps):
        if np.random.rand() < epsilon or i < args.warmup_steps:
            action = env.action_space.sample()
        else:
            action = agent.get_action(torch.from_numpy(state))
            action = action.item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward

        # 修改奖励，加速训练。
        episode_length += 1
        if done is True and episode_length < 200:
            reward = 250 + episode_reward
        else:
            reward = 5 * abs(next_state[0] - state[0]) + 3 * abs(state[1])
        replay_buffer.push(state, action, reward, done, next_state)
        state = next_state

        if done is True:
            log_ep_rewards.append(episode_reward)
            log_ep_length.append(episode_length)

            epsilon = max(epsilon * args.epsilon_decay, 1e-3)

            print(f"i={i}, reward={episode_reward:.0f}, length={episode_length}, max_reward={max_episode_reward}, loss={log_losses[-1]:.1e}, epsilon={epsilon:.3f}")

            if episode_length < 180 and episode_reward > max_episode_reward:
                save_path = os.path.join(args.output_dir, "model.bin")
                torch.save(agent.model.state_dict(), save_path)
                max_episode_reward = episode_reward

            episode_reward = 0
            episode_length = 0
            state, _ = env.reset()

        if i > args.warmup_steps:
            bs, ba, br, bd, bns = replay_buffer.sample(n=args.batch_size)
            bs = torch.tensor(bs, dtype=torch.float32)
            ba = torch.tensor(ba, dtype=torch.long)
            br = torch.tensor(br, dtype=torch.float32)
            bd = torch.tensor(bd, dtype=torch.float32)
            bns = torch.tensor(bns, dtype=torch.float32)

            loss = agent.compute_loss(bs, ba, br, bd, bns)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            log_losses.append(loss.item())

            # 更新目标网络。
            for target_param, param in zip(agent.target_model.parameters(), agent.model.parameters()):
                target_param.data.copy_(args.lr_target * param.data + (1 - args.lr_target) * target_param.data)

    plt.plot(log_losses)
    plt.yscale("log")
    plt.savefig(f"{args.output_dir}/loss.png", bbox_inches="tight")
    plt.close()

    plt.plot(np.cumsum(log_ep_length), log_ep_rewards)
    plt.savefig(f"{args.output_dir}/episode_reward.png", bbox_inches="tight")
    plt.close()


def eval(args, env, agent):
    model_path = os.path.join(args.output_dir, "model.bin")
    agent.model.load_state_dict(torch.load(model_path))

    episode_length = 0
    episode_reward = 0

    agent.model.eval()
    state, _ = env.reset()
    for i in range(5000):
        episode_length += 1
        action = agent.get_action(torch.from_numpy(state)).item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward

        state = next_state
        if done is True:
            print(f"episode reward={episode_reward}, episode length={episode_length}")
            state, _ = env.reset()
            episode_length = 0
            episode_reward = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="MountainCar-v0", type=str, help="Environment name.")
    parser.add_argument("--dim_obs", default=2, type=int, help="Dimension of observation.")
    parser.add_argument("--num_act", default=3, type=int, help="Number of actions.")
    parser.add_argument("--discount", default=0.95, type=float, help="Discount coefficient.")
    parser.add_argument("--max_steps", default=100_000, type=int, help="Maximum steps for interaction.")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate.")
    parser.add_argument("--lr_target", default=1e-3, type=float, help="Update target net.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--warmup_steps", default=10_000, type=int, help="Warmup steps without training.")
    parser.add_argument("--output_dir", default="output", type=str, help="Output directory.")
    parser.add_argument("--epsilon_decay", default=0.99, type=float, help="Epsilon-greedy algorithm decay coefficient.")
    parser.add_argument("--do_train", action="store_true", help="Train policy.")
    parser.add_argument("--do_eval", action="store_true", help="Evaluate policy.")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    env = gym.make(args.env)
    set_seed(args)
    agent = DoubleDQN(dim_obs=args.dim_obs, num_act=args.num_act, discount=args.discount)
    agent.model.to(args.device)

    if args.do_train:
        train(args, env, agent)

    if args.do_eval:
        eval(args, env, agent)


if __name__ == "__main__":
    main()
