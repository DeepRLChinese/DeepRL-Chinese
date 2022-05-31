import argparse
import os
import random
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):
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


class SARSA:
    def __init__(self, dim_obs, num_act, discount=0.99):
        self.discount = discount
        self.model = QNet(dim_obs, num_act)

    def get_action(self, obs):
        qvals = self.model(obs)
        return qvals.argmax()

    def compute_loss(self, s_batch, a_batch, r_batch, d_batch, ns_batch):
        # Compute current Q value based on current states and actions.
        qvals = self.model(s_batch)[a_batch]  # .gather(1, a_batch.unsqueeze(1)).squeeze()
        na_batch = self.get_action(ns_batch)
        # next state的value不参与导数计算，避免不收敛。
        next_qvals = self.model(ns_batch)[na_batch]  # .gather(1, na_batch.unsqueeze(1)).squeeze().detach()
        loss = F.mse_loss(r_batch + self.discount * next_qvals * (1 - d_batch), qvals)
        return loss


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda:
        torch.cuda.manual_seed(args.seed)


def train(args, env, agent):

    optimizer = torch.optim.Adam(agent.model.parameters(), lr=args.lr)

    epsilon = 1
    max_episode_reward = -float("inf")
    episode_reward = 0
    episode_length = 0
    log_reward = []
    log_length = []
    log_losses = []

    agent.model.zero_grad()
    state = env.reset()
    for step in range(args.max_steps):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = agent.get_action(torch.from_numpy(state))
            action = action.item()
        next_state, reward, done, info = env.step(action)

        episode_reward += reward
        episode_length += 1

        # 修改奖励，加速训练。
        if done is True and episode_length < 200:
            reward = 250 + episode_reward
        else:
            reward = 5 * abs(next_state[0] - state[0]) + 3 * abs(state[1])
        state = next_state

        if done is True:

            if episode_length < 150 and episode_reward > max_episode_reward:
                save_path = os.path.join(args.output_dir, "model.bin")
                torch.save(agent.model.state_dict(), save_path)
                max_episode_reward = episode_reward
            print(f"{step=}, reward={episode_reward}, length={episode_length}, max_reward={max_episode_reward}, epsilon={epsilon:.2f}")

            epsilon = epsilon = max(epsilon * args.epsilon_decay, 1e-3)
            log_reward.append(episode_reward)
            log_length.append(episode_length)
            episode_reward = 0
            episode_length = 0
            state = env.reset()

        s = torch.tensor(state, dtype=torch.float32)
        a = torch.tensor(action, dtype=torch.long)
        r = torch.tensor(reward, dtype=torch.float32)
        d = torch.tensor(done, dtype=torch.float32)
        ns = torch.tensor(next_state, dtype=torch.float32)
        loss = agent.compute_loss(s, a, r, d, ns)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        log_losses.append(loss.item())

    # 3. 画图。
    plt.plot(log_losses)
    plt.yscale("log")
    plt.savefig(f"{args.output_dir}/loss.png", bbox_inches="tight")
    plt.close()

    plt.plot(np.cumsum(log_length), log_reward)
    plt.savefig(f"{args.output_dir}/episode_reward.png", bbox_inches="tight")
    plt.close()


def eval(args, env, agent):
    agent = SARSA(args.dim_obs, args.num_act)
    model_path = os.path.join(args.output_dir, "model.bin")
    agent.model.load_state_dict(torch.load(model_path))

    episode_length = 0
    episode_reward = 0
    state = env.reset()
    for i in range(5000):
        episode_length += 1
        action = agent.get_action(torch.from_numpy(state)).item()
        next_state, reward, done, info = env.step(action)
        env.render()
        episode_reward += reward

        state = next_state
        if done is True:
            print(f"{episode_reward=}, {episode_length=}")
            state = env.reset()
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
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--output_dir", default="output", type=str, help="Output directory.")
    parser.add_argument("--epsilon_decay", default=0.99, type=float, help="Epsilon-greedy algorithm decay coefficient.")
    parser.add_argument("--do_train", action="store_true", help="Train policy.")
    parser.add_argument("--do_eval", action="store_true", help="Evaluate policy.")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    env = gym.make(args.env)
    env.seed(args.seed)
    set_seed(args)
    agent = SARSA(dim_obs=args.dim_obs, num_act=args.num_act, discount=args.discount)
    agent.model.to(args.device)

    if args.do_train:
        train(args, env, agent)

    if args.do_eval:
        eval(args, env, agent)


if __name__ == "__main__":
    main()
