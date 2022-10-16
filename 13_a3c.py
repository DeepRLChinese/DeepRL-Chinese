"""13.3节A3C算法实现。"""
import argparse
import os
import gym
import numpy as np
import ray
import torch
import torch.nn.functional as F
from torch import nn
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


@ray.remote(num_cpus=2)
class A3C(nn.Module):
    def __init__(self, args, id):
        super().__init__()
        self.args = args
        self.id = id

        self.V = ValueNet(args.dim_state)
        self.V_target = ValueNet(args.dim_state)
        self.pi = PolicyNet(args.dim_state, args.num_action)
        self.V_target.load_state_dict(self.V.state_dict())
        self.env = gym.make(args.env)

        self.ep_reward = 0

    def get_action(self, state):
        probs = self.pi(state)
        m = Categorical(probs)
        action = m.sample()
        logp_action = m.log_prob(action)
        return action, logp_action

    def play_one_rollout(self):
        self.ep_reward = 0
        rollout = Rollout()
        state = self.env.reset()
        while True:
            action, logp_action = self.get_action(torch.tensor(state).float())
            next_state, reward, done, _ = self.env.step(action.item())

            rollout.put(
                state,
                action,
                logp_action,
                reward,
                done,
                next_state,
            )
            state = next_state
            self.ep_reward += reward

            if done is True:
                break
        return rollout

    def compute_gradient(self, pi_state_dict, V_state_dict):
        """计算网络梯度，送回给Master节点。"""
        # 更新策略网络，值网络，目标值网络参数。
        self.zero_grad()
        self.pi.load_state_dict(pi_state_dict)
        self.V.load_state_dict(V_state_dict)
        self.soft_update()

        # 与环境进行一个完整回合的游戏。
        rollout = self.play_one_rollout()

        # 计算网络参数梯度。
        bs, ba, blogp_a, br, bd, bns = rollout.torch()

        value_loss = self.compute_value_loss(bs, blogp_a, br, bd, bns)
        policy_loss = self.compute_policy_loss(bs, blogp_a, br, bd, bns)

        loss = value_loss + policy_loss
        loss.backward()

        grad_lst = []
        for param in self.parameters():
            grad_lst.append(param.grad)

        return (self.id, self.ep_reward, grad_lst)

    def compute_value_loss(self, bs, blogp_a, br, bd, bns):
        # 累积奖励。
        r_lst = []
        R = 0
        for i in reversed(range(len(br))):
            R = self.args.discount * R + br[i]
            r_lst.append(R)
        r_lst.reverse()
        batch_r = torch.tensor(r_lst)

        # 目标价值。
        with torch.no_grad():
            target_value = batch_r + self.args.discount * torch.logical_not(bd) * self.V_target(bns).squeeze()

        # 计算value loss。
        value_loss = F.mse_loss(self.V(bs).squeeze(), target_value)
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

        # 目标价值。
        with torch.no_grad():
            target_value = batch_r + self.args.discount * torch.logical_not(bd) * self.V_target(bns).squeeze()

        # 计算policy loss。
        with torch.no_grad():
            advantage = target_value - self.V(bs).squeeze()
        policy_loss = 0
        for i, logp_a in enumerate(blogp_a):
            policy_loss += -logp_a * advantage[i]
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

    def torch(self):
        bs = torch.as_tensor(self.state_lst).float()
        ba = torch.as_tensor(self.action_lst).float()
        blogp_a = self.logp_action_lst
        br = torch.as_tensor(self.reward_lst).float()
        bd = torch.as_tensor(self.done_lst)
        bns = torch.as_tensor(self.next_state_lst).float()
        return bs, ba, blogp_a, br, bd, bns


class Master(nn.Module):
    def __init__(self, args):
        super().__init__()
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


def train(args):
    master = Master(args)
    optimizer = torch.optim.Adam(master.parameters(), lr=1e-3)

    # 启动N个Workers。
    worker_dst = {i: A3C.remote(args, i) for i in range(args.num_workers)}

    # 每个Worker接受Master的网络权重，分别计算梯度。
    remaining = [worker_dst[i].compute_gradient.remote(master.pi.state_dict(), master.V.state_dict()) for i in range(args.num_workers)]

    max_ep_reward = {i: 0 for i in range(args.num_workers)}
    cnt = 0
    ready_id = []
    for _ in range(1000):
        # 当有Worker完成梯度计算时，传回给Master节点。
        ready, remaining = ray.wait(remaining)
        cnt += 1

        id, ep_reward, grad_lst = ray.get(ready[0])

        if max_ep_reward[id] < ep_reward:
            save_path = os.path.join(args.output_dir, "model.bin")
            torch.save(master.pi.state_dict(), save_path)

        max_ep_reward[id] = max(max_ep_reward[id], ep_reward)
        print("id=%d, ep_reward=%d, max ep_reward=%d" % (id, ep_reward, max_ep_reward[id]))
        ready_id.append(id)

        for master_param, grad in zip(master.parameters(), grad_lst):
            if master_param.grad is None:
                master_param.grad = grad
            else:
                master_param.grad += grad

        # 每次收集到两个完成的Worker，计算梯度均值，并更新Master模型权重。
        if cnt % args.m == 0 and cnt != 0:
            # print("hello")

            cnt = 0
            for param in master.parameters():
                if param.grad is not None:
                    param.grad /= 2
            optimizer.step()
            master.zero_grad()

            # 让完成梯度的Worker使用新的网络权重继续训练。
            for id in ready_id:
                remaining.append(worker_dst[id].compute_gradient.remote(master.pi.state_dict(), master.V.state_dict()))
                ready_id = []


def eval(args):
    env = gym.make(args.env)
    agent = Master(args)
    model_path = os.path.join(args.output_dir, "model.bin")
    agent.pi.load_state_dict(torch.load(model_path))

    episode_length = 0
    episode_reward = 0
    state = env.reset()
    for i in range(5000):
        episode_length += 1
        action, _ = agent.get_action(torch.from_numpy(state))
        next_state, reward, done, info = env.step(action.item())
        env.render()
        episode_reward += reward

        state = next_state
        if done is True:
            print(f"{episode_reward=}, {episode_length=}")
            state = env.reset()
            episode_length = 0
            episode_reward = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1", type=str, help="Environment name.")
    parser.add_argument("--dim_state", default=4, type=int, help="Dimension of state.")
    parser.add_argument("--num_action", default=2, type=int, help="Number of action.")
    parser.add_argument("--output_dir", default="output", type=str, help="Output directory.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")

    parser.add_argument("--num_workers", default=4, type=int, help="Number of workers.")
    parser.add_argument("--m", default=2, type=int, help="Mean gradients when every m workers get ready.")
    parser.add_argument("--max_steps", default=100_000, type=int, help="Maximum steps for interaction.")
    parser.add_argument("--discount", default=0.99, type=float, help="Discount coefficient.")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    parser.add_argument("--do_train", action="store_true", help="Train policy.")
    parser.add_argument("--do_eval", action="store_true", help="Evaluate policy.")
    args = parser.parse_args()

    if args.do_train:
        train(args)

    if args.do_eval:
        eval(args)
