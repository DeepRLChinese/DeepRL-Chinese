"""参考https://github.com/blackredscarf/pytorch-DDPG。去掉了target网络。"""
import argparse
import gym
from torch import nn
import torch
import numpy as np
import random
import pickle
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
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(h1, h2)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(h2, action_dim)
        self.fc3.weight.data.uniform_(-eps, eps)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state):
        """
        return actor policy function Pi(s)
        :param state: state [n, state_dim]
        :return: action [n, action_dim]
        """
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        action = self.tanh(self.fc3(x))  # tanh limit (-1, 1)
        return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, h1=400, h2=300, eps=0.03):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, h1)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(h1 + action_dim, h2)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(h2, 1)
        self.fc3.weight.data.uniform_(-eps, eps)

        self.relu = nn.ReLU()

    def forward(self, state, action):
        """return critic Q(s,a)
        :param state: state [n, state_dim] (n is batch_size)
        :param action: action [n, action_dim]
        :return: Q(s,a) [n, 1]
        """
        s1 = self.relu(self.fc1(state))
        x = torch.cat((s1, action), dim=1)

        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NormalizedEnv(gym.ActionWrapper):
    # https://github.com/openai/gym/blob/master/gym/core.py，将[-1,1]的action映射到动作空间。
    def action(self, action):
        act_k = (self.action_space.high - self.action_space.low) / 2.0
        act_b = (self.action_space.high + self.action_space.low) / 2.0
        return act_k * action + act_b

    # def reverse_action(self, action):
    #     act_k_inv = 2.0 / (self.action_space.high - self.action_space.low)
    #     act_b = (self.action_space.high + self.action_space.low) / 2.0
    #     return act_k_inv * (action - act_b)


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

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    # def clear(self):
    #     self.buffer = []
    #     self.count = 0

    # def save(self):
    #     file = open("replay_buffer.obj", "wb")
    #     pickle.dump(self.buffer, file)
    #     file.close()

    # def load(self):
    #     try:
    #         filehandler = open("replay_buffer.obj", "rb")
    #         self.buffer = pickle.load(filehandler)
    #         self.count = len(self.buffer)
    #     except:
    #         print("there was no file to load")


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


def soft_update(target, source, tau=0.001):
    """
    update target by target = tau * source + (1 - tau) * target
    :param target: Target network
    :param source: source network
    :param tau: 0 < tau << 1
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    """
    update target by target = source
    :param target: Target network
    :param source: source network
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class DDPG:
    def __init__(self, config):
        self.config = config
        self.init()

    def init(self):
        self.state_dim = self.config.state_dim
        self.action_dim = self.config.action_dim
        self.batch_size = self.config.batch_size
        self.gamma = self.config.gamma
        self.epsilon = self.config.epsilon
        self.is_training = True
        self.randomer = OUNoise(self.action_dim)
        self.buffer = ReplayBuffer(self.config.max_buff)

        self.actor = Actor(self.state_dim, self.action_dim)
        self.actor_target = Actor(self.state_dim, self.action_dim)
        # self.actor_target = self.actor
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.config.learning_rate_actor)

        self.critic = Critic(self.state_dim, self.action_dim)
        self.critic_target = Critic(self.state_dim, self.action_dim)
        # self.critic_target = self.critic
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.config.learning_rate)

        # 屏蔽hard_update。
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        if self.config.use_cuda:
            self.cuda()

    def learning(self):
        s1, a1, r1, t1, s2 = self.buffer.sample_batch(self.batch_size)
        # bool -> int
        t1 = (t1 == False) * 1
        s1 = torch.tensor(s1, dtype=torch.float)
        a1 = torch.tensor(a1, dtype=torch.float)
        r1 = torch.tensor(r1, dtype=torch.float)
        t1 = torch.tensor(t1, dtype=torch.float)
        s2 = torch.tensor(s2, dtype=torch.float)
        if self.config.use_cuda:
            s1 = s1.cuda()
            a1 = a1.cuda()
            r1 = r1.cuda()
            t1 = t1.cuda()
            s2 = s2.cuda()

        a2 = self.actor_target(s2).detach()
        target_q = self.critic_target(s2, a2).detach()
        y_expected = r1[:, None] + t1[:, None] * self.config.gamma * target_q
        y_predicted = self.critic.forward(s1, a1)

        # critic gradient
        critic_loss = nn.MSELoss()
        loss_critic = critic_loss(y_predicted, y_expected)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # actor gradient
        pred_a = self.actor.forward(s1)
        loss_actor = (-self.critic.forward(s1, pred_a)).mean()
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        # Notice that we only have gradient updates for actor and critic, not target
        # actor_optimizer.step() and critic_optimizer.step()

        # 屏蔽target网络更新。
        soft_update(self.actor_target, self.actor, self.config.tau)
        soft_update(self.critic_target, self.critic, self.config.tau)

        return loss_actor.item(), loss_critic.item()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def decay_epsilon(self):
        self.epsilon -= self.config.eps_decay

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)

        if self.config.use_cuda:
            state = state.cuda()

        action = self.actor(state).detach()
        action = action.squeeze(0).cpu().numpy()
        action += self.is_training * max(self.epsilon, self.config.epsilon_min) * self.randomer.noise()
        action = np.clip(action, -1.0, 1.0)

        self.action = action
        return action

    def reset(self):
        self.randomer.reset()

    def load_weights(self, output):
        if output is None:
            return
        self.actor.load_state_dict(torch.load("{}/actor.pkl".format(output)))
        self.critic.load_state_dict(torch.load("{}/critic.pkl".format(output)))

    def save_model(self, output):
        torch.save(self.actor.state_dict(), "{}/actor.pkl".format(output))
        torch.save(self.critic.state_dict(), "{}/critic.pkl".format(output))

    # def save_config(self, output, save_obj=False):

    #     with open(output + "/config.txt", "w") as f:
    #         attr_val = get_class_attr_val(self.config)
    #         for k, v in attr_val.items():
    #             f.write(str(k) + " = " + str(v) + "\n")

    #     if save_obj:
    #         file = open(output + "/config.obj", "wb")
    #         pickle.dump(self.config, file)
    #         file.close()

    # def save_checkpoint(self, ep, total_step, output):

    #     checkpath = output + "/checkpoint_model"
    #     os.makedirs(checkpath, exist_ok=True)

    #     torch.save({"episodes": ep, "total_step": total_step, "actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, "%s/checkpoint_ep_%d.tar" % (checkpath, ep))

    # def load_checkpoint(self, model_path):
    #     checkpoint = torch.load(model_path)
    #     episode = checkpoint["episodes"]
    #     total_step = checkpoint["total_step"]
    #     self.actor.load_state_dict(checkpoint["actor"])
    #     self.critic.load_state_dict(checkpoint["critic"])

    #     return episode, total_step


def train(args, env=None, agent=None):

    env = gym.make("MountainCarContinuous-v0")
    env = NormalizedEnv(env)

    args.action_dim = int(env.action_space.shape[0])
    args.action_lim = float(env.action_space.high[0])
    args.state_dim = int(env.observation_space.shape[0])

    agent = DDPG(args)

    total_step = 0
    all_rewards = []
    for ep in range(args.episodes):
        s0 = env.reset()
        agent.reset()

        done = False
        step = 0
        actor_loss, critic_loss, reward = 0, 0, 0

        agent.decay_epsilon()

        while not done:
            action = agent.get_action(s0)
            s1, r1, done, info = env.step(action)
            agent.buffer.add(s0, action, r1, done, s1)
            s0 = s1

            if agent.buffer.size() > args.batch_size:
                loss_a, loss_c = agent.learning()
                actor_loss += loss_a
                critic_loss += loss_c

            reward += r1
            step += 1
            total_step += 1

            if step + 1 > args.max_steps:
                break

        all_rewards.append(reward)
        print(f"total_step={total_step}, episodes={ep}, episode_step:{step}, episode_reward={reward}")

        # if args.checkpoint and ep % args.checkpoint_interval == 0:
        #     agent.save_checkpoint(ep, total_step, args.output)

    agent.save_model(args.output)


def test(args):
    env = gym.make("MountainCarContinuous-v0")
    env = NormalizedEnv(env)

    args.action_dim = int(env.action_space.shape[0])
    args.action_lim = float(env.action_space.high[0])
    args.state_dim = int(env.observation_space.shape[0])

    agent = DDPG(args)
    agent.load_weights("out")
    agent.is_training = False

    avg_reward = 0
    for ep in range(50):
        s0 = env.reset()
        episode_steps = 0
        episode_reward = 0

        done = False
        while not done:
            env.render()
            action = agent.get_action(s0)
            s0, reward, done, info = env.step(action)

            episode_reward += reward
            episode_steps += 1

            if episode_steps + 1 > 100:
                done = True

        print(f"episode_reward={episode_reward}")

        avg_reward += episode_reward
    avg_reward /= 50
    print(f"avg reward={avg_reward}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Learn MountainCarContinuous-v0 with DDPG.")
    parser.add_argument("--train", dest="train", action="store_true", help="train model")
    parser.add_argument("--test", dest="test", action="store_true", help="test model")
    parser.add_argument("--env", default="MountainCarContinuous-v0", type=str, help="gym environment")
    parser.add_argument("--gamma", default=0.99, type=float, help="discount")
    parser.add_argument("--episodes", default=200, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--learning_rate_actor", default=1e-4, type=float)
    parser.add_argument("--tau", default=0.001, type=float, help="target network update coefficient")
    parser.add_argument("--epsilon", default=1.0, type=float, help="noise epsilon")
    parser.add_argument("--eps_decay", default=0.001, type=float, help="epsilon decay")
    parser.add_argument("--epsilon_min", default=0.001, type=float, help="minimum epsilon")
    parser.add_argument("--max_buff", default=1000000, type=int, help="replay buff size")
    parser.add_argument("--output", default="out", type=str, help="result output dir")
    parser.add_argument("--use_cuda", dest="use_cuda", action="store_true", help="use cuda")
    parser.add_argument("--model_path", type=str, help="if test mode, import the model")
    parser.add_argument("--load_config", type=str, help="load the config from obj file")

    step_group = parser.add_argument_group("step")
    step_group.add_argument("--customize_step", dest="customize_step", action="store_true", help="customize max step per episode")
    step_group.add_argument("--max_steps", default=1000, type=int, help="max steps per episode")

    # record_group = parser.add_argument_group("record")
    # record_group.add_argument("--record", dest="record", action="store_true", help="record the video")
    # record_group.add_argument("--record_ep_interval", default=20, type=int, help="record episodes interval")

    # checkpoint_group = parser.add_argument_group("checkpoint")
    # checkpoint_group.add_argument("--checkpoint", dest="checkpoint", action="store_true", help="use model checkpoint")
    # checkpoint_group.add_argument("--checkpoint_interval", default=500, type=int, help="checkpoint interval")

    # retrain_group = parser.add_argument_group("retrain")
    # retrain_group.add_argument("--retrain", dest="retrain", action="store_true", help="retrain model")
    # retrain_group.add_argument("--retrain_model", type=str, help="retrain model path")

    args = parser.parse_args()

    # args.learning_rate = 1e-3
    # args.learning_rate_actor = 1e-4
    # args.epsilon_min = 0.001
    # args.epsilon = 1.0
    # args.tau = 0.001

    os.makedirs(args.output, exist_ok=True)

    if args.train:
        train(args)

    if args.test:
        test(args)
