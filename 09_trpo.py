"""参考https://github.com/ajlangley/trpo-pytorch。
"""
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
from torch.distributions.categorical import Categorical
from torch.autograd import grad


class CategoricalLayer(nn.Module):
    """
    Implements a layer that outputs a multinomial distribution
    Methods
    ------
    __call__(log_action_probs)
        Takes as input log probabilities and outputs a pytorch multinomail
        distribution
    """

    def __init__(self):
        super().__init__()

    def __call__(self, log_action_probs):
        return Categorical(logits=log_action_probs)


class PolicyNet(nn.Module):
    def __init__(self, dim_obs, num_act):
        super().__init__()
        self.fc1 = nn.Linear(dim_obs, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_act)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.categorical = CategoricalLayer()

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # logits
        x = self.log_softmax(x)
        x = self.categorical(x)
        return x


class ValueNet(nn.Module):
    """QNet.
    Input: feature
    Output: num_act of values
    """

    def __init__(self, dim_obs):
        super().__init__()
        self.fc1 = nn.Linear(dim_obs, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TRPO:
    def __init__(self, args):
        self.discount = args.discount
        self.policy_net = PolicyNet(args.dim_obs, args.num_act)
        self.value_net = ValueNet(args.dim_obs)
        self.value_optimizer = torch.optim.AdamW(self.value_net.parameters(), lr=args.lr_value_net)

        self.max_kl_div = 0.01
        self.cg_max_iters = 10
        self.line_search_accept_ratio = 0.1

    def get_action(self, obs):
        action_dist = self.policy_net(obs)
        act = action_dist.sample()
        return act

    def surrogate_loss(self, log_action_probs, imp_sample_probs, advantages):
        return torch.mean(torch.exp(log_action_probs - imp_sample_probs) * advantages)

    def get_max_step_len(self, search_dir, Hvp_fun, max_step, retain_graph=False):
        num = 2 * max_step
        denom = torch.matmul(search_dir, Hvp_fun(search_dir, retain_graph))
        max_step_len = torch.sqrt(num / denom)
        return max_step_len

    def update_policy_net(self, s_batch, a_batch, r_batch, d_batch, next_s_batch):
        cumsum_rewards = [0]  # 加上0，方便计算。
        for i in reversed(range(len(r_batch))):
            cumsum_current = cumsum_rewards[-1] * self.discount * (1 - d_batch[i]) + r_batch[i]
            cumsum_rewards.append(cumsum_current)
        cumsum_rewards.pop(0)
        cumsum_rewards = list(reversed(cumsum_rewards))
        cumsum_rewards = torch.tensor(cumsum_rewards, dtype=torch.float32)

        action_dists = self.policy_net(s_batch)
        log_action_probs = action_dists.log_prob(a_batch)

        loss = self.surrogate_loss(log_action_probs, log_action_probs.detach(), cumsum_rewards)
        loss_grad = flat_grad(loss, self.policy_net.parameters(), retain_graph=True)

        mean_kl = mean_kl_first_fixed(action_dists, action_dists)

        Fvp_fun = get_Hvp_fun(mean_kl, self.policy_net.parameters())
        search_dir = cg_solver(Fvp_fun, loss_grad, self.cg_max_iters)

        expected_improvement = torch.matmul(loss_grad, search_dir)

        def constraints_satisfied(step, beta):
            apply_update(self.policy_net, step)

            with torch.no_grad():
                new_action_dists = self.policy_net(s_batch)
                new_log_action_probs = new_action_dists.log_prob(a_batch)

                new_loss = self.surrogate_loss(new_log_action_probs, log_action_probs, cumsum_rewards)

                mean_kl = mean_kl_first_fixed(action_dists, new_action_dists)

            actual_improvement = new_loss - loss
            improvement_ratio = actual_improvement / (expected_improvement * beta)

            apply_update(self.policy_net, -step)

            surrogate_cond = improvement_ratio >= self.line_search_accept_ratio and actual_improvement > 0.0
            kl_cond = mean_kl <= self.max_kl_div

            # print(f"kl contidion = {kl_cond}, mean_kl = {mean_kl}")

            return surrogate_cond and kl_cond

        max_step_len = self.get_max_step_len(search_dir, Fvp_fun, self.max_kl_div, retain_graph=True)
        step_len = line_search(search_dir, max_step_len, constraints_satisfied)

        opt_step = step_len * search_dir
        apply_update(self.policy_net, opt_step)

    def update_value_net(self, args, states, r_batch, d_batch):
        cumsum_rewards = [0]  # 加上0，方便计算。
        for i in reversed(range(len(r_batch))):
            cumsum_current = cumsum_rewards[-1] * self.discount * (1 - d_batch[i]) + r_batch[i]
            cumsum_rewards.append(cumsum_current)
        cumsum_rewards.pop(0)
        cumsum_rewards = list(reversed(cumsum_rewards))
        cumsum_rewards = torch.tensor(cumsum_rewards, dtype=torch.float32)

        for i in range(args.num_update_value):

            def mse():
                self.value_optimizer.zero_grad()
                state_values = self.value_net(states).view(-1)
                loss = F.mse_loss(state_values, cumsum_rewards)
                loss.backward(retain_graph=True)
                return loss

            self.value_optimizer.step(mse)


def flat_grad(functional_output, inputs, retain_graph=False, create_graph=False):
    """
    Return a flattened view of the gradients of functional_output w.r.t. inputs
    Parameters
    ----------
    functional_output : torch.FloatTensor
        The output of the function for which the gradient is to be calculated
    inputs : torch.FloatTensor (with requires_grad=True)
        the variables w.r.t. which the gradient will be computed
    retain_graph : bool
        whether to keep the computational graph in memory after computing the
        gradient (not required if create_graph is True)
    create_graph : bool
        whether to create a computational graph of the gradient computation
        itself
    Return
    ------
    flat_grads : torch.FloatTensor
        a flattened view of the gradients of functional_output w.r.t. inputs
    """

    if create_graph == True:
        retain_graph = True

    grads = grad(functional_output, inputs, retain_graph=retain_graph, create_graph=create_graph)
    flat_grads = torch.cat([v.view(-1) for v in grads])
    return flat_grads


def detach_dist(dist):
    detached_dist = Categorical(logits=dist.logits.detach())
    return detached_dist


def mean_kl_first_fixed(dist_1, dist_2):
    """
    Calculate the kl-divergence between dist_1 and dist_2 after detaching dist_1
    from the computational graph
    Parameters
    ----------
    dist_1 : torch.distributions.distribution.Distribution
        the first argument to the kl-divergence function (will be fixed)
    dist_2 : torch.distributions.distribution.Distribution
        the second argument to the kl-divergence function (will not be fixed)
    Returns
    -------
    mean_kl : torch.float
        the kl-divergence between dist_1 and dist_2
    """
    dist_1_detached = detach_dist(dist_1)
    mean_kl = torch.mean(torch.distributions.kl.kl_divergence(dist_1_detached, dist_2))
    return mean_kl


def get_Hvp_fun(functional_output, inputs, damping_coef=0.0):
    """
    Returns a function that calculates a Hessian-vector product with the Hessian
    of functional_output w.r.t. inputs
    Parameters
    ----------
    functional_output : torch.FloatTensor (with requires_grad=True)
        the output of the function of which the Hessian is calculated
    inputs : torch.FloatTensor
        the inputs w.r.t. which the Hessian is calculated
    damping_coef : float
        the multiple of the identity matrix to be added to the Hessian
    """

    inputs = list(inputs)
    grad_f = flat_grad(functional_output, inputs, create_graph=True)

    def Hvp_fun(v, retain_graph=True):
        gvp = torch.matmul(grad_f, v)
        Hvp = flat_grad(gvp, inputs, retain_graph=retain_graph)
        Hvp += damping_coef * v
        return Hvp

    return Hvp_fun


def cg_solver(Avp_fun, b, max_iter=10):
    """
    Finds an approximate solution to a set of linear equations Ax = b
    Parameters
    ----------
    Avp_fun : callable
        a function that right multiplies a matrix A by a vector
    b : torch.FloatTensor
        the right hand term in the set of linear equations Ax = b
    max_iter : int
        the maximum number of iterations (default is 10)
    Returns
    -------
    x : torch.FloatTensor
        the approximate solution to the system of equations defined by Avp_fun
        and b
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.zeros_like(b).to(device)
    r = b.clone()
    p = b.clone()

    for i in range(max_iter):
        Avp = Avp_fun(p, retain_graph=True)

        alpha = torch.matmul(r, r) / torch.matmul(p, Avp)
        x += alpha * p

        if i == max_iter - 1:
            return x

        r_new = r - alpha * Avp
        beta = torch.matmul(r_new, r_new) / torch.matmul(r, r)
        r = r_new
        p = r + beta * p


def apply_update(parameterized_fun, update):
    """
    Add update to the weights of parameterized_fun
    Parameters
    ----------
    parameterized_fun : torch.nn.Sequential
        the function approximator to be updated
    update : torch.FloatTensor
        a flattened version of the update to be applied
    """

    n = 0
    for param in parameterized_fun.parameters():
        numel = param.numel()
        param_update = update[n : n + numel].view(param.size())
        param.data += param_update
        n += numel


def line_search(search_dir, max_step_len, constraints_satisfied, line_search_coef=0.9, max_iter=10):
    """
    Perform a backtracking line search that terminates when constraints_satisfied
    return True and return the calculated step length. Return 0.0 if no step
    length can be found for which constraints_satisfied returns True
    Parameters
    ----------
    search_dir : torch.FloatTensor
        the search direction along which the line search is done
    max_step_len : torch.FloatTensor
        the maximum step length to consider in the line search
    constraints_satisfied : callable
        a function that returns a boolean indicating whether the constraints
        are met by the current step length
    line_search_coef : float
        the proportion by which to reduce the step length after each iteration
    max_iter : int
        the maximum number of backtracks to do before return 0.0
    Returns
    -------
    the maximum step length coefficient for which constraints_satisfied evaluates
    to True
    """

    step_len = max_step_len / line_search_coef

    for i in range(max_iter):
        step_len *= line_search_coef

        if constraints_satisfied(step_len * search_dir, step_len):
            return step_len

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.tensor(0.0).to(device)


@dataclass
class Trajectory:
    state: list = field(default_factory=list)
    action: list = field(default_factory=list)
    next_state: list = field(default_factory=list)
    reward: list = field(default_factory=list)
    done: list = field(default_factory=list)

    def push(self, state, action, reward, done, next_state):
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.done.append(done)
        self.next_state.append(next_state)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda:
        torch.cuda.manual_seed(args.seed)


def train(args, env, agent):
    trajectory = Trajectory()

    max_episode_reward = -float("inf")
    episode_reward = 0
    episode_length = 0
    log_ep_rewards = []
    log_ep_length = []

    agent.policy_net.train()
    agent.policy_net.zero_grad()
    agent.value_net.train()
    agent.value_net.zero_grad()
    state = env.reset()
    for i in range(args.max_steps):
        action = agent.get_action(torch.from_numpy(state)).item()
        next_state, reward, done, info = env.step(action)
        episode_reward += reward
        episode_length += 1

        trajectory.push(state, action, reward, done, next_state)
        state = next_state

        if done is True:
            print(f"{i=}, reward={episode_reward:.0f}, length={episode_length}, max_reward={max_episode_reward}")
            log_ep_rewards.append(episode_reward)
            log_ep_length.append(episode_length)

            if episode_length < 150 and episode_reward > max_episode_reward:
                save_path = os.path.join(args.output_dir, "model.bin")
                torch.save(agent.policy_net.state_dict(), save_path)
                max_episode_reward = episode_reward

            episode_reward = 0
            episode_length = 0
            state = env.reset()

            # Update policy and value nets.
            s_batch = torch.tensor(trajectory.state, dtype=torch.float32)
            a_batch = torch.tensor(trajectory.action, dtype=torch.int64)
            r_batch = torch.tensor(trajectory.reward, dtype=torch.float32)
            d_batch = torch.tensor(trajectory.done, dtype=torch.float32)
            ns_batch = torch.tensor(trajectory.next_state, dtype=torch.float32)

            agent.update_policy_net(s_batch, a_batch, r_batch, d_batch, ns_batch)
            agent.update_value_net(args, s_batch, r_batch, d_batch)

            trajectory = Trajectory()

    # 3. 画图。
    plt.plot(np.cumsum(log_ep_length), log_ep_rewards, label="length")
    plt.savefig(f"{args.output_dir}/episode_reward.png", bbox_inches="tight")
    plt.close()


def eval(args, env, agent):
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
    parser.add_argument("--env", default="CartPole-v1", type=str, help="Environment name.")
    parser.add_argument("--dim_obs", default=4, type=int, help="Dimension of observation.")
    parser.add_argument("--num_act", default=2, type=int, help="Number of actions.")
    parser.add_argument("--discount", default=0.95, type=float, help="Discount coefficient.")
    parser.add_argument("--max_steps", default=100_000, type=int, help="Maximum steps for interaction.")
    parser.add_argument("--lr_value_net", default=1e-3, type=float, help="Learning rate of value net.")
    parser.add_argument("--num_update_value", default=10, type=int, help="Number of updating value net per episode.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--output_dir", default="output", type=str, help="Output directory.")
    parser.add_argument("--do_train", action="store_true", help="Train policy.")
    parser.add_argument("--do_eval", action="store_true", help="Evaluate policy.")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    env = gym.make(args.env)
    env.seed(args.seed)
    set_seed(args)
    agent = TRPO(args)
    agent.policy_net.to(args.device)
    agent.value_net.to(args.device)

    if args.do_train:
        train(args, env, agent)

    if args.do_eval:
        eval(args, env, agent)


if __name__ == "__main__":
    main()
