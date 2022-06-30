import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN_MIN = -9.0
MEAN_MAX = 9.0
LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPS = 1e-7


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu_head = nn.Linear(256, action_dim)
        self.sigma_head = nn.Linear(256, action_dim)

        torch.nn.init.orthogonal_(self.fc1.weight.data, gain=math.sqrt(2.0))
        torch.nn.init.orthogonal_(self.fc2.weight.data, gain=math.sqrt(2.0))
        torch.nn.init.orthogonal_(self.mu_head.weight.data, gain=1e-2)
        torch.nn.init.orthogonal_(self.sigma_head.weight.data, gain=1e-2)

    def forward(self, state):
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        mu = self.mu_head(a)
        mu = torch.clip(mu, MEAN_MIN, MEAN_MAX)
        log_sigma = self.sigma_head(a)
        log_sigma = torch.clip(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = TransformedDistribution(
            Normal(mu, sigma), TanhTransform(cache_size=1)
        )
        action = a_distribution.rsample()
        logp_pi = a_distribution.log_prob(action).sum(axis=-1)

        mu = torch.tanh(mu)
        return action, logp_pi, mu

    def get_log_density(self, state, action):
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        mu = self.mu_head(a)
        mu = torch.clip(mu, MEAN_MIN, MEAN_MAX)
        log_sigma = self.sigma_head(a)
        log_sigma = torch.clip(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = TransformedDistribution(
            Normal(mu, sigma), TanhTransform(cache_size=1)
        )
        action_clip = torch.clip(action, -1. + EPS, 1. - EPS)
        logp_action = a_distribution.log_prob(action_clip)
        return logp_action


class WBC(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            alpha=0.0,
    ):

        self.policy = Actor(state_dim, action_dim).to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)

        self.alpha = alpha

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        _, _, action = self.policy(state)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer_e, replay_buffer_o, batch_size=256):
        self.total_it += 1

        # Sample from D_e and D_o
        state_e, action_e, _, _, _ = replay_buffer_e.sample(batch_size)
        state_o, action_o, _, _, _ = replay_buffer_o.sample(batch_size)
        log_pi_e = self.policy.get_log_density(state_e, action_e)
        log_pi_o = self.policy.get_log_density(state_o, action_o)

        # Compute policy loss
        bc_e_loss = -log_pi_e.sum(1)
        bc_o_loss = -log_pi_o.sum(1)
        p_loss = bc_e_loss.mean() + self.alpha * bc_o_loss.mean()

        # Optimize the policy
        self.policy_optimizer.zero_grad()
        p_loss.backward()
        self.policy_optimizer.step()

    def save(self, filename):
        torch.save(self.policy.state_dict(), filename + "_policy")
        torch.save(self.policy_optimizer.state_dict(), filename + "_policy_optimizer")

    def load(self, filename):
        self.policy.load_state_dict(torch.load(filename + "_policy"))
        self.policy_optimizer.load_state_dict(torch.load(filename + "_policy_optimizer"))
