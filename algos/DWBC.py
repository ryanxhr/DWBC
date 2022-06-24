import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN_MIN = -9.0
MEAN_MAX = 9.0
LOG_STD_MIN = -5
LOG_STD_MAX = 10
EPS = 1e-7


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu_head = nn.Linear(256, action_dim)
        self.sigma_head = nn.Linear(256, action_dim)

        torch.nn.init.orthogonal_(self.fc1.weight.data, gain=math.sqrt(2.0))
        torch.nn.init.orthogonal_(self.fc2.weight.data, gain=math.sqrt(2.0))
        torch.nn.init.orthogonal_(self.mu_head.weight.data, gain=math.sqrt(2.0))
        torch.nn.init.orthogonal_(self.sigma_head.weight.data, gain=math.sqrt(2.0))

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        mu = self.mu_head(a)
        mu = torch.clip(mu, MEAN_MIN, MEAN_MAX)
        log_sigma = self.sigma_head(a)
        log_sigma = torch.clip(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = Normal(mu, sigma)
        action = a_distribution.rsample()

        logp_pi = a_distribution.log_prob(action).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=1)
        logp_pi = torch.unsqueeze(logp_pi, dim=1)

        action = self.max_action * torch.tanh(action)
        mu = torch.tanh(mu) * self.max_action
        return action, logp_pi, mu

    def get_log_density(self, state, action):
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        mu = self.mu_head(a)
        mu = torch.clip(mu, MEAN_MIN, MEAN_MAX)
        log_sigma = self.sigma_head(a)
        log_sigma = torch.clip(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)
        a_distribution = Normal(mu, sigma)

        action_clip = torch.clip(action, -1. + EPS, 1. - EPS)
        raw_action = torch.atanh(action_clip)

        # logp_action = a_distribution.log_prob(raw_action).sum(axis=-1)
        # logp_action -= (2 * (np.log(2) - raw_action - F.softplus(-2 * raw_action))).sum(axis=1)
        # logp_action = torch.unsqueeze(logp_action, dim=1)
        logp_action = a_distribution.log_prob(raw_action)
        logp_action -= (2 * (np.log(2) - raw_action - F.softplus(-2 * raw_action)))
        return logp_action


class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()

        self.fc1_1 = nn.Linear(state_dim + action_dim, 128)
        self.fc1_2 = nn.Linear(action_dim, 128)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        torch.nn.init.orthogonal_(self.fc1_1.weight.data, gain=math.sqrt(2.0))
        torch.nn.init.orthogonal_(self.fc1_2.weight.data, gain=math.sqrt(2.0))
        torch.nn.init.orthogonal_(self.fc2.weight.data, gain=math.sqrt(2.0))
        torch.nn.init.orthogonal_(self.fc3.weight.data, gain=math.sqrt(2.0))

    def forward(self, state, action, log_pi):
        sa = torch.cat([state, action], 1)
        d1 = F.relu(self.fc1_1(sa))
        d2 = F.relu(self.fc1_2(log_pi))
        d = torch.cat([d1, d2], 1)
        d = self.fc2(d)
        d = self.fc3(d)
        return d


class DWBC(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            alpha=2.5,
            eta=0.5,
    ):

        self.policy = Actor(state_dim, action_dim, max_action).to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)

        self.discriminator = Discriminator(state_dim, action_dim).to(device)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=3e-4)

        self.max_action = max_action
        self.alpha = alpha
        self.eta = eta

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

        # Compute discriminator loss
        d_e = self.discriminator(state_e, action_e, log_pi_e.detach())
        d_o = self.discriminator(state_o, action_o, log_pi_o.detach())

        d_loss_e = -torch.log(d_e) + torch.log(1 - d_e)
        d_loss_o = -torch.log(1 - d_o)
        d_loss = self.eta * d_loss_e.mean() + d_loss_o.mean()

        # Optimize the discriminator
        self.discriminator_optimizer.zero_grad()
        d_loss.backward()
        self.discriminator_optimizer.step()

        # Compute policy loss
        d_e_clip = torch.clip(d_e, 0.1, 0.9).detach()
        d_o_clip = torch.clip(d_o, 0.1, 0.9).detach()
        bc_loss = -log_pi_e.sum(1)
        corr_loss_e = -log_pi_e.sum(1) * (self.eta / d_e_clip + self.eta / (1 - d_e_clip))
        corr_loss_o = -log_pi_o.sum(1) * (1 / (1 - d_o_clip))
        p_loss = self.alpha * bc_loss.mean() - corr_loss_e.mean() + corr_loss_o.mean()

        # Optimize the policy
        self.policy_optimizer.zero_grad()
        p_loss.backward()
        self.policy_optimizer.step()

    def save(self, filename):
        torch.save(self.discriminator.state_dict(), filename + "_discriminator")
        torch.save(self.discriminator_optimizer.state_dict(), filename + "_discriminator_optimizer")

        torch.save(self.policy.state_dict(), filename + "_policy")
        torch.save(self.policy_optimizer.state_dict(), filename + "_policy_optimizer")

    def load(self, filename):
        self.discriminator.load_state_dict(torch.load(filename + "_discriminator"))
        self.discriminator_optimizer.load_state_dict(torch.load(filename + "_discriminator_optimizer"))

        self.policy.load_state_dict(torch.load(filename + "_policy"))
        self.policy_optimizer.load_state_dict(torch.load(filename + "_policy_optimizer"))
