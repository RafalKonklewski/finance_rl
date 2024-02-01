import torch
import torch.nn as nn
from classes.utils import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym
import math, os

class A3C(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(A3C, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim, 200)
        self.fc_mu = nn.Linear(200, action_dim)
        self.fc_sigma = nn.Linear(200, action_dim)
        self.fc2 = nn.Linear(state_dim, 100)
        self.fc_value = nn.Linear(100, 1)
        set_init([self.fc1, self.fc_mu, self.fc_sigma, self.fc2, self.fc_value])
        self.distribution = torch.distributions.Normal

    def forward(self, x):
        x1 = F.relu6(self.fc1(x))
        mu = 2 * F.tanh(self.fc_mu(x1))
        sigma = F.softplus(self.fc_sigma(x1)) + 0.001  # avoid 0
        x2 = F.relu6(self.fc2(x))
        values = self.fc_value(x2)
        return mu, sigma, values

    def choose_action(self, state):
        self.training = False
        mu, sigma, _ = self.forward(state)
        m = self.distribution(mu.view(1, ).data, sigma.view(1, ).data)
        return m.sample().numpy()

    def loss_func(self, state, action, value_target):
        self.train()
        mu, sigma, values = self.forward(state)
        td = value_target - values
        critic_loss = td.pow(2)

        m = self.distribution(mu, sigma)
        log_prob = m.log_prob(action)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale)  # exploration
        exp_v = log_prob * td.detach() + 0.005 * entropy
        actor_loss = -exp_v
        total_loss = (actor_loss + critic_loss).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(self, input_size, output_size,
                 global_net, env, max_episodes,
                 gamma, update_global_steps, optimizer,
                 global_episode, global_episode_reward,
                 result_queue, name):
        super(Worker, self).__init__()
        self.name = 'worker-%d' % name
        self.global_episode = global_episode
        self.global_episode_reward = global_episode_reward
        self.result_queue = result_queue
        self.global_net = global_net
        self.optimizer = optimizer
        self.max_episodes = max_episodes
        self.update_global_steps = update_global_steps
        self.local_net = A3C(input_size, output_size)  # local network
        self.env = env
        self.gamma = gamma

    def run(self):
        total_steps = 1
        while self.global_episode.value < self.max_episodes:
            state = self.env.reset()
            buffer_state, buffer_action, buffer_reward = [], [], []
            episode_reward = 0.0

            for t in range(self.max_episodes):
                if self.name == 'worker-0':
                    self.env.reset()

                action = self.local_net.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                if t == self.max_episodes - 1:
                    done = True

                episode_reward += reward
                buffer_state.append(state)
                buffer_action.append(action)
                buffer_reward.append(reward)  # normalize rewards

                if total_steps % self.update_global_steps == 0 or done:
                    # Update the global network and assign to the local network
                    push_and_pull(self.optimizer, self.local_net, self.global_net, done, next_state, buffer_state,
                                  buffer_action, buffer_reward, self.gamma)
                    buffer_state, buffer_action, buffer_reward = [], [], []

                    if done:
                        record(self.global_episode, self.global_episode_reward, episode_reward, self.result_queue,
                               self.name)
                        break

                state = next_state
                total_steps += 1

        self.result_queue.put(None)