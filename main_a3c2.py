import torch
import torch.nn as nn
from classes.utils import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
from classes.shared_adam import SharedAdam
import gym
import math, os

os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 3000
MAX_EP_STEP = 200

env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]


class Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Net, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim, 200)
        self.mu = nn.Linear(200, action_dim)
        self.sigma = nn.Linear(200, action_dim)
        self.fc2 = nn.Linear(state_dim, 100)
        self.value = nn.Linear(100, 1)
        set_init([self.fc1, self.mu, self.sigma, self.fc2, self.value])
        self.distribution = torch.distributions.Normal

    def forward(self, x):
        fc1 = F.relu6(self.fc1(x))
        mu = 2 * F.tanh(self.mu(fc1))
        sigma = F.softplus(self.sigma(fc1)) + 0.001  # avoid 0
        fc2 = F.relu6(self.fc2(x))
        values = self.value(fc2)
        return mu, sigma, values

    def choose_action(self, state):
        self.training = False
        mu, sigma, _ = self.forward(state)
        distribution = self.distribution(mu.view(1, ).data, sigma.view(1, ).data)
        return distribution.sample().numpy()

    def loss_func(self, state, action, value_target):
        self.train()
        mu, sigma, values = self.forward(state)
        td_error = value_target - values
        critic_loss = td_error.pow(2)

        distribution = self.distribution(mu, sigma)
        log_prob = distribution.log_prob(action)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(distribution.scale)  # exploration
        actor_loss = -log_prob * td_error.detach() + 0.005 * entropy
        total_loss = (actor_loss + critic_loss).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(self, global_net, optimizer, global_episode, global_episode_reward, result_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.global_episode, self.global_episode_reward, self.result_queue = global_episode, global_episode_reward, result_queue
        self.global_net, self.optimizer = global_net, optimizer
        self.local_net = Net(state_dim, action_dim)  # local network
        self.env = gym.make('Pendulum-v1').unwrapped

    def run(self):
        total_step = 1
        while self.global_episode.value < MAX_EP:
            state = self.env.reset()
            buffer_state, buffer_action, buffer_reward = [], [], []
            episode_reward = 0.0
            for t in range(MAX_EP_STEP):
                if self.name == 'w0':
                    self.env.render()
                action = self.local_net.choose_action(v_wrap(state[None, :]))
                state_next, reward, done, _ = self.env.step(action.clip(-2, 2))
                if t == MAX_EP_STEP - 1:
                    done = True
                episode_reward += reward
                buffer_action.append(action)
                buffer_state.append(state)
                buffer_reward.append((reward + 8.1) / 8.1)  # normalize

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    # Update the global network and assign to the local network
                    push_and_pull(self.optimizer, self.local_net, self.global_net, done, state_next, buffer_state, buffer_action, buffer_reward, GAMMA)
                    buffer_state, buffer_action, buffer_reward = [], [], []

                    if done:
                        # Episode finished, record information
                        record(self.global_episode, self.global_episode_reward, episode_reward, self.result_queue, self.name)
                        break
                state = state_next
                total_step += 1

        self.result_queue.put(None)


if __name__ == "__main__":
    global_net = Net(state_dim, action_dim)  # global network
    global_net.share_memory()  # share the global parameters in multiprocessing
    optimizer = SharedAdam(global_net.parameters(), lr=1e-4, betas=(0.95, 0.999))  # global optimizer
    global_episode, global_episode_reward, result_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    num_processes = 1

    # Parallel training
    workers = [Worker(global_net, optimizer, global_episode, global_episode_reward, result_queue, i) for i in range(num_processes)]
    [w.start() for w in workers]
    episode_rewards = []  # Record episode rewards to plot
    while True:
        reward = result_queue.get()
        if reward is not None:
            episode_rewards.append(reward)
        else:
            break
    [w.join() for w in workers]