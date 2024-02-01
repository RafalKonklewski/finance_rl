import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch import distributions
from tqdm import tqdm
import torch.multiprocessing as mp
from queue import Empty


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()

        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.view(1, -1)[:, :7]
        x = self.shared_layers(x)
        value = self.critic(x)
        mean = self.actor(x)
        dist = distributions.Normal(mean, torch.ones_like(mean))
        return dist, value

    
class Agent:
    def __init__(self, num_inputs, num_outputs, gamma=0.6, lr=3e-4):
        self.actor_critic = ActorCritic(num_inputs, num_outputs)
        self.optimizer = SharedAdam(self.actor_critic.parameters(), lr=lr)
        self.gamma = gamma

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        dist, value = self.actor_critic(state)

        action = dist.sample().squeeze().detach()
        action = np.array([action], dtype='float32')
        return action

    def get_value(self, state):
        state = torch.FloatTensor(state)
        dist, value = self.actor_critic(state)
        return value.item()

    def update(self, rewards, log_probs, values):
        returns = []
        discounted_reward = 0
        for reward in reversed(rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)

        returns = torch.tensor(returns, dtype=torch.float32, requires_grad=True)
        log_probs = torch.tensor(log_probs, dtype=torch.float32, requires_grad=True)
        values = [torch.tensor(value, dtype=torch.float32, requires_grad=True) for value in values]

        advantage = returns - torch.stack(values)

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        self.optimizer.zero_grad()
        loss = actor_loss + critic_loss
        loss.backward()
        self.optimizer.step()

    def train_episode(self, env, result_queue):
        state = env.reset()
        log_probs = []
        values = []
        rewards = []
        episode_reward = 0
        max_capital = env.current_capital  # Track highest capital achieved
        max_value = env.current_value
        done = False
        while not done:
            action = self.act(state)
            next_state, reward, done, _ = env.step(action)

            episode_reward += reward
            if env.current_capital > max_capital:  # Update max_capital if a higher value is reached
                max_capital = env.current_capital

            if env.current_value > max_value:  # Update max_value if a higher value is reached
                max_value = env.current_value

            log_prob = np.log(action + 1e-10)  # Log probability of the action (adjust for tanh output)
            value = self.get_value(state)

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)

            state = next_state

            if done:
                # print(f"Episode: {episode+1}, Reward: {episode_reward}, Max Capital: {max_capital}")
                break
        log_probs = np.array(log_probs)
        result_queue.put((episode_reward, max_capital, max_value, rewards, action, log_probs, values))

    def train(self, env, num_episodes, num_processes, run):
        best_capital = env.current_capital
        value_10 = env.current_capital
        value_100 = env.current_capital
        value_1000 = int(env.current_capital)
        value_2500 = env.current_capital
        value_5000 = env.current_capital
        value_7500 = env.current_capital
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        processes = []
        for rank in range(num_processes):
            p = ctx.Process(target=self.train_episode, args=(env, result_queue))
            p.start()
            processes.append(p)
        for episode in tqdm(range(num_episodes)):
            try:
                episode_reward, max_capital, max_value, rewards, action, log_probs, values = result_queue.get(timeout=5)
            except Empty:
                if all(not p.is_alive() for p in processes):
                    break
            if episode == 10:
                value_10 = max_value
            
            if episode == 100:
                value_100 = max_value
            
            if episode == 1000:
                value_1000 = max_value
            
            if episode == 2500:
                value_2500 = max_value

            if episode == 5000:
                value_5000 = max_value
            
            if episode == 7500:
                value_7500 = max_value
            
            if episode % 5 == 0:
                print(f"Episode: {episode+1}, Reward: {episode_reward}, Capital: {env.current_capital}, Value: {env.current_value}, Action: {action}")
            if max_capital > best_capital:
                best_capital = max_capital

            self.update(rewards, log_probs, values)

            run["train/episode_reward"].append(episode_reward)
            run["train/max_capital"].append(max_capital)
            run["train/max_value"].append(max_value)
            run["train/current_capital"].append(env.current_capital)
            run["train/current_value"].append(env.current_value)
            run["train/action"].append(action)

        run["train/best_capital"].append(best_capital)
        run["train/best_value"].append(max_value)
        run["train/value_10"].append(value_10)
        run["train/value_100"].append(value_100)
        run["train/value_1000"].append(value_1000)
        run["train/value_2500"].append(value_2500)
        run["train/value_5000"].append(value_5000)
        run["train/value_7500"].append(value_7500)

        run.stop()

        print(f"Training finished, Best Capital: {best_capital}, best Value: {max_value}, Value 10: {value_10}, Value 100: {value_100}, Value 1000: {value_1000}"
              f"Value 2500: {value_2500}, Value 5000: {value_5000}, Value 7500: {value_7500}")
        
        
