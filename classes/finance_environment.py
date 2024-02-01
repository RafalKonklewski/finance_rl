import gymnasium as gym
import numpy as np
import pandas as pd
import math
from classes.ticker_downloader import TickerDownloader


class StockDataEnv(gym.Env):
    def __init__(self, tickers, start_date, end_date, window_size, initial_capital):
        super(StockDataEnv, self).__init__()
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

        self.downloader = TickerDownloader(self.tickers, self.start_date, self.end_date)
        self.data = self.downloader.download_data()[['Open', 'High', 'Low', 'Close', 'Volume']]
        self.current_step = 0
        self.max_steps = len(self.data) - 1
        self.num_features = self.data.shape[1]

        self.initial_capital = initial_capital
        self.current_value = initial_capital
        self.current_capital = initial_capital
        self.current_shares = 0
        self.window_size = window_size
        self.action_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(window_size * self.num_features + 2,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.current_capital = self.initial_capital
        self.current_shares = 0
        return self._get_observation()

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        prev_shares = self.current_shares
        prev_capital = self.current_capital

        close_price = self.data.iloc[self.current_step]['Close']
        action = action[0]

        if action > 0:  # Buy shares
            available_shares = self.current_capital / close_price
            buy_shares = min(action, available_shares)
            self.current_shares += buy_shares
            self.current_capital -= buy_shares * close_price
        elif action < 0:  # Sell shares
            sell_shares = min(-action, self.current_shares)
            self.current_shares -= sell_shares
            self.current_capital += sell_shares * close_price

        self.current_step += 1

        new_observation = self._get_observation()
        reward = self._get_reward(action, prev_shares, prev_capital)
        self.current_value = self.current_capital + self.current_shares * close_price
        done = self.current_step == len(self.data) - 1

        return new_observation, reward, done, {}

    def _get_observation(self):
        if self.current_step < self.window_size:
            price_history = self.data.iloc[:self.current_step + 1].values.flatten()
        else:
            price_history = self.data.iloc[self.current_step - self.window_size + 1:self.current_step + 1].values.flatten()
        price_history = price_history.reshape((1, -1))  # Reshape to a 2D array
        current_info = np.array([self.current_shares, self.current_capital]).reshape((1, -1))
        observation = np.concatenate([price_history, current_info], axis=1)
        return observation
    
    def _get_reward(self, action, prev_shares, prev_capital):
        close_price = self.data.iloc[self.current_step]['Close']
        # reward = (self.current_shares - prev_shares) * close_price
        prev_stock_value = prev_shares * close_price
        current_stock_value = self.current_shares * close_price
        capital_factor = math.exp(self.current_capital / (self.current_value / math.log(200))) - 100
        reward = self.current_capital - prev_capital + (current_stock_value - prev_stock_value) + capital_factor
        return reward
    
    
