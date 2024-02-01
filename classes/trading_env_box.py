import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt


class Actions(Enum):
    Sell = 0
    Buy = 1


class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size, available_capital):
        assert df.ndim == 2

        self.seed()
        self.df = df
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1] + 1)
        self.available_capital = available_capital
        self.starting_available_capital = available_capital
        self.last_action = 0
        self.stocks = 0
        self.total_asset = 0

        # spaces
        self.action_space = spaces.Box(low=-5, high=5, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float64)

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self.position = 0.0  # position in terms of quantity of stocks
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self._previous_profit = None
        self._first_rendering = None
        self.history = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None):
        super().reset(seed=seed)

        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self.position = 0.0  # Initial position is no stocks
        self._position_history = (self.window_size * [0]) + [self.position]
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self._first_rendering = True
        self.available_capital = self.starting_available_capital
        self.history = {}
        info = self._get_info()
        obs = self._get_observation()
        return obs, info

    def step(self, action):
        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True

        self.last_action = action

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        self._update_profit(action)

        # The sign of the action indicates whether we are buying or selling
        # The absolute value indicates the amount of stocks to buy/sell
        action_type = Actions.Buy if action >= 0 else Actions.Sell
        amount = abs(action)

        # Update the position by adding or subtracting the quantity of stocks traded
        if action_type == Actions.Buy:
            self.position += amount
        elif action_type == Actions.Sell:
            self.position -= amount

        self._position_history.append(self.position)
        observation = self._get_observation()
        truncated = False
        info = dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            position=self.position,  # position is now a quantity of stocks
            available_capital=self.available_capital,
            step_reward=step_reward
        )
        self._update_history(info)
        step_reward = float(step_reward[0]) if isinstance(step_reward, np.ndarray) else float(step_reward)

        return observation, step_reward, self._done, truncated, info

    def _get_observation(self):
        observation = self.signal_features[(self._current_tick-self.window_size+1):self._current_tick+1]
        scaled_capital = np.full((self.window_size, 1), self.available_capital)
        observation = np.hstack((observation, scaled_capital))
        return observation

    def _get_info(self):
        return {
            "total_reward": self._total_reward,
            "total_profit": self._total_profit,
            "position": self.position,  # position is now a quantity of stocks
            "total_asset": self.total_asset,
            "last_trade_tick": self._last_trade_tick,
            "done": self._done
        }

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def _plot_position(self, position, tick):
        marker_size = abs(position) * 10  # Marker size proportional to quantity of stocks
        color = 'green' if position >= 0 else 'red'  # Green for buying, red for selling

        plt.scatter(tick, self.prices[tick], color=color, s=marker_size)

    def render(self, mode='human'):
        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self._position_history[self._start_tick]
            self._plot_position(start_position, self._start_tick)

        self._plot_position(self.position, self._current_tick)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

        plt.pause(0.01)

    def render_all(self, mode='human'):
        window_ticks = np.arange(len(self._position_history))
        plt.plot(self.prices)

        for i, tick in enumerate(window_ticks):
            position = self._position_history[i]
            self._plot_position(position, tick)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    def _process_data(self):
        raise NotImplementedError

    def _calculate_reward(self, action):
        raise NotImplementedError

    def _update_profit(self, action):
        raise NotImplementedError

    def max_possible_profit(self):  # trade fees are ignored
        raise NotImplementedError
