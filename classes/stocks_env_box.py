from classes.trading_env_box import TradingEnv, Actions
import numpy as np

class StocksEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound, available_capital):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        super().__init__(df, window_size, available_capital)

        self.trade_fee_bid_percent = 0.01  # unit
        self.trade_fee_ask_percent = 0.005  # unit

    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()

        prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]

        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))

        return prices, signal_features

    def _calculate_reward(self, action):
        step_reward = 0

        current_price = self.prices[self._current_tick]
        last_price = self.prices[self._current_tick - 1]

        total_asset = self.available_capital + (self.position * current_price).sum()
        reward = total_asset - self.total_asset
        self.total_asset = total_asset

        return reward

    def _update_profit(self, action):
        current_price = self.prices[self._current_tick]
        last_trade_price = self.prices[self._current_tick -1]

        if action > 0:  # Buying shares
            shares_bought = min(abs(action), self.available_capital / (1 + self.trade_fee_bid_percent))
            self._total_profit -= shares_bought * current_price * (1 + self.trade_fee_bid_percent)
            self.available_capital -= shares_bought * current_price * (1 + self.trade_fee_bid_percent)
            self.position += shares_bought
        elif action < 0:  # Selling shares
            shares_sold = min(self.position, abs(action))
            self._total_profit += shares_sold * current_price * (1 - self.trade_fee_ask_percent)
            self.available_capital += shares_sold * current_price * (1 - self.trade_fee_ask_percent)
            self.position -= shares_sold

    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            position = 0.0
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                position = -5.0  # Maximum short position
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = 5.0  # Maximum long position

            if position > 0:  # Buying shares
                shares = (profit * (1 - self.trade_fee_ask_percent)) / self.prices[last_trade_tick]
                profit = shares * self.prices[current_tick - 1]
            elif position < 0:  # Selling shares
                shares = (profit * (1 - self.trade_fee_bid_percent)) / self.prices[last_trade_tick]
                profit = shares * self.prices[current_tick - 1]

            last_trade_tick = current_tick - 1

        return profit