#Finance reinforcement learning app
from classes.ticker_downloader import TickerDownloader
from classes.rl_agent import ActorCritic, Agent
from classes.finance_environment import StockDataEnv
import neptune


def main():
    # Define hyperparameters
    num_episodes = 2500
    num_processes = 1
    gamma = 0.6
    lr = 0.01

    # Create the environment
    tickers = ["^IXIC"]
    start_date = "2021-01-31"
    end_date = "2023-01-01"
    window_size = 10
    initial_capital = 1000
    env = StockDataEnv(tickers, start_date, end_date, window_size, initial_capital)

    # Create the agent
    num_features = env.data.shape[1] + 2
    num_outputs = env.action_space.shape[0]
    agent = Agent(num_features, num_outputs, gamma=gamma, lr=lr)
    run = neptune.init_run(
        project="rafal-konklewski/FinanceRL",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkNmZiZGEzNS02YzFhLTQwMGMtYmYyYi1iMzAyNjU0YjAwNjEifQ==",
        name="Finance RL Agent"
        )

    agent.train(env=env,
                num_episodes=num_episodes,
                num_processes=num_processes,
                run=run)


if __name__ == '__main__':
    main()
