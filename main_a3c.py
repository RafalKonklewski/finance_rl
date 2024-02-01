from classes.a3c import A3C, Worker
from classes.shared_adam import SharedAdam
import torch.multiprocessing as mp
from classes.finance_environment import StockDataEnv
from classes.ticker_downloader import TickerDownloader


def main():
    # Define hyperparameters
    num_episodes = 2500
    num_processes = 2
    gamma = 0.6
    lr = 0.01

    # Create the environment
    tickers = ["^IXIC"]
    start_date = "2021-01-31"
    end_date = "2023-01-01"
    window_size = 10
    initial_capital = 1000
    update_steps = 5
    env = StockDataEnv(tickers, start_date, end_date, window_size, initial_capital)

    # Create the agent
    num_features = env.data.shape[1] + 2
    num_outputs = env.action_space.shape[0]
    agent = A3C(num_features, num_outputs)        # global network
    agent.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(agent.parameters(), lr=1e-4, betas=(0.95, 0.999))  # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [Worker(num_features, num_outputs, agent,
                      env, num_episodes, gamma, update_steps,
                      opt, global_ep, global_ep_r,
                      res_queue, i) for i in range(num_processes)]
    [w.start() for w in workers]
    res = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

if __name__ == "__main__":
    main()