
import gymnasium as gym
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import random
import torch
from torch import nn
import yaml

from experience_replay import ReplayMemory
from dqn import DQN

from datetime import datetime, timedelta
import argparse
import itertools
import os

TIME_FMT = "%m-%d %H:%M:%S"

OUTPUT_DIR = "runs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

matplotlib.use('Agg')

compute_device = 'cuda' if torch.cuda.is_available() else 'cpu'
compute_device = 'cpu'

class RLAgent:

    def __init__(self, hp_key):
        with open('hyperparameters.yml', 'r') as f:
            hp_all = yaml.safe_load(f)
            hp = hp_all[hp_key]

        self.hp_key = hp_key

        self.env_name        = hp['env_id']
        self.lr              = hp['learning_rate_a']
        self.gamma           = hp['discount_factor_g']
        self.sync_steps      = hp['network_sync_rate']
        self.buffer_cap      = hp['replay_memory_size']
        self.batch_sz        = hp['mini_batch_size']
        self.eps_start       = hp['epsilon_init']
        self.eps_decay       = hp['epsilon_decay']
        self.eps_min         = hp['epsilon_min']
        self.reward_target   = hp['stop_on_reward']
        self.hidden_units    = hp['fc1_nodes']
        self.env_kwargs      = hp.get('env_make_params', {})
        self.use_double      = hp['enable_double_dqn']
        self.use_dueling     = hp['enable_dueling_dqn']

        self.criterion = nn.SmoothL1Loss()
        self.opt = None

        self.LOG_PATH   = os.path.join(OUTPUT_DIR, f'{self.hp_key}.log')
        self.MODEL_PATH = os.path.join(OUTPUT_DIR, f'{self.hp_key}.pt')
        self.PLOT_PATH  = os.path.join(OUTPUT_DIR, f'{self.hp_key}.png')

    def run(self, train_mode=True, show=False):
        total_episodes = 10000 if train_mode else 10
        sync_counter = 0
        top_avg = -1e9

        if train_mode:
            t0 = datetime.now()
            last_plot = t0
            msg = f"{t0.strftime(TIME_FMT)} | run started"
            print(msg)
            with open(self.LOG_PATH, 'w') as f:
                f.write(msg + '\n')

        env = gym.make(self.env_name, render_mode='human' if show else None, **self.env_kwargs)

        n_actions = env.action_space.n
        n_states = env.observation_space.shape[0]

        ep_returns = []

        policy_net = DQN(n_states, n_actions, self.hidden_units, self.use_dueling).to(compute_device)

        if train_mode:
            eps = self.eps_start
            replay = ReplayMemory(self.buffer_cap)

            target_net = DQN(n_states, n_actions, self.hidden_units, self.use_dueling).to(compute_device)
            target_net.load_state_dict(policy_net.state_dict())

            self.opt = torch.optim.Adam(policy_net.parameters(), lr=self.lr)

            eps_trace = []
            best_ep = -1e9
        else:
            policy_net.load_state_dict(torch.load(self.MODEL_PATH))
            policy_net.eval()

        for ep in range(total_episodes):
            LIMIT = 1000
            obs, _ = env.reset()

            obs = torch.tensor(obs, dtype=torch.float, device=compute_device)
            obs = obs / torch.tensor([1.5, 1.5, 5.0, 5.0, 3.14, 5.0, 1.0, 1.0], device=compute_device)

            done = False
            score = 0.0
            steps = 0

            while (not done) and steps < LIMIT:
                steps += 1

                if train_mode and random.random() < eps:
                    act = env.action_space.sample()
                    act = torch.tensor(act, dtype=torch.int64, device=compute_device)
                else:
                    with torch.no_grad():
                        act = policy_net(obs.unsqueeze(0)).squeeze().argmax()

                nxt_obs, rew, done, trunc, info = env.step(act.item())
                score += rew

                nxt_obs = torch.tensor(nxt_obs, dtype=torch.float, device=compute_device)
                nxt_obs = nxt_obs / torch.tensor([1.5, 1.5, 5.0, 5.0, 3.14, 5.0, 1.0, 1.0], device=compute_device)
                rew = torch.tensor(rew, dtype=torch.float, device=compute_device)

                if train_mode:
                    replay.append((obs, act, nxt_obs, rew, done))
                    sync_counter += 1

                obs = nxt_obs

                if train_mode and len(replay) > self.batch_sz and sync_counter % 4 == 0:
                    batch = replay.sample(self.batch_sz)
                    self.update(batch, policy_net, target_net)

            ep_returns.append(score)

            if train_mode:
                if len(ep_returns) >= 100:
                    avg100 = np.mean(ep_returns[-100:])
                    if avg100 > top_avg:
                        torch.save(policy_net.state_dict(), self.MODEL_PATH)
                        top_avg = avg100
                        print("checkpoint saved (best avg)")

                if score > best_ep:
                    msg = f"{datetime.now().strftime(TIME_FMT)} | new peak reward {score:.1f} at ep {ep}"
                    print(msg)
                    with open(self.LOG_PATH, 'a') as f:
                        f.write(msg + '\n')
                    best_ep = score

                if ep % 100 == 0:
                    print(f"progress: {ep} episodes")

                if ep % 500 == 0:
                    torch.save(policy_net.state_dict(), f"{OUTPUT_DIR}/ckpt_{ep}.pt")
                    print("intermediate save done")

                now = datetime.now()
                if now - last_plot > timedelta(seconds=10):
                    self.save_plot(ep_returns, eps_trace)
                    last_plot = now

                eps = max(eps * self.eps_decay, self.eps_min)
                eps_trace.append(eps)

                if sync_counter >= self.sync_steps:
                    target_net.load_state_dict(policy_net.state_dict())
                    sync_counter = 0

    def save_plot(self, returns, eps_trace):
        fig = plt.figure(1)

        avg = np.zeros(len(returns))
        for i in range(len(avg)):
            avg[i] = np.mean(returns[max(0, i-99):(i+1)])

        plt.subplot(121)
        plt.ylabel('Avg Reward')
        plt.plot(avg)

        plt.subplot(122)
        plt.ylabel('Epsilon')
        plt.plot(eps_trace)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)
        fig.savefig(self.PLOT_PATH)
        plt.close(fig)

    def update(self, batch, policy_net, target_net):
        s, a, ns, r, d = zip(*batch)

        s = torch.stack(s)
        a = torch.stack(a)
        ns = torch.stack(ns)
        r = torch.stack(r)
        d = torch.tensor(d).float().to(compute_device)

        with torch.no_grad():
            if self.use_double:
                best_a = policy_net(ns).argmax(dim=1)
                tgt = r + (1 - d) * self.gamma * \
                      target_net(ns).gather(1, best_a.unsqueeze(1)).squeeze()
            else:
                tgt = r + (1 - d) * self.gamma * target_net(ns).max(dim=1)[0]

        cur = policy_net(s).gather(1, a.unsqueeze(1)).squeeze()

        loss = self.criterion(cur, tgt)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('hyperparameters')
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    agent = RLAgent(hp_key=args.hyperparameters)

    if args.train:
        agent.run(train_mode=True)
    else:
        agent.run(train_mode=False, show=True)
        
        
        
        
        
        
        
        
        

