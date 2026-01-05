# import gymnasium as gym
# import numpy as np

# import matplotlib
# import matplotlib.pyplot as plt

# import random
# import torch
# from torch import nn
# import yaml

# from experience_replay import ReplayMemory
# from dqn import DQN

# from datetime import datetime, timedelta
# import argparse
# import itertools


# import os

# # For printing date and time
# DATE_FORMAT = "%m-%d %H:%M:%S"

# # Directory for saving run info
# RUNS_DIR = "runs"
# os.makedirs(RUNS_DIR, exist_ok=True)

# # 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
# matplotlib.use('Agg')

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu' # force cpu, sometimes GPU not always faster than CPU due to overhead of moving data to GPU

# # Deep Q-Learning Agent
# class Agent():

#     def __init__(self, hyperparameter_set):
#         with open('hyperparameters.yml', 'r') as file:
#             all_hyperparameter_sets = yaml.safe_load(file)
#             hyperparameters = all_hyperparameter_sets[hyperparameter_set]
#             # print(hyperparameters)

#         self.hyperparameter_set = hyperparameter_set

#         # Hyperparameters (adjustable)
#         self.env_id             = hyperparameters['env_id']
#         self.learning_rate_a    = hyperparameters['learning_rate_a']        # learning rate (alpha)
#         self.discount_factor_g  = hyperparameters['discount_factor_g']      # discount rate (gamma)
#         self.network_sync_rate  = hyperparameters['network_sync_rate']      # number of steps the agent takes before syncing the policy and target network
#         self.replay_memory_size = hyperparameters['replay_memory_size']     # size of replay memory
#         self.mini_batch_size    = hyperparameters['mini_batch_size']        # size of the training data set sampled from the replay memory
#         self.epsilon_init       = hyperparameters['epsilon_init']           # 1 = 100% random actions
#         self.epsilon_decay      = hyperparameters['epsilon_decay']          # epsilon decay rate
#         self.epsilon_min        = hyperparameters['epsilon_min']            # minimum epsilon value
#         self.stop_on_reward     = hyperparameters['stop_on_reward']         # stop training after reaching this number of rewards
#         self.fc1_nodes          = hyperparameters['fc1_nodes']
#         self.env_make_params    = hyperparameters.get('env_make_params',{}) # Get optional environment-specific parameters, default to empty dict
#         self.enable_double_dqn  = hyperparameters['enable_double_dqn']      # double dqn on/off flag
#         self.enable_dueling_dqn = hyperparameters['enable_dueling_dqn']     # dueling dqn on/off flag

#         # Neural Network
#         # self.loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
#         self.loss_fn = nn.SmoothL1Loss()
#         self.optimizer = None                # NN Optimizer. Initialize later.

#         # Path to Run info
#         self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
#         self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
#         self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

#     def run(self, is_training=True, render=False):
#         episodes=5000 if is_training else 10
#         step_count=0
#         best_mean_reward = -9999999
        
#         if is_training:
#             start_time = datetime.now()
#             last_graph_update_time = start_time

#             log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
#             print(log_message)
#             with open(self.LOG_FILE, 'w') as file:
#                 file.write(log_message + '\n')

#         # Create instance of the environment.
#         # Use "**self.env_make_params" to pass in environment-specific parameters from hyperparameters.yml.
#         env = gym.make(self.env_id, render_mode='human' if render else None, **self.env_make_params)

#         # Number of possible actions
#         num_actions = env.action_space.n

#         # Get observation space size
#         num_states = env.observation_space.shape[0] # Expecting type: Box(low, high, (shape0,), float64)

#         # List to keep track of rewards collected per episode.
#         rewards_per_episode = []

#         # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
#         policy_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(device)

#         if is_training:
#             # Initialize epsilon
#             epsilon = self.epsilon_init

#             # Initialize replay memory
#             memory = ReplayMemory(self.replay_memory_size)

#             # Create the target network and make it identical to the policy network
#             target_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(device)
#             target_dqn.load_state_dict(policy_dqn.state_dict())

#             # Policy network optimizer. "Adam" optimizer can be swapped to something else.
#             self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

#             # List to keep track of epsilon decay
#             epsilon_history = []

#             # Track number of steps taken. Used for syncing policy => target network.
            

#             # Track best reward
#             best_reward = -9999999
#         else:
#             # Load learned policy
#             policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))

#             # switch model to evaluation mode
#             policy_dqn.eval()

#         # Train INDEFINITELY, manually stop the run when you are satisfied (or unsatisfied) with the results
#         # for episode in itertools.count():
#         for episode in range(episodes):
#             MAX_STEPS = 1000
            
#             state, _ = env.reset()  # Initialize environment. Reset returns (state,info).
           
                                        
#             state = torch.tensor(state, dtype=torch.float, device=device) # Convert state to tensor directly on device
#             state = state / torch.tensor(
#                     [1.5, 1.5, 5.0, 5.0, 3.14, 5.0, 1.0, 1.0],
#                     device=device)
            
#             terminated = False      # True when agent reaches goal or fails
#             episode_reward = 0.0    # Used to accumulate rewards per episode
#             episode_steps=0
#             # Perform actions until episode terminates or reaches max rewards
#             # (on some envs, it is possible for the agent to train to a point where it NEVER terminates, so stop on reward is necessary)
#             while(not terminated and episode_steps<MAX_STEPS):
#                 episode_steps+=1

#                 # Select action based on epsilon-greedy
#                 if is_training and random.random() < epsilon:
#                     # select random action
#                     action = env.action_space.sample()
#                     action = torch.tensor(action, dtype=torch.int64, device=device)
#                 else:
#                     # select best action
#                     with torch.no_grad():
#                         # state.unsqueeze(dim=0): Pytorch expects a batch layer, so add batch dimension i.e. tensor([1, 2, 3]) unsqueezes to tensor([[1, 2, 3]])
#                         # policy_dqn returns tensor([[1], [2], [3]]), so squeeze it to tensor([1, 2, 3]).
#                         # argmax finds the index of the largest element.
#                         action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

#                 # Execute action. Truncated and info is not used.
#                 new_state,reward,terminated,truncated,info = env.step(action.item())

#                 # Accumulate rewards
#                 episode_reward += reward

#                 # Convert new state and reward to tensors on device
#                 new_state = torch.tensor(new_state, dtype=torch.float, device=device)
#                 new_state= new_state / torch.tensor(
#                     [1.5, 1.5, 5.0, 5.0, 3.14, 5.0, 1.0, 1.0],
#                     device=device)
#                 reward = torch.tensor(reward, dtype=torch.float, device=device)

#                 if is_training:
#                     # Save experience into memory
#                     memory.append((state, action, new_state, reward, terminated))

#                     # Increment step counter
#                     step_count+=1

#                 # Move to the next state
#                 state = new_state
                
#                 if is_training and len(memory) > self.mini_batch_size and step_count % 4 == 0:
#                     mini_batch = memory.sample(self.mini_batch_size)
#                     self.optimize(mini_batch, policy_dqn, target_dqn)

#             # Keep track of the rewards collected per episode.
#             rewards_per_episode.append(episode_reward)

#             # Save model when new best reward is obtained.
#             if is_training:
                
#                 if len(rewards_per_episode) >= 100:
#                     mean_100 = np.mean(rewards_per_episode[-100:])
#                     if mean_100 > best_mean_reward:
#                         torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
#                         best_mean_reward = mean_100
#                         print("BEST MEAN REWARD-MODEL SAVED")
                        
#                 if episode_reward > best_reward:
                    
#                     log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f}  at episode {episode}"
#                     print(log_message)
#                     with open(self.LOG_FILE, 'a') as file:
#                         file.write(log_message + '\n')
#                     best_reward = episode_reward

                    
                    
#                 if episode%100==0:
#                     print(f"{episode} episodes done")
#                 if episode % 500 == 0:
#                     torch.save(policy_dqn.state_dict(), f"{RUNS_DIR}/ckpt_{episode}.pt")
#                     print("500 MORE EPISODES DONE-MODEL SAVED")


#                 # Update graph every x seconds
#                 current_time = datetime.now()
#                 if current_time - last_graph_update_time > timedelta(seconds=10):
#                     self.save_graph(rewards_per_episode, epsilon_history)
#                     last_graph_update_time = current_time

#                 # If enough experience has been collected
#                 # if len(memory)>self.mini_batch_size:
#                 #     mini_batch = memory.sample(self.mini_batch_size)
#                 #     self.optimize(mini_batch, policy_dqn, target_dqn)

#                 # Decay epsilon
#                 epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
#                 epsilon_history.append(epsilon)

#                 # Copy policy network to target network after a certain number of steps
#                 if step_count >= self.network_sync_rate:
#                     target_dqn.load_state_dict(policy_dqn.state_dict())
#                     step_count=0
                    


#     def save_graph(self, rewards_per_episode, epsilon_history):
#         # Save plots
#         fig = plt.figure(1)

#         # Plot average rewards (Y-axis) vs episodes (X-axis)
#         mean_rewards = np.zeros(len(rewards_per_episode))
#         for x in range(len(mean_rewards)):
#             mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
#         plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
#         # plt.xlabel('Episodes')
#         plt.ylabel('Mean Rewards')
#         plt.plot(mean_rewards)

#         # Plot epsilon decay (Y-axis) vs episodes (X-axis)
#         plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
#         # plt.xlabel('Time Steps')
#         plt.ylabel('Epsilon Decay')
#         plt.plot(epsilon_history)

#         plt.subplots_adjust(wspace=1.0, hspace=1.0)

#         # Save plots
#         fig.savefig(self.GRAPH_FILE)
#         plt.close(fig)


#     # Optimize policy network
#     def optimize(self, mini_batch, policy_dqn, target_dqn):

#         # Transpose the list of experiences and separate each element
#         states, actions, new_states, rewards, terminations = zip(*mini_batch)

#         # Stack tensors to create batch tensors
#         # tensor([[1,2,3]])
#         states = torch.stack(states)

#         actions = torch.stack(actions)

#         new_states = torch.stack(new_states)

#         rewards = torch.stack(rewards)
#         terminations = torch.tensor(terminations).float().to(device)

#         with torch.no_grad():
#             if self.enable_double_dqn:
#                 best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)

#                 target_q = rewards + (1-terminations) * self.discount_factor_g * \
#                                 target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
#             else:
#                 # Calculate target Q values (expected returns)
#                 target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]
#                 '''
#                     target_dqn(new_states)  ==> tensor([[1,2,3],[4,5,6]])
#                         .max(dim=1)         ==> torch.return_types.max(values=tensor([3,6]), indices=tensor([3, 0, 0, 1]))
#                             [0]             ==> tensor([3,6])
#                 '''

#         # Calcuate Q values from current policy
#         current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
#         '''
#             policy_dqn(states)  ==> tensor([[1,2,3],[4,5,6]])
#                 actions.unsqueeze(dim=1)
#                 .gather(1, actions.unsqueeze(dim=1))  ==>
#                     .squeeze()                    ==>
#         '''

#         # Compute loss
#         loss = self.loss_fn(current_q, target_q)

#         # Optimize the model (backpropagation)
#         self.optimizer.zero_grad()  # Clear gradients
#         loss.backward()             # Compute gradients
#         self.optimizer.step()       # Update network parameters i.e. weights and biases

# if __name__ == '__main__':
#     # Parse command line inputs
#     parser = argparse.ArgumentParser(description='Train or test model.')
#     parser.add_argument('hyperparameters', help='')
#     parser.add_argument('--train', help='Training mode', action='store_true')
#     args = parser.parse_args()

#     dql = Agent(hyperparameter_set=args.hyperparameters)

#     if args.train:
#         dql.run(is_training=True)
#     else:
#         dql.run(is_training=False, render=True)



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
        
        
        
        
        
        
        
        
        

