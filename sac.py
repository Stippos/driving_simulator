import argparse
import random
from collections import deque
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import game2


# Parameters

parser = argparse.ArgumentParser(description="Soft Actor-Critic",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seed', default=1, type=int, metavar='N',
                    help='random seed')
parser.add_argument('--env', default='HalfCheetah-v2',
                    help='environment id')
parser.add_argument('--gamma', default=0.99, type=float, metavar='GAMMA',
                    help='discount factor')
parser.add_argument('--tau', default=0.005, type=float, metavar='TAU',
                    help='smoothing coefficient for target network')
parser.add_argument('--lr', default=0.0003, type=float, metavar='LR',
                    help='learning rate')
parser.add_argument('--replay_buffer_size', default=1000000, type=int, metavar='N',
                    help='size of the replay buffer')
parser.add_argument('--hidden_size', default=256, type=int, metavar='N',
                    help='number of units in a hidden layer')
parser.add_argument('--batch_size', default=256, type=int, metavar='N',
                    help='batch size')
parser.add_argument('--n_episodes', default=1000, type=int, metavar='N',
                    help='total number of training episodes')
parser.add_argument('--n_random_episodes', default=10, type=int, metavar='N',
                    help='number of initial episodes for random exploration')

parser.add_argument('--throttle_min', default=1, type=float)
parser.add_argument('--throttle_max', default=2, type=float)
parser.add_argument('--reward', default='speed')
parser.add_argument('--vision', default='normal')

args = parser.parse_args()

env = game2.game(throttle_min=args.throttle_min, throttle_max=args.throttle_max, reward_type=args.reward, vision="simple")
obs_size, act_size = env.observation_space.shape[0], env.action_space.shape[0]

#env.seed(args.seed)
#env.action_space.seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Networks

class MLP(nn.Module):
    def __init__(self, input_size, output_size, init_w=3e-3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, args.hidden_size), nn.ReLU(),
            nn.Linear(args.hidden_size, args.hidden_size), nn.ReLU(),
            nn.Linear(args.hidden_size, args.hidden_size), nn.ReLU(),
            nn.Linear(args.hidden_size, output_size)
        )
        self.net[-1].weight.data.uniform_(-init_w, init_w)
        self.net[-1].bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    """ Twin Q-networks """
    def __init__(self):
        super().__init__()
        self.net1 = MLP(obs_size+act_size, 1)
        self.net2 = MLP(obs_size+act_size, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        return self.net1(state_action), self.net2(state_action)


class Actor(nn.Module):
    """ Gaussian Policy """
    def __init__(self):
        super().__init__()
        self.net = MLP(obs_size, act_size*2)

    def forward(self, state):
        x = self.net(state)
        mean, log_std = x[:, :act_size], x[:, act_size:]
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        normal = Normal(mean, log_std.exp())
        x = normal.rsample()

        # Enforcing action bounds
        action = torch.tanh(x)
        log_prob = normal.log_prob(x) - torch.log(1 - action**2 + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        action, _ = self.sample(state)
        return action[0].detach().cpu().numpy()


critic = Critic().to(device)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.lr)

critic_target = Critic().to(device)
for target_param, param in zip(critic_target.parameters(), critic.parameters()):
    target_param.data.copy_(param.data)

actor = Actor().to(device)
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)

target_entropy = -act_size
log_alpha = torch.zeros(1, requires_grad=True, device=device)
alpha_optimizer = torch.optim.Adam([log_alpha], lr=args.lr)


def update_parameters(replay_buffer):
    batch = random.sample(replay_buffer, k=args.batch_size)
    state, action, reward, next_state, not_done = [torch.FloatTensor(t).to(device) for t in zip(*batch)]

    alpha = log_alpha.exp().item()

    # Update critic

    with torch.no_grad():
        next_action, next_action_log_prob = actor.sample(next_state)
        q1_next, q2_next = critic_target(next_state, next_action)
        q_next = torch.min(q1_next, q2_next)
        value_next = q_next - alpha * next_action_log_prob
        q_target = reward + not_done * args.gamma * value_next

    q1, q2 = critic(state, action)
    q1_loss = 0.5*F.mse_loss(q1, q_target)
    q2_loss = 0.5*F.mse_loss(q2, q_target)
    critic_loss = q1_loss + q2_loss

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    for target_param, param in zip(critic_target.parameters(), critic.parameters()):
        target_param.data.copy_((1.0-args.tau)*target_param.data + args.tau*param.data)

    # Update actor

    action_new, action_new_log_prob = actor.sample(state)
    q1_new, q2_new = critic(state, action_new)
    q_new = torch.min(q1_new, q2_new)
    actor_loss = (alpha*action_new_log_prob - q_new).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # Update alpha

    alpha_loss = -(log_alpha * (action_new_log_prob + target_entropy).detach()).mean()

    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()


# Training loop

replay_buffer = deque(maxlen=args.replay_buffer_size)

for episode in range(args.n_episodes):
    state = env.reset()
    episode_reward = 0
    for episode_step in range(env._max_episode_steps):
        if episode < args.n_random_episodes:
            action = env.action_space.sample()
        else:
            action = actor.select_action(state)

        next_state, reward, done, i = env.step(action)
        episode_reward += reward
	
        print("{} {} {:.2f}".format(i, episode, episode_reward))
        not_done = 1.0 if (episode_step+1) == env._max_episode_steps else float(not done)
        replay_buffer.append([state, action, [reward], next_state, [not_done]])
        state = next_state

        if len(replay_buffer) > args.batch_size:
            update_parameters(replay_buffer)

        if done:
            break

    print("Episode {}. Reward {}".format(episode, episode_reward))
