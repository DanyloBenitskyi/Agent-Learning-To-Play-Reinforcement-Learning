"""
dqn_agent.py

Deep Q-Network (DQN) agent.

This is the same core idea as Q-learning, but instead of a table, we use
a small neural network to *predict* Q[state, action] for any state -
this is necessary because CartPole's state (cart position, cart velocity,
pole angle, pole angular velocity) is made of continuous numbers, so
there's no way to build a finite table with a row for every possible
state.

Two tricks make this actually train stably instead of blowing up:

1. Replay buffer: instead of learning only from the most recent step
   (which is highly correlated with the step before it), we store past
   experiences in a buffer and train on random batches sampled from it.
   This breaks the correlation and reuses data more efficiently.

2. Target network: we keep a second, slightly-delayed copy of the
   network to compute the "target" Q-value during training. If we used
   the same network that's currently being updated, the target would
   shift under us every single step, making training unstable. The
   target network's weights are periodically synced from the main
   network instead of updated every step.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """A small feedforward network: state -> Q-value for each action."""

    def __init__(self, state_dim, action_dim, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states), np.array(actions), np.array(rewards),
            np.array(next_states), np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.995,
                 buffer_capacity=10000, batch_size=64, target_update_every=10):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_every = target_update_every

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

        self.train_steps = 0

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return int(torch.argmax(q_values).item())

    def remember(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        """One gradient update, sampled from the replay buffer. Does
        nothing until the buffer has enough experiences for a full batch."""
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Q-value the network currently predicts for the action we actually took
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # target: reward + discounted best future value, using the target
        # network (not the live one) for stability, as explained above
        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        loss = nn.functional.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
