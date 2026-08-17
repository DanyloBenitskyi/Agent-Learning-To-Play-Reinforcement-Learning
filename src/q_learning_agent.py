"""
q_learning_agent.py

A simple tabular Q-learning agent.

Q-learning keeps a big table (the "Q-table") of size [n_states x n_actions].
Each entry Q[state, action] is the agent's current estimate of "how good
is it to take this action from this state, considering all future rewards
too, not just the immediate one".

The agent starts out knowing nothing (table full of zeros) and updates it
after every single step it takes in the environment, using this rule:

    Q[state, action] = Q[state, action] + alpha * (
        reward + gamma * max(Q[next_state]) - Q[state, action]
    )

- alpha (learning rate): how much we adjust our estimate after each step
- gamma (discount factor): how much we care about future rewards vs
  immediate ones (close to 1 = plan far ahead, close to 0 = short-sighted)
- reward + gamma * max(Q[next_state]) is our new "target" estimate of
  the value of that state-action pair, and we nudge our old estimate
  toward it

This only works when there's a small, finite number of states (here, 16
tiles on the FrozenLake grid) — that's why this is a "table". For
environments with continuous states (like CartPole, where position and
velocity are real numbers), you can't build a table with infinite rows,
which is why we use a neural network instead in dqn_agent.py.
"""

import numpy as np


class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        """
        Epsilon-greedy action selection: most of the time pick the best
        known action, but sometimes (with probability epsilon) pick a
        random action instead, so the agent keeps exploring instead of
        getting stuck always doing the same thing early on when its
        Q-table estimates are still bad.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        best_next_value = 0 if done else np.max(self.q_table[next_state])
        target = reward + self.gamma * best_next_value
        current = self.q_table[state, action]
        self.q_table[state, action] = current + self.alpha * (target - current)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
