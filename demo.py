"""
demo.py

Quick way to see both agents in action without waiting for a full
training run. This trains a smaller/faster version of each and prints
the results to the terminal.

Run from the project root:
    python3 demo.py
"""

import sys
import os
import numpy as np
import gymnasium as gym

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from q_learning_agent import QLearningAgent
from dqn_agent import DQNAgent


def demo_q_learning():
    print("=== Q-Learning on FrozenLake (2000 episodes, quick version) ===")
    env = gym.make("FrozenLake-v1", is_slippery=False)
    agent = QLearningAgent(n_states=env.observation_space.n, n_actions=env.action_space.n)

    rewards = []
    for episode in range(2000):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        agent.decay_epsilon()
        rewards.append(total_reward)

    print(f"Success rate, first 200 episodes:  {np.mean(rewards[:200]):.1%}")
    print(f"Success rate, last 200 episodes:   {np.mean(rewards[-200:]):.1%}")
    env.close()


def demo_dqn():
    print("\n=== DQN on CartPole (80 episodes, quick version) ===")
    env = gym.make("CartPole-v1")
    agent = DQNAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)

    rewards = []
    for episode in range(80):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.remember(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            total_reward += reward
        agent.decay_epsilon()
        rewards.append(total_reward)

    print(f"Avg reward, first 10 episodes: {np.mean(rewards[:10]):.1f} / 500")
    print(f"Avg reward, last 10 episodes:  {np.mean(rewards[-10:]):.1f} / 500")
    print("(80 episodes isn't enough to fully solve CartPole - run train_dqn.py "
          "for the full 300-episode version, which gets much closer to 500)")
    env.close()


if __name__ == "__main__":
    demo_q_learning()
    demo_dqn()
