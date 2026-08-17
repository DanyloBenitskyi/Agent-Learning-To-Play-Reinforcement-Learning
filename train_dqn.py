"""
train_dqn.py

Trains a DQN agent to solve CartPole: a pole is balanced on a cart, and
the agent has to push the cart left or right to keep the pole from
falling over. One point of reward per timestep the pole stays upright.
CartPole-v1 caps episodes at 500 steps, so a "solved" agent should be
regularly hitting close to 500.

Run from the project root:
    python3 train_dqn.py

Note: this takes a few minutes since it's training a neural network,
unlike the near-instant tabular Q-learning script.
"""

import os
import sys
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from dqn_agent import DQNAgent

N_EPISODES = 300


def train():
    env = gym.make("CartPole-v1")
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
    )

    rewards_per_episode = []

    for episode in range(N_EPISODES):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.remember(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        rewards_per_episode.append(total_reward)

        if (episode + 1) % 20 == 0:
            recent_avg = np.mean(rewards_per_episode[-20:])
            print(f"Episode {episode + 1}/{N_EPISODES} | "
                  f"avg reward (last 20 eps): {recent_avg:.1f} | "
                  f"epsilon: {agent.epsilon:.3f}")

    env.close()
    return agent, rewards_per_episode


def plot_results(rewards_per_episode, window=20):
    rolling_avg = np.convolve(rewards_per_episode, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(8, 4))
    plt.plot(rewards_per_episode, alpha=0.3, label="raw reward")
    plt.plot(range(window - 1, len(rewards_per_episode)), rolling_avg, label=f"rolling avg (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Total reward (steps survived)")
    plt.title("DQN on CartPole")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), "outputs", "dqn_rewards.png")
    plt.savefig(out_path)
    print(f"\nSaved training plot to {out_path}")


if __name__ == "__main__":
    agent, rewards = train()
    plot_results(rewards)

    final_avg = np.mean(rewards[-20:])
    print(f"\nFinal average reward (last 20 episodes): {final_avg:.1f} / 500 max")
