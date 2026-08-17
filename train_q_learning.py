"""
train_q_learning.py

Trains a tabular Q-learning agent to solve FrozenLake, a simple grid
world game:

    S F F F        S = start
    F H F H        F = frozen (safe to walk on)
    F F F H        H = hole (fall in = game over, reward 0)
    H F F G        G = goal (reward 1)

The agent starts at S and needs to reach G without falling into a hole.
We use the non-slippery version so the agent's moves are deterministic
(no random slipping), which makes it a cleaner example of Q-learning
actually converging to the optimal policy.

Run from the project root:
    python3 train_q_learning.py
"""

import os
import sys
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from q_learning_agent import QLearningAgent

N_EPISODES = 5000
MAX_STEPS_PER_EPISODE = 100


def train():
    env = gym.make("FrozenLake-v1", is_slippery=False)
    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
    )

    rewards_per_episode = []

    for episode in range(N_EPISODES):
        state, _ = env.reset()
        total_reward = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                break

        agent.decay_epsilon()
        rewards_per_episode.append(total_reward)

        if (episode + 1) % 500 == 0:
            recent_success_rate = np.mean(rewards_per_episode[-500:])
            print(f"Episode {episode + 1}/{N_EPISODES} | "
                  f"success rate (last 500 eps): {recent_success_rate:.2%} | "
                  f"epsilon: {agent.epsilon:.3f}")

    env.close()
    return agent, rewards_per_episode


def plot_results(rewards_per_episode, window=100):
    """Plot a rolling average of success rate over training, since raw
    per-episode reward (0 or 1) is too noisy to read on its own."""
    rolling_avg = np.convolve(rewards_per_episode, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(8, 4))
    plt.plot(rolling_avg)
    plt.xlabel("Episode")
    plt.ylabel(f"Success rate (rolling avg, window={window})")
    plt.title("Q-Learning on FrozenLake")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), "outputs", "q_learning_rewards.png")
    plt.savefig(out_path)
    print(f"\nSaved training plot to {out_path}")


def show_learned_policy(agent, env_name="FrozenLake-v1"):
    """Print out the grid with arrows showing what the agent learned to
    do in each state - a nice way to sanity check the result visually."""
    arrows = {0: "<", 1: "v", 2: ">", 3: "^"}  # LEFT, DOWN, RIGHT, UP
    policy = [arrows[np.argmax(agent.q_table[s])] for s in range(agent.n_states)]

    print("\nLearned policy (arrow = action the agent picks in that tile):")
    for row in range(4):
        print(" ".join(policy[row * 4:(row + 1) * 4]))


if __name__ == "__main__":
    agent, rewards = train()
    plot_results(rewards)
    show_learned_policy(agent)

    final_success_rate = np.mean(rewards[-500:])
    print(f"\nFinal success rate (last 500 episodes): {final_success_rate:.2%}")
