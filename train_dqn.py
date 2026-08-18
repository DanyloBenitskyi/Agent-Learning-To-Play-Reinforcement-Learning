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
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from dqn_agent import DQNAgent

N_EPISODES = 300
ROLLING_WINDOW = 20  # how many recent episodes to average when checking for a new best
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "outputs", "best_dqn_model.pt")


def train():
    env = gym.make("CartPole-v1")
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
    )

    rewards_per_episode = []
    best_rolling_avg = -float("inf")
    best_episode = None

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

        # check if this is a new best, every episode (not just every 20,
        # print interval), so we don't miss saving a brief peak
        if len(rewards_per_episode) >= ROLLING_WINDOW:
            current_rolling_avg = np.mean(rewards_per_episode[-ROLLING_WINDOW:])
            if current_rolling_avg > best_rolling_avg:
                best_rolling_avg = current_rolling_avg
                best_episode = episode + 1
                torch.save(agent.q_network.state_dict(), MODEL_SAVE_PATH)

        if (episode + 1) % 20 == 0:
            recent_avg = np.mean(rewards_per_episode[-20:])
            print(f"Episode {episode + 1}/{N_EPISODES} | "
                  f"avg reward (last 20 eps): {recent_avg:.1f} | "
                  f"epsilon: {agent.epsilon:.3f}")

    env.close()
    return agent, rewards_per_episode, best_rolling_avg, best_episode


def plot_results(rewards_per_episode, best_rolling_avg, best_episode, window=20):
    """
    Marks the best point the rolling average ever reached during
    training, not just wherever it happened to land at the very end.

    DQN training isn't monotonic - it's normal for performance to peak
    partway through and then dip afterward (a known instability called
    "catastrophic forgetting", where a bad batch of replay samples nudges
    the network in a worse direction). So the final episode's score isn't
    necessarily the agent's best achieved performance, and it's worth
    reporting both.
    """
    rolling_avg = np.convolve(rewards_per_episode, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(8, 4))
    plt.plot(rewards_per_episode, alpha=0.3, label="raw reward")
    plt.plot(range(window - 1, len(rewards_per_episode)), rolling_avg, label=f"rolling avg (window={window})")
    plt.scatter([best_episode], [best_rolling_avg], color="red", zorder=5,
                label=f"best rolling avg ({best_rolling_avg:.0f} @ episode {best_episode})")
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
    agent, rewards, best_rolling_avg, best_episode = train()
    plot_results(rewards, best_rolling_avg, best_episode)

    final_avg = np.mean(rewards[-20:])
    print(f"\nFinal average reward (last 20 episodes):        {final_avg:.1f} / 500 max")
    print(f"Best rolling average reached during training:    {best_rolling_avg:.1f} / 500 max (around episode {best_episode})")
    print(f"Best model weights saved to:                      {MODEL_SAVE_PATH}")
    print("\n(DQN training isn't monotonic - it's normal for performance to peak "
          "partway through and dip afterward. The model saved above is a snapshot "
          "from the agent's best-performing point during training, not the final "
          "episode, so it's the version worth using if you want to actually run "
          "the agent afterward.)")
