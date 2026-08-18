"""
run_best_dqn.py

Loads the best saved model from outputs/best_dqn_model.pt (produced by
train_dqn.py) and runs it for a few episodes with no exploration and no
further learning, just to see how well it actually performs.

Run from the project root, after running train_dqn.py at least once:
    python3 run_best_dqn.py
"""

import os
import sys
import numpy as np
import gymnasium as gym
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from dqn_agent import QNetwork

MODEL_PATH = os.path.join(os.path.dirname(__file__), "outputs", "best_dqn_model.pt")
N_EVAL_EPISODES = 10


def run():
    if not os.path.exists(MODEL_PATH):
        print(f"No saved model found at {MODEL_PATH}")
        print("Run train_dqn.py first to train and save a model.")
        return

    env = gym.make("CartPole-v1")
    net = QNetwork(env.observation_space.shape[0], env.action_space.n)
    net.load_state_dict(torch.load(MODEL_PATH))
    net.eval()  # no dropout/batchnorm here, but good practice regardless

    rewards = []
    for episode in range(N_EVAL_EPISODES):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # always pick the best known action - no random exploration,
            # since we're evaluating the trained agent, not training it
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = int(torch.argmax(net(state_tensor)).item())

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        rewards.append(total_reward)
        print(f"Episode {episode + 1}: {total_reward:.0f} steps")

    env.close()
    print(f"\nAverage over {N_EVAL_EPISODES} episodes: {np.mean(rewards):.1f} / 500")


if __name__ == "__main__":
    run()
