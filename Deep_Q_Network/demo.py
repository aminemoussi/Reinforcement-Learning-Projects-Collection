import time

import gymnasium as gym
import numpy as np
from agent import Agent


def demo_cartpole(model_path, num_episodes=50):
    env = gym.make("CartPole-v1", render_mode="human")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = Agent(state_size, action_size)
    agent.load_model(model_path)
    agent.epsilon = 0.0  # No exploration

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = np.array(state)
        total_reward = 0
        done = False

        print(f"Starting episode {episode + 1}")

        while not done:
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            state = np.array(next_state)
            total_reward += reward

            # Slow down for visualization
            time.sleep(0.02)

            if done or truncated:
                break

        print(f"Episode {episode + 1} completed with score: {total_reward}")
        time.sleep(1)  # Pause between episodes

    env.close()


if __name__ == "__main__":
    demo_cartpole("Deep_Q_Network/cartpole_model/final_model.pth")
