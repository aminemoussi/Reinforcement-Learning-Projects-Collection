import gymnasium as gym
import numpy as np
from agent import Agent


def test_cartpole(model_path, num_episodes=10, render=True):
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Initialize agent
    agent = Agent(state_size, action_size)

    # Load trained model
    agent.load_model(model_path)

    # Set epsilon to 0 for pure exploitation
    agent.epsilon = 0.0

    scores = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = np.array(state)
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            state = np.array(next_state)
            total_reward += reward

            if done or truncated:
                break

        scores.append(total_reward)
        print(f"Test Episode {episode + 1}: Score = {total_reward}")

    env.close()

    print(f"\nAverage test score: {np.mean(scores):.2f}")
    print(f"Max test score: {max(scores)}")
    print(f"Min test score: {min(scores)}")

    return scores


# Run testing
if __name__ == "__main__":
    test_scores = test_cartpole(
        "Deep_Q_Network/cartpole_model/final_model.pth", num_episodes=5, render=True
    )
    test_scores = test_cartpole(
        "Deep_Q_Network/cartpole_model/model_950.pth", num_episodes=5, render=True
    )
