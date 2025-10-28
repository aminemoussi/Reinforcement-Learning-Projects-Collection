import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from agent import Agent


def train_cartpole():
    # env setup
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print(f"State size: {state_size}, Action size: {action_size}")

    # training params
    batch_size = 32
    n_episodes = 500
    gamma = 0.99  # Slightly higher for long-term thinking
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    learning_rate = 0.001
    update_target_every = 5

    agent = Agent(
        state_size=state_size,
        action_size=action_size,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        learning_rate=learning_rate,
    )

    print(f"Model first layer weight shape: {agent.model.d1.weight.shape}")
    print(f"Expected: (24, {state_size})")

    # output directory
    output_dir = "./cartpole_model/"
    os.makedirs(output_dir, exist_ok=True)

    # Trackings
    scores = []
    epsilons = []
    losses = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_losses = []

        for time_step in range(500):  # max steps
            action = agent.act(state)

            next_state, reward, done, truncated, _ = env.step(action)

            if not done:
                reward = reward
            else:
                reward = -10  # penalty for messing up

            # save as experience
            agent.remember(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            if done or truncated:
                break

        # target model should be updated regularely
        if episode % update_target_every == 0:
            agent.update_target_model()

        # tracking progress
        scores.append(total_reward)
        epsilons.append(agent.epsilon)

        # print progress
        if episode % 10 == 0:
            avg_score = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)

            print(
                f"Episodes: {episode}/{n_episodes}, Score: {total_reward}",
                f"Avg Score (last 50): {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}",
            )

        # Save model periodically
        if episode % 50 == 0:
            agent.save_model(os.path.join(output_dir, f"model_{episode}.pth"))

        # Early stopping if solved (CartPole is considered solved at 195+ avg)
        if len(scores) >= 100 and np.mean(scores[-100:]) >= 195:
            print(
                f"🎉 Solved at episode {episode}! Average score: {np.mean(scores[-100:]):.2f}"
            )
            agent.save_model(os.path.join(output_dir, "solved_model.pth"))
            break

    # Save final model
    agent.save_model(os.path.join(output_dir, "final_model.pth"))

    # Plot results
    plot_training_results(scores, epsilons, output_dir)

    env.close()
    return agent, scores


def plot_training_results(scores, epsilons, output_dir):
    """Plot training progress"""
    plt.figure(figsize=(15, 5))

    # Plot scores
    plt.subplot(1, 3, 1)
    plt.plot(scores)
    plt.title("Training Scores")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.grid(True)

    # Plot moving average
    plt.subplot(1, 3, 2)
    window = 50
    moving_avg = [
        np.mean(scores[max(0, i - window) : i + 1]) for i in range(len(scores))
    ]
    plt.plot(moving_avg)
    plt.title(f"Moving Average (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Average Score")
    plt.grid(True)

    # Plot epsilon decay
    plt.subplot(1, 3, 3)
    plt.plot(epsilons)
    plt.title("Epsilon Decay")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_results.png"))
    plt.show()


if __name__ == "__main__":
    trained_agent, scores = train_cartpole()
