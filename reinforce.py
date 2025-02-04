import numpy as np
import gymnasium as gym
import random

#this is an implementation of reinforcement learning using gymnasium..


def main():
    # Initialize environment with ANSI rendering for compatibility
    env = gym.make('Taxi-v3', render_mode="ansi")

    # Q-table parameters
    state_size = env.observation_space.n  # Number of possible states
    action_size = env.action_space.n      # Number of possible actions
    qtable = np.zeros((state_size, action_size))

    # Hyperparameters (tuned for better training)
    learning_rate = 0.7    # Adjusted for more stable updates
    discount_rate = 0.95   # Increased to value long-term rewards more
    epsilon = 1.0          # Exploration rate
    decay_rate = 0.001     # Slower decay
    num_episodes = 5000    # More episodes for better learning
    max_steps = 99         # Max steps per episode

    # Training loop
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False

        for step in range(max_steps):
            # Exploration vs exploitation
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Random action
            else:
                action = np.argmax(qtable[state, :])  # Best known action

            # Take action (returns new state, reward, done flags, and info)
            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Update Q-table using Q-learning formula
            qtable[state, action] = qtable[state, action] + learning_rate * (
                reward + discount_rate * np.max(qtable[new_state, :]) - qtable[state, action]
            )

            # Move to new state
            state = new_state

            if done:
                break  # End episode if task is done

        # Decay exploration rate (exponential decay)
        epsilon = max(0.1, np.exp(-decay_rate * episode))  # Ensuring it doesn't go below 0.1

    print(f"Training completed over {num_episodes} episodes")

    # Test the trained agent
    input("Press Enter to watch the trained agent...")
    state, info = env.reset()
    done = False
    total_reward = 0

    for step in range(max_steps):
        action = np.argmax(qtable[state, :])  # Always use best action
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        print(f"Step {step + 1}")
        print(f"Total Reward: {total_reward}")
        print(env.render())  # Print environment state

        state = new_state
        if done:
            break

    env.close()

if __name__ == "__main__":
    main()
