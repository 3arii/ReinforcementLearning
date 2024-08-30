import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

# Create the environment using Gymnasium
env = gym.make('Humanoid-v4')

# Reset the environment to start a new episode
obs, info = env.reset()

# Create an empty list to store the trajectory
observations = []
actions = []

model = PPO.load("ppo_humanoid")

# Run the trained agent in the environment for a set number of steps
for _ in range(1000):  # or the desired number of steps
    # Predict the next action using the trained model
    action, _states = model.predict(obs)
    
    # Apply the action in the environment
    obs, reward, done, truncated, info = env.step(action)
    
    # Append the observation and action to their respective lists
    observations.append(obs)
    actions.append(action)
    
    # If the episode ends, break the loop
    if done or truncated:
        break

# Convert the lists to NumPy arrays
observations = np.array(observations)
actions = np.array(actions)

# Save the trajectory to a file
np.save("humanoid_observations.npy", observations)
np.save("humanoid_actions.npy", actions)