import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make('Humanoid-v4')

model = PPO('MlpPolicy', env, verbose=1)

# train the agent for 100000 timesteps
model.learn(total_timesteps=100000)

model.save("ppo_humanoid")

# Use this if you want to load the model and use it later
# model = PPO.load("ppo_humanoid")
