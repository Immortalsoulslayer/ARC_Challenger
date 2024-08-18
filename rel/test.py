import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

# Create the BipedalWalker environment
env = gym.make("BipedalWalker-v3")

# Create action noise for exploration
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Initialize the TD3 model
model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)

# Train the model
model.learn(total_timesteps=200000)

# Test the trained model
obs, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done or truncated:
        obs, info = env.reset()

env.close()
