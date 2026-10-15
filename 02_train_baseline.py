import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("LunarLander-v3")
model = PPO("MlpPolicy", env, verbose=1, seed=0, device="cpu")
model.learn(total_timesteps=1_000_000)
model.save("models/baseline")
env.close()