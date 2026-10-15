import sys
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

path = sys.argv[1]
name = sys.argv[2]

env = gym.make("LunarLander-v3", render_mode="rgb_array")
env = RecordVideo(env, video_folder="videos", name_prefix=name,
                  episode_trigger=lambda ep: True)   # record every attempt
model = PPO.load(path)

for _ in range(3):
    obs, info = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
env.close()