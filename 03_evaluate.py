import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

path = sys.argv[1] if len(sys.argv) > 1 else "models/baseline"

env = gym.make("LunarLander-v3")
model = PPO.load(path)
mean, std = evaluate_policy(model, env, n_eval_episodes=20, deterministic=True)
print(f"{path}: {mean:.1f} +/- {std:.1f}  (200+ means solved)")
env.close()