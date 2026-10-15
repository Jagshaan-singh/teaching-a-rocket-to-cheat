import sys
import gymnasium as gym
from stable_baselines3 import PPO
from wrappers import TimePenaltyWrapper, HoverBonusWrapper, ProgressWrapper

WRAPPERS = {
    "time_penalty": TimePenaltyWrapper,
    "hover_hack": HoverBonusWrapper,
    "fixed": ProgressWrapper,
}

name = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0

env = WRAPPERS[name](gym.make("LunarLander-v3"))
model = PPO("MlpPolicy", env, verbose=1, seed=seed, device="cpu")
model.learn(total_timesteps=1_000_000)
model.save(f"models/{name}_s{seed}")
env.close()