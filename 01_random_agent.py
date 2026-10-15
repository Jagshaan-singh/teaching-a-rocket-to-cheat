import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")
obs, info = env.reset(seed=0)

for i in range(500):
    action = env.action_space.sample()   # random pick from the 4 actions
    obs, reward, terminated, truncated, info = env.step(action)
    print(i, obs.round(2), round(reward, 2))
    if terminated or truncated:
        obs, info = env.reset()

env.close()