import numpy as np
import gymnasium as gym


def is_upright_and_slow(obs):
    # obs: 0=x, 1=y, 2=vx, 3=vy, 4=tilt, 5=spin, 6/7=legs
    speed = np.sqrt(obs[2] ** 2 + obs[3] ** 2)
    return abs(obs[4]) < 0.2 and speed < 0.3


class TimePenaltyWrapper(gym.Wrapper):
    """v1: tiny fee every frame. Should still learn to land."""
    def __init__(self, env, fee=0.02):
        super().__init__(env)
        self.fee = fee

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward - self.fee, terminated, truncated, info


class HoverBonusWrapper(gym.Wrapper):
    """v2: the hack. +1 per frame for being upright and slow."""
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if is_upright_and_slow(obs):
            reward += 1.0
        return obs, reward, terminated, truncated, info


class ProgressWrapper(gym.Wrapper):
    """v3: the fix. Pay for the CHANGE in 'upright and slow', plus a per-frame fee."""
    def __init__(self, env, fee=0.02):
        super().__init__(env)
        self.fee = fee
        self.prev = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev = 1.0 if is_upright_and_slow(obs) else 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        now = 1.0 if is_upright_and_slow(obs) else 0.0
        reward += (now - self.prev) - self.fee
        self.prev = now
        return obs, reward, terminated, truncated, info