import gymnasium as gym
import numpy as np

env = gym.make('CarRacing-v2')

def collect_data(env, episodes=1000):
    frames = []
    for _ in range(episodes):
        observation, info = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()  # Random action
            action = np.array(action, dtype=np.float32)  # Ensure action is float32
            observation, reward, done, truncated, info = env.step(action)
            frames.append(observation)
    return frames

frames = collect_data(env)
