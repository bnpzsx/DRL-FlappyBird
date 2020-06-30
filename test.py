from time import sleep

import cv2
import gym
import gym_ple
import numpy as np
from gym.wrappers import Monitor

from train import DQN, Agent, Model, env_reset, env_steps

env = gym.make('FlappyBird-v0')
act_dim = env.action_space.n
model = Model(act_dim)
algorithm = DQN(model, act_dim=act_dim, gamma=0., lr=0.)
agent = Agent(algorithm, act_dim=act_dim)

save_path = './dqn_model.ckpt'
agent.restore(save_path)

def run_evaluate_episode(env, agent, render=True):
    obs = env_reset(env, render=render)
    total_reward = 0
    while True:
        action = agent.predict(obs)
        obs, reward, isOver, _ = env_steps(env, action, render=render)
        total_reward += reward
        env.viewer.window.set_caption(f'得分: {total_reward}')
    return total_reward

if __name__ == "__main__":
    run_evaluate_episode(env, agent)
