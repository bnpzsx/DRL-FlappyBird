from collections import deque
from time import sleep

import cv2
import gym
import gym_ple
import numpy as np
import paddle.fluid as fluid
import parl
from gym.wrappers import Monitor
from parl.algorithms import DQN
from parl.utils import logger, summary
from tqdm import tqdm

from agent import Agent
from model import Model
from replay_memory import ReplayMemory

CONTEXT_LEN = 4
FRAME_SKIP = 4
IMAGE_SIZE = (80, 80)
MEMORY_SIZE = int(1e4)
MEMORY_WARMUP_SIZE = MEMORY_SIZE // 20
BATCH_SIZE = 64
UPDATE_FREQ = 5
GAMMA = 0.99
LEARNING_RATE = 1e-3
E_GREED = 0.3
E_GREED_DECREMENT = 3e-6

train_total_steps = int(1e5)
test_every_steps = 2000

def preprocess(image):
    image = cv2.resize(image, IMAGE_SIZE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.expand_dims(image, axis=0)
    return image / 255

def env_reset(env, action=0, render=False):
    global recent_obs
    recent_obs = deque(maxlen=CONTEXT_LEN)
    recent_obs.append(preprocess(env.reset()))
    for i in range(CONTEXT_LEN - 1):
        next_obs, _, _, _ = env.step(action)
        recent_obs.append(preprocess(next_obs))
        if render:
            env.render()
            sleep(0.03)
    obs = list(recent_obs)
    return np.concatenate(obs)

def env_steps(env, action=0, render=False):
    global recent_obs
    total_reward = 0
    for i in range(FRAME_SKIP):
        next_obs, reward, isOver, info = env.step(action)
        total_reward += reward
        recent_obs.append(preprocess(next_obs))
        if render:
            env.render()
            sleep(0.03)
        if isOver:
            break
    obs = list(recent_obs)
    return np.concatenate(obs), total_reward, isOver, info

def run_train_episode(env, agent, rpm):
    total_reward = 0
    all_cost = []
    obs = env_reset(env, env.action_space.sample())
    steps = 0
    while True:
        steps += 1
        action = agent.sample(obs)
        next_obs, reward, isOver, _ = env_steps(env, action)
        rpm.append((obs, action, reward, next_obs, isOver))
        # start training
        if len(rpm) > MEMORY_WARMUP_SIZE:
            if steps % UPDATE_FREQ == 0:
                (batch_obs, batch_action, batch_reward, batch_next_obs,
                 batch_isOver) = rpm.sample(BATCH_SIZE)
                cost = agent.learn(batch_obs, batch_action, batch_reward,
                                   batch_next_obs, batch_isOver)
                all_cost.append(float(cost))
        total_reward += reward
        obs = next_obs
        if isOver:
            break
    if all_cost:
        logger.info('[Train]total_reward: {}, mean_cost: {}'.format(
            total_reward, np.mean(all_cost)))
    return total_reward, steps, np.mean(all_cost)

def run_evaluate_episode(env, agent, render=False):
    obs = env_reset(env)
    total_reward = 0
    while True:
        action = agent.predict(obs)
        obs, reward, isOver, _ = env_steps(env, action)
        total_reward += reward
        if render:
            env.render()
        if isOver:
            break
    return total_reward


def main():
    env = gym.make('FlappyBird-v0')
    test_env = Monitor(env, directory='test', video_callable=lambda x: True, force=True)
    rpm = ReplayMemory(MEMORY_SIZE)
    act_dim = env.action_space.n

    model = Model(act_dim)
    algorithm = DQN(model, act_dim=act_dim, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(algorithm,
                  act_dim=act_dim,
                  e_greed=E_GREED,
                  e_greed_decrement=E_GREED_DECREMENT)

    # 加载模型
    #save_path = './dqn_model.ckpt'
    #agent.restore(save_path)

    with tqdm(total=MEMORY_WARMUP_SIZE, desc='[Replay Memory Warm Up]') as pbar:
        while len(rpm) < MEMORY_WARMUP_SIZE:
            total_reward, steps, _ = run_train_episode(env, agent, rpm)
            pbar.update(steps)

    # train
    best_reward = -5
    pbar = tqdm(total=train_total_steps)
    test_flag = 0
    total_steps = 0
    while total_steps < train_total_steps:
        # start epoch
        total_reward, steps, loss = run_train_episode(env, agent, rpm)
        total_steps += steps
        pbar.set_description('[train]exploration:{}'.format(agent.e_greed))
        summary.add_scalar('dqn/score', total_reward, total_steps)
        summary.add_scalar('dqn/loss', loss, total_steps)  # mean of total loss
        summary.add_scalar('dqn/exploration', agent.e_greed, total_steps)
        pbar.update(steps)

        if total_steps // test_every_steps >= test_flag:
            while total_steps // test_every_steps >= test_flag:
                test_flag += 1
            pbar.write("testing")
            eval_rewards = []
            for _ in tqdm(range(3), desc='eval agent'):
                eval_reward = run_evaluate_episode(test_env, agent)
                eval_rewards.append(eval_reward)
            logger.info(
                "eval_agent done, (steps, eval_reward): ({}, {})".format(
                    total_steps, np.mean(eval_rewards)))
            eval_test = np.mean(eval_rewards)
            summary.add_scalar('dqn/eval', eval_test, total_steps)
            if eval_test > best_reward:
                agent.save('./best_dqn_model.ckpt')
                best_reward = eval_test

    pbar.close()
    # 训练结束，保存模型
    save_path = './dqn_model.ckpt'
    agent.save(save_path)

if __name__ == '__main__':
    main()
