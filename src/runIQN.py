#-*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import random
import gym

from Agent import IQNAgent
from collections import deque
# from matplotlib import pyplot as plt

seed = 0
np.random.seed(seed)
random.seed(seed)

if __name__ == '__main__':
    '''
    define env. and agent
    '''
    env = gym.make('CartPole-v1')
    # env = gym.make('FrozenLake-v0')
    obs_space = env.observation_space
    print('Observation space')
    print(type(obs_space))
    print(obs_space.shape)
    print()

    act_space = env.action_space
    print('Action space')
    print(type(act_space))
    print("Total {} actions".format(act_space.n))
    print()

    env.seed(seed)
    max_t = env.spec.max_episode_steps

    ''' AGENT '''
    agent = IQNAgent(env.observation_space.high.shape[0],env.action_space.n,
                     N=8, N_target = 8, K = 32, n_embedding_dim = 64, huber_k=1, learning_rate=5e-5)
    # agent = IQNAgent(1, env.action_space.n, N=None, k=1, learning_rate=5e-4, hidden_unit_size=32)
    RETURN_MAX, LOSS_MAX = 2, 10.0

    ''' train agent '''
    avg_return_list = deque(maxlen=10)
    avg_loss_list = deque(maxlen=10)
    nepisodes = 2000
    step = 0

    MAX_STEP = 1000000
    episode = 0

    rewards_history = []
    loss_history = []

    # plt.style.use('ggplot')
    # plt.figure(figsize=(14,10))

    while (step < MAX_STEP):
        obs = env.reset()
        done = False
        total_reward = 0
        total_loss = 0
        episode_len = 0

        for t in range(max_t):
            episode_len += 1
            step += 1
            action = agent.get_action(obs)
            # action = agent.get_action([obs])
            next_obs, reward, done, info = env.step(action)

            agent.add_experience(obs,action,reward,next_obs,done)

            loss = agent.train_model()
            agent.update_policy()

            obs = next_obs
            total_reward += reward
            total_loss += loss

            if (step % 1000 == 0):
                ''' target network 업데이트 '''
                agent.update_target()

            # if (np.mean(avg_return_list) >= 0.6 and (done)) or ((done) and (episode % 1000 == 0) and step > 10000):
                # for act in range(4):
                    # _ = plt.subplot(2,4,act+5)
                    # _.cla()
                    # qs = agent.get_prediction([[obs]])
                    # plt.plot(qs[0][act], 'o-', color='green', alpha=0.8)
                    # plt.axis([0, agent.N, np.min(qs[0][act]), np.max(qs[0][act])])
                    # plt.xlabel('quantile')
                    # plt.ylabel('val')
                    # plt.draw()
                    # plt.tight_layout()
                    # plt.pause(0.02)

            if done:
                episode += 1
                rewards_history.append(total_reward)
                loss_history.append(total_loss)
                print(' [{:5d}/{:5d}] eps={:.3f} epi={:4d}, epi_len={:3d}, reward={:.3f}, loss={:.5f}').format(step, MAX_STEP, agent.epsilon, episode, episode_len, total_reward, total_loss)
                # if (episode % 10 == 0):
                #     plt.hold()
                #     plt.subplot(2,4,(1,2))
                #     plt.plot(range(0, len(rewards_history)), rewards_history, 'o', color='red', alpha=0.6)
                #     plt.axis([episode-50, episode+50, -1, RETURN_MAX])
                #     plt.xlabel('episode')
                #     plt.ylabel('returns')
                #     plt.draw()
                #     plt.subplot(2,4,(3,4))
                #     plt.plot(range(0, len(loss_history)), loss_history, 'o', color='blue', alpha=0.6)
                #     plt.axis([episode-50, episode+50, 0, LOSS_MAX])
                #     plt.xlabel('episode')
                #     plt.ylabel('loss')
                #     plt.draw()
                #     plt.tight_layout()
                #     plt.pause(0.05)
                break

        avg_return_list.append(total_reward)
        avg_loss_list.append(total_loss)

        if (np.mean(avg_return_list) >= 480):
            print('The problem is solved with {} episodes'.format(episode))

    ''' test agent '''
    env = gym.make('CartPole-v1')
    # env = gym.make('FrozenLake-v0')
    obs = env.reset()
    total_reward = 0
    frames = []
    for t in range(10000):
        # Render into buffer.
        frames.append(env.render())
        action = agent.get_action(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    env.close()

    raise NotImplementedError
