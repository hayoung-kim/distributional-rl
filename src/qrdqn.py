#-*- coding: utf-8 -*-

import gym
import random
import numpy as np
import tensorflow as tf

from collections import deque

from matplotlib import pyplot as plt


seed = 1
np.random.seed(seed)
random.seed(seed)

def huber_loss(x, delta=1.0):
  """Apply the function:
  ```
  0.5*x^2 if |x| < delta else delta*(|x| - 0.5*delta)
  ```
  """
  abs_x = tf.abs(x)
  return tf.where(
    abs_x < delta,
    tf.square(x) * 0.5,
    delta * (abs_x - 0.5 * delta)
  )

class PrioritizedReplayMemory(object): #PER memory
    def __init__(self, memory_size=10000, per_alpha=0.2, per_beta0=0.4):
        self.memory = SumTree(capacity=memory_size) # Use sumtree
        self.memory_size = memory_size

        # hyperparameter for importance probability
        self.per_alpha = per_alpha

        # hyperparameter for importance weight
        self.per_beta0 = per_beta0
        self.per_beta = per_beta0

        self.per_epsilon = 1E-6
        self.prio_max = 0

    def anneal_per_importance_sampling(self, step, max_step): # Anneal beta
        self.per_beta = self.per_beta0 + step*(1-self.per_beta0)/max_step

    def error2priority(self, errors): # Get priority from TD error
        return np.power(np.abs(errors) + self.per_epsilon, self.per_alpha)

    def save_experience(self, state, action, reward, state_next, done):
        experience = (state, action, reward, state_next, done)
        self.memory.add(np.max([self.prio_max, self.per_epsilon]), experience) # Add experience with importance weight

    def retrieve_experience(self, batch_size):
        idx = None
        priorities = None
        w = None

        idx, priorities, experience = self.memory.sample(batch_size) # Sample batch from memory
        sampling_probabilities = priorities / self.memory.total() # Make priorities to be sum to one
        w = np.power(self.memory.n_entries * sampling_probabilities, -self.per_beta) # Importance weight
        w = w / w.max()
        return idx, priorities, w, experience

    def update_experience_weight(self, idx, errors ):
        priorities = self.error2priority(errors)
        for i in range(len(idx)):
            self.memory.update(idx[i], priorities[i])
        self.prio_max = max(priorities.max(), self.prio_max)

class SumTree(object): # Sum Tree Memory
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

        self.write = 0
        self.n_entries = 0

        self.tree_len = len(self.tree)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1

        if left >= self.tree_len:
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            right = left + 1
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1

        return idx, self.tree[idx], self.data[data_idx]

    def sample(self, batch_size):
        batch_idx = [None] * batch_size
        batch_priorities = [None] * batch_size
        batch = [None] * batch_size
        segment = self.total() / batch_size

        a = [segment*i for i in range(batch_size)]
        b = [segment * (i+1) for i in range(batch_size)]
        s = np.random.uniform(a, b)

        for i in range(batch_size):
            (batch_idx[i], batch_priorities[i], batch[i]) = self.get(s[i])

        return batch_idx, batch_priorities, batch

class QRDQNPERAgent(object):
    def __init__(self, observation_dim, n_actions, N, k, seed=0,
                 discount_factor = 0.995, epsilon_decay = 0.999, epsilon_min = 0.01,
                 learning_rate = 1e-4, # STEP SIZE
                 batch_size = 32,
                 memory_size = 15000, hidden_unit_size = 128):

        self.seed = seed
        self.observation_dim = observation_dim
        self.n_actions = n_actions
        self.N = N
        self.k = k           #

        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.train_start = 10000

        self.memory = PrioritizedReplayMemory(memory_size=memory_size)

        self.hidden_unit_size = hidden_unit_size

        self.g = tf.Graph()
        with self.g.as_default():
            self.build_placeholders()
            self.build_model()
            self.build_loss()
            self.build_update_operation()
            self.init_session()

    def build_placeholders(self):
        self.observation_ph   = tf.placeholder(tf.float32, (None, self.observation_dim), 'observation')
        # self.quantile_diff_ph = tf.placeholder(tf.float32, (None, self.N, self.N), 'quantile_differences')    # quantile_difference matrix between quantiles_target and quantiles_pred
        self.quantile_target_ph = tf.placeholder(tf.float32, (None, self.N), 'quantile_target')
        # self.batch_weight_ph  = tf.placeholder(tf.float32, (None, self.n_actions), name='batch_weights')
        self.actions_ph = tf.placeholder(tf.int32, (None), name='action_selected')
        self.batch_weight_ph  = tf.placeholder(tf.float32, (None, ), name='batch_weights')
        self.learning_rate_ph = tf.placeholder(tf.float32, (), 'lr')

    def build_model(self): # Build networks
        hid1_size = self.hidden_unit_size
        hid2_size = self.hidden_unit_size
        hid3_size = self.hidden_unit_size

        with tf.variable_scope('predction_network'): # Prediction Network / Two layered perceptron / Training Parameters
            out = tf.layers.dense(self.observation_ph, hid1_size, tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='hidden1')
            out = tf.layers.dense(out, hid2_size, tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='hidden2')
            quantiles = tf.layers.dense(out, self.n_actions * self.N, # Linear Layer
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='predicted_quantiles')
            self.quantiles_pred = tf.reshape(quantiles, [-1, self.n_actions, self.N])
            self.q_pred = tf.reduce_mean(self.quantiles_pred, axis=-1, name='q_pred')


        with tf.variable_scope('target_network'): # Target Network / Two layered perceptron / Old Parameters
            out = tf.layers.dense(self.observation_ph, hid1_size, tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='hidden1')
            out = tf.layers.dense(out, hid2_size, tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='hidden2')
            quantiles = tf.layers.dense(out, self.n_actions * self.N, # Linear Layer
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='predicted_quantiles')
            self.quantiles_pred_old = tf.reshape(quantiles, [-1, self.n_actions, self.N])
            self.q_pred_old = tf.reduce_mean(self.quantiles_pred_old, axis=-1, name='q_pred')

        self.weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='predction_network') # Get Prediction network's Parameters
        self.weights_old = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_network') # Get Target network's Parameters

    def build_loss(self):
        '''
        quantile_loss = quantile_diff(theta_j - theta_i) -> huber loss -> E_j() -> sum_i
        '''
        quantile_target      = self.quantile_target_ph    # out: [None, N]
        quantile_pred_action = self.quantiles_pred        # out: [None, n_actions, N]
        actions              = self.actions_ph

        action_mask   = tf.one_hot(actions, self.n_actions, dtype=tf.float32)  # [None, n_actions]
        action_mask   = tf.expand_dims(action_mask, axis=-1) # [None, n_actions, 1]
        quantile_pred = tf.reduce_sum(quantile_pred_action * action_mask, axis=1) # [None, N]

        # compute mid-quantiles
        mid_quantiles = (np.arange(0, self.N, 1, dtype=np.float64) + 0.5) / float(self.N)
        mid_quantiles = np.asarray(mid_quantiles, dtype=np.float32)
        mid_quantiles = tf.constant(mid_quantiles[None, None, :], dtype=tf.float32)

        # target
        quantile_diff = tf.expand_dims(quantile_target, axis=-2) - tf.expand_dims(quantile_pred, axis=-1)  # [None, N, N]

        # compute quantile penalty weights
        indicator_fn     = tf.to_float(quantile_diff < 0.0)
        quantile_weights = tf.abs(mid_quantiles - indicator_fn)
        quantile_weights = tf.stop_gradient(quantile_weights)

        # Quantile Regression Loss
        if self.k == 0:
            quantile_loss = quantile_weights * quantile_diff
        else:
            _huber_loss = huber_loss(quantile_diff, delta=self.k)
            quantile_loss = quantile_weights * _huber_loss   # out: [None, n_actions, N]

        quantile_loss = tf.reduce_mean(quantile_loss, axis=-1)     # E_j(), out: [None, N]
        self.errors   = tf.reduce_sum(quantile_loss, axis=-1)      # sum_i(), out: [None]
        self.loss     = tf.reduce_mean(tf.multiply(self.batch_weight_ph, self.errors))               # PRIORITIZED, out: []
        self.optim    = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=0.01/32).minimize(self.loss) # optimizer

    def build_update_operation(self):
        update_ops = []
        for var, var_old in zip(self.weights, self.weights_old):
            update_ops.append(var_old.assign(var))
        self.update_ops = update_ops

    def init_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config,graph=self.g)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.update_ops)

    def update_target(self):
        self.sess.run(self.update_ops)

    def update_memory(self, step, max_step):
        self.memory.anneal_per_importance_sampling(step,max_step)

    def update_policy(self):
        if (self.epsilon > self.epsilon_min) and (self.memory.memory.n_entries > self.train_start):
            self.epsilon *= self.epsilon_decay

    def get_prediction_old(self, obs):
        quantiles_pred_old = self.sess.run(self.quantiles_pred_old,feed_dict={self.observation_ph:obs})
        return quantiles_pred_old

    def get_prediction(self, obs):
        quantiles_pred = self.sess.run(self.quantiles_pred,feed_dict={self.observation_ph:obs})
        return quantiles_pred

    def get_action(self, obs):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.n_actions)
        else:
            quantiles = self.get_prediction([obs]) # out: [1, n_actions, N]
            q_value = np.mean(quantiles, axis=-1)  # out: [1, n_actions]
            return np.argmax(q_value[0])

    def add_experience(self, obs, action, reward, next_obs, done):
        self.memory.save_experience(obs, action, reward, next_obs, done)

    def train_model(self):
        loss = np.nan

        n_entries = self.memory.memory.n_entries

        if n_entries >= self.train_start:

            # PRIORITIZED EXPERIENCE REPLAY
            idx, priorities, w, mini_batch = self.memory.retrieve_experience(self.batch_size)
            # batch_weights = np.transpose(np.tile(w, (self.n_actions, 1)))
            batch_weights = w

            observations = np.zeros((self.batch_size, self.observation_dim))
            next_observations = np.zeros((self.batch_size, self.observation_dim))
            actions, rewards, dones = [], [], []

            for i in range(self.batch_size):
                observations[i] = mini_batch[i][0]
                actions.append(mini_batch[i][1])
                rewards.append(mini_batch[i][2])
                next_observations[i] = mini_batch[i][3]
                dones.append(mini_batch[i][4])

            # calculate Q(s',a')
            quantiles_pred_old, Q = self.sess.run([self.quantiles_pred_old, self.q_pred_old],
                                                    feed_dict={self.observation_ph:next_observations})
            best_actions = np.argmax(Q, axis=-1)   # best actions for each batch, out: [None]
            quantile_target = np.zeros((self.batch_size, self.N))

            # BELLMAN UPDATE RULE
            for i in range(self.batch_size):
                # print('i', dones[i])
                if dones[i]:
                    quantile_target[i] = rewards[i] * np.ones(self.N)

                else:
                    best_action = best_actions[i]
                    quantile_target[i] = rewards[i] + self.discount_factor * quantiles_pred_old[i, best_action]

            n_REPEAT_TRAIN = 1
            for _n_train in range(n_REPEAT_TRAIN):
                loss, errors, _ = self.sess.run([self.loss, self.errors, self.optim],
                                     feed_dict={self.observation_ph: observations,
                                                self.actions_ph: actions,
                                                self.quantile_target_ph: quantile_target,
                                                self.learning_rate_ph: self.learning_rate,
                                                self.batch_weight_ph: batch_weights})
            # errors = errors[np.arange(len(errors)), actions]

            self.memory.update_experience_weight(idx, errors)

        return loss

if __name__ == '__main__':
    '''
    define env. and agent
    '''
    # env = gym.make('CartPole-v1')
    env = gym.make('FrozenLake-v0')
    obs_space = env.observation_space
    print('Observation space')
    print(type(obs_space))
    print(obs_space.shape)
    # print("Dimension:{}".format(obs_space.shape[0]))
    # print("Dimension:{}".format(obs_space.shape))
    # print("High: {}".format(obs_space.high))
    # print("Low: {}".format(obs_space.low))
    print()

    act_space = env.action_space
    print('Action space')
    print(type(act_space))
    print("Total {} actions".format(act_space.n))
    print()

    env.seed(seed)
    max_t = env.spec.max_episode_steps

    ''' AGENT '''
    # agent = QRDQNPERAgent(env.observation_space.high.shape[0],env.action_space.n, N=10, k=1, learning_rate=5e-5)
    agent = QRDQNPERAgent(1,env.action_space.n, N=10, k=1, learning_rate=5e-4, hidden_unit_size=32)
    RETURN_MAX, LOSS_MAX = 2, 10.0

    '''
    train agent
    '''
    avg_return_list = deque(maxlen=10)
    avg_loss_list = deque(maxlen=10)
    nepisodes = 2000
    step = 0

    MAX_STEP = 1000000
    episode = 0

    rewards_history = []
    loss_history = []

    plt.style.use('ggplot')
    plt.figure(figsize=(14,10))

    while (step < MAX_STEP):
        obs = env.reset()
        done = False
        total_reward = 0
        total_loss = 0
        episode_len = 0

        for t in range(max_t):
            episode_len += 1
            step += 1
            # action = agent.get_action(obs)
            action = agent.get_action([obs])
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

            if (np.mean(avg_return_list) >= 0.6 and (done)) or ((done) and (episode % 1000 == 0) and step > 10000):
                for act in range(4):
                    _ = plt.subplot(2,4,act+5)
                    _.cla()
                    qs = agent.get_prediction([[obs]])
                    plt.plot(qs[0][act], 'o-', color='green', alpha=0.8)
                    plt.axis([0, agent.N, np.min(qs[0][act]), np.max(qs[0][act])])
                    plt.xlabel('quantile')
                    plt.ylabel('val')
                    plt.draw()
                    plt.tight_layout()
                    plt.pause(0.02)


            if done:
                episode += 1
                rewards_history.append(total_reward)
                loss_history.append(total_loss)
                print(' [{:5d}/{:5d}] eps={:.3f} epi={:4d}, epi_len={:3d}, reward={:.3f}, loss={:.5f}').format(step, MAX_STEP, agent.epsilon, episode, episode_len, total_reward, total_loss)
                if (episode % 10 == 0):
                    plt.hold()
                    plt.subplot(2,4,(1,2))
                    plt.plot(range(0, len(rewards_history)), rewards_history, 'o', color='red', alpha=0.6)
                    plt.axis([episode-50, episode+50, -1, RETURN_MAX])
                    plt.xlabel('episode')
                    plt.ylabel('returns')
                    plt.draw()
                    plt.subplot(2,4,(3,4))
                    plt.plot(range(0, len(loss_history)), loss_history, 'o', color='blue', alpha=0.6)
                    plt.axis([episode-50, episode+50, 0, LOSS_MAX])
                    plt.xlabel('episode')
                    plt.ylabel('loss')
                    plt.draw()
                    plt.tight_layout()
                    plt.pause(0.05)
                break


        avg_return_list.append(total_reward)
        avg_loss_list.append(total_loss)

        if (np.mean(avg_return_list) >= 0.7):
            print('The problem is solved with {} episodes'.format(episode))
            # print('estimated quantiles:')
            # print(agent.get_prediction([obs]))
            # plt.hold()
            # plt.subplot(2,4,(1,2))
            # plt.plot(range(0, len(rewards_history)), rewards_history, '.-', color='red', alpha=0.6)
            # plt.axis([episode-2500, episode+2500, 0, RETURN_MAX])
            # plt.xlabel('episode')
            # plt.ylabel('returns')
            # plt.draw()
            # plt.subplot(2,4,(3,4))
            # plt.plot(range(0, len(loss_history)), loss_history, '.-', color='blue', alpha=0.6)
            # plt.axis([episode-2500, episode+2500, 0, LOSS_MAX])
            # plt.xlabel('episode')
            # plt.ylabel('loss')
            # plt.draw()
            # plt.tight_layout()
            # plt.show()
            # break


    '''
    test agent
    '''
    # env = gym.make('CartPole-v1')
    env = gym.make('FrozenLake-v0')
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
