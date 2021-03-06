#-*- coding: utf-8 -*-
import random
import numpy as np
import tensorflow as tf

from collections import deque
from matplotlib import pyplot as plt

from helper import huber_loss
from replayMemory import PrioritizedReplayMemory

class IQNAgent(object):
    def __init__(self, observation_dim, n_actions,
                 N = 8, N_target = 8, K = 32, huber_k = 1, n_embedding_dim = 64, seed=0,
                 discount_factor = 0.995, epsilon_decay = 0.999, epsilon_min = 0.01,
                 learning_rate = 1e-4, # STEP SIZE
                 batch_size = 32,
                 memory_size = 15000, hidden_unit_size = [64, 64]):

        self.seed = seed
        self.observation_dim = observation_dim
        self.n_actions = n_actions
        self.N = N     # prediction network에서 출력할 샘플 수 (N)
        self.N_target = N_target   # target network에서 출력할 샘플 수 (N')
        self.K = K     # policy 계산시 사용할 샘플 수
        self.huber_k = huber_k     # huber loss 차원
        self.n_embedding_dim = n_embedding_dim


        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.train_start = 500

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
        ''' 네트워크 관련 placeholders '''
        self.tau_ph = tf.placeholder(tf.float32, (None, self.N))    # samples의 갯수 [batch, N]
        self.observation_ph   = tf.placeholder(tf.float32, (None, self.observation_dim), 'observation')

        ''' loss 계산 관련 placeholders '''
        self.actions_ph = tf.placeholder(tf.int32, (None), name='action_selected')  # (s,a,s',r) 중 a
        self.td_target_ph = tf.placeholder(tf.float32, (None, self.N), 'td_target') # sample당 td_errors
        self.batch_weight_ph  = tf.placeholder(tf.float32, (None, ), name='batch_weights') # prioritized experience replay
        self.learning_rate_ph = tf.placeholder(tf.float32, (), 'lr') # learning rate

    def build_model(self):
        hidden_size = self.hidden_unit_size
        n_embedding_dim = self.n_embedding_dim

        n_layers = len(hidden_size)

        tau = tf.transpose(tf.expand_dims(self.tau_ph, axis=0), [2,1,0])     # out: [N, batch, 1]

        with tf.variable_scope('predction_network'):
            ''' states to feature vector '''
            out = tf.layers.dense(self.observation_ph, hidden_size[0], tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.seed), name='hidden1')
            for i in range(1, n_layers):
                out = tf.layers.dense(out, hidden_size[i], tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.seed), name='hidden%d' % (i+1))

            ''' random sample embedding '''
            ''' 원래는 cos basis로 해야하지만 귀찮으므로 linear embedding'''
            embedding = tf.layers.dense(tau, n_embedding_dim, tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.seed), name='random_to_embedding')
            embedding = tf.layers.dense(embedding, hidden_size[-1], tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.seed), name='embedding_out')

            ''' embedding * feature vector'''
            f = tf.multiply(out, embedding, name='f')
            samples_per_action = tf.layers.dense(f, self.n_actions, None, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.seed), name='samples')   # out: [N, batch, n_actions]
            self.samples_per_action_pred = tf.transpose(samples_per_action, [1,2,0])    # out: [batch, n_actions, N]

        with tf.variable_scope('target_network'):
            ''' states to feature vector '''
            out = tf.layers.dense(self.observation_ph, hidden_size[0], tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.seed), name='hidden1')
            for i in range(1, n_layers):
                out = tf.layers.dense(out, hidden_size[i], tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.seed), name='hidden%d' % (i+1))

            ''' random sample embedding '''
            ''' 원래는 cos basis로 해야하지만 귀찮으므로 linear embedding'''
            embedding = tf.layers.dense(tau, n_embedding_dim, tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.seed), name='random_to_embedding')
            embedding = tf.layers.dense(embedding, hidden_size[-1], tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.seed), name='embedding_out')

            ''' embedding * feature vector'''
            f = tf.multiply(out, embedding, name='f')
            samples_per_action = tf.layers.dense(f, self.n_actions, None, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=self.seed), name='samples')   # out: [N, batch, n_actions]
            self.samples_per_action_target = tf.transpose(samples_per_action, [1,2,0])    # out: [batch, n_actions, N]

        self.weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='predction_network') # Get Prediction network's Parameters
        self.weights_old = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_network') # Get Target network's Parameters

    def build_loss(self): # build loss function for IQNAgent
        ''' td prediction 계산 '''
        pred_action = self.samples_per_action_pred        # out: [None, n_actions, N]
        actions     = self.actions_ph

        action_mask   = tf.one_hot(actions, self.n_actions, dtype=tf.float32)   # [None, n_actions]
        action_mask   = tf.expand_dims(action_mask, axis=-1)    # [None, n_actions, 1]
        td_prediction = tf.reduce_sum(pred_action * action_mask, axis=1)    # [None, N]

        ''' td error 계산 '''
        td_target = self.td_target_ph    # out: [None, N]
        td_error  = tf.expand_dims(td_target, axis=-2) - tf.expand_dims(td_prediction, axis=-1)  # [None, N, N]

        ''' quantile weight 계산'''
        tau = self.tau_ph   # [None, self.N]
        indicator_fn     = tf.to_float(td_error < 0.0)   # [None, N, N]
        quantile_weights = tf.abs(tf.expand_dims(tau, axis=-1) - indicator_fn)
        quantile_weights = tf.stop_gradient(quantile_weights)

        ''' huber quantile loss 계산 '''
        if self.huber_k == 0:
            quantile_loss = quantile_weights * td_error
        else:
            _huber_loss = huber_loss(td_error, delta=self.huber_k)
            quantile_loss = quantile_weights * _huber_loss   # out: [None, N, N]

        quantile_loss = tf.reduce_mean(quantile_loss, axis=-1)     # target (j)에 대해 평균치 구하기, out: [None, N]
        self.errors   = tf.reduce_sum(quantile_loss, axis=-1)      # prediction에 대해 합(=quantile weighted 평균), out: [None]
        self.loss     = tf.reduce_mean(tf.multiply(self.batch_weight_ph, self.errors))      # PRIORITIZED, out: []
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
        ''' target network 업데이트'''
        self.sess.run(self.update_ops)

    def update_memory(self, step, max_step):
        self.memory.anneal_per_importance_sampling(step,max_step)

    def update_policy(self):
        ''' epsilon greedy '''
        if (self.epsilon > self.epsilon_min) and (self.memory.memory.n_entries > self.train_start):
            self.epsilon *= self.epsilon_decay

    def get_prediction_old(self, obs):
        samples_per_action = self.sess.run(self.samples_per_action_target,feed_dict={self.observation_ph:obs})
        return samples_per_action

    def get_prediction(self, obs):
        ''' N개의 random sample을 이용해서 action별 reward sample 뽑기 '''
        tau = np.random.rand(self.N)
        samples_per_action = self.sess.run(self.samples_per_action_pred, feed_dict={self.observation_ph:obs, self.tau_ph:[tau]})
        return samples_per_action

    def get_action(self, obs):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.n_actions)
        else:
            samples = self.get_prediction([obs]) # out: [1, n_actions, N]
            q_value = np.mean(samples, axis=-1)  # out: [1, n_actions]
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

            ''' random sample 만들고 prediction, target network 결과 추출 '''
            # 아직 risk 고려해서 sampling 안함. 앞으로 코드에 추가해야할 일

            for k in range(self.K // self.N):
                tau = np.random.rand(self.batch_size, self.N)  # 32개의 sample 뽑기
                samples_from_target = self.sess.run(self.samples_per_action_target, feed_dict={self.observation_ph: next_observations, self.tau_ph: tau})   # out: [batch, n_actions, N]
                if k == 0:
                    Z_tau_pred = np.array(samples_from_target)
                else:
                    Z_tau_pred = np.concatenate((Z_tau_pred, samples_from_target), axis=-1)

            ''' Q(s',a')을 최대화 하는 a' 찾기 '''
            Q = np.mean(Z_tau_pred, axis=-1)   # out: [batch, n_actions]
            best_actions = np.argmax(Q, axis=-1)        # best actions for each batch, out: [None]

            ''' TD target 생성 [None, N]'''
            N = 8
            tau = np.random.rand(self.batch_size, N)
            samples_from_target = self.sess.run(self.samples_per_action_pred,
                                                feed_dict={self.observation_ph: next_observations,
                                                           self.tau_ph: tau})   # out: [batch, n_actions, N]

            td_target = np.zeros((self.batch_size, N))

            # BELLMAN UPDATE RULE
            for i in range(self.batch_size):
                if dones[i]:
                    td_target[i] = rewards[i] * np.ones(N)

                else:
                    best_action = best_actions[i]
                    td_target[i] = rewards[i] + self.discount_factor * samples_from_target[i, best_action]

            ''' 최적화 '''
            tau = np.random.rand(self.batch_size, N)
            loss, errors, _ = self.sess.run([self.loss, self.errors, self.optim],
                                 feed_dict={self.observation_ph: observations,
                                            self.actions_ph: actions,
                                            self.td_target_ph: td_target,
                                            self.tau_ph: tau,
                                            self.learning_rate_ph: self.learning_rate,
                                            self.batch_weight_ph: batch_weights})

            self.memory.update_experience_weight(idx, errors)

        return loss
