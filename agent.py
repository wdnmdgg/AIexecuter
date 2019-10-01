from env import Env
import shift
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import Q_function as Qf
from datetime import datetime


class LSTMAgent:

    def __init__(self,
                 sess_,
                 observations_dim,
                 action_space,
                 batch_size,
                 Q_function,
                 optimizer,
                 GAMMA,
                 EPSILON,
                 learning_rate=0.001):

        # Env setup
        self.sess = sess_
        # self.task_info = task_info
        # '''
        # task_info:  1. 'market_price0'
        #             2. 'limit_time'
        #             3. 'target_size'
        # '''
        self.GAMMA = GAMMA
        self.EPSILON = EPSILON
        self.tmp_buffer = []
        self.Q_function = Q_function
        self.training_times = 0
        # self.raw_obs = None

        # Initialize actions, observations and rewards
        self.actions_space = action_space
        self.batch_size = batch_size
        self.observations = tf.placeholder(dtype=tf.float32,
                                           shape=[None, observations_dim])
        self.observations1 = tf.placeholder(dtype=tf.float32,
                                            shape=[None, observations_dim])
        self.actions = tf.placeholder(dtype=tf.float32,
                                      shape=[None, action_space])
        self.rewards = tf.placeholder(dtype=tf.float32,
                                      shape=[None, 1])
        self.terminal = tf.placeholder(dtype=tf.float32,
                                       shape=[None, 1])
        self.episode_rewards = []
        self.discounted_episode_rewards = []

        # Whole structure setup
        """
        Observation_t and A_t concatenated as a whole state
        """
        self.q_hat = Q_function(self.observations, self.actions)
        self.optimizer = optimizer(learning_rate=learning_rate)

        self.obj = self.objective_function()
        self.loss = tf.losses.mean_squared_error(self.obj, self.q_hat)
        self.min_opt = self.optimizer.minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())

    def save_buffer(self, data_list, keepGoing):
        if keepGoing:
            self.tmp_buffer = []
            self.tmp_buffer.append(data_list[0])
            self.tmp_buffer.append(data_list[1])
        else:
            self.tmp_buffer.append(data_list[0])
            self.tmp_buffer.append(data_list[1])
            self.tmp_buffer.append(data_list[2])

    def get_buffer(self):
        return self.tmp_buffer

    # def obs_preprocessor(self):
    #     """
    #     :param raw_obs: a dict with a price vector 'order_book',
    #                                     and 2 scalar ('rt', 'rs')
    #     :return: 3D array [1, 1, n]
    #     """
    #     price_vec, remained_time, remained_size = self.raw_obs['order_book'] / self.task_info['market_price0'], \
    #                                               self.raw_obs['rt'] / self.task_info['limit_time'], \
    #                                               self.raw_obs['rs'] / self.task_info['target_size']
    #     res = np.concatenate([price_vec, [remained_time, remained_size]])
    #
    #     return [[res]]

    def act_preprocessor(self, num_act):
        """
        :param num_act: a scalar
        :return: 1D array
        """
        res = np.zeros(self.actions_space)
        res[num_act] = 1.0
        return res

    def objective_function(self):
        """
        :param r0:
        :param s1:
        :param terminal:
        :return:
        """

        tmp = []
        for raw_action in range(self.actions_space):
            action = tf.one_hot(raw_action, self.actions_space)
            action = tf.tile([action], [self.batch_size, 1])
            tmp_q = self.Q_function(self.observations1, action)
            tmp.append(tmp_q)
        tmp = tf.concat(tmp, axis=1)
        max_q = tf.math.reduce_max(tmp, axis=1, keepdims=True)
        res = self.rewards + self.terminal * self.GAMMA * max_q
        return res

    def train(self, obs_, act_, rwd_, obs_plus1_, terminal_):
        self.training_times += 1
        optimizer, loss = self.sess.run((self.min_opt, self.loss),
                                        feed_dict={self.observations: obs_,
                                                   self.actions: act_,
                                                   self.rewards: rwd_,
                                                   self.observations1: obs_plus1_,
                                                   self.terminal: terminal_})
        print('Training times:\t', self.training_times)
        print('Loss:\t', loss)

    # def discount_rewards(self, rewards):
    #     discount_rewards = np.zeros_like(rewards)
    #     running_add = 0
    #     for t in reversed(range(len(rewards))):
    #         running_add = running_add * self.GAMMA + rewards[t]
    #         discount_rewards[t] = running_add
    #
    #     mean = np.mean(discount_rewards)
    #     std = np.std(discount_rewards)
    #     discount_rewards = (discount_rewards - mean) / std
    #
    #     return discount_rewards

    def get_action(self, raw_obs):
        """
        :param raw_obs: 1D array
        :return: an int e.g. 8
        """
        start = datetime.now()
        prob = np.random.uniform(0, 1)
        if prob < self.EPSILON:
            action = np.random.randint(0, self.actions_space)
            act_ = self.act_preprocessor(action)
            self.save_buffer([raw_obs, act_], True)
            return action
        else:
            q_list = []
            for i in range(self.actions_space):
                actions = self.act_preprocessor(i)
                q = self.sess.run(self.q_hat,
                                  feed_dict={self.observations: [raw_obs],
                                             self.actions: [actions]})
                q_list.append(q[0][0])

            action = np.argmax(q_list)
            act_ = self.act_preprocessor(action)
            self.save_buffer([raw_obs, act_], True)
            print('Action:\t{}%'.format(100 * action / (self.actions_space - 1)))
            end = datetime.now()
            print('Time cost:\t{}'.format(end - start))
            return action  # little problem may happen here

    # def store_episode_hist(self, raw_obs, action):
    #     self.get_observation(raw_obs)
    #     obs, act = self.obs_preprocessor(), self.act_preprocessor(action)
    #     self.episode_hist['obs'].append(obs)
    #     self.episode_hist['act'].append(act)


sess = tf.Session()
obs = np.array([[1.2, 1.19, 1.17, 1.15, 1.1, 1.09],
                [1.23, 1.22, 1.2, 1.17, 1.13, 1.11],
                [1.1, 1.08, 1.02, 0.99, 0.97, 0.93]])
act = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
rwd = np.array([[1],
                [0.8],
                [0]])
obs_plus1 = np.array([[1.3, 1.21, 1.11, 1.03, 0.93, 0.83],
                      [1.2, 1.11, 1.01, 1.0, 0.99, 0.92],
                      [0.92, 0.91, 0.9, 0.83, 0.81, 0.8]])
terminal = np.array([[0],
                     [0],
                     [1]])

agent = LSTMAgent(sess_=sess,
                  observations_dim=6,
                  action_space=11,
                  batch_size=3,
                  Q_function=Qf.ann,
                  optimizer=tf.train.AdamOptimizer,
                  GAMMA=0.95,
                  EPSILON=0.1,
                  learning_rate=0.001)
for i in range(20):
    agent.train(obs, act, rwd, obs_plus1, terminal)
    print(agent.get_action(obs[0]))
