from env import Env
from agent import LSTMAgent
from pool import SimpleReplayPool
import shift
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import Q_function as Qf
import time

GAMMA = 0.95
EPSILON = 0.1
EPISODES = 1000
trader = shift.Trader("username")
symbol = None
commission = None
sess = tf.Session()
execute_time = 3600
exe_times = 120
exe_interval = execute_time / exe_times


def main():
    env = Env(trader=trader,
              symbol=symbol,
              commission=commission)
    agent = LSTMAgent(sess_=sess,
                      observations_dim=101,
                      action_space=11,
                      batch_size=5,
                      Q_function=Qf.ann,
                      optimizer=tf.train.AdamOptimizer,
                      GAMMA=GAMMA,
                      EPSILON=EPSILON,
                      learning_rate=0.001)
    pool = SimpleReplayPool(max_pool_size=1000,
                            pop_size=100)

    for i in range(EPISODES):
        # Deal with the initialization for each episode
        for j in range(exe_times):
            time.sleep(exe_interval)
# ---------------------------------------------
            if j == 0:
                env.get_obs()
            else:
                env.step()
            agent.get_action()
            agent.save_buffer()
            pool.add_sample()
        pool.random_batch()
        agent.train()
# ---------------------------------------------


if __name__ == '__main__':
    main()
