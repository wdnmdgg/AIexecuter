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
action_space = 11


def main():
    env = Env(trader=trader,
              symbol=symbol,
              commission=commission,
              action_space=action_space)
    agent = LSTMAgent(sess_=sess,
                      observations_dim=101,
                      action_space=action_space,
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

        ob = env.reset()
        act = agent.get_action(ob['states'])
        ob = env.step(act)
        for j in range(1, exe_times):
            agent.save_buffer([ob['reward'], ob['states'], ob['isdone']],
                              False)
            pool.add_sample(agent.tmp_buffer[0],
                            agent.tmp_buffer[1],
                            agent.tmp_buffer[2],
                            agent.tmp_buffer[3],
                            agent.tmp_buffer[4])
            act = agent.get_action(ob['states'])
            ob = env.step(act)
        s0, acts, r, s1, ter = pool.random_batch()
        agent.train(s0, acts, r, s1, ter)



if __name__ == '__main__':
    main()
