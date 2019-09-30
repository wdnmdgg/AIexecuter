from env import Env
from agent import LSTMAgent
import shift
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import Q_function as Qf
from datetime import datetime

GAMMA = 0.95
EPSILON = 0.1
EPISODES = 1000
trader = shift.Trader("username")
symbol = None
commission = None
sess = tf.Session()


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

    for i in range(EPISODES):
        pass






if __name__ == '__main__':
    main()
