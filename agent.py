import numpy as np
import tensorflow as tf
import rnn
import output_functions as of
import tensorflow.keras.activations as act
import shift
print(tf.__version__)


class Agent:

    def __init__(self,
                 ):

sess = tf.Session()

x = tf.constant([[[1, 1, 1], [2, 2, 2]],
                 [[3, 3, 3], [4, 4, 4]],
                 [[5, 5, 5], [6, 6, 6]]], dtype=tf.float32)

labels = tf.placeholder(dtype=tf.float32, shape=(3, 2, 3))

initstate = tf.constant(0.0, shape=(3, 8))

tmp = rnn.simple_rnn_layer(x, 2, of.I, act.relu)
outcome = rnn.simple_rnn_layer(tmp, 2, of.I, act.softmax)

mse = tf.losses.mean_squared_error(labels, outcome)

opt = tf.train.AdamOptimizer(learning_rate=0.01)
min_opt = opt.minimize(mse)

sess.run(tf.global_variables_initializer())
for i in range(100000):
    print(sess.run((min_opt, mse), feed_dict={labels: [[[1, 1, 0], [0, 1, 0]],
                                                    [[0, 0, 0], [0, 0, 1]],
                                                    [[1, 1, 1], [1, 0, 1]]]}))
