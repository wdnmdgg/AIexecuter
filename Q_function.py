import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


def lstm(s, a):
    x = tf.concat([s, a], axis=1)
    lstm_1 = keras.layers.LSTMCell(units=10,
                                   activation='tanh',
                                   recurrent_activation='relu',
                                   use_bias=True,
                                   kernel_initializer='glorot_uniform',
                                   recurrent_initializer='orthogonal',
                                   bias_initializer='zeros',
                                   unit_forget_bias=True,
                                   kernel_regularizer=None,
                                   recurrent_regularizer=None,
                                   bias_regularizer=None,
                                   kernel_constraint=None,
                                   recurrent_constraint=None,
                                   bias_constraint=None,
                                   dropout=0,
                                   recurrent_dropout=0,
                                   implementation=2)

    lstm_2 = keras.layers.LSTMCell(units=1,
                                   activation='softmax',
                                   recurrent_activation='relu',
                                   use_bias=True,
                                   kernel_initializer='glorot_uniform',
                                   recurrent_initializer='orthogonal',
                                   bias_initializer='zeros',
                                   unit_forget_bias=True,
                                   kernel_regularizer=None,
                                   recurrent_regularizer=None,
                                   bias_regularizer=None,
                                   kernel_constraint=None,
                                   recurrent_constraint=None,
                                   bias_constraint=None,
                                   dropout=0,
                                   recurrent_dropout=0,
                                   implementation=2)

    vf = keras.layers.RNN(cell=[lstm_1, lstm_2],
                          return_sequences=False,
                          return_state=False,
                          go_backwards=False)

    y = vf(inputs=x,
           initial_state=None,
           constants=None)

    return y


def ann(s, a):
    x = tf.concat([s, a], axis=1)
    l1 = tf.layers.dense(inputs=x,
                         units=40,
                         activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=l1,
                             units=1,
                             activation=tf.nn.sigmoid)
    return logits
