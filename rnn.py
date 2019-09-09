import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import RNN
import output_functions as of
import tensorflow.keras.activations as act
from tensorflow import keras
print(tf.__version__)


def simple_rnn_layer(x,
                     state_size,
                     output_func,
                     activation_func,
                     init_state=None,
                     one_timestep_output=False):

    """
    :param activation_func:
    :param one_timestep_output: if only output the final time step
    :param state_size: determine the output size to some extend
    :param init_state: (batch_size, n)
    :param x: (batch_size, features, timesteps)
    :param output_func: transfer states to outputs
    :return: the same form as the input, a 3D tensor
    """

    # Weights initialization with bias
    batch_size, Wx_size, timesteps = x.get_shape().as_list()
    if init_state:
        Wr_size = init_state.get_shape().as_list()[1]
    else:
        Wr_size = state_size

    W = tf.Variable(tf.truncated_normal((Wx_size + Wr_size, Wr_size)))
    b = tf.constant(0.0, shape=(1, Wr_size))

    # State initialization
    if init_state:
        Sk = init_state
    else:
        Sk = tf.constant(0.0, shape=(batch_size, state_size))

    # Computation

    # Edit activation function here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    if one_timestep_output:

        for k in range(timesteps):
            temp = tf.concat([x[:, :, k], Sk], axis=1)
            if activation_func == act.relu:
                Sk = activation_func(tf.linalg.matmul(temp, W) + b)
            if activation_func == act.softmax:
                Sk = activation_func(tf.linalg.matmul(temp, W) + b, axis=1)
        y = output_func(tf.expand_dims(Sk, axis=2))

    else:

        tl = []
        for k in range(timesteps):
            temp = tf.concat([x[:, :, k], Sk], axis=1)
            if activation_func == act.relu:
                Sk = activation_func(tf.linalg.matmul(temp, W) + b)
            if activation_func == act.softmax:
                Sk = activation_func(tf.linalg.matmul(temp, W) + b, axis=1)
            tl.append(tf.expand_dims(Sk, axis=2))
        y = output_func(tf.concat(tl, axis=2))

    return y




def bi_rnn_layer():
    None


def LSTM_rnn_layer():
    None


# ------------------------------------------------------------------------


# First, let's define a RNN Cell, as a layer subclass.
class MinimalRNNCell(keras.layers.Layer):
  def __init__(self, units, **kwargs):
      self.units = units
      self.state_size = units
      super(MinimalRNNCell, self).__init__(**kwargs)
  def build(self, input_shape):
      self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                    initializer='uniform',
                                    name='kernel')
      self.recurrent_kernel = self.add_weight(
          shape=(self.units, self.units),
          initializer='uniform',
          name='recurrent_kernel')
      self.built = True
  def call(self, inputs, states):
      prev_output = states[0]
      h = K.dot(inputs, self.kernel)
      output = h + K.dot(prev_output, self.recurrent_kernel)
      return output, [output]
# # Let's use this cell in a RNN layer:
# cell = MinimalRNNCell(32)
# x = keras.Input((None, 5))
# layer = RNN(cell)
# y = layer(x)
# # Here's how to use the cell to build a stacked RNN:
# cells = [MinimalRNNCell(32), MinimalRNNCell(64)]
# x = keras.Input((None, 5))
# layer = RNN(cells)
# y = layer(x)