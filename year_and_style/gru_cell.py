#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys

import tensorflow as tf
import numpy as np

#logger = logging.getLogger("hw3.q3.1")
#logger.setLevel(logging.DEBUG)
#logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class GRUCell(tf.nn.rnn_cell.RNNCell):
    """Wrapper around our GRU cell implementation that allows us to play
    nicely with TensorFlow.
    """
    def __init__(self, input_size, state_size, regularizer):
        self.input_size = input_size
        self._state_size = state_size
        self.regularizer = regularizer
        self.regularization_loss = 0

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, inputs, state, scope=None):
        """Updates the state using the previous @state and @inputs.
        Remember the GRU equations are:

        z_t = sigmoid(x_t U_z + h_{t-1} W_z + b_z)
        r_t = sigmoid(x_t U_r + h_{t-1} W_r + b_r)
        o_t = tanh(x_t U_o + r_t * h_{t-1} W_o + b_o)
        h_t = z_t * h_{t-1} + (1 - z_t) * o_t

        TODO: In the code below, implement an GRU cell using @inputs
        (x_t above) and the state (h_{t-1} above).
            - Define W_r, U_r, b_r, W_z, U_z, b_z and W_o, U_o, b_o to
              be variables of the apporiate shape using the
              `tf.get_variable' functions.
            - Compute z, r, o and @new_state (h_t) defined above
        Tips:
            - Remember to initialize your matrices using the xavier
              initialization as before.
        Args:
            inputs: is the input vector of size [None, self.input_size]
            state: is the previous state vector of size [None, self.state_size]
            scope: is the name of the scope to be used when defining the variables inside.
        Returns:
            a pair of the output vector and the new state vector.
        """
        scope = scope or type(self).__name__

        # It's always a good idea to scope variables in functions lest they
        # be defined elsewhere!
        with tf.variable_scope(scope):
            ### YOUR CODE HERE (~20-30 lines)
            xavier = tf.contrib.layers.xavier_initializer()
            W_r = tf.get_variable("W_r", initializer=xavier, shape=[self.state_size, self.state_size], dtype=tf.float32)
            U_r = tf.get_variable("U_r", initializer=xavier, shape=[self.input_size, self.state_size], dtype=tf.float32)
            b_r = tf.get_variable("b_r", initializer=xavier, shape=[self.state_size], dtype=tf.float32)
            W_z = tf.get_variable("W_z", initializer=xavier, shape=[self.state_size, self.state_size], dtype=tf.float32)
            U_z = tf.get_variable("U_z", initializer=xavier, shape=[self.input_size, self.state_size], dtype=tf.float32)
            b_z = tf.get_variable("b_z", initializer=xavier, shape=[self.state_size], dtype=tf.float32)
            W_o = tf.get_variable("W_o", initializer=xavier, shape=[self.state_size, self.state_size], dtype=tf.float32)
            U_o = tf.get_variable("U_o", initializer=xavier, shape=[self.input_size, self.state_size], dtype=tf.float32)
            b_o = tf.get_variable("b_o", initializer=xavier, shape=[self.state_size], dtype=tf.float32)
            z_t = tf.nn.sigmoid(tf.matmul(inputs, U_z) + tf.matmul(state, W_z) + b_z, name='z_t')
            r_t = tf.nn.sigmoid(tf.matmul(inputs, U_r) + tf.matmul(state, W_r) + b_r, name='r_t')
            o_t = tf.nn.tanh(tf.matmul(inputs, U_o) + tf.matmul(tf.mul(r_t, state), W_o) + b_o, name='o_t')
            h_t = tf.mul(z_t, state) + tf.mul((tf.ones(shape=[1,self.state_size])-z_t), o_t, name='h_t')

            new_state = h_t
            #self.regularization_loss = tf.contrib.layers.apply_regularization(self.regularizer, [W_r, U_r, W_z, U_z, W_o, U_o])
            ### END YOUR CODE ###
        # For a GRU, the output and state are the same (N.B. this isn't true
        # for an LSTM, though we aren't using one of those in our
        # assignment)
        output = new_state
        return output, new_state
