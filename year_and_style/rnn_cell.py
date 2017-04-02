#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q2(c): Recurrent neural nets for NER
"""

from __future__ import absolute_import
from __future__ import division

import logging
import sys

import tensorflow as tf
import numpy as np

logger = logging.getLogger("hw3.q2.1")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class RNNCell(tf.nn.rnn_cell.RNNCell):
    """Wrapper around our RNN cell implementation that allows us to play
    nicely with TensorFlow.
    """
    def __init__(self, input_size, state_size, regularizer):
        self.input_size = input_size
        self._state_size = state_size
        self.regularizer = regularizer

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, inputs, state, scope=None):
        """Updates the state using the previous @state and @inputs.
        Remember the RNN equations are:

        h_t = sigmoid(x_t W_x + h_{t-1} W_h + b)

        TODO: In the code below, implement an RNN cell using @inputs
        (x_t above) and the state (h_{t-1} above).
            - Define W_x, W_h, b to be variables of the apporiate shape
              using the `tf.get_variable' functions. Make sure you use
              the names "W_x", "W_h" and "b"!
            - Compute @new_state (h_t) defined above
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
            ### YOUR CODE HERE (~6-10 lines)
            xavier = tf.contrib.layers.xavier_initializer()
            W_x = tf.get_variable("W_x", initializer=xavier, shape=[self.input_size, self.state_size], dtype=tf.float32)
            W_h = tf.get_variable("W_h", initializer=xavier, shape=[self.state_size, self.state_size], dtype=tf.float32)
            b = tf.get_variable("b", initializer = tf.constant_initializer(0.0), shape=[self.state_size], dtype=tf.float32)
            new_state = tf.nn.sigmoid(tf.matmul(state,W_h) + tf.matmul(inputs, W_x) + b)

            ### END YOUR CODE ###
        # For an RNN , the output and state are the same (N.B. this
        # isn't true for an LSTM, though we aren't using one of those in
        # our assignment)
        output = new_state
        return output, new_state

