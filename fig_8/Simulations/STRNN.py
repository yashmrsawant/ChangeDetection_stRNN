import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import datasets
#from sklearn.model_selection import train_test_split
import sys
import skimage
from sklearn import preprocessing
from skimage import transform
from skimage import io
from skimage.color import rgb2gray
import numpy as np
import os
import cv2
import matplotlib.image as mpimg
import random
import time


# Vanilla RNN class and functions
class RNN_cell_1(object):

    """
    RNN cell object which takes 3 arguments for initialization.
    input_size = Input Vector size
    hidden_layer_size = Hidden layer size
    target_size = Output vector size

    """

    def __init__(self, input_size, hidden_layer_size, target_size,weights_x,weights_h,weights_o,bias_i,bias_o):
        # Initialization of given values
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.target_size = target_size

	self.Wx = tf.constant(weights_x)
        self.Wh = tf.constant(weights_h)
	self.Wo = tf.constant(weights_o)
        self.bi = tf.constant(bias_i)
	self.bo = tf.constant(bias_o)

        # Placeholder for input vector with shape[batch, seq, embeddings]
        self._inputs = tf.placeholder(tf.float32,
                                      shape=[None, None, self.input_size],
                                      name='inputs')

        # Processing inputs to work with scan function
        self.processed_input = process_batch_input_for_RNN(self._inputs)

        '''
        Initial hidden state's shape is [1,self.hidden_layer_size]
        In First time stamp, we are doing dot product with weights to
        get the shape of [batch_size, self.hidden_layer_size].
        For this dot product tensorflow use broadcasting. But during
        Back propagation a low level error occurs.
        So to solve the problem it was needed to initialize initial
        hiddden state of size [batch_size, self.hidden_layer_size].
        So here is a little hack !!!! Getting the same shaped
        initial hidden state of zeros.
        '''

        self.initial_hidden = self._inputs[:, 0, :]
        self.initial_hidden = tf.matmul(
            self.initial_hidden, tf.zeros([input_size, hidden_layer_size]))

    # Function for vhanilla RNN.
    def vanilla_rnn(self, previous_hidden_state, x):
        """
        This function takes previous hidden state and input and
        outputs current hidden state.
        """
        current_hidden_state = tf.tanh(
        1*( tf.matmul(tf.nn.relu(previous_hidden_state), self.Wh) + tf.matmul(x,  self.Wx)) )
        return current_hidden_state

    # Function for getting all hidden state.
    def get_states(self):
        """
        Iterates through time/ sequence to get all hidden state
        """

        # Getting all hidden state throuh time
        all_hidden_states = tf.scan(self.vanilla_rnn,
                                    self.processed_input,
                                    initializer=self.initial_hidden,
                                    name='states')
        return all_hidden_states

    # Function to get output from a hidden layer
    def get_output(self, hidden_state):
        """
        This function takes hidden state and returns output
        """
        output = tf.sigmoid(tf.matmul(tf.nn.relu(hidden_state), self.Wo) + self.bo )
        return output

    # Function for getting all output layers
    def get_outputs(self):
        """
        Iterating through hidden states to get outputs for all timestamp
        """
        all_hidden_states = self.get_states()
        all_outputs = tf.map_fn(self.get_output, all_hidden_states)
        return all_outputs


# Vanilla RNN class and functions
class RNN_cell_2(object):

    """
    RNN cell object which takes 3 arguments for initialization.
    input_size = Input Vector size
    hidden_layer_size = Hidden layer size
    target_size = Output vector size

    """

    def __init__(self, input_size, hidden_layer_size, target_size,weights_x,weights_h,weights_o,bias_i,bias_o):
        # Initialization of given values
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.target_size = target_size
	self.Wx = tf.constant(weights_x)
        self.Wh = tf.constant(weights_h)
        self.bi = tf.constant(bias_i)

        # Weights for output layers
        self.Wo = tf.constant(weights_o)
        self.bo = tf.constant(bias_o)

        # Placeholder for input vector with shape[batch, seq, embeddings]
        self._inputs = tf.placeholder(tf.float32,shape=[None, None, self.input_size], name='inputs')

        # Processing inputs to work with scan function
        self.processed_input = process_batch_input_for_RNN(self._inputs)

        '''
        Initial hidden state's shape is [1,self.hidden_layer_size]
        In First time stamp, we are doing dot product with weights to
        get the shape of [batch_size, self.hidden_layer_size].
        For this dot product tensorflow use broadcasting. But during
        Back propagation a low level error occurs.
        So to solve the problem it was needed to initialize initial
        hiddden state of size [batch_size, self.hidden_layer_size].
        So here is a little hack !!!! Getting the same shaped
        initial hidden state of zeros.
        '''

        self.initial_hidden = self._inputs[:, 0, :]
        self.initial_hidden = tf.matmul(self.initial_hidden, tf.zeros([input_size, hidden_layer_size]))

    # Function for vhanilla RNN.
    def vanilla_rnn(self, previous_hidden_state, x):
        """
        This function takes previous hidden state and input and
        outputs current hidden state.
        """
        current_hidden_state = tf.tanh(
            1*( tf.matmul(tf.nn.relu(previous_hidden_state), self.Wh) + tf.matmul(x,  self.Wx)) )  # + self.bi

        return current_hidden_state

    # Function for getting all hidden state.
    def get_states(self):
        """
        Iterates through time/ sequence to get all hidden state
        """

        # Getting all hidden state throuh time
        all_hidden_states = tf.scan(self.vanilla_rnn,
                                    self.processed_input,
                                    initializer=self.initial_hidden,
                                    name='states')

        return all_hidden_states

    # Function to get output from a hidden layer
    def get_output(self, hidden_state):
        """
        This function takes hidden state and returns output
        """
        output = tf.sigmoid(tf.matmul(tf.nn.relu(hidden_state), self.Wo) + self.bo)

        return output

    # Function for getting all output layers
    def get_outputs(self):
        """
        Iterating through hidden states to get outputs for all timestamp
        """
        all_hidden_states = self.get_states()
        all_outputs = tf.map_fn(self.get_output, all_hidden_states)
        return all_outputs

# Function to convert batch input data to use scan ops of tensorflow.
def process_batch_input_for_RNN(batch_input):
    """
    Process tensor of size [5,3,2] to [3,5,2]
    """
    batch_input_ = tf.transpose(batch_input, perm=[2, 0, 1])
    X = tf.transpose(batch_input_)
    return X


##################################################################################
##################################################################################

# Vanilla RNN class and functions
class RNN_cell_1_g(object):

    """
    RNN cell object which takes 3 arguments for initialization.
    input_size = Input Vector size
    hidden_layer_size = Hidden layer size
    target_size = Output vector size

    """

    def __init__(self, input_size, hidden_layer_size, target_size, input_g_size, weights_x,weights_x_g, weights_h,weights_o,bias_i,bias_o):
        # Initialization of given values
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.target_size = target_size
        self.input_g_size = input_g_size
        ###
        self.Wx = tf.constant(weights_x)
        self.Wh = tf.constant(weights_h)
        self.Wg = tf.constant(weights_x_g)
        self.Wo = tf.constant(weights_o)
        self.bi = tf.constant(bias_i)
        self.bo = tf.constant(bias_o)

        # Placeholder for input vector with shape[batch, seq, embeddings]
        self._inputs = tf.placeholder(tf.float32,
                                  shape=[None, None, self.input_size],
                                  name='inputs')
        self._inputs_g = tf.placeholder(tf.float32,
                                  shape=[None, None, self.input_g_size],
                                  name='inputs_g')

        # Processing inputs to work with scan function
        self.processed_input, self.processed_input_g = process_batch_input_for_RNN_g(self._inputs, self._inputs_g)

        '''
        Initial hidden state's shape is [1,self.hidden_layer_size]
        In First time stamp, we are doing dot product with weights to
        get the shape of [batch_size, self.hidden_layer_size].
        For this dot product tensorflow use broadcasting. But during
        Back propagation a low level error occurs.
        So to solve the problem it was needed to initialize initial
        hiddden state of size [batch_size, self.hidden_layer_size].
        So here is a little hack !!!! Getting the same shaped
        initial hidden state of zeros.
        '''

        self.initial_hidden = self._inputs[:, 0, :]
        self.initial_hidden = tf.matmul(
        self.initial_hidden, tf.zeros([input_size, hidden_layer_size]))

    # Function for vhanilla RNN.
    def vanilla_rnn(self, previous_hidden_state, input_tuple): #, x, x_g):
        """
        This function takes previous hidden state and input and
        outputs current hidden state.
        """ ### update to tanh (to make the loss work)
        (x, x_g) = input_tuple
        current_hidden_state = tf.tanh(
        ( tf.matmul(tf.nn.relu(previous_hidden_state), self.Wh) + tf.matmul(x,  self.Wx) + tf.matmul(x_g,  self.Wg)) ) #+ self.bi

        return current_hidden_state

    # Function for getting all hidden state.
    def get_states(self):
        """
        Iterates through time/ sequence to get all hidden state
        """

        # Getting all hidden state throuh time
        all_hidden_states = tf.scan(self.vanilla_rnn,
                                    (self.processed_input, self.processed_input_g),
                                    initializer = self.initial_hidden,
                                    name='states')

        return all_hidden_states

    # Function to get output from a hidden layer
    def get_output(self, hidden_state):
        """
        This function takes hidden state and returns output
        """
        output = tf.sigmoid(tf.matmul(tf.nn.relu(hidden_state), self.Wo) + self.bo) ### update 

        return output

    # Function for getting all output layers
    def get_outputs(self):
        """
        Iterating through hidden states to get outputs for all timestamp
        """
        all_hidden_states = self.get_states()

        all_outputs = tf.map_fn(self.get_output, all_hidden_states)

        return all_outputs

# Function to convert batch input data to use scan ops of tensorflow.
def process_batch_input_for_RNN_g(batch_input, batch_input_g):
    """
    Process tensor of size [5,3,2] to [3,5,2]
    """
    batch_input_ = tf.transpose(batch_input, perm=[2, 0, 1])
    batch_input_g_ = tf.transpose(batch_input_g, perm=[2, 0, 1])
    X = tf.transpose(batch_input_)
    X_g = tf.transpose(batch_input_g_)

    return X, X_g