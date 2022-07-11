# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 21:29:39 2019

@author: dhawa
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import h5py
from scipy.misc import imsave


def plot_matrix(data,cm=plt.cm.RdYlBu,vmin=-1.0,vmax=1.0):
    plt.matshow(data,cmap=cm,vmin=vmin,vmax=vmax)
    plt.show()
    
class RNN_cell(object):

    """
    RNN cell object which takes 3 arguments for initialization.
    input_size = Input Vector size
    hidden_layer_size = Hidden layer size
    target_size = Output vector size

    """

    def __init__(self, input_size, hidden_layer_size, target_size,weights_x,weights_h,weights_o,bias_i,bias_o):
    #def __init__(self, input_size, hidden_layer_size, target_size):
        # Initialization of given values
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.target_size = target_size
        
#        ################################
#        self.Wi_mask_big = tf.constant(Wi_mask_big)
#        self.Wh_mask_big = tf.constant(Wh_mask_big)
#        self.Wo_mask_big = tf.constant(Wo_mask_big)
#        self.Wi_sign_mask = tf.constant(Wi_sign_mask)
#        self.Wh_sign_mask = tf.constant(Wh_sign_mask)
#        self.Wo_sign_mask = tf.constant(Wo_sign_mask)
#
#        # Weights and Bias for input and hidden tensor
#
#        self.Wx =  tf.matmul(self.Wi_sign_mask,tf.nn.relu(Wi_mask_big * tf.Variable(tf.truncated_normal([self.input_size, self.hidden_layer_size],mean = 0.2,stddev = 0.01))))
#
#        self.Wh = tf.matmul(self.Wh_sign_mask,tf.nn.relu(Wh_mask_big * tf.Variable(tf.truncated_normal([self.hidden_layer_size, self.hidden_layer_size],mean = 0.2,stddev = 0.01))))
#
#        self.bi = tf.Variable(tf.zeros([self.hidden_layer_size]))
#
#        # Weights for output layers ### update
#        self.Wo = tf.matmul(self.Wo_sign_mask,tf.nn.relu(Wo_mask_big * tf.Variable(tf.truncated_normal([self.hidden_layer_size, self.target_size],mean=0.1,stddev=0.01))))
#
#        self.bo = tf.Variable(tf.truncated_normal([self.target_size], mean=0.0, stddev=0.05))   
        ##############################
        
        self.Wx = tf.Variable(tf.constant(weights_x))
        self.Wh = tf.Variable(tf.constant(weights_h))
        self.Wo = tf.Variable(tf.constant(weights_o))
        self.bi = tf.Variable(tf.constant(bias_i))
        self.bo = tf.Variable(tf.constant(bias_o))
        
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
        """ ### update to tanh (to make the loss work)
        current_hidden_state = tf.tanh(
        ( tf.matmul(tf.nn.relu(previous_hidden_state), self.Wh) + tf.matmul(x,  self.Wx) ) ) #+ self.bi

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
def process_batch_input_for_RNN(batch_input):
    """
    Process tensor of size [5,3,2] to [3,5,2]
    """
    batch_input_ = tf.transpose(batch_input, perm=[2, 0, 1])
    X = tf.transpose(batch_input_)

    return X

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
hidden_layer_size = 256+64
input_size =  64
target_size = 64

# importing learned weights
# persistence

perweights = './new_weights//'
Wxp = np.loadtxt(perweights + 'Wx2237_per.csv').astype(np.float32)
Whp = np.loadtxt(perweights + 'Wh2237_per.csv').astype(np.float32)
Wop = np.loadtxt(perweights + 'Wo2237_per.csv').astype(np.float32)
Bop = np.loadtxt(perweights + 'bo2237_per.csv').astype(np.float32)
Bip = np.zeros((hidden_layer_size)).astype(np.float32)

# change
chaweights = './new_weights/'
Wxc = np.loadtxt(chaweights + 'Wx9999_cd.csv').astype(np.float32)
Whc = np.loadtxt(chaweights + 'Wh9999_cd.csv').astype(np.float32)
Woc = np.loadtxt(chaweights + 'Wo9999_cd.csv').astype(np.float32)
Bic = np.loadtxt(chaweights + 'bi9999_cd.csv').astype(np.float32)
Boc = np.loadtxt(chaweights + 'bo9999_cd.csv').astype(np.float32)

#

y = tf.placeholder(tf.float32, shape=[None, 10, target_size], name='inputs')
rnn_per = RNN_cell(input_size, hidden_layer_size, target_size, Wxp, Whp, Wop, Bip, Bop)
#weights_x,weights_h,weights_o,bias_i,bias_o
rnn_change = RNN_cell_2(input_size, hidden_layer_size, target_size, Wxc, Whc, Woc, Bic, Boc)
per_outputs = rnn_per.get_outputs()

cha_outputs = rnn_change.get_outputs()


## Load data file
matfile = h5py.File('./toyimages.mat', 'r')
toyimages = matfile['toyimages']
toyimages.shape
from os import mkdir
try:
    mkdir('./Figures/')
except:
    print("Already Created Folder")

t1 = 0
t2 = 5


with tf.Session() as sess:

    for j in range(5):
    
        I = 2 * j
        toy_1 = np.transpose(toyimages[:, :, I])
        toy_2 = np.zeros((24, 24))
        toy_3 = np.transpose(toyimages[:, :, I + 1])
        seq_l = 12
        s1, s2 = toy_1.shape
        data = np.zeros((seq_l, s1, s2), dtype = np.float32)
        data.fill(0.0)
    
        data[t1, :, :] = toy_1
        data[t2, :, :] = toy_3
        sample_size = 9
        M_1 = np.zeros((sample_size, seq_l, 64))
        M_g = np.zeros((sample_size, seq_l, 80))
    
        counter = 0
    
        for i in range(3):
            for j in range(3):
                t = data[:, i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8]
                for k in range(data.shape[0]):
                    M_1[counter, k, :] = np.reshape(t[k, :, :], (1, 64))
                counter += 1
    
        sess.run(tf.global_variables_initializer())
        p_predict_1 = sess.run(per_outputs, feed_dict={rnn_per._inputs: M_1.astype(np.float32)})
            
        p_predict_1 = np.float32(p_predict_1 > 0.5) 
        c_predict_1 = sess.run(cha_outputs, feed_dict={rnn_change._inputs: np.swapaxes(p_predict_1,0,1)})
        
        z = tf.placeholder(tf.float32,shape=[None, 64])
        q = tf.reshape(z,(9,8,8))
        out1_a = sess.run(q, feed_dict = {z : p_predict_1[1, :, :]}) 
        out1_a.shape
    
        z = tf.placeholder(tf.float32,shape=[None, 64])
        q = tf.reshape(z,(9,8,8))
    
        result1_a, result2_a = np.zeros((seq_l,24,24)), np.zeros((seq_l,24,24))
    
        for itr in range(seq_l):
            out1_a, out2_a = sess.run(q, feed_dict={z:p_predict_1[itr,:,:]}), sess.run(q, feed_dict={z:c_predict_1[itr,:,:]})
            for i in range(9):
                m = int(i / 3)
                n = i % 3
                result1_a[itr, m*8 : (m+1)*8, n*8 : (n+1)*8]= out1_a[i,:,:] ### update np.abs
                result2_a[itr, m*8 : (m+1)*8, n*8 : (n+1)*8]= out2_a[i,:,:] ### update np.abs
    
    
        result_1 = result1_a 
        result_2 = result2_a
    
        result_final_per = np.zeros((seq_l,240,240))
        result_final_cd = np.zeros((seq_l,240,240))
    
        for i in range(seq_l):
            result_final_per[i, :, :] = cv2.resize(result_1[i, :, :], (240, 240), interpolation = cv2.INTER_NEAREST)
            result_final_cd[i, :, :] = cv2.resize(result_2[i, :, :], (240, 240), interpolation = cv2.INTER_NEAREST)
    
        im1 = ((1 - cv2.resize(toy_1, (240, 240), interpolation = cv2.INTER_NEAREST)) * 255.0).astype(np.uint8)
        im2 = ((1 - np.zeros((240, 240))) * 255.0).astype(np.uint8)
        im3 = ((1 - cv2.resize(toy_3, (240, 240), interpolation = cv2.INTER_NEAREST)) * 255.0).astype(np.uint8)
        
        per1 = ((1 - result_final_per[t1, :, :]) * 255.0).astype(np.uint8)
        per2 = ((1 - result_final_per[t2 - 1, :, :]) * 255.0).astype(np.uint8)
        per3 = ((1 - result_final_per[t2, :, :]) * 255.0).astype(np.uint8)
        
        ch1 = ((1 - result_final_cd[t1, :, :]) * 255.0).astype(np.uint8)
        ch2 = ((1 - result_final_cd[t2 - 1, :, :]) * 255.0).astype(np.uint8)
        ch3 = ((1 - result_final_cd[t2, :, :]) * 255.0).astype(np.uint8)
        pathf = './Figures/TOY' + str(I) + 'WOGI/'
        
        try:
            mkdir(pathf)
        except:
            print("Already Existed Folder")    
        imsave(pathf + 'img1.png', im1)
        imsave(pathf + 'img2.png', im2)
        imsave(pathf + 'img3.png', im3)
        
        imsave(pathf + 'ynormal_per_1.png', per1)
        imsave(pathf + 'ynormal_per_2.png', per2)
        imsave(pathf + 'ynormal_per_3.png', per3)
        
        imsave(pathf + 'ynormal_cd_1.png', ch1)
        imsave(pathf + 'ynormal_cd_2.png', ch2)
        imsave(pathf + 'ynormal_cd_3.png', ch3)
