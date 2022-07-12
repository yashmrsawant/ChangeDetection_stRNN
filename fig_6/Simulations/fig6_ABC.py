# -*- coding: utf-8 -*-
"""
@author(s): Jogendra Kundu      jogendrak@iisc.ac.in
            Yash M. Sawant     yashsawant@iisc.ac.in

@last edited: Dec, 2021
"""

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import os
import random
import time
from os import chdir
# from STRNN import *
from numpy import concatenate as concat
from os import mkdir, chdir

import h5py
from matplotlib.pyplot import imsave

#path = '...'

chdir(path)

# In[]:
# -*- coding: utf-8 -*-
"""
@description: for defining mc-stRNN with GI layer having excitatory topographic connections from stRNN E units and the
                global inhibitory connections from GI layer units to stRNN units.
"""

import numpy as np

# Vanilla RNN class and functions
class RNN_cell_1_g(object):

    """
    RNN cell object which takes 3 arguments for initialization.
    input_size = Input Vector size
    hidden_layer_size = Hidden layer size
    target_size = Output vector size

    """

    def __init__(self, input_size, hidden_layer_size, target_size, input_g_size, hidden_gi_size, hidden_layer_size_E, \
            weights_x, weights_x_g, weights_h,weights_o,bias_i,bias_o, weights_g_x, weights_g_s, weights_g_g):
        # Initialization of given values
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.target_size = target_size
        self.input_g_size = input_g_size
        self.hidden_gi_size = hidden_gi_size
        self.hidden_layer_size_E = hidden_layer_size_E
        ###
        self.Wg_x = tf.constant(weights_g_x)
        self.Wg_s = tf.constant(weights_g_s)
        self.Wg_g = tf.constant(weights_g_g)
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
        self.initial_hidden_states_gi = tf.matmul(self._inputs[:, 0, :], tf.zeros([self.input_size, self.hidden_gi_size]))
        self.initial_hidden_states_stRNN = tf.matmul(self._inputs[:, 0, :], tf.zeros([self.input_size, self.hidden_layer_size]))
        self.initial_hidden_states = tf.concat([self.initial_hidden_states_gi, self.initial_hidden_states_stRNN], axis = 1)

    # Function for vhanilla RNN.
    def vanilla_rnn(self, previous_network_state, input_tuple): #, x, x_g):
        """
        This function takes previous hidden state and input and
        outputs current hidden state.
        """ ### update to tanh (to make the loss work)
        (x, x_g) = input_tuple
        previous_gi_state = previous_network_state[:, :self.hidden_gi_size]
        previous_stRNN_hidden_state = previous_network_state[:, self.hidden_gi_size:]
        previous_stRNN_E_hidden_state = tf.reshape(tf.reduce_mean(tf.nn.relu(previous_stRNN_hidden_state[:, :self.hidden_layer_size_E]), \
                                                                  axis = 0), shape = [1, self.hidden_layer_size_E])
        
        
        current_gi_state = tf.tanh(tf.matmul(x_g, self.Wg_x) + tf.matmul(tf.nn.relu(previous_stRNN_E_hidden_state), self.Wg_s) +\
                                   tf.matmul(tf.nn.relu(previous_gi_state), self.Wg_g))
        current_stRNN_hidden_state = tf.tanh((tf.matmul(tf.nn.relu(previous_stRNN_hidden_state), self.Wh) + \
                                   tf.matmul(x, self.Wx) + tf.matmul(tf.nn.relu(current_gi_state), self.Wg)))
        current_network_state = tf.concat([current_gi_state, current_stRNN_hidden_state], axis = 1)
        return current_network_state

    # Function for getting all hidden state.
    def get_states(self):
        """
        Iterates through time/ sequence to get all hidden state
        """

        # Getting all hidden state throuh time
        all_hidden_states = tf.scan(self.vanilla_rnn,
                                    (self.processed_input, self.processed_input_g),
                                    initializer = self.initial_hidden_states,
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

        all_outputs = tf.map_fn(self.get_output, all_hidden_states[:, :, self.hidden_gi_size:])

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



"""
    stRNN class without GI layer
"""
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


# In[]:
with h5py.File("./fig6_stimuli.h5", "r") as f:
    data = f['/data_seq'][()]
    data_g = f['/data_g_seq'][()]

seq_l = data.shape[0]
cp = 15

data_g[0, :, :] = data_g[0, :, :] + np.random.random((10, 10)) > 0.7
data_g[cp, :, :] = data_g[cp, :, :] + np.random.random((10, 10)) > 0.7
print(np.max(data[cp]))

# In[]: Half trained; loading mc-stRNN with GI layer weights (see RNN_cell_1_g)
with h5py.File('./weights_mc/weights_epoch100.h5', 'r') as f:
    wg_x_v = f['/Wg_x'][()]
    wg_s_v = f['/Wg_s'][()]
    wg_g_v = f['/Wg_g'][()]
    wg_v = f['/Wg'][()]
    wh_v = f['/Wh'][()]
    wx_v = f['/Wx'][()]
    wo_v = f['/Wo'][()]
    bo_v = f['/bo'][()]
# In[]: load cd-stRNN weights

Wx2 = np.loadtxt('./Weights_cd/Wx9999_cd.csv').astype(np.float32)
Wh2 = np.loadtxt('./Weights_cd/Wh9999_cd.csv').astype(np.float32)
Bi2 = np.loadtxt('./Weights_cd/bi9999_cd.csv').astype(np.float32)
Wo2 = np.loadtxt('./Weights_cd/Wo9999_cd.csv').astype(np.float32)
Bo2 = np.loadtxt('./Weights_cd/bo9999_cd.csv').astype(np.float32)


# SC stimulation
Wh1_g_s = 1.1 * np.copy(wh_v)
Wo1_g_s = 1.1 * np.copy(wo_v)

# In[]:
sample_size = 12500
BX = np.zeros((2,64))
BX.fill(0.0)
M_1 = np.zeros((sample_size,seq_l,64))
M_2 = np.zeros((sample_size,seq_l,64))
M_3 = np.zeros((sample_size,seq_l,64))
M_4 = np.zeros((sample_size,seq_l,64))
M_g = np.zeros((sample_size,seq_l,100))
N = np.zeros((sample_size,9,64))
counter = 0

ts = time.time()
for i in range(0, 100-1):
    for j in range(0, 125-1):
        t_1 = data[:,   i*8:(i+1)*8,           j*8:(j+1)*8]
        t_2 = data[:,   4+ i*8: 4+ (i+1)*8,    j*8:(j+1)*8]
        t_3 = data[:,   i*8:(i+1)*8,       4+j*8:4+(j+1)*8]
        t_4 = data[:,   4+i*8:4+(i+1)*8,    4+j*8:4+(j+1)*8]
        for k in range(data.shape[0]):
            M_1[counter,k,:] = np.reshape(t_1[k,:,:],(1,64))
            M_2[counter,k,:] = np.reshape(t_2[k,:,:],(1,64))
            M_3[counter,k,:] = np.reshape(t_3[k,:,:],(1,64))
            M_4[counter,k,:] = np.reshape(t_4[k,:,:],(1,64))
            M_g[counter,k,:] = np.reshape(data_g[k,:,:].astype(np.float32),(1,100))
        counter += 1
        #print('Loading Data Point :', counter)

N = N.astype(int)
te = time.time()
print('%2.2f sec' % (te-ts))


# In[]: change detection network
tf.reset_default_graph()
hidden_layer_size = 256 + 64
hidden_layer_size_E = 256
hidden_gi_size = 100
input_size, input_g_size = 64, 100
target_size = input_size
tf.reset_default_graph()
#y = tf.placeholder(tf.float32, shape = [None, 3, target_size], name = 'inputs')

rnn_1 = RNN_cell_1_g(input_size, hidden_layer_size, target_size, input_g_size, \
                     hidden_gi_size, hidden_layer_size_E, wx_v, wg_v, wh_v, wo_v, 0, bo_v, wg_x_v, wg_s_v, wg_g_v)
rnn_1_s = RNN_cell_1_g(input_size, hidden_layer_size, target_size, input_g_size, \
                     hidden_gi_size, hidden_layer_size_E, wx_v, wg_v, Wh1_g_s, Wo1_g_s, 0, bo_v, wg_x_v, wg_s_v, wg_g_v)

rnn_2 = RNN_cell_2(input_size, hidden_layer_size, target_size, Wx2, Wh2, Wo2, Bi2, Bo2)
outputs = rnn_2.get_outputs()

p_outputs = rnn_1.get_outputs()
p_outputs_s = rnn_1_s.get_outputs()

p_h_states = rnn_1.get_states()
p_h_states_s = rnn_1_s.get_states()


# In[]:
def create_folder(folderpath):
    try:
        mkdir(folderpath)
    except:
        print("Some error creating folder...")

try:
    sess.close()
except:
    print("Session already closed")

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

path = "./Figures/"
create_folder(path)
folder = path + "GratingsExp_Normal/"
create_folder(folder)


# In[]: without micro-stimulation 
p_predict_1 = sess.run(p_outputs, feed_dict={rnn_1._inputs: M_1.astype(np.float32), rnn_1._inputs_g:(M_g>0.5).astype(np.float32)})
p_predict_2 = sess.run(p_outputs, feed_dict={rnn_1._inputs: M_2.astype(np.float32), rnn_1._inputs_g:(M_g>0.5).astype(np.float32)})
p_predict_3 = sess.run(p_outputs, feed_dict={rnn_1._inputs: M_3.astype(np.float32), rnn_1._inputs_g:(M_g>0.5).astype(np.float32)})
p_predict_4 = sess.run(p_outputs, feed_dict={rnn_1._inputs: M_4.astype(np.float32), rnn_1._inputs_g:(M_g>0.5).astype(np.float32)})
p_states_v = sess.run(p_h_states, feed_dict={rnn_1._inputs: M_1.astype(np.float32), rnn_1._inputs_g:(M_g>0.5).astype(np.float32)})
p_predict_1_un = p_predict_1
p_predict_1 = np.float32(p_predict_1 > 0.5) 
p_predict_2 = np.float32(p_predict_2 > 0.5) 
p_predict_3 = np.float32(p_predict_3 > 0.5) 
p_predict_4 = np.float32(p_predict_4 > 0.5) 
#input_cd = np.swapaxes(p_predict,0,1)
predict_1 = sess.run(outputs, feed_dict={rnn_2._inputs: np.swapaxes(p_predict_1,0,1)})
predict_2 = sess.run(outputs, feed_dict={rnn_2._inputs: np.swapaxes(p_predict_2,0,1)})
predict_3 = sess.run(outputs, feed_dict={rnn_2._inputs: np.swapaxes(p_predict_3,0,1)})
predict_4 = sess.run(outputs, feed_dict={rnn_2._inputs: np.swapaxes(p_predict_4,0,1)})

z = tf.placeholder(tf.float32,shape=[None, 64])
q = tf.reshape(z,(12500,8,8))

result1_a, result2_a = np.zeros((seq_l,800,1000)), np.zeros((seq_l,800,1000))
result1_b, result2_b = np.zeros((seq_l,800,1000)), np.zeros((seq_l,800,1000))
result1_c, result2_c = np.zeros((seq_l,800,1000)), np.zeros((seq_l,800,1000))
result1_d, result2_d = np.zeros((seq_l,800,1000)), np.zeros((seq_l,800,1000))

for itr in range(seq_l):
    out1_a, out2_a = sess.run(q, feed_dict={z:p_predict_1[itr,:,:]}), sess.run(q, feed_dict={z:predict_1[itr,:,:]})
    out1_b, out2_b = sess.run(q, feed_dict={z:p_predict_2[itr,:,:]}), sess.run(q, feed_dict={z:predict_2[itr,:,:]})
    out1_c, out2_c = sess.run(q, feed_dict={z:p_predict_3[itr,:,:]}), sess.run(q, feed_dict={z:predict_3[itr,:,:]})
    out1_d, out2_d = sess.run(q, feed_dict={z:p_predict_4[itr,:,:]}), sess.run(q, feed_dict={z:predict_4[itr,:,:]}) 
    for i in range(12276):
        m = int(i / (125-1))
        n = int(i % (125-1))
        result1_a[itr,m*8:(m+1)*8,n*8:(n+1)*8]= out1_a[i,:,:] ### update np.abs
        result2_a[itr,m*8:(m+1)*8,n*8:(n+1)*8]= out2_a[i,:,:] ### update np.abs
        
        result1_b[itr, 4+m*8:4+(m+1)*8, n*8:(n+1)*8]= out1_b[i,:,:] ### update np.abs
        result2_b[itr, 4+m*8:4+(m+1)*8, n*8:(n+1)*8]= out2_b[i,:,:] ### update np.abs
        
        result1_c[itr,m*8:(m+1)*8, 4+n*8:4+(n+1)*8]= out1_c[i,:,:] ### update np.abs
        result2_c[itr,m*8:(m+1)*8, 4+n*8:4+(n+1)*8]= out2_c[i,:,:] ### update np.abs

        result1_d[itr, 4+m*8:4+(m+1)*8, 4+n*8:4+(n+1)*8]= out1_d[i,:,:] ### update np.abs
        result2_d[itr, 4+m*8:4+(m+1)*8, 4+n*8:4+(n+1)*8]= out2_d[i,:,:] ### update np.abs
        
result_1 = result1_a + result1_b + result1_c + result1_d
result_2 = result2_a + result2_b + result2_c + result2_d

result_final_per, result_final_cd = np.copy(result_1), np.copy(result_2)

result_final_per = np.float32((result_final_per / 4.0) > 0.6)
result_final_cd = np.float32((result_final_cd / 4.0) > 0.6)


folder_per = folder + 'Per/'
create_folder(folder_per)

folder_cd = folder + 'CD/'
create_folder(folder_cd)

for i in range(seq_l):
    imsave("%sPer_Normal_%d.png"%(folder_per, i), ((1 - result_final_per[i, :, :])*255.0).astype(np.uint8), cmap = 'gray', vmin = 0, vmax = 255)
    imsave("%sCD_Normal_%d.png"%(folder_cd, i), ((1 - result_final_cd[i, :, :])*255.0).astype(np.uint8), cmap = 'gray', vmin = 0, vmax = 255)


# In[]: simulating micro-stimulation (Scaled Wh & Wo weight matrices)
try:
    sess.close()
except:
    print("Session already closed")

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

Input0_1 = M_1[:, 0, :].reshape((12500, 1, 64)).astype(np.float32)
Input0_g = M_g[:, 0, :].reshape((12500, 1, 100)).astype(np.float32)

Input0_2 = M_2[:, 0, :].reshape((12500, 1, 64)).astype(np.float32)

Input0_3 = M_3[:, 0, :].reshape((12500, 1, 64)).astype(np.float32)

Input0_4 = M_4[:, 0, :].reshape((12500, 1, 64)).astype(np.float32)

InputCP_1 = M_1[:, cp, :].reshape((12500, 1, 64)).astype(np.float32)
Input9_g = M_g[:, cp, :].reshape((12500, 1, 100)).astype(np.float32)

InputCP_2 = M_2[:, cp, :].reshape((12500, 1, 64)).astype(np.float32)

InputCP_3 = M_3[:, cp, :].reshape((12500, 1, 64)).astype(np.float32)

InputCP_4 = M_4[:, cp, :].reshape((12500, 1, 64)).astype(np.float32)

blankperiod = cp-1
# 1
p_h_states_v0_1 = np.swapaxes(sess.run(p_h_states, feed_dict = {rnn_1._inputs : Input0_1, rnn_1._inputs_g: Input0_g}), 0, 1)
p_predict0_1 = np.swapaxes(sess.run(p_outputs, feed_dict = {rnn_1._inputs : Input0_1, rnn_1._inputs_g : Input0_g}), 0, 1)

p_h_states_sv_blanks_1 = np.swapaxes(sess.run(p_h_states_s, feed_dict = {rnn_1_s._inputs : np.zeros((12500, blankperiod, 64)).astype(np.float32), \
                                                                      rnn_1_s._inputs_g : np.zeros((12500, blankperiod, 100)).astype(np.float32), \
                                                                      rnn_1_s.initial_hidden_states : p_h_states_v0_1[:, 0, :]}), 0, 1)
p_predict_sv_blanks_1 = np.swapaxes(sess.run(p_outputs_s, feed_dict = {rnn_1_s._inputs : np.zeros((12500, blankperiod, 64)).astype(np.float32), \
                                                                      rnn_1_s._inputs_g : np.zeros((12500, blankperiod, 100)).astype(np.float32), \
                                                                      rnn_1_s.initial_hidden_states : p_h_states_v0_1[:, 0, :]}), 0, 1)

    
p_h_states_vcp_1 = np.swapaxes(sess.run(p_h_states, feed_dict = {rnn_1._inputs : InputCP_1, \
                                                              rnn_1._inputs_g : Input9_g, \
                                                              rnn_1.initial_hidden_states : p_h_states_sv_blanks_1[:, -1, :]}), 0, 1)
p_predictcp_1 = np.swapaxes(sess.run(p_outputs, feed_dict = {rnn_1._inputs : InputCP_1, rnn_1._inputs_g : Input9_g, \
                                                            rnn_1.initial_hidden_states : p_h_states_sv_blanks_1[:, -1, :]}), 0, 1)

p_h_states_sva_blanks_1 = np.swapaxes(sess.run(p_h_states_s, feed_dict = {rnn_1_s._inputs : np.zeros((12500, blankperiod, 64)).astype(np.float32), \
                                                                      rnn_1_s._inputs_g : np.zeros((12500, blankperiod, 100)).astype(np.float32), \
                                                                      rnn_1_s.initial_hidden_states : p_h_states_vcp_1[:, 0, :]}), 0, 1)
p_predict_sva_blanks_1 = np.swapaxes(sess.run(p_outputs_s, feed_dict = {rnn_1_s._inputs : np.zeros((12500, blankperiod, 64)).astype(np.float32), \
                                                                      rnn_1_s._inputs_g : np.zeros((12500, blankperiod, 100)).astype(np.float32), \
                                                                      rnn_1_s.initial_hidden_states : p_h_states_vcp_1[:, 0, :]}), 0, 1)


p_predict_1 = concat((p_predict0_1, p_predict_sv_blanks_1, p_predictcp_1, p_predict_sva_blanks_1), axis = 1)
p_h_statesV_1 = concat((p_h_states_v0_1, p_h_states_sv_blanks_1, p_h_states_vcp_1, p_h_states_sva_blanks_1), axis = 1)

# 2
p_h_states_v0_2 = np.swapaxes(sess.run(p_h_states, feed_dict = {rnn_1._inputs : Input0_2, rnn_1._inputs_g: Input0_g}), 0, 1)
p_predict0_2 = np.swapaxes(sess.run(p_outputs, feed_dict = {rnn_1._inputs : Input0_2, rnn_1._inputs_g : Input0_g}), 0, 1)

p_h_states_sv_blanks_2 = np.swapaxes(sess.run(p_h_states_s, feed_dict = {rnn_1_s._inputs : np.zeros((12500, blankperiod, 64)).astype(np.float32), \
                                                                      rnn_1_s._inputs_g : np.zeros((12500, blankperiod, 100)).astype(np.float32), \
                                                                      rnn_1_s.initial_hidden_states : p_h_states_v0_2[:, 0, :]}), 0, 1)
p_predict_sv_blanks_2 = np.swapaxes(sess.run(p_outputs_s, feed_dict = {rnn_1_s._inputs : np.zeros((12500, blankperiod, 64)).astype(np.float32), \
                                                                      rnn_1_s._inputs_g : np.zeros((12500, blankperiod, 100)).astype(np.float32), \
                                                                      rnn_1_s.initial_hidden_states : p_h_states_v0_2[:, 0, :]}), 0, 1)

    
p_h_states_vcp_2 = np.swapaxes(sess.run(p_h_states, feed_dict = {rnn_1._inputs : InputCP_2, \
                                                              rnn_1._inputs_g : Input9_g, \
                                                              rnn_1.initial_hidden_states : p_h_states_sv_blanks_2[:, -1, :]}), 0, 1)
p_predictcp_2 = np.swapaxes(sess.run(p_outputs, feed_dict = {rnn_1._inputs : InputCP_2, rnn_1._inputs_g : Input9_g, \
                                                            rnn_1.initial_hidden_states : p_h_states_sv_blanks_2[:, -1, :]}), 0, 1)

p_h_states_sva_blanks_2 = np.swapaxes(sess.run(p_h_states_s, feed_dict = {rnn_1_s._inputs : np.zeros((12500, blankperiod, 64)).astype(np.float32), \
                                                                      rnn_1_s._inputs_g : np.zeros((12500, blankperiod, 100)).astype(np.float32), \
                                                                      rnn_1_s.initial_hidden_states : p_h_states_vcp_2[:, 0, :]}), 0, 1)
p_predict_sva_blanks_2 = np.swapaxes(sess.run(p_outputs_s, feed_dict = {rnn_1_s._inputs : np.zeros((12500, blankperiod, 64)).astype(np.float32), \
                                                                      rnn_1_s._inputs_g : np.zeros((12500, blankperiod, 100)).astype(np.float32), \
                                                                      rnn_1_s.initial_hidden_states : p_h_states_vcp_2[:, 0, :]}), 0, 1)

p_predict_2 = concat((p_predict0_2, p_predict_sv_blanks_2, p_predictcp_2, p_predict_sva_blanks_2), axis = 1)
p_h_statesV_2 = concat((p_h_states_v0_2, p_h_states_sv_blanks_2, p_h_states_vcp_2, p_h_states_sva_blanks_2), axis = 1)

# 3
p_h_states_v0_3 = np.swapaxes(sess.run(p_h_states, feed_dict = {rnn_1._inputs : Input0_3, rnn_1._inputs_g: Input0_g}), 0, 1)
p_predict0_3 = np.swapaxes(sess.run(p_outputs, feed_dict = {rnn_1._inputs : Input0_3, rnn_1._inputs_g : Input0_g}), 0, 1)

p_h_states_sv_blanks_3 = np.swapaxes(sess.run(p_h_states_s, feed_dict = {rnn_1_s._inputs : np.zeros((12500, blankperiod, 64)).astype(np.float32), \
                                                                      rnn_1_s._inputs_g : np.zeros((12500, blankperiod, 100)).astype(np.float32), \
                                                                      rnn_1_s.initial_hidden_states : p_h_states_v0_3[:, 0, :]}), 0, 1)
p_predict_sv_blanks_3 = np.swapaxes(sess.run(p_outputs_s, feed_dict = {rnn_1_s._inputs : np.zeros((12500, blankperiod, 64)).astype(np.float32), \
                                                                      rnn_1_s._inputs_g : np.zeros((12500, blankperiod, 100)).astype(np.float32), \
                                                                      rnn_1_s.initial_hidden_states : p_h_states_v0_3[:, 0, :]}), 0, 1)

    
p_h_states_vcp_3 = np.swapaxes(sess.run(p_h_states, feed_dict = {rnn_1._inputs : InputCP_3, \
                                                              rnn_1._inputs_g : Input9_g, \
                                                              rnn_1.initial_hidden_states : p_h_states_sv_blanks_3[:, -1, :]}), 0, 1)
p_predictcp_3 = np.swapaxes(sess.run(p_outputs, feed_dict = {rnn_1._inputs : InputCP_3, rnn_1._inputs_g : Input9_g, \
                                                            rnn_1.initial_hidden_states : p_h_states_sv_blanks_3[:, -1, :]}), 0, 1)

p_h_states_sva_blanks_3 = np.swapaxes(sess.run(p_h_states_s, feed_dict = {rnn_1_s._inputs : np.zeros((12500, blankperiod, 64)).astype(np.float32), \
                                                                      rnn_1_s._inputs_g : np.zeros((12500, blankperiod, 100)).astype(np.float32), \
                                                                      rnn_1_s.initial_hidden_states : p_h_states_vcp_3[:, 0, :]}), 0, 1)
p_predict_sva_blanks_3 = np.swapaxes(sess.run(p_outputs_s, feed_dict = {rnn_1_s._inputs : np.zeros((12500, blankperiod, 64)).astype(np.float32), \
                                                                      rnn_1_s._inputs_g : np.zeros((12500, blankperiod, 100)).astype(np.float32), \
                                                                      rnn_1_s.initial_hidden_states : p_h_states_vcp_3[:, 0, :]}), 0, 1)


p_predict_3 = concat((p_predict0_3, p_predict_sv_blanks_3, p_predictcp_3, p_predict_sva_blanks_3), axis = 1)
p_h_statesV_3 = concat((p_h_states_v0_3, p_h_states_sv_blanks_3, p_h_states_vcp_3, p_h_states_sva_blanks_3), axis = 1)

# 4
p_h_states_v0_4 = np.swapaxes(sess.run(p_h_states, feed_dict = {rnn_1._inputs : Input0_4, rnn_1._inputs_g: Input0_g}), 0, 1)
p_predict0_4 = np.swapaxes(sess.run(p_outputs, feed_dict = {rnn_1._inputs : Input0_4, rnn_1._inputs_g : Input0_g}), 0, 1)

p_h_states_sv_blanks_4 = np.swapaxes(sess.run(p_h_states_s, feed_dict = {rnn_1_s._inputs : np.zeros((12500, blankperiod, 64)).astype(np.float32), \
                                                                      rnn_1_s._inputs_g : np.zeros((12500, blankperiod, 100)).astype(np.float32), \
                                                                      rnn_1_s.initial_hidden_states : p_h_states_v0_4[:, 0, :]}), 0, 1)
p_predict_sv_blanks_4 = np.swapaxes(sess.run(p_outputs_s, feed_dict = {rnn_1_s._inputs : np.zeros((12500, blankperiod, 64)).astype(np.float32), \
                                                                      rnn_1_s._inputs_g : np.zeros((12500, blankperiod, 100)).astype(np.float32), \
                                                                      rnn_1_s.initial_hidden_states : p_h_states_v0_4[:, 0, :]}), 0, 1)

    
p_h_states_vcp_4 = np.swapaxes(sess.run(p_h_states, feed_dict = {rnn_1._inputs : InputCP_4, \
                                                              rnn_1._inputs_g : Input9_g, \
                                                              rnn_1.initial_hidden_states : p_h_states_sv_blanks_4[:, -1, :]}), 0, 1)
p_predictcp_4 = np.swapaxes(sess.run(p_outputs, feed_dict = {rnn_1._inputs : InputCP_4, rnn_1._inputs_g : Input9_g, \
                                                            rnn_1.initial_hidden_states : p_h_states_sv_blanks_4[:, -1, :]}), 0, 1)

p_h_states_sva_blanks_4 = np.swapaxes(sess.run(p_h_states_s, feed_dict = {rnn_1_s._inputs : np.zeros((12500, blankperiod, 64)).astype(np.float32), \
                                                                      rnn_1_s._inputs_g : np.zeros((12500, blankperiod, 100)).astype(np.float32), \
                                                                      rnn_1_s.initial_hidden_states : p_h_states_vcp_4[:, 0, :]}), 0, 1)
p_predict_sva_blanks_4 = np.swapaxes(sess.run(p_outputs_s, feed_dict = {rnn_1_s._inputs : np.zeros((12500, blankperiod, 64)).astype(np.float32), \
                                                                      rnn_1_s._inputs_g : np.zeros((12500, blankperiod, 100)).astype(np.float32), \
                                                                      rnn_1_s.initial_hidden_states : p_h_states_vcp_4[:, 0, :]}), 0, 1)

p_predict_4 = concat((p_predict0_4, p_predict_sv_blanks_4, p_predictcp_4, p_predict_sva_blanks_4), axis = 1)
p_h_statesV_4 = concat((p_h_states_v0_4, p_h_states_sv_blanks_4, p_h_states_vcp_4, p_h_states_sva_blanks_4), axis = 1)



predict_1 = sess.run(outputs, feed_dict={rnn_2._inputs: (p_predict_1 > 0.5).astype(np.float32)})
predict_2 = sess.run(outputs, feed_dict={rnn_2._inputs: (p_predict_2 > 0.5).astype(np.float32)})
predict_3 = sess.run(outputs, feed_dict={rnn_2._inputs: (p_predict_3 > 0.5).astype(np.float32)})
predict_4 = sess.run(outputs, feed_dict={rnn_2._inputs: (p_predict_4 > 0.5).astype(np.float32)})

p_predict_1 = np.swapaxes(np.float32(p_predict_1 > 0.5), 0, 1)
p_predict_2 = np.swapaxes(np.float32(p_predict_2 > 0.5), 0, 1)
p_predict_3 = np.swapaxes(np.float32(p_predict_3 > 0.5), 0, 1)
p_predict_4 = np.swapaxes(np.float32(p_predict_4 > 0.5), 0, 1)
#input_cd = np.swapaxes(p_predict,0,1)


z = tf.placeholder(tf.float32,shape=[None, 64])
q = tf.reshape(z,(12500,8,8))

result1_a, result2_a = np.zeros((seq_l,800,1000)), np.zeros((seq_l,800,1000))
result1_b, result2_b = np.zeros((seq_l,800,1000)), np.zeros((seq_l,800,1000))
result1_c, result2_c = np.zeros((seq_l,800,1000)), np.zeros((seq_l,800,1000))
result1_d, result2_d = np.zeros((seq_l,800,1000)), np.zeros((seq_l,800,1000))

for itr in range(seq_l):
    out1_a, out2_a = sess.run(q, feed_dict={z:p_predict_1[itr,:,:]}), sess.run(q, feed_dict={z:predict_1[itr,:,:]})
    out1_b, out2_b = sess.run(q, feed_dict={z:p_predict_2[itr,:,:]}), sess.run(q, feed_dict={z:predict_2[itr,:,:]})
    out1_c, out2_c = sess.run(q, feed_dict={z:p_predict_3[itr,:,:]}), sess.run(q, feed_dict={z:predict_3[itr,:,:]})
    out1_d, out2_d = sess.run(q, feed_dict={z:p_predict_4[itr,:,:]}), sess.run(q, feed_dict={z:predict_4[itr,:,:]}) 
    for i in range(12276):
        m = int(i / (125-1))
        n = int(i % (125-1))
        result1_a[itr,m*8:(m+1)*8,n*8:(n+1)*8]= out1_a[i,:,:] ### update np.abs
        result2_a[itr,m*8:(m+1)*8,n*8:(n+1)*8]= out2_a[i,:,:] ### update np.abs
        
        result1_b[itr, 4+m*8:4+(m+1)*8, n*8:(n+1)*8]= out1_b[i,:,:] ### update np.abs
        result2_b[itr, 4+m*8:4+(m+1)*8, n*8:(n+1)*8]= out2_b[i,:,:] ### update np.abs
        
        result1_c[itr,m*8:(m+1)*8, 4+n*8:4+(n+1)*8]= out1_c[i,:,:] ### update np.abs
        result2_c[itr,m*8:(m+1)*8, 4+n*8:4+(n+1)*8]= out2_c[i,:,:] ### update np.abs

        result1_d[itr, 4+m*8:4+(m+1)*8, 4+n*8:4+(n+1)*8]= out1_d[i,:,:] ### update np.abs
        result2_d[itr, 4+m*8:4+(m+1)*8, 4+n*8:4+(n+1)*8]= out2_d[i,:,:] ### update np.abs
        
result_1_s = result1_a + result1_b + result1_c + result1_d
result_2_s = result2_a + result2_b + result2_c + result2_d


# In[]: micro-stimulation results in UR (upper-right) field location in 1000 x 800 space
path = "./Figures/"
create_folder(path)
folder = path + "GratingsExp_StimUR_BlankPeriod/"
create_folder(folder)

result_final_per_s, result_final_cd_s = np.copy(result_1), np.copy(result_2)

result_final_per_s[:, :400,500:] = result_1_s[:,:400,500:]
result_final_cd_s[:,:400,500:] = result_2_s[:,:400,500:]

# result_final_per_s = ((result_final_per_s / 4) > 0.5).astype(np.float32)
# result_final_cd_s = ((result_final_cd_s / 4) > 0.5).astype(np.float32)

result_final_per_s = np.float32((result_final_per_s / 4.0) > 0.5)
result_final_cd_s = np.float32((result_final_cd_s / 4.0) > 0.6)

folder_per = folder + 'Per/'
create_folder(folder_per)

folder_cd = folder + 'CD/'
create_folder(folder_cd)

for i in range(seq_l):
    imsave("%sPer_stimulated_%d.png"%(folder_per, i), ((1 - result_final_per_s[i, :, :])*255.0).astype(np.uint8), cmap = 'gray', vmin = 0, vmax = 255)
    imsave("%sCD_stimulated_%d.png"%(folder_cd, i), ((1 - result_final_cd_s[i, :, :])*255.0).astype(np.uint8), cmap = 'gray', vmin = 0, vmax = 255)
   
   
# In[]: micro-stimulation results for LL (lower-left) field in a 1000 x 800 space
path = "./Figures/"
create_folder(path)
folder = path + "GratingsExp_StimLL_BlankPeriod/"
create_folder(folder)

result_final_per_s, result_final_cd_s = np.copy(result_1), np.copy(result_2)

result_final_per_s[:, 400:, :500] = result_1_s[:, 400:,:500]
result_final_cd_s[:, 400:, :500] = result_2_s[:, 400:,:500]

# result_final_per_s = ((result_final_per_s / 4) > 0.5).astype(np.float32)
# result_final_cd_s = ((result_final_cd_s / 4) > 0.5).astype(np.float32)

result_final_per_s = np.float32((result_final_per_s / 4.0) > 0.5)
result_final_cd_s = np.float32((result_final_cd_s / 4.0) > 0.6)

folder_per = folder + 'Per/'
create_folder(folder_per)

folder_cd = folder + 'CD/'
create_folder(folder_cd)

for i in range(seq_l):
    imsave("%sPer_stimulated_%d.png"%(folder_per, i), ((1 - result_final_per_s[i, :, :])*255.0).astype(np.uint8), cmap = 'gray', vmin = 0, vmax = 255)
    imsave("%sCD_stimulated_%d.png"%(folder_cd, i), ((1 - result_final_cd_s[i, :, :])*255.0).astype(np.uint8), cmap = 'gray', vmin = 0, vmax = 255)
    
with h5py.File("./ll_stim.h5", "w") as f:
    f.create_dataset("/per", data = result_final_per_s)
    f.create_dataset("/cd", data = result_final_cd_s)