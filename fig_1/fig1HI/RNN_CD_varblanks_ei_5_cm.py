# -*- coding: utf-8 -*-
"""
@author(s): Jogendra Kundu,
            Yash M. Sawant

@last edited: Dec, 2021
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from numpy.matlib import repmat
from matplotlib import pyplot as plt
import random
np.random.seed(21)
import h5py
import os
import sys
from os import mkdir
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


CM = np.zeros((64,64),dtype=np.float32)
l = [(0,3),(2,5),(4,7)]
for i in range(3):
        for j in range(3):
                for p in range(l[i][0],l[i][1]+1):
                        for q in range(l[j][0],l[j][1]+1):
                                for r in range(l[i][0],l[i][1]+1):
                                        for s in range(l[j][0],l[j][1]+1):
                                                m = (p)*8 + q
                                                n = (r)*8 + s
                                                CM[m,n] = 1.0

################## define all connectivity matrix #########################
###########################################################################
Wi_sign_mask = np.zeros((64,64), dtype = np.float32) # always row wise +ve or -ve
for i in range(64): Wi_sign_mask[i,i] = 1.0

Wh_sign_mask = np.zeros((320,320), dtype = np.float32)
for i in range(256):    Wh_sign_mask[i,i] = 1.0

for i in range(256,320):    Wh_sign_mask[i,i] = -1.0

Wo_sign_mask = np.zeros((320,320), dtype = np.float32)
for i in range(256):    Wo_sign_mask[i,i] = 1.0

for i in range(256,320):    Wo_sign_mask[i,i] = -1.0

######### connectivity matrices ###########
CM_64_256 = np.zeros((64,256),dtype = np.float32)
i,j = 1,1
while i<=256:
    for aa in [i, i+1, i+16, i+16+1]:
        CM_64_256[:,aa-1] = CM[:,j-1]
    j=j+1
    if (i+1)%16 == 0: i=i+1+16+1
    else: i=i+2


CM_256_256 = np.zeros((256,256),dtype=np.float32)
l = [(0,3),(3,6),(6,9),(9,12),(12,15)]
for i in range(5):
        for j in range(5):
                for p in range(l[i][0],l[i][1]+1):
                        for q in range(l[j][0],l[j][1]+1):
                                for r in range(l[i][0],l[i][1]+1):
                                        for s in range(l[j][0],l[j][1]+1):
                                                m = (p)*16 + q
                                                n = (r)*16 + s
                                                CM_256_256[m,n] = 1.0

CM_256_64 = np.zeros((256,64),dtype = np.float32)
CM_256_64 = CM_64_256.T

CM_64_64 = CM
Wi_mask_big = np.append(CM_64_256, CM_64_64, axis=1)
Wh_mask_big = np.append( np.append(CM_256_256, CM_256_64, axis=1), 
                np.append(CM_64_256, CM_64_64, axis=1), axis=0)
Wo_mask_big = np.append(CM_256_64, CM_64_64, axis=0)

# connecting 8 x 8 topological input space to 8 x 8 output space; DEFINE CONNECTIVITY MATRICES USING rd (radius of connections) and st (sparsity in connections) 

sz = 8
rd = 4
st = 3

l = [[0, rd-1]]
n = (np.floor((10-rd)/st) + 1).astype(np.int32)

for i in range(n-1):
    l.append([l[-1][0] + st, l[-1][0] + st + rd - 1])

l = np.array(l)
##print(l)

matr_tmp = np.zeros((10 ** 2, 10 ** 2))
for i in range(l.shape[0]):
    for j in range(l.shape[0]):
        for p in range(l[i, 0], l[i, 1] + 1):
            for q in range(l[j, 0], l[j, 1] + 1):
                for r in range(l[i, 0], l[i, 1] + 1):
                    for s in range(l[j, 0], l[j, 1] + 1):
                        m = p * 10 + q
                        n = r * 10 + s
                        matr_tmp[m, n] = 1
                        

idxs = []
for i in range(10):
    for j in range(10):
        if i < 8 and j < 8:
            idxs.append(i * 10 + j)
idxs = np.array(idxs).reshape((64))
idxs_x = repmat(idxs, 64, 1)
idxs_y = np.transpose(idxs_x)

cm_64_64 = matr_tmp[idxs_x, idxs_y]


# connectivity of 8 x 8 topological space (t.s.) to 16 x 16 t.s.

cm_64_256 = np.zeros((64, 256), dtype = np.float32)

i,j = 1,1
while i<=256:
    for aa in [i, i+1, i+16, i+16+1]:
        cm_64_256[:,aa-1] = cm_64_64[:,j-1]
    j=j+1
    if (i+1)%16 == 0: i=i+1+16+1
    else: i=i+2

        
rd = 4
st = 4
# connectivity of 16 x 16 t.s. to 16 x 16 t.s.
cm_256_256 = np.zeros((256, 256), dtype = np.float32)
l = [[0, rd-1]]
n = (np.floor((18-rd)/st) + 1).astype(np.int32)

for i in range(n-1):
    l.append([l[-1][0] + st, l[-1][0] + st + rd - 1])

l = np.array(l)
##print(l)

matr_tmp = np.zeros((18 ** 2, 18 ** 2))
for i in range(l.shape[0]):
    for j in range(l.shape[0]):
        for p in range(l[i, 0], l[i, 1] + 1):
            for q in range(l[j, 0], l[j, 1] + 1):
                for r in range(l[i, 0], l[i, 1] + 1):
                    for s in range(l[j, 0], l[j, 1] + 1):
                        m = p * 18 + q
                        n = r * 18 + s
                        matr_tmp[m, n] = 1

idxs = []
for i in range(18):
    for j in range(18):
        if i < 16 and j < 16:
            idxs.append(i * 18 + j)

idxs_x = repmat(idxs, 256, 1)
idxs_y = np.transpose(idxs_x)

cm_256_256 = matr_tmp[idxs_x, idxs_y]


Wi_mask = np.append(cm_64_256, cm_64_64, axis = 1)
Wh_mask = np.append(np.append(cm_256_256, cm_64_256.T, axis = 1), 
                    np.append(cm_64_256, cm_64_64, axis = 1), axis = 0)
Wo_mask = np.append(cm_64_256.T, cm_64_64, axis = 0)

# In[]:
class RNN_cell(object):

    """
    RNN cell object which takes 3 arguments for initialization.
    input_size = Input Vector size
    hidden_layer_size = Hidden layer size
    target_size = Output vector size

    """

    #def __init__(self, input_size, hidden_layer_size, target_size,weights_x,weights_h,weights_o,bias_i,bias_o):
    def __init__(self, input_size, hidden_layer_size, target_size):
        # Initialization of given values
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.target_size = target_size

        ################################
        self.Wi_mask_big = tf.constant(Wi_mask_big)
        self.Wh_mask_big = tf.constant(Wh_mask_big)
        self.Wo_mask_big = tf.constant(Wo_mask_big)
        self.Wi_sign_mask = tf.constant(Wi_sign_mask)
        self.Wh_sign_mask = tf.constant(Wh_sign_mask)
        self.Wo_sign_mask = tf.constant(Wo_sign_mask)
        
        # Weights and Bias for input and hidden tensor

        self.Wx =  tf.matmul(self.Wi_sign_mask,tf.nn.relu(Wi_mask * tf.Variable(tf.truncated_normal([self.input_size, self.hidden_layer_size],mean = 0.2,stddev = 0.01))))

        self.Wh = tf.matmul(self.Wh_sign_mask,tf.nn.relu(Wh_mask * tf.Variable(tf.truncated_normal([self.hidden_layer_size, self.hidden_layer_size],mean = 0.2,stddev = 0.01))))

        self.bi = tf.Variable(tf.zeros([self.hidden_layer_size]))

        # Weights for output layers ### update
        self.Wo = tf.matmul(self.Wo_sign_mask,tf.nn.relu(Wo_mask * tf.Variable(tf.truncated_normal([self.hidden_layer_size, self.target_size],mean=0.1,stddev=0.01))))

        self.bo = tf.Variable(tf.truncated_normal([self.target_size], mean=0.0, stddev=0.05))   
        '''
        ##############################
        self.Wx =  tf.matmul(self.Wi_sign_mask,tf.nn.relu(Wi_mask_big * tf.Variable(tf.constant(weights_x))))
        self.Wh = tf.matmul(self.Wh_sign_mask,tf.nn.relu(Wh_mask_big * tf.Variable(tf.constant(weights_h))))
        self.bi = tf.Variable(tf.zeros([self.hidden_layer_size]))
        self.Wo = tf.matmul(self.Wo_sign_mask,tf.nn.relu(Wo_mask_big * tf.Variable(tf.constant(weights_o))))
        self.bo = tf.Variable(tf.constant(bias_o))
        '''
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
        output = tf.sigmoid(tf.matmul(tf.nn.relu(hidden_state), self.Wo) + self.bo) ### update + self.bo

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


# # Placeholder and initializers

hidden_layer_size = 256+64
input_size =  64
target_size = 64
# In[]:
# Initializing rnn object
tf.reset_default_graph()
rnn = RNN_cell(input_size, hidden_layer_size, target_size)
#rnn = RNN_cell(input_size, hidden_layer_size, target_size, Wx, Wh, Wo, Bi, Bo)
y = tf.placeholder(tf.float32, shape=[None, 10, target_size], name='inputs')

# Getting all outputs from rnn
outputs = rnn.get_outputs()

# Getting final output through indexing after reversing
output_0 = tf.reverse(outputs, [0])[0, :, :]
output_1 = tf.reverse(outputs, [0])[1, :, :]
output_2 = tf.reverse(outputs, [0])[2, :, :]
output_3 = tf.reverse(outputs, [0])[3, :, :]
output_4 = tf.reverse(outputs, [0])[4, :, :]
output_5 = tf.reverse(outputs, [0])[5, :, :]
output_6 = tf.reverse(outputs, [0])[6, :, :]
output_7 = tf.reverse(outputs, [0])[7, :, :]
output_8 = tf.reverse(outputs, [0])[8, :, :]
output_9 = tf.reverse(outputs, [0])[9, :, :]

# Getting all outputs from rnn
h_states  = rnn.get_states()

# Getting final output through indexing after reversing
h_0 = tf.reverse(h_states, [0])[0, :, :]
h_1 = tf.reverse(h_states, [0])[1, :, :]
h_2 = tf.reverse(h_states, [0])[2, :, :]
h_3 = tf.reverse(h_states, [0])[3, :, :]
h_4 = tf.reverse(h_states, [0])[4, :, :]
h_5 = tf.reverse(h_states, [0])[5, :, :]
h_6 = tf.reverse(h_states, [0])[6, :, :]
h_7 = tf.reverse(h_states, [0])[7, :, :]
h_8 = tf.reverse(h_states, [0])[8, :, :]
h_9 = tf.reverse(h_states, [0])[9, :, :]


target_0 = y[:,9,:]
target_1 = y[:,8,:]
target_2 = y[:,7,:]
target_3 = y[:,6,:]
target_4 = y[:,5,:]
target_5 = y[:,4,:]
target_6 = y[:,3,:]
target_7 = y[:,2,:]
target_8 = y[:,1,:]
target_9 = y[:,0,:]

### update bce
loss = (tf.losses.mean_squared_error(target_0, output_0) +
    tf.losses.mean_squared_error(target_1, output_1) +
    tf.losses.mean_squared_error(target_2, output_2) +
    tf.losses.mean_squared_error(target_3, output_3) +
    tf.losses.mean_squared_error(target_4, output_4) +
    tf.losses.mean_squared_error(target_5, output_5) +
    tf.losses.mean_squared_error(target_6, output_6) +
    tf.losses.mean_squared_error(target_7, output_7) +
    tf.losses.mean_squared_error(target_8, output_8))
train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)


# In[]:
sample_size = 200000
M = np.load('./x_full_data_exp_5_modified.npy')
N = np.load('./y_full_data_exp_5_modified.npy')
print('Finished loading dataset')

# Getting Train and test Dataset
ll_idx = np.random.permutation(sample_size)
ll_idx_train, ll_idx_test = ll_idx[:180000], ll_idx[-20000:]
X_train, X_test, y_train, y_test = M[ll_idx_train],  M[ll_idx_test], N[ll_idx_train], N[ll_idx_test]

X_train = X_train.astype(np.float32)
y_train = np.absolute(y_train.astype(np.float32)) ### update
X_test = X_test.astype(np.float32)
y_test = np.absolute(y_test.astype(np.float32)) ### update

# Cuttting for simple iteration
X_train = X_train[:180000]
y_train = y_train[:180000]


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# In[]:
try:
    sess.close()
except:
    print('Error closing session')
sess = tf.InteractiveSession(config=config)
    
#sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
Wh_v = sess.run(rnn.Wh)
plt.imshow(Wh_v)
print(path)
for it in range(1):

    num_epochs = 1000
    test_epoch_start = 39995  
    data = np.zeros(num_epochs)
    
    # Iterations to do trainning
    # while (epoch < num_epochs): ### update
    epoch = -1
    #for epoch in range(num_epochs): 
    while epoch<(num_epochs-1):
        epoch+=1
        start = 0
        end = 128
        for i in range((sample_size-20000)//128):
        #print(epoch,':-',i)
            X = X_train[start:end]
            Y = y_train[start:end]
            start = end
            end = start + 128
            sess.run(train_step,feed_dict={rnn._inputs: X, y: Y})   
    
        L = sess.run([loss], feed_dict={rnn._inputs: X_test, y: y_test})
        data[epoch] = L[-1] 
    
        sys.stdout.flush()
        print("\rIteration: %s Loss: %f " % (epoch, L[-1])),
        sys.stdout.flush()
        start = 0
        if epoch % 10 == 0:
            w_x = sess.run(rnn.Wx)
            w_h = sess.run(rnn.Wh)
            w_o = sess.run(rnn.Wo)
            b_o = sess.run(rnn.bo)
            # stride for 16 x 16 to 16 x 16 is one more than for 8 x 8 to 8 x 8
            with h5py.File(path + "/Epoch_%d_iter_%d.h5"%(epoch, it), "w") as f:
                f.create_dataset("Wx", data = w_x)
                f.create_dataset("Wh", data = w_h)
                f.create_dataset("Wo", data = w_o)
                f.create_dataset("bo", data = b_o)
                f.create_dataset("epoch_losses", data = data)
