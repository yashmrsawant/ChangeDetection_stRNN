# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 11:23:22 2022

@author: yashm
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import sys
import numpy as np
#plt.rcParams['image.cmap'] = 'gray'
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#np.random.seed(41)

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
for i in range(64):
        CM_64_256[:,4*i] = CM[i,:]
        CM_64_256[:,4*i + 1] = CM[i,:]
        CM_64_256[:,4*i + 2] = CM[i,:]
        CM_64_256[:,4*i + 3] = CM[i,:]

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
for i in range(64):
        CM_256_64[4*i,:] = CM[i,:]
        CM_256_64[4*i + 1,:] = CM[i,:]
        CM_256_64[4*i + 2,:] = CM[i,:]
        CM_256_64[4*i + 3,:] = CM[i,:]

CM_64_64 = CM

for iters in range(1, 8):
    
    Wi_mask_big = np.append(CM_64_256, CM_64_64, axis=1)
    Wh_mask_big = np.append( np.append(CM_256_256, CM_256_64, axis=1), 
                    np.append(CM_64_256, CM_64_64, axis=1), axis=0)
    Wo_mask_big = np.append(CM_256_64, CM_64_64, axis=0)
    topographic = True
    if np.random.random() < 1:
    
        Wi_v = Wi_mask_big.ravel()
        Wh_v = Wh_mask_big.ravel()
        Wo_v = Wo_mask_big.ravel()
        np.random.shuffle(Wi_v)
        np.random.shuffle(Wh_v)
        np.random.shuffle(Wo_v)
        Wi_mask_big = Wi_v.reshape(64,320)
        Wo_mask_big = Wo_v.reshape(320,64)
        Wh_mask_big = Wh_v.reshape(320,320)
        topographic = False
    print("Topographic = %d"%(topographic))
    ##############################################################################
    # Vanilla RNN class and functions
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
    
            self.Wx =  tf.matmul(self.Wi_sign_mask,tf.nn.relu(Wi_mask_big * tf.Variable(tf.truncated_normal([self.input_size, self.hidden_layer_size],mean = 0.2,stddev = 0.01))))
    
            self.Wh = tf.matmul(self.Wh_sign_mask,tf.nn.relu(Wh_mask_big * tf.Variable(tf.truncated_normal([self.hidden_layer_size, self.hidden_layer_size],mean = 0.2,stddev = 0.01))))
    
            self.bi = tf.Variable(tf.zeros([self.hidden_layer_size]))
    
            # Weights for output layers ### update
            self.Wo = tf.matmul(self.Wo_sign_mask,tf.nn.relu(Wo_mask_big * tf.Variable(tf.truncated_normal([self.hidden_layer_size, self.target_size],mean=0.1,stddev=0.01))))
    
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
    
    
    y = tf.placeholder(tf.float32, shape=[None, 10, target_size], name='inputs')
    '''
    # Models ### update
    Wx = np.loadtxt('./rnn_cd_jnk_exp_3_weights/Wx238.csv')
    Wh = np.loadtxt('./rnn_cd_jnk_exp_3_weights/Wh238.csv')
    Wo = np.loadtxt('./rnn_cd_jnk_exp_3_weights/Wo238.csv')
    Bi = np.loadtxt('./rnn_cd_jnk_exp_3_weights/bi238.csv')
    Bo = np.loadtxt('./rnn_cd_jnk_exp_3_weights/bo238.csv')
    Wx = Wx.astype(np.float32)
    Wh = Wh.astype(np.float32)
    Wo = Wo.astype(np.float32)
    Bi = Bi.astype(np.float32)
    Bo = Bo.astype(np.float32)
    '''
    # Initializing rnn object
    rnn = RNN_cell(input_size, hidden_layer_size, target_size)
    #rnn = RNN_cell(input_size, hidden_layer_size, target_size, Wx, Wh, Wo, Bi, Bo)
    
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
    #tf.losses.mean_squared_error(target_9, output_9) )
    
    # Create a summary to monitor loss tensor
    #tf.scalar_summary("loss", loss)
    
    # Merge all summaries into a single op
    #merged_summary_op = tf.merge_all_summaries()
    
    # Trainning with Adadelta Optimizer
    train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
    
    # Calculatio of correct prediction and accuracy
    # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
    # accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32))) * 100
    
    #correct_prediction = np.sum(y==output_0)
    #accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32))) * 100
    # accuracy = tf.cast(correct_prediction,tf.float32)
    
    #logs_path = os.path.join(os.getcwd(),"logdata")
    #summary_writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
    
    # Creating the dataset to train RNN to perform change detection
    #############################################################################################
    '''
    sample_size = 200000
    BX = np.zeros((2,64))
    BX.fill(0.0)
    M = np.zeros((sample_size,10,64))
    N = np.zeros((sample_size,10,64))
    counter = 0
    for i in range(sample_size):
        sparsity = np.random.sample()
        a_t = np.random.choice([0,1],size=(8,8),p=[sparsity,1-sparsity])
        choice = np.random.sample()
        ################### copy the same b_t ######################
        if(choice <= 0.1):
            b_t = a_t.copy()
        elif (choice>0.1) & (choice<=0.4):
            ################### entirly new sample #################
            sparsity_new = np.random.sample()
            b_t = np.random.choice([0,1],size=(8,8),p=[sparsity_new,1-sparsity_new])
        elif (choice>0.4) & (choice<=0.5):
            ################### all blank samples #################
            a_t = BX[1].copy()
        elif (choice>0.5) & (choice<=0.6):
            b_t = BX[1].copy()
        else:
            xx = np.random.randint(0,8)
            yy = np.random.randint(0,8)
            half_width = np.random.randint(1,4)
            half_height = np.random.randint(1,4)
            b_t = a_t.copy()
            for p in range(half_height):
                for q in range(half_width):
                    b_t[np.min([xx+p,7]),np.min([yy+q,7])]=0
                    b_t[np.min([xx+p,7]),np.max([yy-q,0])]=0
                    b_t[np.max([xx-p,0]),np.max([yy-q,0])]=0
                    b_t[np.max([xx-p,0]),np.min([yy+q,7])]=0
                    a_t[np.min([xx+p,7]),np.min([yy+q,7])]=1
                    a_t[np.min([xx+p,7]),np.max([yy-q,0])]=1
                    a_t[np.max([xx-p,0]),np.max([yy-q,0])]=1
                    a_t[np.max([xx-p,0]),np.min([yy+q,7])]=1
            if choice>0.8:
                ################### flip 20% of the time #############
                a_t, b_t = b_t, a_t
        ################### chose blanks in between ##################
        a_t = a_t.astype(np.float32)
        b_t = b_t.astype(np.float32)        
        a_t = np.reshape(a_t,(1,64))
        b_t = np.reshape(b_t,(1,64))
        latest, alternate = a_t, b_t
        N[counter,0,:], M[counter,0,:] = BX[1], latest
        for j in range(9):
            t = np.random.sample()
            if(t>0.65):
                N[counter,j+1,:] = BX[1] 
            else:
                N[counter,j+1,:] = latest - alternate
                latest, alternate = alternate, latest    
            M[counter,j+1,:] = latest
        counter += 1
    ##################################################################################################
    aa = aa+a1
    '''
    ######## modified with more blanks and full blank samples in dataset ############
    #np.save('x_full_data_exp_5_modified.npy', M>0.5)
    #np.save('y_full_data_exp_5_modified.npy', np.absolute(N)>0.5)
    
    sample_size = 200000
    M = np.load('../x_full_data_exp_5_modified.npy')
    N = np.load('../y_full_data_exp_5_modified.npy')
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

    try:
        sess.close()
    except:
        print("Error closing session")
    sess = tf.InteractiveSession(config=config)
    
    #sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    
    num_epochs = 151
    test_epoch_start = 39995  
    data = np.zeros(num_epochs+1)
    
    # Iterations to do trainning
    # while (epoch < num_epochs): ### update
    epoch = -1
    #for epoch in range(num_epochs): 
    while epoch<num_epochs:
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
    
        L = sess.run([loss], feed_dict={rnn._inputs: X, y: Y})
        data[epoch] = L[-1] 
        if epoch>test_epoch_start or epoch%4==0:  
            for j in range(10):
                r = np.random.randint(1,100)
                X_test, y_test = X_test[:100], y_test[:100] ### update
                predict_0 = sess.run(output_0, feed_dict={rnn._inputs: X_test, y: y_test})
                predict_1 = sess.run(output_1, feed_dict={rnn._inputs: X_test, y: y_test})
                predict_2 = sess.run(output_2, feed_dict={rnn._inputs: X_test, y: y_test})
                predict_3 = sess.run(output_3, feed_dict={rnn._inputs: X_test, y: y_test})
                predict_4 = sess.run(output_4, feed_dict={rnn._inputs: X_test, y: y_test})
                predict_5 = sess.run(output_5, feed_dict={rnn._inputs: X_test, y: y_test})
                predict_6 = sess.run(output_6, feed_dict={rnn._inputs: X_test, y: y_test})
                predict_7 = sess.run(output_7, feed_dict={rnn._inputs: X_test, y: y_test})
                predict_8 = sess.run(output_8, feed_dict={rnn._inputs: X_test, y: y_test})
                predict_9 = sess.run(output_9, feed_dict={rnn._inputs: X_test, y: y_test})
    
                diff_0 = sess.run(target_0, feed_dict={rnn._inputs: X_test, y: y_test})
                diff_1 = sess.run(target_1, feed_dict={rnn._inputs: X_test, y: y_test})
                diff_2 = sess.run(target_2, feed_dict={rnn._inputs: X_test, y: y_test})
                diff_3 = sess.run(target_3, feed_dict={rnn._inputs: X_test, y: y_test})
                diff_4 = sess.run(target_4, feed_dict={rnn._inputs: X_test, y: y_test})
                diff_5 = sess.run(target_5, feed_dict={rnn._inputs: X_test, y: y_test})
                diff_6 = sess.run(target_6, feed_dict={rnn._inputs: X_test, y: y_test})
                diff_7 = sess.run(target_7, feed_dict={rnn._inputs: X_test, y: y_test})
                diff_8 = sess.run(target_8, feed_dict={rnn._inputs: X_test, y: y_test})
                diff_9 = sess.run(target_9, feed_dict={rnn._inputs: X_test, y: y_test})
    
                state_0 = sess.run(h_0, feed_dict={rnn._inputs: X_test, y: y_test})
                state_1 = sess.run(h_1, feed_dict={rnn._inputs: X_test, y: y_test})
                state_2 = sess.run(h_2, feed_dict={rnn._inputs: X_test, y: y_test})
                state_3 = sess.run(h_3, feed_dict={rnn._inputs: X_test, y: y_test})
                state_4 = sess.run(h_4, feed_dict={rnn._inputs: X_test, y: y_test})      
                state_5 = sess.run(h_5, feed_dict={rnn._inputs: X_test, y: y_test})
                state_6 = sess.run(h_6, feed_dict={rnn._inputs: X_test, y: y_test})
                state_7 = sess.run(h_7, feed_dict={rnn._inputs: X_test, y: y_test})
                state_8 = sess.run(h_8, feed_dict={rnn._inputs: X_test, y: y_test})
                state_9 = sess.run(h_9, feed_dict={rnn._inputs: X_test, y: y_test})      
                
                prediction = np.zeros((10,64))
                prediction[0,:] = predict_0[r,:]
                prediction[1,:] = predict_1[r,:]
                prediction[2,:] = predict_2[r,:]
                prediction[3,:] = predict_3[r,:]
                prediction[4,:] = predict_4[r,:]
                prediction[5,:] = predict_5[r,:]
                prediction[6,:] = predict_6[r,:]
                prediction[7,:] = predict_7[r,:]
                prediction[8,:] = predict_8[r,:]
                prediction[9,:] = predict_9[r,:]
                
                target = np.zeros((10,64))
                target[0,:] = diff_0[r,:]       
                target[1,:] = diff_1[r,:]       
                target[2,:] = diff_2[r,:]       
                target[3,:] = diff_3[r,:]       
                target[4,:] = diff_4[r,:]
                target[5,:] = diff_5[r,:]       
                target[6,:] = diff_6[r,:]       
                target[7,:] = diff_7[r,:]       
                target[8,:] = diff_8[r,:]       
                target[9,:] = diff_9[r,:]
                
                state = np.zeros((10,320))         
                state[0,:] = state_0[r,:]       
                state[1,:] = state_1[r,:]       
                state[2,:] = state_2[r,:]       
                state[3,:] = state_3[r,:]
                state[4,:] = state_4[r,:]
                state[5,:] = state_5[r,:]       
                state[6,:] = state_6[r,:]       
                state[7,:] = state_7[r,:]       
                state[8,:] = state_8[r,:]
                state[9,:] = state_9[r,:]
    
                fig = plt.figure(figsize=(60,40))
                cbar_ax = fig.add_axes([0.95,0.05,0.02,0.9])    
                plt.xlim(-1,1)
                plt.ylim(-1,1)      
    
                for k in range(10):
                    P = X_test[r][k]
                    P = np.reshape(P,(8,8))
                    ax = fig.add_subplot(3,10,k+1)
                    ax.set_axis_off()
                    im = ax.matshow(P,cmap=plt.cm.RdYlBu,vmin=-1.0,vmax=1.0)    
                    #fig.colorbar(im, ax = ax)  
                    ax.set_title(str(k))
    
                for k in range(10):
                    difference = target[9-k,:]
                    difference = np.reshape(difference,(8,8))
                    ax1 = fig.add_subplot(3,10,10 + k + 1)
                    ax1.set_axis_off()
                    im = ax1.matshow(difference,cmap=plt.cm.RdYlBu,vmin=-1.0,vmax = 1.0)
                    #fig.colorbar(im,ax=ax1)    
                    ax1.set_title('Target_%d'%k)
    
                for k in range(10):
                    predict = prediction[9-k,:]
                    predict = np.reshape(predict,(8,8))
                    ax2 = fig.add_subplot(3,10, 20 + k + 1)
                    ax2.set_axis_off()
                    im = ax2.matshow(predict,cmap=plt.cm.RdYlBu,vmin=-1.0,vmax = 1.0)
                    #fig.colorbar(im, ax = ax2) 
                    ax2.set_title('Prediction_%d'%k)
    
                cbar = fig.colorbar(im,cax = cbar_ax)
                plt.savefig('./figures/result_%d_%d.png'%(epoch,j))
                plt.close('all')
                #plt.show()                                 
        
        sys.stdout.flush()
        print("\rIteration: %s Loss: %f " % (epoch, L[-1])),
        sys.stdout.flush()
        start = 0   
        if epoch%1 == 0 :
            np.savetxt('./loss_rnn_ei_cd_%d_sntRNN%d.csv'%(iters,topographic),data)    




