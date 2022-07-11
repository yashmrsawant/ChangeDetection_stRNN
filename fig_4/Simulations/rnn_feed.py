# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:35:24 2019

@author: Yash M. Sawant

"""

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
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
from STRNN import *
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%r %2.2f sec' % \
              (method.__name__, te-ts))
        return result
    return timed


np.random.seed(17)

def plot_matrix(data,cm=plt.cm.RdYlBu,vmin=-1.0,vmax=1.0):
    plt.matshow(data,cmap=cm,vmin=vmin,vmax=vmax)
    plt.show()
    
    
## Load data file
import h5py
matfile = h5py.File('./toyimages.mat', 'r')
toyimages = matfile['toyimages']
toyimages.shape
    

I = 2 * 0


#
I = 2 * 1
toy_1 = np.transpose(toyimages[:, :, I])
toy_2 = np.zeros((24, 24))
toy_3 = np.transpose(toyimages[:, :, I + 1])

#
I = 2 * 2
toy_1 = np.transpose(toyimages[:, :, I])
toy_2 = np.zeros((24, 24))
toy_3 = np.transpose(toyimages[:, :, I + 1])

#
I = 2 * 3
toy_1 = np.transpose(toyimages[:, :, I])
toy_2 = np.zeros((24, 24))
toy_3 = np.transpose(toyimages[:, :, I + 1])

#
I = 2 * 4
toy_1 = np.transpose(toyimages[:, :, I])
toy_2 = np.zeros((24, 24))
toy_3 = np.transpose(toyimages[:, :, I + 1])

functionGI(toy_1, toy_2, toy_3, 0)
functionGI(toy_1, toy_2, toy_3, 1)

##

t1 = 0
t2 = 2
       
def functionGI(toy_1, toy_2, toy_3, GI = 0):
    seq_l = 12
    s1, s2 = toy_1.shape
    data = np.zeros((seq_l, s1, s2), dtype = np.float32)
    data.fill(0.0)

    data[t1, :, :] = toy_1


    #data[1, pos[0] - 1 : pos[0] + s1 - 1, pos[1] - 1 : pos[1] + s2 - 1] = toy_2
    data[t2, :, :] = toy_3
    #data[3, pos[0] - 1 : pos[0] + s1 - 1, pos[1] - 1 : pos[1] + s2 - 1] = toy_2

    data = np.abs(data)
    data_g = np.zeros((seq_l, 8, 10), dtype = np.float32)
    kernel25 = np.ones((4, 4), np.float32) / 16
    data_g_1p = cv2.filter2D(toy_1, -1, kernel25)
    data_g[t1, :, :] = cv2.resize(data_g_1p, (10, 8), interpolation = cv2.INTER_NEAREST) > 0.00001

    data_g_1p = cv2.filter2D(toy_3, -1, kernel25)
    data_g[t2, :, :] = cv2.resize(data_g_1p, (10, 8), interpolation = cv2.INTER_NEAREST) > 0.00001

    sample_size = 9

    M_1 = np.zeros((sample_size, seq_l, 64))
    M_g = np.zeros((sample_size, seq_l, 80))

    counter = 0

    for i in range(3):
        for j in range(3):
            t = data[:, i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8]
            for k in range(data.shape[0]):
                M_1[counter, k, :] = np.reshape(t[k, :, :], (1, 64))
                M_g[counter, k, :] = np.reshape(data_g[k, :, :].astype(np.float32), (1, 80))
            counter += 1

#    plot_matrix(M_1[1, 0, :].reshape((8, 8)))
    
    hidden_layer_size = 256+64
    input_size, input_g_size =  64, 80
    target_size = 64


#    y = tf.placeholder(tf.float32, shape=[None, 3, target_size], name='inputs')
    if GI == 0:
    # Without GI
        GIStr = 'WOGI'
        Wx1 = np.loadtxt('./new_weights/Wx2237_per.csv').astype(np.float32)
        Wh1 = np.loadtxt('./new_weights/Wh2237_per.csv').astype(np.float32)
        Bi1 = np.loadtxt('./new_weights/bi2237_per.csv').astype(np.float32)
        Wo1 = np.loadtxt('./new_weights/Wo2237_per.csv').astype(np.float32)
        Bo1 = np.loadtxt('./new_weights/bo2237_per.csv').astype(np.float32)

        Wx2 = np.loadtxt('./new_weights/Wx9999_cd.csv').astype(np.float32)
        Wh2 = np.loadtxt('./new_weights/Wh9999_cd.csv').astype(np.float32)
        Bi2 = np.loadtxt('./new_weights/bi9999_cd.csv').astype(np.float32)
        Wo2 = np.loadtxt('./new_weights/Wo9999_cd.csv').astype(np.float32)
        Bo2 = np.loadtxt('./new_weights/bo9999_cd.csv').astype(np.float32)
        rnn_1 = RNN_cell_1(input_size, hidden_layer_size, target_size, Wx1, Wh1, Wo1, Bi1, Bo1)

        # Getting all outputs from rnn
        p_outputs = rnn_1.get_outputs()

        # Getting all outputs from rnn
        p_h_states  = rnn_1.get_states()
        # Initializing rnn object
        rnn_2 = RNN_cell_2(input_size, hidden_layer_size, target_size, Wx2, Wh2, Wo2, Bi2, Bo2)
        # Getting all outputs from rnn
        outputs = rnn_2.get_outputs()
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        p_predict_1 = sess.run(p_outputs, feed_dict={rnn_1._inputs: M_1.astype(np.float32)})
        
        p_predict_1 = np.float32(p_predict_1 > 0.5) 
        predict_1 = sess.run(outputs, feed_dict={rnn_2._inputs: np.swapaxes(p_predict_1,0,1)})
    else:
        GIStr = 'WGI'
        Wx2 = np.loadtxt('./new_weights/Wx9999_cd.csv').astype(np.float32)
        Wh2 = np.loadtxt('./new_weights/Wh9999_cd.csv').astype(np.float32)
        Bi2 = np.loadtxt('./new_weights/bi9999_cd.csv').astype(np.float32)
        Wo2 = np.loadtxt('./new_weights/Wo9999_cd.csv').astype(np.float32)
        Bo2 = np.loadtxt('./new_weights/bo9999_cd.csv').astype(np.float32)

        Wg1_g = np.loadtxt('./new_weights/Wg_stRNN.csv').astype(np.float32)
        Wx1_g = np.loadtxt('./new_weights/Wx_stRNN.csv').astype(np.float32)
        Wh1_g = np.loadtxt('./new_weights/Wh_stRNN.csv').astype(np.float32)
        Bi1_g = np.loadtxt('./new_weights/bi_stRNN.csv').astype(np.float32)
        Wo1_g = np.loadtxt('./new_weights/Wo_stRNN.csv').astype(np.float32)
        Bo1_g = np.loadtxt('./new_weights/bo_stRNN.csv').astype(np.float32)
        # Placeholder and initializers
        rnn_1 = RNN_cell_1_g(input_size, hidden_layer_size, target_size, input_g_size, Wx1_g, Wg1_g, Wh1_g,Wo1_g,Bi1_g,Bo1_g)

        p_outputs = rnn_1.get_outputs()
        p_h_states  = rnn_1.get_states()
        # Initializing rnn object
        rnn_2 = RNN_cell_2(input_size, hidden_layer_size, target_size, Wx2, Wh2, Wo2, Bi2, Bo2)
        # Getting all outputs from rnn
        outputs = rnn_2.get_outputs()
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        p_predict_1 = sess.run(p_outputs, feed_dict={rnn_1._inputs: M_1.astype(np.float32), rnn_1._inputs_g:(M_g>0.5).astype(np.float32)})
        p_predict_1 = np.float32(p_predict_1 > 0.5) 
        predict_1 = sess.run(outputs, feed_dict={rnn_2._inputs: np.swapaxes(p_predict_1,0,1)})

        
        
    saveImages(data, seq_l, sess, outputs, p_outputs, p_predict_1, predict_1, GIStr)


def saveImages(data, seq_l, sess, outputs, p_outputs, p_predict_1, predict_1, GIStr):
        ###################### Forward pass through st-RNNs #####################
#    seq_l = data.shape[0]
    global t1, t2
    z = tf.placeholder(tf.float32,shape=[None, 64])
    q = tf.reshape(z,(9,8,8))
    out1_a = sess.run(q, feed_dict = {z : p_predict_1[1, :, :]}) 
    out1_a.shape

    z = tf.placeholder(tf.float32,shape=[None, 64])
    q = tf.reshape(z,(9,8,8))

    result1_a, result2_a = np.zeros((seq_l,24,24)), np.zeros((seq_l,24,24))

    for itr in range(seq_l):
        out1_a, out2_a = sess.run(q, feed_dict={z:p_predict_1[itr,:,:]}), sess.run(q, feed_dict={z:predict_1[itr,:,:]})
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

#    plot_matrix(cv2.resize(result_1[0, :, :], (240, 240), interpolation = cv2.INTER_NEAREST))
#    plot_matrix(data[2, :, :])
    import scipy.misc
    from os import mkdir
    folder = './NewTOYEXAMPLES' + str(I) + GIStr
    try:
        mkdir(folder)
    except:
        print("Exception possibly directory already exists")
    img1 = cv2.resize(data[t1, :, :].reshape((24, 24)), (240, 240), interpolation = cv2.INTER_NEAREST) > 0.00001
    img2 = np.zeros((240, 240))
    img3 = cv2.resize(data[t2, :, :].reshape((24, 24)), (240, 240), interpolation = cv2.INTER_NEAREST) > 0.00001

    scipy.misc.imsave(folder + '/img1.png', ((1 - img1) * 255.0).astype(np.uint8))
    scipy.misc.imsave(folder + '/img2.png', ((1 - img2) * 255.0).astype(np.uint8))
    scipy.misc.imsave(folder + '/img3.png', ((1 - img3) * 255.0).astype(np.uint8))

    for i in range(seq_l):
        scipy.misc.imsave(folder + './ynorm_per' + str(i) + '.png', ((1 - result_final_per[i, :, :]) * 255.0).astype(np.uint8))
#    scipy.misc.imsave(folder + '/ynormal_per_1.png', ((1 - result_final_per[t1 + 2, :, :]) * 255.0).astype(np.uint8))
#    scipy.misc.imsave(folder + '/ynormal_per_2.png', ((1-result_final_per[t2 - 2,:,:])*255.0).astype(np.uint8))
#    scipy.misc.imsave(folder + '/ynormal_per_3.png', ((1-result_final_per[t2,:,:])*255.0).astype(np.uint8))

        scipy.misc.imsave(folder + '/ynorm_cd' + str(i) + '.png', ((1 - result_final_cd[i, :, :]) * 255.0).astype(np.uint8))
#    scipy.misc.imsave(folder + '/ynormal_cd_2.png', ((1-result_final_cd[t2 - 2,:,:])*255.0).astype(np.uint8))
#    scipy.misc.imsave(folder + '/ynormal_cd_3.png', ((1-result_final_cd[t2,:,:])*255.0).astype(np.uint8))
    sess.close()
    