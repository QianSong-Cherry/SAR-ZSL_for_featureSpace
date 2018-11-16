# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 18:26:22 2016

@author: Administrator
"""

##preTrain.py 
#pretrain the nets for Constructor NN
#Dem. 30 2016 by Qian Song

import tensorflow as tf
import numpy as np
import time

learning_rate = 0.01

def lrelu(x,leaky = 0.03):
    return tf.maximum(x,x*leaky)    

def weights(shape,name1):
    w = tf.Variable(tf.random_normal(shape, stddev=0.02),name = name1)
    return w

def bias(shape,name2):
    b = tf.Variable(tf.constant(0.0,"float32",shape),name = name2)
    return b


c = tf.placeholder(tf.float32,[7,7])
f = tf.placeholder(tf.float32,[7,2])
wfc1_1 = weights([7,20],name1 = 'wfc1_1')
bfc1_1 = bias([20],name2 = 'bfc1_1')
fc1_1 = lrelu(tf.matmul(c,wfc1_1) + bfc1_1)

#--------------
wfc2_1 = weights([20,2],name1 = 'wfc2_1')
bfc2_1 = bias([2],name2 = 'bfc2_1')


fc2_1 =tf.nn.tanh(tf.matmul(fc1_1,wfc2_1) + bfc2_1)

d_loss = tf.reduce_sum(tf.square(tf.sub(f,fc2_1)))   
optim = tf.train.AdamOptimizer(learning_rate = learning_rate, epsilon = 1e-6) \
                          .minimize(d_loss)

init = tf.initialize_all_variables()

saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)

data_c = np.eye(7,7)


start_time = time.time()  
data_f = [[0.5,0.5],[0.5,0],[-0.9,0.9],[0,0],[-0.5,0.0],[0.8,-0.8],[-0.5,-0.5]]
data_f = np.array(data_f)
for epoch in range(200):
    
    dloss,Train_step = sess.run([d_loss,optim], feed_dict={c:data_c,f:data_f})
    print("Epoch: [%2d]  time: %4.4f, d_loss: %.8f" % (epoch,time.time() - start_time,dloss))
    
    saver.save(sess,"./checkpoint/Chairs_64_128/preTrain model-1")
    print("[*]Save Model...")




