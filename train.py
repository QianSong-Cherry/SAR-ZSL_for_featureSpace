# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 15:10:56 2016

@author: Administrator
"""

## Code for paper: Zero-Shot Learning of SAR Target Feature Space With Deep Generative Neural Networks
## by Qian Song on 12 09 2016
## used on MSTAR data


import numpy as np
import time
import utils
from math import pi
import scipy.io as sio


import tensorflow as tf
learning_rate = 0.0005
beta1 = 0.9
beta2 = 0.999
batch_size = 64
lamda = 0.1
d_h = 2
d_w = 2
output_size = 128
dataset_name = "Chairs"
checkpoint_dir = './checkpoint'


def load_data():
    matfn = './data/mstar-tf.mat'
    data1 = sio.loadmat(matfn)
    data = data1['data']
    label = data1['y_']
    angle_temp = data1['angle']   #angles imported are all in radian
    
    
    angle =np.zeros([2049,2])
    angle[0:2049,0] = np.sin(angle_temp[:,1])
    angle[0:2049,1] = np.cos(angle_temp[:,1])

            
    PI = pi*np.ones([64])
    
    deg = np.linspace(0.0,360.0,64)  
    sample_v = np.zeros([batch_size,2])
    sample_v[:,0] = np.sin(deg*PI/180)
    sample_v[:,1] = np.cos(deg*PI/180)
     
    
    seed = 5
    np.random.seed(seed)
    np.random.shuffle(data)
    np.random.seed(seed)
    np.random.shuffle(angle)
    np.random.seed(seed)
    np.random.shuffle(label)
    
    return data,data,angle,label, sample_v

def lrelu(x,leaky = 0.03):
    return tf.maximum(x,x*leaky)    

def weights(shape,name1):
    w = tf.Variable(tf.random_normal(shape, stddev=0.02),name = name1)
    return w

def bias(shape,name2):
    b = tf.Variable(tf.constant(0.0,"float32",shape),name = name2)
    return b

def deconv2d(input_, w, bia_,output_shape,strides=[1, 2, 2, 1]):
    deconv = tf.nn.conv2d_transpose(input_, w, output_shape,strides)
    deconv = tf.reshape(tf.nn.bias_add(deconv, bia_), deconv.get_shape())
    return deconv
 

   
data_X,data_Y, data_v, data_c, sample_v = load_data()


##Build Model
X = tf.placeholder(tf.float32,[batch_size,128,128,1])
c = tf.placeholder(tf.float32,[batch_size,7])
v = tf.placeholder(tf.float32,[batch_size,2])


#--------------
wfc1_1 = weights([7,20],name1 = 'wfc1_1')
bfc1_1 = bias([20],name2 = 'bfc1_1') 
fc1_1 = lrelu(tf.matmul(c,wfc1_1) + bfc1_1)

#--------------
wfc2_1 = weights([20,2],name1 = 'wfc2_1')
bfc2_1 = bias([2],name2 = 'bfc2_1')


fc2_1 =tf.nn.tanh(tf.matmul(fc1_1,wfc2_1) + bfc2_1)
fc2 = tf.concat(axis = 1,values = [fc2_1,v])

#--------------
wfc3 = weights([4,4],name1 = 'wfc3')
bfc3 = bias([4],name2 = 'bfc3')
fc3 = lrelu(tf.matmul(fc2,wfc3) + bfc3)


#--------------
wfc4 = weights([4,400],name1 = 'wfc3')
bfc4 = bias([400],name2 = 'bfc3')
fc4 = lrelu(tf.matmul(fc3,wfc4) + bfc4)



#------------
wfc5_1 = weights([400,8*8*256],name1 = 'wfc5_1')
bfc5_1 = bias([8*8*256],name2 = 'bfc5_1')
fc5_1 = lrelu(tf.matmul(fc4,wfc5_1) + bfc5_1)    
fc5_1  = tf.reshape(fc5_1,[batch_size,8,8,256])   


wuconv1_1 = weights([5,5,256,256],name1 = 'wuconv1_1')
buconv1_1 = bias([256],name2 = 'buconv1_1')
uconv1_1 = lrelu(deconv2d(fc5_1, wuconv1_1, buconv1_1, [batch_size,16,16,256],strides=[1, 2, 2, 1]))


wuconv2_1 = weights([5,5,92,256],name1 = 'wuconv2_1')
buconv2_1 = bias([92],name2 = 'buconv2_1')
uconv2_1 = lrelu(deconv2d(uconv1_1, wuconv2_1, buconv2_1, [batch_size,32,32,92],strides=[1, 2, 2, 1]))


wuconv3_1 = weights([5,5,92,92],name1 = 'wuconv3_1')
buconv3_1 = bias([92],name2 = 'buconv3_1')
uconv3_1 = lrelu(deconv2d(uconv2_1, wuconv3_1, buconv3_1, [batch_size,64,64,92],strides=[1, 2, 2, 1]))


wuconv4_1 = weights([5,5,1,92],name1 = 'wuconv4_1')
buconv4_1 = bias([1],name2 = 'buconv4_1')
uconv4_1 = tf.nn.relu(deconv2d(uconv3_1, wuconv4_1, buconv4_1, [batch_size,128,128,1],strides=[1, 2, 2, 1]))



d_loss = tf.reduce_sum(tf.square(X - uconv4_1))   
optim = tf.train.AdamOptimizer(learning_rate = learning_rate, epsilon = 1e-6) \
                          .minimize(d_loss)
init = tf.initialize_all_variables()


saver1 = tf.train.Saver({'wfc1_1':wfc1_1,'bfc1_1':bfc1_1,'wfc2_1':wfc2_1,'bfc2_1':bfc2_1})
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
start_time = time.time()
print("[*]Loading Model...")
saver.restore(sess, "./checkpoint/Generate model-1")
print("[*]Load successfully!")

#print("[*]Loading Model...")
#saver1.restore(sess,"./checkpoint/preTrain model-1")    
#print("[*]Load successfully!")

#loss_epoch = np.zeros([200])
#counter = 1    
#for epoch in range(200):
#    batch_idxs = len(data_X)//batch_size
#                          
#    for idx in range(batch_idxs):
#        batch_images = data_X[idx*batch_size:(idx+1)*batch_size]      
#        batch_images.shape = batch_size,128,128,1 
#            
#        batch_v = data_v[idx*batch_size:(idx+1)*batch_size]
#        batch_c = data_c[idx*batch_size:(idx+1)*batch_size]
#        loss,train_step = sess.run([d_loss,optim], feed_dict={X: batch_images,c:batch_c, v:batch_v})                            
#        
#        counter += 1        
#        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f" \
#                    % (epoch, idx, batch_idxs,
#                        time.time() - start_time,loss))    
#        if np.mod(counter, 70) == 1:
#            sample, d_loss_1 = sess.run([uconv4_1,d_loss],feed_dict={X: batch_images,
#                                      c:batch_c,v:batch_v})
#            utils.save_images(sample,[8,8],'./samples/train_{:02d}_{:04d}.png'.format(epoch, idx))
#            utils.save_images(batch_images,[8,8],'./samples/true_{:02d}_{:04d}.png'.format(epoch, idx))
#            print("[Sample] d_loss: %.8f" % (d_loss_1))
#                                                
#        if np.mod(counter, 200) == 1:
#           saver.save(sess,"./checkpoint/Generate model-1")
#           print("[*]Save Model...")
#    loss_temp = 0
#    for i in range(batch_idxs):
#        batch_images = data_X[i*batch_size:(i+1)*batch_size]      
#        batch_images.shape = batch_size,128,128,1 
#            
#        batch_v = data_v[i*batch_size:(i+1)*batch_size]
#        batch_c = data_c[i*batch_size:(i+1)*batch_size]
#        loss_i = sess.run([d_loss], feed_dict={X: batch_images,c:batch_c, v:batch_v})
#        loss_temp = loss_temp + loss_i[0]
#    loss_epoch[epoch] = loss_temp/len(data_X)
#    sio.savemat("loss_epoch.mat",{'loss_epoch':loss_epoch})

test_c = np.zeros([batch_size,7])
for i in range(7):
    test_c[i,i] = 1
    fc2_1_ = sess.run([fc2_1],feed_dict={c:test_c})
    sio.savemat('./samples/target_feature.mat',{'proto_features':fc2_1_[0:7]})
        
print("**UsedTime**:%.8f"%(time.time() - start_time))

