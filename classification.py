# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 23:19:17 2016

@author: emw
"""

## Code for paper: Zero-Shot Learning of SAR Target Feature Space With Deep Generative Neural Networks
## by Qian Song on 30 December 2016
## used on MSTAR data


import numpy as np
import time
import scipy.io as sio

import tensorflow as tf

learning_rate = 0.0005
beta1 = 0.9
beta2 = 0.999
batch_size = 100
lamda = 0.5
d_h = 2
d_w = 2
total_size = 2049
checkpoint_dir = './checkpoint'
types = 7
fc3_size = 400
train_num = 1639

def load_data():    
    matfn = './data/mstar-tf.mat'
    data1 = sio.loadmat(matfn)
    data = data1['data']
    label = data1['y_']
    angle_temp = data1['angle']   #angles imported are all in radian
    
    matfn = './samples/target_feature.mat'
    data2 = sio.loadmat(matfn)
    class_true = data2['proto_features']
                  
    c_feature = np.zeros([total_size,2])    
    for i in range(total_size):
        temp = list(label[i,:])
        c_feature[i,:] = class_true[0,temp.index(1),:]        

    
    
    angle =np.zeros([2049,4]) 
    angle[:,0] = np.sin(angle_temp[:,1])
    angle[:,1] = np.cos(angle_temp[:,1])
    angle[:,2] = np.sin(angle_temp[:,0])
    angle[:,3] = np.cos(angle_temp[:,0])
            
    
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(data)
    np.random.seed(seed)
    np.random.shuffle(angle)
    np.random.seed(seed)
    np.random.shuffle(c_feature)
    np.random.seed(seed)
    np.random.shuffle(label)
        
    return data[0:train_num,:,:],data[train_num:,:,:],angle[0:train_num,0:2],angle[train_num:,0:2],c_feature[0:train_num,:],c_feature[train_num:,:],label[0:train_num,:],label[train_num:,:]




def lrelu(x,leaky = 0.03):
    return tf.maximum(x,x*leaky)    

def weights(shape,name1):
    w = tf.Variable(tf.random_normal(shape, stddev=0.02),name = name1)
    return w

def bias(shape,name2):
    b = tf.Variable(tf.constant(0.0,"float32",shape),name = name2)
    return b
        
def conv2d(input_, w, bia_,strides=[1, 2, 2, 1],padding='SAME'):
    deconv = tf.nn.conv2d(input_, w,strides,padding=padding)
    deconv = tf.nn.bias_add(deconv, bia_)
    return deconv

def load_testdata(dataset_name):
    matfn = './data/'+dataset_name+'.mat'
    data1 = sio.loadmat(matfn)
    data = data1[dataset_name]
    data_angle = data1[dataset_name+'_angle']
    return data,data_angle

    
data_X, sample_X, data_v, sample_v, data_c, sample_c,data_label,sample_label = load_data()

##Build Model
X = tf.placeholder(tf.float32,[None,128,128,1])
c = tf.placeholder(tf.float32,[None,2])
v = tf.placeholder(tf.float32,[None,2])

wuconv4_1 = weights([5,5,1,92],name1 = 'wuconv4_1')
buconv4_1 = bias([92],name2 = 'b_uconv4_1')
u_conv3_1 = lrelu(conv2d(X,wuconv4_1,buconv4_1))

wuconv3_1 = weights([5,5,92,92],name1 = 'wuconv3_1')
buconv3_1 = bias([92],name2 = 'b_uconv3_1')
u_conv2_1 = lrelu(conv2d(u_conv3_1,wuconv3_1,buconv3_1))


wuconv2_1 = weights([5,5,92,256],name1 = 'wuconv2_1')
buconv2_1 = bias([256],name2 = 'b_uconv2_1')
u_conv1_1 = lrelu(conv2d(u_conv2_1,wuconv2_1,buconv2_1,[1,2,2,1],padding='SAME'))



wuconv1_1 = weights([5,5,256,256],name1 = 'wuconv1_1')
buconv1_1 = bias([256],name2 = 'b_uconv1_1')
fc5_1 = lrelu(conv2d(u_conv1_1,wuconv1_1,buconv1_1,[1,2,2,1],padding='SAME'))

fc5_1 = tf.reshape(fc5_1,[-1,8*8*256])

wfc5_1 = weights([8*8*256,fc3_size],name1 = 'wfc5_1')
bfc5_1 = bias([fc3_size],name2 = 'bfc5_1')
fc4 = lrelu(tf.matmul(fc5_1 ,wfc5_1) + bfc5_1)


wfc4 = weights([fc3_size,4],name1 = 'wfc4')
bfc4 = bias([4],name2 = 'bfc4')
fc3 = lrelu(tf.matmul(fc4 ,wfc4) + bfc4)

wfc3 = weights([4,4],name1 = 'wfc3')
bfc3 = bias([4],name2 = 'bfc3')
fc2 = tf.nn.tanh(tf.matmul(fc3,wfc3) + bfc3)
fc2_1 = fc2[:,0:2]
fc2_2 = fc2[:,2:4]

d_loss = tf.reduce_sum((c - fc2_1)**2)/2.0 + lamda*tf.reduce_sum((v - fc2_2)**2)/2.0

optim = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = beta1, beta2 = beta2, epsilon = 1e-6) \
                         .minimize(d_loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
start_time = time.time()
saver1 = tf.train.Saver({'wuconv4_1':wuconv4_1,'wuconv3_1':wuconv3_1,'wuconv2_1':wuconv2_1,'wuconv1_1':wuconv1_1})
saver  = tf.train.Saver()

#print("[*]Loading Model...")
#saver1.restore(sess, "./checkpoint/Generate model-1")
#print("[*]Load successfully!")

print("[*]Loading Model...")
saver.restore(sess, "./checkpoint/Inference model-1")
print("[*]Load successfully!")

#counter = 1    
#for epoch in range(20):
#    batch_idxs = len(data_X)//batch_size
#    for idx in range(batch_idxs):
#        learning_rate = learning_rate/(epoch//4+1)
#        
#        batch_images = data_X[idx*batch_size:(idx+1)*batch_size]      
#        batch_images.shape = batch_size,128,128,1            
#        
#        batch_v = data_v[idx*batch_size:(idx+1)*batch_size]
#        batch_c = data_c[idx*batch_size:(idx+1)*batch_size]
#        loss,fc2_,train_step = sess.run([d_loss ,fc2,optim], feed_dict={X: batch_images,c:batch_c,v:batch_v})
#        
#        counter += 1
#        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f" \
#                    % (epoch, idx, batch_idxs,
#                        time.time() - start_time,loss))
#
#        if np.mod(counter,20)==1:                   
#            Error = tf.sqrt(tf.reduce_sum((fc2_[:,0:2]-batch_c)**2)/(batch_size*4.0))
#            print("error c:")
#            print sess.run(Error,feed_dict={X: batch_images,c:batch_c,v:batch_v})   
#                                                    
#            Error2 = tf.sqrt(tf.reduce_sum((fc2_[:,2:4]-batch_v)**2)/(batch_size*4.0))
#            print("error v:")
#            print sess.run(Error2,feed_dict={X: batch_images,c:batch_c, v:batch_v})      
#            
#        if np.mod(counter, 500) == 499:
#           saver.save(sess,"./checkpoint/Inference model-1")
#           print("[*]Save Model...")

#saver.save(sess,"./checkpoint/Inference model-1")
#print("[*]Save Model...")

#print("**UsedTime**:%.8f"%(time.time() - start_time))

##train error
data_X.shape = -1,128,128,1
Error = tf.sqrt(tf.reduce_sum((fc2[:,0:2]-c)**2)/( train_num*4.0))
Error2 = tf.sqrt(tf.reduce_sum((fc2[:,2:4]-v)**2)/(train_num*4.0))
#print("error c:")
error1,error2,fc2_ = sess.run([Error,Error2,fc2],feed_dict={X: data_X,c:data_c,v:data_v})
sio.savemat('dis_train2.mat',{'fc2':fc2_,'label':data_label,'data_v':data_v})
#print error1
#print("error v:")
#print error2

##test the distribution of training data
#sample_X.shape = -1,128,128,1
#fc2_ = sess.run([fc2], feed_dict={X: sample_X})
#sio.savemat('dis_test.mat',{'fc2':fc2_[0],'label':sample_label,'data_v':sample_v})
#Error = tf.sqrt(tf.reduce_sum((fc2[:,0:2]-c)**2)/((total_size - train_num)*4.0))
#Error2 = tf.sqrt(tf.reduce_sum((fc2[:,2:4]-v)**2)/((total_size - train_num)*4.0))
#print("error c:")
#error1,error2 = sess.run([Error,Error2],feed_dict={X: sample_X,c:sample_c,v:sample_v})
#print error1
#print("error v:")
#print error2


##project t72 variants to feature space
#dataset_name = 'T72_A04'
#sample_X,data_v = load_testdata(dataset_name)
#sample_X.shape = -1,128,128,1
#fc2_sample = sess.run([fc2], feed_dict={X: sample_X})
#fc2_sample = fc2_sample[0]
#sio.savemat(dataset_name+'_features2.mat',{'fc2':fc2_sample,'data_v':data_v})
#
#dataset_name = 'T72_A05'
#sample_X,data_v = load_testdata(dataset_name)
#sample_X.shape = -1,128,128,1
#fc2_sample = sess.run([fc2], feed_dict={X: sample_X})
#fc2_sample = fc2_sample[0]
#sio.savemat(dataset_name+'_features2.mat',{'fc2':fc2_sample,'data_v':data_v})


### Calculate the execution time of testing one SAR image
#start_time = time.time()
#sample_X,data_v = load_testdata(dataset_name)
#sample_X = sample_X[0]
#sample_X.shape = 1,128,128,1
#fc2_sample = sess.run([fc2], feed_dict={X: sample_X})
#print("**UsedTime**:%.8f"%(time.time() - start_time))  
