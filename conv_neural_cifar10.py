# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 21:47:38 2016

@author: engrs
"""


    




import tester_cnn_cifar10

from tester_cnn_cifar10 import Network
print 'wait.......'
from tester_cnn_cifar10 import ConvPoolLayer,FullyConnectedLayer, SoftmaxLayer

training_data, test_data =tester_cnn_cifar10.load_data_shared()

mini_batchsize=10



net = Network([ConvPoolLayer(filter_shape=(20,3,5,5),
                           image_shape=(mini_batchsize,3,32,32),
                            pool_size=(2,2)),
             ConvPoolLayer(image_shape=(mini_batchsize, 20, 14, 14), 
                      filter_shape=(40, 20, 5, 5), 
                      pool_size=(2, 2)),
        FullyConnectedLayer(n_in=40*5*5, n_out=100),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batchsize)
        
print'Network is going to be Trained'
net.SGD(training_data, 60, mini_batchsize, 0.05, 
             test_data) 
             
             








#