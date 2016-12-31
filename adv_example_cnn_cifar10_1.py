# -*- coding: utf-8 -*-
"""
Created on Wed Mar 02 17:02:09 2016

@author: engrs
"""

import tester_adv_ex_cnn_cifar10_1
from tester_adv_ex_cnn_cifar10_1 import ReLU

from tester_adv_ex_cnn_cifar10_1 import Network
print 'wait.......'
from tester_adv_ex_cnn_cifar10_1 import ConvPoolLayer,ConvPoolLayer1,ConvPoolLayer2,FullyConnectedLayer, SoftmaxLayer

training_data, test_data =tester_adv_ex_cnn_cifar10_1.load_data_shared()

mini_batchsize=1
epislon=0.02
net = Network([
        ConvPoolLayer(image_shape=(mini_batchsize, 3, 32, 32), 
                      filter_shape=(32, 3, 5, 5), 
                      pool_size=(2, 2), 
                      activation_fn=ReLU),
        ConvPoolLayer1(image_shape=(mini_batchsize, 32, 14, 14), 
                      filter_shape=(32,32, 5, 5), 
                      pool_size=(2, 2), 
                      activation_fn=ReLU),
        ConvPoolLayer2(image_shape=(mini_batchsize, 32, 5, 5), 
                      filter_shape=(64, 32, 4, 4), 
                      pool_size=(2, 2), 
                      activation_fn=ReLU),
                      
        FullyConnectedLayer(n_in=64*1*1, n_out=128, activation_fn=ReLU),
#        FullyConnectedLayer(n_in=64*1*1, n_out=10, activation_fn=ReLU),

        SoftmaxLayer(n_in=128, n_out=10)], mini_batchsize)
        
        
        
        
net.getting_cost(training_data,test_data,mini_batchsize)
                         
net.forwarding_testdata(test_data,mini_batchsize)
""" Create correctly classified image adversial"""
net.create_adversial_inpt(test_data,mini_batchsize,epislon)
net.check_adv_inpt(mini_batchsize)
#net.check_adv_inpt(mini_batchsize,test_data)