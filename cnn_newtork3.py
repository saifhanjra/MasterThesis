# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 13:23:19 2016

@author: engrs
"""
import network3_1
from network3_1 import ReLU

from network3_1 import Network
print 'wait.......'
from network3_1 import ConvPoolLayer,FullyConnectedLayer, SoftmaxLayer

training_data, test_data =network3_1.load_data_shared()

mini_batchsize=10
net = Network([
        ConvPoolLayer(image_shape=(mini_batchsize, 3, 32, 32), 
                      filter_shape=(32, 3, 5, 5), 
                      pool_size=(2, 2), 
                      activation_fn=ReLU),
        ConvPoolLayer(image_shape=(mini_batchsize, 32, 14, 14), 
                      filter_shape=(64,32, 5, 5), 
                      pool_size=(2, 2), 
                      activation_fn=ReLU),
        ConvPoolLayer(image_shape=(mini_batchsize, 64, 5, 5), 
                      filter_shape=(128, 64, 4, 4), 
                      pool_size=(2, 2), 
                      activation_fn=ReLU),
                      
        FullyConnectedLayer(n_in=128*1*1, n_out=64, activation_fn=ReLU),
#        FullyConnectedLayer(n_in=64*1*1, n_out=10, activation_fn=ReLU),

        SoftmaxLayer(n_in=64, n_out=10)], mini_batchsize)
net.SGD(training_data, 50, mini_batchsize, 0.01, test_data, lmbda=0.0,momentum=0.9)