# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 16:56:48 2016

@author: engrs
"""

    
import tester_conv_pretrained_1

from tester_conv_pretrained_1 import Network
from tester_conv_pretrained_1 import ConvPoolLayer,ConvPoolLayer1, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data =tester_conv_pretrained_1.load_data_shared()


mini_batchsize=10



        
net=Network([ConvPoolLayer(filter_shape=(20,1,5,5),
                           image_shape=(mini_batchsize,1,28,28),
                            pool_size=(2,2)),
             ConvPoolLayer1(image_shape=(mini_batchsize, 20, 12, 12), 
                      filter_shape=(40, 20, 5, 5), 
                      pool_size=(2, 2)),
            FullyConnectedLayer(n_in=40*4*4,
                                n_out=100,
                                ),
            SoftmaxLayer(n_in=100,
                         n_out=10)],mini_batchsize)
#net.SGD(training_data, 2, mini_batchsize, 0.1, 
#            validation_data, test_data)
            