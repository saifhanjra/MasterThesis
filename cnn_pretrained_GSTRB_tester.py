# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:49:40 2016

@author: engrs
"""

import tester_cnn_pretrained_GSTRB

from tester_cnn_pretrained_GSTRB import Network
from tester_cnn_pretrained_GSTRB import ConvPoolLayer,ConvPoolLayer1, FullyConnectedLayer, SoftmaxLayer
training_data, test_data =tester_cnn_pretrained_GSTRB.load_data_shared()


mini_batchsize=1


net = Network([ConvPoolLayer(filter_shape=(20,3,5,5),
                           image_shape=(mini_batchsize,3,48,48),
                            pool_size=(2,2)),
             ConvPoolLayer1(image_shape=(mini_batchsize, 20, 22, 22), 
                      filter_shape=(40, 20, 5, 5), 
                      pool_size=(2, 2)),
        FullyConnectedLayer(n_in=40*9*9, n_out=100),
        SoftmaxLayer(n_in=100, n_out=43)], mini_batchsize)
        

net.Lets_check_Performance(mini_batchsize, test_data)