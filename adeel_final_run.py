# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 18:51:46 2016

@author: engrs
"""

import adeel_final

from adeel_final import Network
from adeel_final import ConvPoolLayer,ConvPoolLayer1, FullyConnectedLayer, SoftmaxLayer
training_data, test_data =adeel_final.load_data_shared()


mini_batchsize=1


net = Network([ConvPoolLayer(filter_shape=(10,3,5,5),
                           image_shape=(mini_batchsize,3,76,76),
                            pool_size=(2,2)),
             ConvPoolLayer1(image_shape=(mini_batchsize, 10, 36, 36), 
                      filter_shape=(30, 10, 5, 5), 
                      pool_size=(2, 2)),
        FullyConnectedLayer(n_in=30*16*16, n_out=100),
        SoftmaxLayer(n_in=100, n_out=5)], mini_batchsize)
        

net.Lets_check_Performance(mini_batchsize, test_data)
net.forwarding_testdata(test_data,mini_batchsize)