# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:46:43 2016

@author: engrs
"""

import tester_adv_ex_cnn_distance_calc

from tester_adv_ex_cnn_distance_calc import Network
from tester_adv_ex_cnn_distance_calc import ConvPoolLayer,ConvPoolLayer1, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data =tester_adv_ex_cnn_distance_calc.load_data_shared()


mini_batchsize=1
epislon=5 #### hyper parameter  used for the creation of adversial example



        
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
                         
                         
"""Towards the creation of adversial example"""
                         
net.getting_cost(training_data,test_data,mini_batchsize)
                         
net.forwarding_testdata(test_data,mini_batchsize)
""" Create rightly classified image adversial"""
net.create_adversial_inpt(test_data,mini_batchsize,epislon)
