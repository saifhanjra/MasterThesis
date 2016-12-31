# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 17:24:57 2016

@author: saif
"""



import tester_adv_ex_cnn_gstrb_1
#from tester_adv_ex_cnn_gstrb_1 import ReLU

from tester_adv_ex_cnn_gstrb_1 import Network
print 'wait.......'
from tester_adv_ex_cnn_gstrb_1 import ConvPoolLayer,ConvPoolLayer1,FullyConnectedLayer, SoftmaxLayer

training_data, test_data =tester_adv_ex_cnn_gstrb_1.load_data_shared()

mini_batchsize=1
epislon=1
net = Network([ConvPoolLayer(filter_shape=(20,3,5,5),
                           image_shape=(mini_batchsize,3,48,48),
                            pool_size=(2,2)),
             ConvPoolLayer1(image_shape=(mini_batchsize, 20, 22, 22), 
                      filter_shape=(40, 20, 5, 5), 
                      pool_size=(2, 2)),
        FullyConnectedLayer(n_in=40*9*9, n_out=100),
        SoftmaxLayer(n_in=100, n_out=43)], mini_batchsize)
        
        
        
        
net.getting_cost(training_data,mini_batchsize)
#                         
net.forwarding_testdata(test_data,mini_batchsize)
""" Create correctly classified image adversial"""
net.create_adversial_inpt(test_data,mini_batchsize,epislon)
net.check_adv_inpt(mini_batchsize)
#net.check_adv_inpt(mini_batchsize,test_data)