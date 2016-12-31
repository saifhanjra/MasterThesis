# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 19:59:12 2016

@author: engrs
"""
import tester_pretrained_cultivated_adv_inpt

from tester_pretrained_cultivated_adv_inpt import Network
from tester_pretrained_cultivated_adv_inpt import ConvPoolLayer,ConvPoolLayer1, FullyConnectedLayer, SoftmaxLayer
test_data =tester_pretrained_cultivated_adv_inpt.load_data_shared()


mini_batchsize=1



        
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
                         
                         
                         
net.Lets_check_Performance(mini_batchsize, test_data)
