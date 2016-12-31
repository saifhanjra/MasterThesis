# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 21:57:26 2016

@author: engrs
"""


import tester_cnn_gstrb

from tester_cnn_gstrb import Network
print 'wait.......'
from tester_cnn_gstrb import ConvPoolLayer,FullyConnectedLayer, SoftmaxLayer
#
training_data, test_data =tester_cnn_gstrb.load_data_shared()
#
mini_batchsize=10
#
#
#
net = Network([ConvPoolLayer(filter_shape=(20,3,5,5),
                           image_shape=(mini_batchsize,3,48,48),
                            pool_size=(2,2)),
             ConvPoolLayer(image_shape=(mini_batchsize, 20, 22, 22), 
                      filter_shape=(40, 20, 5, 5), 
                      pool_size=(2, 2)),
        FullyConnectedLayer(n_in=40*9*9, n_out=100),
        SoftmaxLayer(n_in=100, n_out=43)], mini_batchsize)
#        
#print'Network is going to be Trained'
net.SGD(training_data, 10, mini_batchsize, 0.2, 
             test_data)
             
             
             
             
             
#import tester_cnn_gstrb
#
#from tester_cnn_gstrb import Network
#print 'wait.......'
#from tester_cnn_gstrb import ConvPoolLayer,FullyConnectedLayer, SoftmaxLayer
##
#training_data, test_data =tester_cnn_gstrb.load_data_shared()
##
#mini_batchsize=10
##
##
##
#net = Network([ConvPoolLayer(filter_shape=(100,3,7,7),
#                           image_shape=(mini_batchsize,3,48,48),
#                            pool_size=(2,2)),
#             ConvPoolLayer(image_shape=(mini_batchsize, 100, 21, 21), 
#                      filter_shape=(150, 100, 4, 4), 
#                      pool_size=(2, 2)),
#        ConvPoolLayer(image_shape=(mini_batchsize, 150, 9, 9), 
#                      filter_shape=(250, 150, 4, 4), 
#                      pool_size=(2, 2)),
#        FullyConnectedLayer(n_in=250*3*3, n_out=300),
#        SoftmaxLayer(n_in=300, n_out=43)], mini_batchsize)
##        
##print'Network is going to be Trained'
#net.SGD(training_data, 60, mini_batchsize, 0.05, 
#             test_data) 