# -*- coding: utf-8 -*-
"""
Created on Mon May 09 13:47:11 2016

@author: engrs
"""

import mnist_loader
import tester
#import sgd_backprop


training_data, validation_data, test_data= mnist_loader.load_data_wrapper()
net=tester.Network([784,200,30,10])
net.SGD(training_data,30, 100, 1.0, test_data=test_data)