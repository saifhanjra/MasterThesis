# -*- coding: utf-8 -*-
"""
Created on Mon May 09 13:38:11 2016

@author: engrs
"""

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import tester_mnist_cost
net = tester_mnist_cost.Network([784, 30,200, 10], cost=tester_mnist_cost.CrossEntropyCost)
net.SGD(training_data, 20, 100, 1.0,
evaluation_data=test_data,
monitor_evaluation_accuracy=True,
monitor_evaluation_cost=True,
monitor_training_accuracy=True,
monitor_training_cost=True)