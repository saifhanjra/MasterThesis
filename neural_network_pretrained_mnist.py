# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 16:49:03 2016

@author: engrs
"""
import cPickle
#######################################################
input_file=open('tester_params.pkl','rb')
weights=cPickle.load(input_file)
biases=cPickle.load(input_file)
input_file.close()
######################################################
import mnist_loader
training_data, validation_data, test_data= mnist_loader.load_data_wrapper()
#############################################################


import tester_pre_train
net=tester_pre_train.Pre_trained_tester(weights,biases)
net.evaluate(test_data)
net.print_result()

