# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 16:50:59 2015

@author: Saif ur Rehman
"""

import mnist_loader
import tester
#import sgd_backprop


training_data, validation_data, test_data= mnist_loader.load_data_wrapper()
net=tester.Network([784,200,30,10])
net.SGD(training_data,30, 100, 1.0, test_data=test_data)














        
        
        
        
    
    



    


        
        
        

        
        
        
        
            
        
        
        




    









