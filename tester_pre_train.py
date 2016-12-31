# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 14:26:50 2016

@author: engrs
"""



""""
The network defined below has been pretrained and it has following specefication   
net=tester.Network([784,30,10])
net.SGD(training_data,30, 100, 3.0, test_data=test_data)
"""



import matplotlib.pyplot as plt


# Third-party libraries
import numpy as np


#############################################################
class Pre_trained_tester():
    def __init__(self,weights,biases):
        self.weights=weights
        self.biases=biases
    def feedforward_network(self,a):
        for b,w in zip(self.biases,self.weights):
            a=sigmoid_vec(np.dot(w,a)+b)
        return a
    
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        
        
        test_results=[]
        for x,y in test_data:
         test_results_cpy=(np.argmax(self.feedforward_network(x)),y)
         test_results.append(test_results_cpy)
        self.res= sum(int(x == y) for (x, y) in test_results)
         
        return self.res
        
        
    def print_result(self):
        a=self.res
        b=10000.0
        c=a/b
        d=c*100
        print "Efficeny of pretrained network is",d,"%"
        
        
        
        

    
        
        
        
            
            
            
            
            
            
            
            






#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

sigmoid_vec = np.vectorize(sigmoid)