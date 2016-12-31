# -*- coding: utf-8 -*-
"""
Created on Tue Mar 01 12:29:25 2016

@author: engrs
"""

#### Libraries
# Standard library
import cPickle
import gzip

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

import cv2



#### Load the MNIST data
def load_data_shared(filename="cifar10.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data,test_data = cPickle.load(f)
    f.close()
    train_x, train_y=training_data
    
    for i in xrange(50000):
        a=train_x[i]
        rimg=cv2.flip(a,1)
        train_x.append(rimg)
  
    b=train_y   
    for j in xrange(50000):
      c=b[j]
      train_y.append(c)
    training_data=(train_x,train_y)
    a=(training_data,test_data)
    
    training_data,test_data=a
    
    
    
    print 'good'
    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.

        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data), shared(test_data)]

#### Main class used to construct and train networks
class Network(object):
    def __init__(self, layers, mini_batchsize):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.
        """
        
        self.layers = layers
        self.mini_batchsize = mini_batchsize
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.tensor4("x")
        self.y = T.ivector("y")
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x,self.mini_batchsize)
        for j in xrange(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt( prev_layer.output, self.mini_batchsize)
        self.output = self.layers[-1].output
        
        
    
        
        

    def SGD(self, training_data, epochs, mini_batchsize, eta,
             test_data,lmbda,momentum):
        """Train the network using mini-batch stochastic gradient descent."""
        training_x, training_y = training_data
        
        test_x, test_y = test_data

        # compute number of minibatches for training, validation and testing
        num_training_batches = size(training_data)/mini_batchsize
        num_test_batches = size(test_data)/mini_batchsize

        # define the cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self)+0.5*lmbda*l2_norm_squared/num_training_batches
               
        """To implement Mini batch stochastic gradient in my earlier program when using numpy
        i have to derive the  complex equations for calculating  rate of change of cost function
        wrt weights and biases here in theano it is very simple we dont need derive any eqaution
        it calculate symbolic graident easily"""
              
        grads = T.grad(cost, self.params)
        """ Next, i want to update all the  parameters after calcualting the gradient wrt to 
        every parametere  layer by layer """
        updates=[]
        for param,grad in zip(self.params,grads):
            param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
            updates.append((param, param - eta*param_update))            
            updates.append((param_update, momentum*param_update + (1. - momentum)*grad))            
#        updates = [(param, param-eta*grad)
#                   for param, grad in zip(self.params, grads)]

        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.
        i = T.lscalar() # mini-batch index
        """ I am done with defining my network symbolically and calculation of gradient wrt to ever
        -y parameter of my network i am done with updates of paramters after each iteration(
        mini_batch) of training
        """
        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x:
                training_x[i*self.mini_batchsize: (i+1)*self.mini_batchsize],
                self.y:
                training_y[i*self.mini_batchsize: (i+1)*self.mini_batchsize]
            })
        
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                test_x[i*self.mini_batchsize: (i+1)*self.mini_batchsize],
                self.y:
                test_y[i*self.mini_batchsize: (i+1)*self.mini_batchsize]
            })
            
        
        
        iterations= epochs*num_training_batches
        print " Netwok is going to be trained for epochs:  " , epochs, " with iteartions", iterations
        
        """Now i have define the function which will calculate the cost function  (train_mb) and 
        and i also have defined the function which is calculating efficnecy(test_mb_accuracy) by 
        using test_data only thing left is to provide i/p to the function so that evalution can 
        be done of network
            """
            ### start training
        test_accuracy_accum=[]    ###local variable used to store accuracy of test_data
                                ### of one epoch

        
        for epoch in xrange(epochs):
            if epoch<28:
                eta=eta
                print 'The Value of eta is:',eta

            if  epoch>=28:
                eta=0.001
                print'The value of eta is',eta
                
            print "Epoch:{0}/{1}".format(epoch+1,epochs)
            print "training..........."
                

                
            for minibatch_index in xrange(num_training_batches):               
                cost_fn=train_mb(minibatch_index)
#                print minibatch_index
            if test_data:
                            test_accuracy = np.mean(
                                [test_mb_accuracy(j) for j in xrange(num_test_batches)])
                            test_accuracy_accum.append(test_accuracy)
                            print('The corresponding test accuracy is {0:.2%}'.format(
                                test_accuracy))
            
                            
                            
                                
                                
        if test_data:
                x_test_inpt=[]
                for x in range(epochs):
                    x=x+1
                    x_test_inpt.append(x)
                    
                
#                plt.plot(x_test_inpt,test_accuracy_accum,label='Convolutinary neural Netork')
#                plt.xlabel('Epochs')
#                plt.ylabel('Accuracy')
#                plt.legend()
#                plt.grid()
#                
#                plt.show()

    

#### Define layer types

class ConvPoolLayer(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.
    """

    def __init__(self, filter_shape, image_shape,activation_fn, pool_size=(2, 2),
                 ):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.
        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.
        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.
        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.pool_size = pool_size
        self.activation_fn=activation_fn
        # initialize weights and biases
        n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(pool_size))
#        self.w = theano.shared(
#            np.asarray(
#                np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape),
#                dtype=theano.config.floatX),
#            borrow=True)
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=0.1, size=filter_shape),
                dtype=theano.config.floatX),
            borrow=True)
        self.b = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=0.1, size=(filter_shape[0],)),
                dtype=theano.config.floatX),
            borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = conv.conv2d(
            input=self.inpt, filters=self.w, filter_shape=self.filter_shape,
            image_shape=self.image_shape)
        pooled_out = downsample.max_pool_2d(
            input=conv_out, ds=self.pool_size, ignore_border=True)
        self.output = self.activation_fn(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))

class SoftmaxLayer(object):

    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out
        
        # Initialize weights and biases
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax (T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        

    def cost(self, net):
        "Return the log-likelihood cost."
        return -T.mean(T.log(self.output)[T.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))





















#### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]
