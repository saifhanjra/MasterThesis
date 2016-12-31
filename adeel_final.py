# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 18:44:19 2016

@author: engrs
"""

import cPickle
#####################################################
input_file=open('params_cnn_gstrb_mb_10_tester.pkl','rb')
params=cPickle.load(input_file)
input_file.close()


# Theano and numpy
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

# Activation functions

from theano.tensor.nnet import sigmoid
from theano.tensor.nnet import softmax




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
        init_layer.set_inpt(self.x, self.mini_batchsize)
        for j in xrange(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt( prev_layer.output, self.mini_batchsize)
        self.output = self.layers[-1].output
        
        
    
        
        

    def Lets_check_Performance(self,mini_batchsize, test_data):
        """Train the network using mini-batch stochastic gradient descent."""

        
        test_x, test_y = test_data

 
        num_test_batches = size(test_data)/mini_batchsize

  
        
        i = T.lscalar()
        """ I am done with defining my network symbolically and calculation of gradient wrt to ever
        -y parameter of my network i am done with updates of paramters after each iteration(
        mini_batch) of training
        """

        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                test_x[i*self.mini_batchsize: (i+1)*self.mini_batchsize],
                self.y:
                test_y[i*self.mini_batchsize: (i+1)*self.mini_batchsize]
            })
            
        
        

        
        """Now i have define the function which will calculate the cost function  (train_mb) and 
        and i also have defined the function which is calculating efficnecy(test_mb_accuracy) by 
        using test_data only thing left is to provide i/p to the function so that evalution can 
        be done of network
            """
          
        test_accuracy_accum=[]    ###local variable used to store accuracy of test_data
                                ### of one epoch

        

            
        if test_data:
            test_accuracy = np.mean(
                                [test_mb_accuracy(j) for j in xrange(num_test_batches)])
            test_accuracy_accum.append(test_accuracy)
            print('The corresponding test accuracy is {0:.2%}'.format(
                                test_accuracy))
                                
                                
    def forwarding_testdata(self,test_data,mini_batchsize):
        
        
        
        test_x,test_y=test_data
        
        i=T.lscalar()
        
        
        test_network=theano.function(inputs=[i],outputs=self.layers[-1].accuracy(self.y),
                                     givens={
                                     self.x:
                                         test_x[i*mini_batchsize:(i+1)*mini_batchsize],
                                    self.y:
                                        test_y[i*mini_batchsize:(i+1)*mini_batchsize]})
                                        
        self.s=input('Please enter the Index of Test input for which you want to evaluate the Network = ')
        self.accurracy=test_network(self.s)
                                
        
            
                
            
            
#### Define layer types

class ConvPoolLayer(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.
    """

    def __init__(self, filter_shape, image_shape, pool_size=(2, 2),
                 activation_fn=sigmoid,params=params):
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
        self.w=params[0]
        self.b=params[1]

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
            
class ConvPoolLayer1(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.
    """

    def __init__(self, filter_shape, image_shape, pool_size=(2, 2),
                 activation_fn=sigmoid,params=params):
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
        self.w=params[2]
        self.b=params[3]
       
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

    def __init__(self, n_in, n_out, activation_fn=sigmoid,params=params):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.w=params[4]
        self.b=params[5]
        
       
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))

class SoftmaxLayer(object):

    def __init__(self, n_in, n_out,params=params):
        self.n_in = n_in
        self.n_out = n_out
        self.w=params[6]
        self.b=params[7]
     
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax (T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        



    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))
        
        
        
        
        
        
def load_data_shared(filename="C:\Users\Adeel\Desktop\Thesis Work\First SIM RESULTS\Classes Check\Classes_new.pkl"):
    f = open(filename, 'rb')
    training_data, test_data = cPickle.load(f)
    f.close()
    train_x,train_y= training_data
    test_x,test_y=test_data
    
        
#    training_data=(train_x,train_y)
    
    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.
        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
#        shared_m = np.reshape(shared_x, (76,76,3),"float64")    
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data), shared(test_data)]




























































































### for importing data
#import cPickle
#import gzip
#
#
##### Miscellanea
#def size(data):
#    "Return the size of the dataset `data`."
#    return data[0].get_value(borrow=True).shape[0]
##### Lthe MNIST data #### 
#import csv
#import sys
#import matplotlib.pyplot as plt
#from scipy.misc import imresize
#
## function for reading the images
## arguments: path to the traffic sign data, for example './GTSRB/Training'
## returns: list of images, list of corresponding labels 
#def readTrafficSigns_training(rootpath_training='Images'):
#    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.
#
#    Arguments: path to the traffic sign data, for example './GTSRB/Training'
#    Returns:   list of images, list of corresponding labels'''
#    print rootpath_training
#    images = [] # images
#    labels = [] # corresponding labels
#    # loop over all 42 classes
#    for c in range(0,43):
#        
#        print c
#        prefix = "./" + rootpath_training + '/' + format(c, '05d') + '/' # subdirectory for class
#        fullpath = prefix + 'GT-'+ format(c, '05d') + '.csv'        
#        print (str(fullpath))
#        gtFile = open(fullpath) # annotations file
#        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
#        gtReader.next() # skip header
#        # loop over all images in current annotations file
#        for row in gtReader:
#            images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
#            labels.append(row[7]) # the 8th column is the label
#        gtFile.close()
##    return images, labels
#    
#    
#
##### got trainig data in two list
#
#
#    train_inpt=images
#
#
#    train_x=[]
###### as the size of each impage is differet, worth to resize each image. here 48*48
#    for i in xrange(39209):
#        a=imresize(train_inpt[i],(48,48))
#        img = np.asarray(a, dtype='float64') / 256.
#        train_x.append(img)
#    
#    
#    training_tuple_list=[]    
#    for i in zip(train_x,labels):
#        a=i
#        training_tuple_list.append(a)
#    
#    
#    
#    np.random.shuffle(training_tuple_list)
#
#### now again i want to break tuple in to two differetn lists
#
#    training_x=[]
#    training_y=[]
#
#    for i in xrange(len(training_tuple_list)):
#        b=training_tuple_list[i]
#        c=b[0]
#        d=b[1]
#        training_x.append(c)
#        training_y.append(d)
#    
#    
#    
#    return training_x, training_y
#
#
#    
#    
#def readTrafficSigns_test(rootpath_test='test_images'):
#    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.
#
#    Arguments: path to the traffic sign data, for example './GTSRB/Training'
#    Returns:   list of images, list of corresponding labels'''
#    print rootpath_test
#    images_test = [] # images
#    labels_test = [] # corresponding labels
#    # loop over all 42 classes
##    print c
#    prefix = "./" + rootpath_test + '/' + format(0, '05d') + '/' # subdirectory for class
#    fullpath = prefix + 'GT-'+ format(0, '05d') + '.csv'        
#    print (str(fullpath))
#    gtFile = open(fullpath) # annotations file
#    gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
#    gtReader.next() # skip header
##         loop over all images in current annotations file
#    for row in gtReader:
#        images_test.append(plt.imread(prefix + row[0])) # the 1th column is the filename
#        labels_test.append(row[7]) # the 8th column is the label
#    gtFile.close()
#    
#    test_inpt=images_test
#    test_x=[]
#    for i in xrange(len(test_inpt)):
#        a=imresize(test_inpt[i],(48,48))
#        img_test = np.asarray(a, dtype='float64') / 256.
#        test_x.append(img_test)
#        
#    test_tuple_list=[]
#    for i in zip(test_x,labels_test):
#        a=i
#        test_tuple_list.append(a)
#        
#    test_data_x=[]
#    test_data_y=[]
#    
#    
#    for i in xrange(len(test_tuple_list)):
#        b=test_tuple_list[i]
#        c=b[0]
#        d=b[1]
#        test_data_x.append(c)
#        test_data_y.append(d)
#        
#        
#    return test_data_x,test_data_y
#    
#    
#    
#        
#    
#        
#    
#
#    
#    
#trainImages,trainLabels = readTrafficSigns_training('Images')
#    
#testImages, testLabels = readTrafficSigns_test('test_images')
#    
#tes_data=(testImages, testLabels)
#trai_data=(trainImages,trainLabels)
#data=trai_data,tes_data
#
#
#def load_data_shared(filename=data):
#    training_data,test_data=filename
#    
#    
#    def shared(data):
#        """Place the data into shared variables.  This allows Theano to copy
#        the data to the GPU, if one is available.
#        """
#        shared_x = theano.shared(
#            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
#        shared_y = theano.shared(
#            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
#        return shared_x, T.cast(shared_y, "int32")
#    return [shared(training_data), shared(test_data)]