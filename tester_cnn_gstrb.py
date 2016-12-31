# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 19:48:49 2016

@author: engrs
"""

### For plotting
import matplotlib.pyplot as plt

# Theano and numpy
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

# Activation functions

from theano.tensor.nnet import sigmoid
from theano.tensor.nnet import softmax




#GPU = True
#try: theano.config.device = 'gpu'
#except: pass # it's already set
#theano.config.floatX = 'float32'







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
        init_layer.set_inpt(self.x, self.mini_batchsize)
        for j in xrange(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt( prev_layer.output, self.mini_batchsize)
        self.output = self.layers[-1].output
        
        
    
        
        

    def SGD(self, training_data, epochs, mini_batchsize, eta,
             test_data):
        """Train the network using mini-batch stochastic gradient descent."""
        training_x, training_y = training_data
        
        test_x, test_y = test_data

        # compute number of minibatches for training, validation and testing
        num_training_batches = size(training_data)/mini_batchsize
        num_test_batches = size(test_data)/mini_batchsize

        # define the cost function, symbolic gradients, and updates
        cost = self.layers[-1].cost(self)
               
        """To implement Mini batch stochastic gradient in my earlier program when using numpy
        i have to derive the  complex equations for calculating  rate of change of cost function
        wrt weights and biases here in theano it is very simple we dont need derive any eqaution
        it calculate symbolic graident easily"""
              
        grads = T.grad(cost, self.params)
        """ Next, i want to update all the  parameters after calcualting the gradient wrt to 
        every parametere  layer by layer """
        updates = [(param, param-eta*grad)
                   for param, grad in zip(self.params, grads)]

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
                    
                
                plt.plot(x_test_inpt,test_accuracy_accum,label='Convolutinary neural Netork')
                plt.xlabel('Epochs')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.grid()
                
                plt.show()
                
                
                
    
                
        
                     
                    
                
                
                    
                
                

                            
                                
                                
            
                            
                
            
                
            
                
               
                
                

#### Define layer types

class ConvPoolLayer(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.
    """

    def __init__(self, filter_shape, image_shape, pool_size=(2, 2),
                 activation_fn=sigmoid):
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
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape),
                dtype=theano.config.floatX),
            borrow=True)
        self.b = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
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

    def __init__(self, n_in, n_out, activation_fn=sigmoid):
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





### for importing data
#import cPickle
#import gzip
#import cv2

#### Miscellanea

def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]
#### Lthe MNIST data #### 
    
    

import csv
import sys
from scipy.misc import imresize

# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 
def readTrafficSigns_training(rootpath_training='Images'):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    print rootpath_training
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    for c in range(0,43):
        
        print c
        prefix = "./" + rootpath_training + '/' + format(c, '05d') + '/' # subdirectory for class
        fullpath = prefix + 'GT-'+ format(c, '05d') + '.csv'        
        print (str(fullpath))
        gtFile = open(fullpath) # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        gtReader.next() # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
            labels.append(row[7]) # the 8th column is the label
        gtFile.close()
#    return images, labels
    
    

#### got trainig data in two list


    train_inpt=images


    train_x=[]
##### as the size of each impage is differet, worth to resize each image. here 48*48
    for i in xrange(39209):
        a=imresize(train_inpt[i],(48,48))
        img = np.asarray(a, dtype='float64') / 256.
        train_x.append(img)
    
    
    training_tuple_list=[]    
    for i in zip(train_x,labels):
        a=i
        training_tuple_list.append(a)
    
    
    
    np.random.shuffle(training_tuple_list)

### now again i want to break tuple in to two differetn lists

    training_x=[]
    training_y=[]

    for i in xrange(len(training_tuple_list)):
        b=training_tuple_list[i]
        c=b[0]
        d=b[1]
        training_x.append(c)
        training_y.append(d)
    
    
    
    return training_x, training_y


    
    
def readTrafficSigns_test(rootpath_test='test_images'):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    print rootpath_test
    images_test = [] # images
    labels_test = [] # corresponding labels
    # loop over all 42 classes
#    print c
    prefix = "./" + rootpath_test + '/' + format(0, '05d') + '/' # subdirectory for class
    fullpath = prefix + 'GT-'+ format(0, '05d') + '.csv'        
    print (str(fullpath))
    gtFile = open(fullpath) # annotations file
    gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
    gtReader.next() # skip header
#         loop over all images in current annotations file
    for row in gtReader:
        images_test.append(plt.imread(prefix + row[0])) # the 1th column is the filename
        labels_test.append(row[7]) # the 8th column is the label
    gtFile.close()
    
    test_inpt=images_test
    test_x=[]
    for i in xrange(len(test_inpt)):
        a=imresize(test_inpt[i],(48,48))
        img_test = np.asarray(a, dtype='float64') / 256.
        test_x.append(img_test)
        
    test_tuple_list=[]
    for i in zip(test_x,labels_test):
        a=i
        test_tuple_list.append(a)
        
    test_data_x=[]
    test_data_y=[]
    
    
    for i in xrange(len(test_tuple_list)):
        b=test_tuple_list[i]
        c=b[0]
        d=b[1]
        test_data_x.append(c)
        test_data_y.append(d)
        
        
    return test_data_x,test_data_y
    
    
    
        
    
        
    

    
    
trainImages,trainLabels = readTrafficSigns_training('Images')
    
testImages, testLabels = readTrafficSigns_test('test_images')
    
tes_data=(testImages, testLabels)
trai_data=(trainImages,trainLabels)
data=trai_data,tes_data


def load_data_shared(filename=data):
    training_data,test_data=filename
    
    
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