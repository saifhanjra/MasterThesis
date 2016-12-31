# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 00:31:49 2016

@author: engrs
"""


import cPickle
import gzip
import numpy as np
import theano 
from theano import tensor as t
import PIL
from PIL import Image

from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

# Activation functions

from theano.tensor.nnet import sigmoid
#from theano.tensor.nnet import softmax


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
            
        
            
            
            
            
            

        
        
        
#def load_data_shared(filename="cifar10.pkl.gz"):
#    f = gzip.open(filename, 'rb')
#    training_data, test_data = cPickle.load(f)
#    f.close()
#    print'external function'
#    def shared(data):
#        """Place the data into shared variables.  This allows Theano to copy
#        the data to the GPU, if one is available.
#        """
#        shared_x = theano.shared(
#            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
#        shared_y = theano.shared(
#            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
#        return shared_x, t.cast(shared_y, "int32")
#    return [shared(training_data), shared(test_data)]
#        






















