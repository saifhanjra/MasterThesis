# -*- coding: utf-8 -*-
"""
Created on Thu Mar 03 12:49:57 2016

@author: engrs
"""

import theano
from theano import tensor as T
from theano.tensor.nnet import conv
from PIL import Image
import matplotlib.pyplot as plt

import numpy
rng = numpy.random.RandomState(23455)
input = T.tensor4(name='input')
w_shp = (1, 3, 3, 3)
w_bound = numpy.sqrt(3 * 3 * 3)
W = theano.shared( numpy.asarray(
            rng.uniform(
                low=-1.0 / w_bound,
                high=1.0 / w_bound,
                size=w_shp),
            dtype=input.dtype), name ='W')
            
            
            
b_shp = (1,)


b = theano.shared(numpy.asarray(
            rng.uniform(low=-.5, high=.5, size=b_shp),
            dtype=input.dtype), name ='b')
            
            
            
conv_out = conv.conv2d(input, W)
output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))
f = theano.function([input], output)


import cPickle
f=open('data_batch_1','rb')
dictt=cPickle.load(f)
a=dictt['data'][0]
b=dictt['labels'][0]
img=a.reshape(3,32,32).transpose(1,2,0)
image=img



img = numpy.asarray(image, dtype='float64') / 256.
img_ = img.transpose(2, 0, 1).reshape(1, 3, 32, 32)

h=img_.reshape(32,32,3).transpose(3,2,1)
#filtered_img = f(img_)
#

           
