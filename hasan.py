# -*- coding: utf-8 -*-
"""
Created on Wed Mar 02 17:40:33 2016

@author: engrs
"""

import theano
from theano import tensor as T
from theano.tensor.nnet import conv
import pylab
from PIL import Image
import numpy



img = Image.open(('3wolfmoon.jpg'))

img = numpy.asarray(img, dtype='float64') / 256.


img_ = img.transpose(2, 0, 1).reshape(1, 3, 639, 516)
filtered_img = img_
inpt = T.tensor4()


rng = numpy.random.RandomState(23455)


w_shp = (1, 3, 3, 3)

w_bound = numpy.sqrt(3 * 3 * 3)

W = theano.shared( numpy.asarray(
            rng.uniform(
                low=-1.0 / w_bound,
                high=1.0 / w_bound,
                size=w_shp),
            dtype=inpt.dtype), name ='W')


b_shp = (1,)
b = theano.shared(numpy.asarray(
            rng.uniform(low=-.5, high=.5, size=b_shp),
            dtype=inpt.dtype), name ='b')

# build symbolic expression that computes the convolution of input with filters in w
conv_out = conv.conv2d(inpt, W)
#
#
output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))
#
## create theano function to compute filtered images
f = theano.function([inpt], output)
#f()
g=f(filtered_img)
