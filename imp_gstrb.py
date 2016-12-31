# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 15:53:45 2016

@author: engrs
"""

import theano
from theano import tensor as T
from theano.tensor.nnet import conv
from PIL import Image
import matplotlib.pyplot as plt
import csv
import sys
import numpy as np
from scipy.misc import imresize

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



#############################################################
###################GSTRB#####################################
#############################################################
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