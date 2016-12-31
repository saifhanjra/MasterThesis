# -*- coding: utf-8 -*-
"""
Created on Wed Mar 09 14:04:55 2016

@author: engrs
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
import gzip
import cPickle
f=gzip.open('cifar10.pkl.gz','rb')
training_data,test_data=cPickle.load(f)
train_x,train_y=training_data
test_x,test_y=test_data

for i in xrange(50000):
    a=train_x[i]
    rimg=cv2.flip(a,1)
    train_x.append(rimg)
  
b=train_y   
for j in xrange(50000):
    c=b[j]
    train_y.append(c)
    
print 'good'
    
for k in xrange(50000):
    d=train_x[k]
    limg=cv2.flip(d,0)
    train_x.append(limg)
    
e=train_y    
for l in xrange(50000):
    f=e[l]
    train_y.append(f)
    

    
    
    