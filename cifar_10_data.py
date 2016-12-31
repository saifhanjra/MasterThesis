# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:16:23 2016

@author: engrs
"""





#def unpickle (f='cifar-10-python.tar.gz'):
#    import cPickle
#    fo = open(f, 'rb')
#    dict = cPickle.load(fo)
#    fo.close()
#    return dict
#    
#    
#
#dict['data']

import cPickle
import numpy as np
train_x=[]
train_y=[]

for i in xrange(5):
    if i ==0:
        f=open('data_batch_1','rb')
    if i==1:
        f=open('data_batch_2','rb')
        
    if i==2:
        f=open('data_batch_3','rb')
    if i==3:
        f=open('data_batch_4','rb')
    if i==4:
        f=open('data_batch_5','rb')
    dictt=cPickle.load(f)
    for j in xrange(10000):
        a=dictt['data'][j]
        img=a.reshape(3,32,32).transpose(1,2,0)
        img = np.asarray(img, dtype='float64') / 256.
#        img_ = img.transpose(2, 0, 1).reshape(1, 3, 32, 32)
        b=dictt['labels'][j]
        train_x.append(img)
        train_y.append(b)
        
        
training_data=(train_x,train_y)
        
        
        
        
test_x=[]
test_y=[]
f=open('test_batch','rb')
dict=cPickle.load(f)
for k in xrange(10000):
    a=dict['data'][k]
    img=a.reshape(3,32,32).transpose(1,2,0)
    img = np.asarray(img, dtype='float64') / 256.
#    img_ = img.transpose(2, 0, 1).reshape(1, 3, 32, 32)
    
    b=dict['labels'][k]
    test_x.append(img)
    test_y.append(b)
    
test_data=(test_x,test_y)

cifar_10=(training_data,test_data)



    
    
    
    
    
    
    

        
        
        
        
        
        