# -*- coding: utf-8 -*-
"""
Created on Tue May 24 18:04:45 2016

@author: engrs
"""

import cPickle
import numpy as np
import matplotlib.pyplot as plt

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

train_x, train_y=training_data
test_x,test_y=test_data

a=test_x[0]
plt.subplot(331)
plt.imshow(a,plt.cm.gray)

b=train_x[200]
plt.subplot(332)
plt.imshow(b,plt.cm.gray)

c=test_x[120]
plt.subplot(333)
plt.imshow(c, plt.cm.gray)

d=train_x[36]
plt.subplot(334)
plt.imshow(d,plt.cm.gray)

e=test_x[349]
plt.subplot(335)
plt.imshow(e,plt.cm.gray)

f=test_x[470]
plt.subplot(336)
plt.imshow(f,plt.cm.gray)

g=train_x[590]
plt.subplot(337)
plt.imshow(g,plt.cm.gray)


h=test_x[690]
plt.subplot(338)
plt.imshow(h,plt.cm.gray)


i=train_x[790]
plt.subplot(339)
plt.imshow(i,plt.cm.gray)
