# -*- coding: utf-8 -*-
"""
Created on Tue May 24 14:08:08 2016

@author: engrs
"""

import cPickle 
import gzip
import matplotlib.pyplot as plt

f=gzip.open('mnist.pkl.gz','rb')
training_data,validation_data, test_data=cPickle.load(f)

train_x, train_y=training_data
test_x,test_y=test_data

a=test_x[0].reshape(28,28)
plt.subplot(331)
plt.imshow(a,plt.cm.gray)

b=train_x[200].reshape(28,28)
plt.subplot(332)
plt.imshow(b,plt.cm.gray)

c=test_x[120].reshape(28,28)
plt.subplot(333)
plt.imshow(c, plt.cm.gray)

d=train_x[36].reshape(28,28)
plt.subplot(334)
plt.imshow(d,plt.cm.gray)

e=test_x[349].reshape(28,28)
plt.subplot(335)
plt.imshow(e,plt.cm.gray)

f=test_x[470].reshape(28,28)
plt.subplot(336)
plt.imshow(f,plt.cm.gray)

g=train_x[590].reshape(28,28)
plt.subplot(337)
plt.imshow(g,plt.cm.gray)


h=test_x[690].reshape(28,28)
plt.subplot(338)
plt.imshow(h,plt.cm.gray)


i=train_x[790].reshape(28,28)
plt.subplot(339)
plt.imshow(i,plt.cm.gray)


 

