# -*- coding: utf-8 -*-
"""
Created on Thu May 12 11:31:31 2016

@author: engrs
"""
import matplotlib.pyplot as plt
import cPickle
import gzip
from matplotlib import rc

rc('text', usetex=True)

f=gzip.open('mnist.pkl.gz','rb')
training_data,validation_data,test_data= cPickle.load(f)

test_x, test_y=test_data
#############
a=test_x[412]
a_label=test_y[412] ###5
a=a.reshape(28,28)

#plt.plot(a, plt.cm.gray)

plt.subplot(231)
plt.suptitle('Subset of wrongly classified Test inputs', fontsize=30, fontweight='bold')
#plt.text(0.05, -0.90, r'\underline{Parameters}: ', fontsize=12)
plt.title('Predicted output= 3, True output= 5', fontweight='bold')
plt.imshow(a, plt.cm.gray)
############

b=test_x[583]
b_label=test_y[583]####2
b=b.reshape(28,28)
plt.subplot(232)
plt.title('Predicted output= 7, True output= 2', fontweight='bold')
plt.imshow(a, plt.cm.gray)
plt.imshow(b,plt.cm.gray)

#############
c=test_x[625]
c_label=test_y[625] #####6
c=c.reshape(28,28)
plt.subplot(233)
plt.title('Predicted output= 4, True output= 6', fontweight='bold')
plt.imshow(a, plt.cm.gray)
plt.imshow(c,plt.cm.gray)
#########
d=test_x[740]
d_label=test_y[740] ######1
d=d.reshape(28,28)
plt.subplot(234)
plt.title('Predicted output= 9, True output= 1', fontweight='bold')
plt.imshow(a, plt.cm.gray)
plt.imshow(d,plt.cm.gray)
##########
e=test_x[947]
e_label=test_y[947]########8
e=e.reshape(28,28)
plt.subplot(235)
plt.title('Predicted output= 9, True output= 8', fontweight='bold')
plt.imshow(a, plt.cm.gray)
plt.imshow(e,plt.cm.gray)
#############

f=test_x[1260]
f_label=test_y[1260] #####7
f=f.reshape(28,28)
plt.subplot(236)
plt.title('Predicted output= 1, True output= 7', fontweight='bold')
plt.imshow(a, plt.cm.gray)
plt.imshow(f,plt.cm.gray)
############



