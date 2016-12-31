# -*- coding: utf-8 -*-
"""
Created on Sat May 14 02:31:38 2016

@author: engrs
"""
import matplotlib.pyplot as plt

import cPickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
#f_adv=gzip.open('mnist_adv_sign_report.pkl.gz','rb')
f_adv=gzip.open('mnist_adv_norm_report.pkl.gz','rb')
f_test=gzip.open('mnist.pkl.gz','rb')
train_inpt,validation_inpt,test_inpt=cPickle.load(f_test)
test_x,test_y=test_inpt


plt.rc('text', usetex=True)
plt.rc('font', family='serif')

adv_inputs=cPickle.load(f_adv)



#test_104=test_x[104]
#test_104=test_104.reshape(28,28)
#plt.subplotl(231)
#plt.suptitle('Some examples of  correcty classified test inputs with their adversarial versions(Gradient norm method)', fontsize=20, fontweight='bold')
#plt.imshow(test_104,plt.cm.gray)
#plt.xlabel(r'Predicted  as 9 with 74.9\% confidence',fontsize=15, color='black')
#plt.subplot(234)
#plt.imshow(adv_inputs[0].reshape(28,28), plt.cm.gray)
#plt.xlabel(r'Predicted  as 5 with 99.9\% confidence',fontsize=15, color='black')
#
#test_158=test_x[158]
#test_158=test_158.reshape(28,28)
#plt.subplot(232)
#plt.xlabel(r'Predicted  as 3 with 36.0\% confidence',fontsize=15, color='black')
#plt.imshow(test_158,plt.cm.gray)
#
#plt.subplot(235)
#plt.xlabel(r'Predicted  as 2 with 59.2\% confidence',fontsize=15, color='black')
#plt.imshow(adv_inputs[1].reshape(28,28), plt.cm.gray)
#
#
#test_543=test_x[543]
#test_543=test_543.reshape(28,28)
#plt.subplot(233)
#plt.xlabel(r'Predicted  as 8 with 95.4\% confidence',fontsize=15, color='black')
#plt.imshow(test_543,plt.cm.gray)
#
#plt.subplot(236)
#plt.xlabel(r'Predicted  as 2 with 99.5\% confidence',fontsize=15, color='black')
#plt.imshow(adv_inputs[2].reshape(28,28), plt.cm.gray)



#########################################################
test_720=test_x[720]
test_720=test_720.reshape(28,28)
plt.subplot(231)
plt.imshow(test_720, plt.cm.gray)
plt.xlabel(r'Predicted  as 5 with 85.0\% confidence',fontsize=15, color='black')
plt.suptitle('Some examples of  correctly classified test inputs with their adversarial versions(Gradient norm method)', fontsize=20, fontweight='bold')
#plt.suptitle('Some examples of  correctly classified test inputs with their adversarial versions', fontsize=30, fontweight='bold')
plt.subplot(234)
plt.xlabel(r'Predicted  as 8 with 99.9\% confidence',fontsize=15, color='black')
plt.imshow(adv_inputs[3].reshape(28,28), plt.cm.gray)




test_1138=test_x[1138].reshape(28,28)
plt.subplot(232)
plt.imshow(test_1138,plt.cm.gray)
plt.xlabel(r'Predicted  as 2 with 93.1\% confidence',fontsize=15, color='black')
plt.subplot(235)
plt.imshow(adv_inputs[4].reshape(28,28),plt.cm.gray)
plt.xlabel(r'Predicted  as 1 with 99.9\% confidence',fontsize=15, color='black')



test_1290=test_x[1290].reshape(28,28)
plt.subplot(233)
plt.imshow(test_1290, plt.cm.gray)
plt.xlabel(r'Predicted  as 3 with 99.3\% confidence',fontsize=15, color='black')
plt.subplot(236)
plt.imshow(adv_inputs[5].reshape(28,28), plt.cm.gray)
plt.xlabel(r'Predicted  as 5 with 99.9\% confidence',fontsize=15, color='black')











