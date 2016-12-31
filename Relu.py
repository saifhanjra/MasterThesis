# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 14:56:50 2016

@author: engrs
"""
import matplotlib.pyplot as plt
import numpy as np
x=np.arange(-8,8,0.001)
y=[]
for i in xrange(len(x)):
    y_single=np.maximum(0,x[i])
    y.append(y_single)
 


plt.rc('text', usetex=True)
plt.rc('font', family='serif') 
plt.plot(x,y)
plt.text(-7.5,7.35,r'$\displaystyle\ g(z)=max(0,z) $',fontsize=20, color='black')
plt.xlabel(r'\textbf{Z}')
plt.ylabel(r'\texbf{g(z)}',fontsize=20)
plt.title(r"\ Rectified Linear Unit(RELU)",
            fontsize=20, color='black')
            
plt.grid()   


























 
#w=np.log(1+np.exp(x))
#plt.plot(x,w,label='(1+exp^(wx+b))')
#plt.legend(bbox_to_anchor=(0,1 ), loc=2, borderaxespad=0.)
#green_patch = mpatches.Patch(color='red', label='(1+exp^(wx+b))')
#plt.legend(handles=[green_patch])
#plt.grid()
#
# 