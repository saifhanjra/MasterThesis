# -*- coding: utf-8 -*-
"""
Created on Fri May 06 15:39:00 2016

@author: engrs
"""

import numpy as np
import matplotlib.pyplot as plt
x=np.arange(0.0001,1,0.001)


''' Cross entropy cost, y=1'''
#y=-np.log(x)
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif') 
#a=plt.plot(x,y)
#plt.text(0.025,9.5,r'$\displaystyle\ y=1 $',fontsize=20, color='black')
#plt.xlabel(r'\textbf{$\tilde{y}$}', fontsize=22)
#plt.ylabel(r'\texbf{$cost(\tilde{y},y)$}',fontsize=22)
#plt.title(r"\ Cross Entropy Cost if True Output is 1",
#            fontsize=20, color='black')
#            
#plt.grid()

'''cross entropy cost, y=0'''



y_0=-np.log(1-x)

plt.rc('text', usetex=True)
plt.rc('font', family='serif') 
a=plt.plot(x,y_0)#
plt.text(0.025,7.65,r'$\displaystyle\ y=0 $',fontsize=20, color='black')
plt.xlabel(r'\textbf{$\tilde{y}$}', fontsize=22)
plt.ylabel(r'\texbf{$cost(\tilde{y},y)$}',fontsize=22)
plt.title(r"\ Cross Entropy Cost if True Output is 0",
            fontsize=20, color='black')
            
plt.grid()