# -*- coding: utf-8 -*-
"""
Created on Fri May 06 14:23:08 2016

@author: engrs
"""

import numpy as np
import matplotlib.pyplot as plt
x=np.arange(0,4*np.pi,0.1)

f1=np.sin(x)
f2=np.exp(-0.1*x)
f3=f1*f2
plt.rc('text', usetex=True)
plt.rc('font', family='serif') 
a=plt.plot(x,f3)
plt.title(r"\ Non-Convex Cost Function ",
            fontsize=20, color='black')
plt.xlabel(r'\textbf{$\theta$}', fontsize=22)
plt.ylabel(r'\texbf{$j(\theta)$}',fontsize=22)
plt.grid()
