# -*- coding: utf-8 -*-
"""
Created on Fri May 06 11:30:43 2016

@author: engrs
"""

import matplotlib.pyplot as plt
import numpy as np
x=np.arange(-8,8,0.01)
'''Sigmoid Activation Function'''

#
#y=1/(1+np.exp(-x)) #### Sigmoid Activation function
#z=y*(1-y)
#
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif') 
#a=plt.plot(x,y)
#plt.text(-7.5,0.875,r'$\displaystyle\ g(z)=\frac{1}{1+\exp{(-z)}} $',fontsize=18, color='black')
#plt.xlabel(r'\textbf{Z}')
#plt.ylabel(r'\texbf{g(z)}',fontsize=16)
#plt.title(r"\ Logistic Sigomoid Activation Function",
#            fontsize=16, color='black')
            
''' Sigmoid Activation Function Deivative'''
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif') 
#a=plt.plot(x,z)
#plt.text(-7.75,0.23,r'$\displaystyle\dot{g}(z)=\ g(z)(1-g(z))$',fontsize=18, color='black')
#plt.xlabel(r'\textbf{Z}', fontsize=16)
#plt.ylabel(r'$\texbf{\displaystyle\dot{g}(z)}$',fontsize=16)
#plt.title(r"\ Logistic Sigomoid Activation Function Derivative",
#            fontsize=16, color='black')
#            
#plt.grid()

''' Hyperbolic Tangent Activation Fucntion'''

y=2/(1+np.exp(-2*x))-1
y=1/(1+np.exp(-x)) #### Sigmoid Activation function


plt.rc('text', usetex=True)
plt.rc('font', family='serif') 
a=plt.plot(x,y)
plt.text(-7.5,0.78,r'$\displaystyle\ g(z)=\frac{2}{1+\exp{(-2z)}}-1 $',fontsize=16, color='black')
plt.xlabel(r'\textbf{Z}')
plt.ylabel(r'\texbf{g(z)}',fontsize=16)
plt.title(r"\ Hyperbolic Tangent Activation Function",
            fontsize=16, color='black')
            
plt.grid()

'''Hyperbolic Tangent Activation Function'''
#z=(1-np.square(y))
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif') 
#a=plt.plot(x,z)
#plt.text(-7.75,0.95,r'$\displaystyle\dot{g}(z)=\ (1-g(z)^2)$',fontsize=18, color='black')
#plt.xlabel(r'\textbf{Z}', fontsize=14)
#plt.ylabel(r'$\texbf{\displaystyle\dot{g}(z)}$',fontsize=16)
#plt.title(r"\ Hyperbolic Tangent Activation Function Derivative",
#            fontsize=16, color='black')
#            
#plt.grid()

         



