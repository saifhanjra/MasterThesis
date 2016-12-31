# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 13:10:24 2016

@author: engrs
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.misc import imresize

#a = Image.open("atiq_sign.jpg") # open colour image
#a= a.convert('1')

a=plt.imread('attiq_sign.jpg')

a=imresize(a,(500,500))
a=a/255.0
a=a.reshape(250000,1)
a=np.asarray(a)

for i in xrange(250000):
    if a[i]>0.57:
        a[i]=1
    if a[i]<0.57:
        a[i]=0
        
        
a=a.reshape(500,500)
plt.imshow(a, plt.cm.gray)







        
        
#a=plt.imshow(a,plt.cm.gray)
#a=a.reshape(2304,1)



        
        
        
        
    
    
        
    

 


#image_file = Image.open("atiq_sign.jpg") # open colour image
#image_file = image_file.convert('1')
#
#
#image_file.reshape(15872256,1)

#from PIL import Image 
#import ImageEnhance
#import ImageFilter
#from scipy.misc import imsave
#image_file = Image.open("atiq_sign.jpg") # open colour image
#image_file= image_file.convert('L') # convert image to monochrome - this works
#image_file= image_file.convert('1') # convert image to black and white
#imsave('result_col.png', image_file)

