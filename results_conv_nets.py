# -*- coding: utf-8 -*-
"""
Created on Wed May 11 12:47:24 2016

@author: engrs
"""
import numpy as np
import matplotlib.pyplot as plt

test_accuracy=[0.098000000000000004,0.95599999999999996,0.97589999999999999,0.97939999999999994,0.98199999999999998,
0.98360000000000003,0.9849,0.98639999999999983,0.9869,0.9869,0.98719999999999997,0.98759999999999992,
0.98780000000000001,0.98840000000000006,0.98850000000000005,0.9890000000000001,0.98939999999999995,
0.98960000000000004,0.98999999999999999,0.98999999999999999,0.99009999999999987,0.99040000000000006,
0.99050000000000016,0.99050000000000016,0.99070000000000003,0.99080000000000013,0.9909,0.99080000000000013,
0.99060000000000004,0.99080000000000001,0.9909,0.99080000000000001,0.99099999999999999,0.99099999999999999,
0.99099999999999999,0.99099999999999999,0.99099999999999999,0.99099999999999999,0.99099999999999999,
 0.99099999999999999]
epochs=np.arange(0,40,1)
#ax=plt.subplot(111)

plt.plot(epochs,test_accuracy ,color='r')
plt.title('Classification acuuracy on test data', fontsize=20)
plt.xlabel('$Epochs$',fontsize=18)
plt.ylabel('$Accuracy$',fontsize=18)


#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#ax.set_position([box.x0, box.y0 + box.height * 0.1,
#                 box.width, box.height * 0.9])
#ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
#          ncol=3, fancybox=True, shadow=True)
#ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


                 
                 
#ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
#          fancybox=True, shadow=True, ncol=5)
         
#ax.set_xlabel('Epochs', fontsize=15,fontweight='bold') 
#ax.set_ylabel('Accuracy', fontsize=15, fontweight='bold')   
plt.grid()
plt.show()

          