# -*- coding: utf-8 -*-
"""
Created on Mon May 09 16:52:12 2016

@author: engrs
"""

evalution_accuracy=[5575, 8732, 9019, 9080, 9157, 9193, 9248, 9277, 9314, 9340, 9369, 9384, 9411, 9431, 9428, 9451, 9468, 9482, 9490, 9491]
evalution_cost=[0.2835156201477404, 0.13733711411048655, 0.092067137563620866, 0.07833265507053884, 0.071159628511364717, 0.067674800590499851, 0.062451467724855901, 0.059794029152328038, 0.056829306277531459, 0.054339088762350704, 0.053137721846760022, 0.051583060021690295, 0.049162884760551778, 0.048017238604669298, 0.047194347926745366, 0.04593490340334417, 0.045459265544861953, 0.043917402572156455, 0.043209046067276553, 0.042841457136116257]
len(evalution_cost)
training_accuracy=[27781, 43378, 44835, 45339, 45693, 45886, 46249, 46418, 46616, 46823, 46942, 47050, 47223, 47287, 47334, 47467, 47548, 47613, 47685, 47711]
training_cost=[0.28442311673227066, 0.14181548384653522, 0.096064923955035381, 0.081066347290303528, 0.073697727394934964, 0.0693857797329515, 0.06366125380590959, 0.060754664732419639, 0.057123538543424938, 0.054151811273675132, 0.052695889228157863, 0.050317917948929071, 0.047666261368703132, 0.046343635950779796, 0.045118305926537379, 0.043412907982441372, 0.042751681452156749, 0.04100080302243491, 0.039984024777602406, 0.039727105873441096]
len(training_cost)
################################################
training_accuracy_crossentropy=[44384, 46387, 47237, 47386, 47968, 48171, 48365, 48472, 48643, 48669, 48846, 48855, 48896, 49013, 49013, 49076, 49055, 49178, 49187, 49178]
training_cost_crossentropy=[0.74225404324445909, 0.46856608180467368, 0.368601945013314, 0.33617147711772472, 0.26607063003454157, 0.24862475869676315, 0.22206531579850508, 0.21515758881531413, 0.19088221045462786, 0.18340348147806559, 0.16382171795027817, 0.16430585567337699, 0.15746563623942259, 0.14198429463563031, 0.1415516436509158, 0.13235823526463095, 0.13629349754177358, 0.11850706577559643, 0.12279293936765648, 0.12012083875188866]
evalution_accuracy_crossentropy=[8897, 9299, 9406, 9454, 9541, 9563, 9569, 9585, 9603, 9609, 9635, 9623, 9614, 9636, 9620, 9637, 9626, 9654, 9641, 9618]
evalution_cost_crossentropy=[0.73898701498908892, 0.46367820294525214, 0.38792531858776402, 0.35477637198880435, 0.29981700536627137, 0.28831273215034969, 0.27296790375758934, 0.26951693544310829, 0.25287869228587523, 0.25664853498680229, 0.2413624563569535, 0.24350137627523336, 0.24818346032867314, 0.23335403629022453, 0.24039595129748254, 0.2366726544791658, 0.24468284725180375, 0.22324554824576753, 0.23444918633168488, 0.24212880940808976]
########################################

import numpy as np
import matplotlib.pyplot as plt
epochs=np.arange(0,20,1)
#
#
#
#train_accuracy = [y/ 50000.0 for y in training_accuracy]
#test_accuracy=newList = [x/10000.0 for x in evalution_accuracy]
#ax=plt.subplot(111)
#ax.plot(epochs,train_accuracy , label='Training accuracy',color='r')
#ax.plot(epochs, test_accuracy,label='Test accuracy',color='g')
#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
#          ncol=3, fancybox=True, shadow=True)
##ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#          
#ax.set_xlabel('Epochs', fontsize=15,fontweight='bold') 
#ax.set_ylabel('Accuracy', fontsize=15, fontweight='bold')   
#plt.grid()
#plt.show()


#ax=plt.subplot(111)
#ax.plot(epochs,training_cost , label='Training Cost',color='r')
#ax.plot(epochs, evalution_cost,label='Test Cost',color='g')
#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
#          ncol=3, fancybox=True, shadow=True)
##ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#          
#ax.set_xlabel('Epochs', fontsize=15,fontweight='bold') 
#ax.set_ylabel('Mean Square Error Cost', fontsize=15, fontweight='bold')   
#plt.grid()
#plt.show()

###################################################################################################

#train_accuracy = [y/ 50000.0 for y in training_accuracy_crossentropy]
#test_accuracy=newList = [x/10000.0 for x in evalution_accuracy_crossentropy]
#ax=plt.subplot(111)
#ax.plot(epochs,train_accuracy , label='Training accuracy',color='r')
#ax.plot(epochs, test_accuracy,label='Test accuracy',color='g')
#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
#          ncol=3, fancybox=True, shadow=True)
##ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#          
#ax.set_xlabel('Epochs', fontsize=15,fontweight='bold') 
#ax.set_ylabel('Accuracy', fontsize=15, fontweight='bold')   
#plt.grid()
#plt.show()

ax=plt.subplot(111)
ax.plot(epochs,training_cost_crossentropy , label='Training Cost',color='r')
ax.plot(epochs, evalution_cost_crossentropy,label='Test Cost',color='g')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
          
ax.set_xlabel('Epochs', fontsize=15,fontweight='bold') 
ax.set_ylabel('Cross Entropy Cost', fontsize=15, fontweight='bold')   
plt.grid()
plt.show()



