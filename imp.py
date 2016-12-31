# -*- coding: utf-8 -*-
"""
Created on Mon Feb 08 14:19:29 2016

@author: engrs
"""

#### wrongly classified
####[412, 583, 625, 646, 691, 740, 900, 947, 965, 1014, 1039, 1072, 1112, 1114, 
### 1178, 1226, 1247, 1260, 1500, 1522, 1611, 1737, 1782, 1878, 1901, 2035, 2070, 
### 2109, 2118, 2129, 2130, 2135, 2293, 2414, 2447, 2582, 2597, 2630, 2654, 2771, 
###2850, 2939, 2953, 3166, 3225, 3289, 3422, 3457, 3520, 3727, 3778, 4027, 4078, 
###,4163, 4176, 4536, 4571, 4740, 4814, 4823, 4860, 5937, 5955, 5997, 6571, 6576 
###6597, 6783, 7154, 8246, 8279, 8375, 8408, 8527, 9700, 9729, 9770, 9850]

##### norm.gard>20
##index=[104, 158, 448, 655, 721, 1216, 1660, 2886, 3427, 3875, 4703, 5593, 7992, 9605, 9624]

##values=[28.152883882152835, 21.106940628482594, 22.466394198176975, 21.576377169754327,
# 48.325651668962543, 40.620381897244954, 22.931037139350153, 26.706585573325054, 
#61.718861046189708, 22.803953151449605, 33.436496747996173, 29.999566479154662, 
#35.540237200852609, 39.736034674312485, 25.295946526225261]

#[494, 510, 685, 711, 1095, 1114, 1272, 1396, 1507, 1534, 1688, 1767, 1824, 1845, 1886, 1993, 
# 2014, 2027, 2080, 2150, 2234, 2629, 2719, 2731, 2770, 2822, 2855, 2987, 3030, 3079,
# 3145, 3548, 3632, 3757, 3767, 3802, 3808, 3892, 3956, 4146, 4193, 4213, 4216, 4239, 
# 4388, 4466, 4518, 4522, 4819, 5104, 5140, 5532, 5594, 5691, 5827, 5909, 5933, 6027, 
# 6093, 6102, 6688, 6827, 7143, 7344, 7403, 7429, 8025, 8059, 8207, 8254, 8255, 8256, 8304, 
# 8446, 8941, 9560, 9590, 9595, 9679, 9904]

########## norm.grad>0.5 and <1


########## norm(grad(cost=wrt_coorectly classified input))>7
#[104, 158, 444, 448, 655, 715, 721, 948, 1124, 1216, 1360, 1660, 2373, 2427, 2704, 2886, 
 #2952, 2962, 3427, 3680, 3855, 3875, 3934, 3966, 4150, 4314, 4642, 4703, 4882, 5396, 5593,
 #5918, 7147, 7857, 7859, 7992, 9605, 9624]

#                ####### just for fun
#                if i==10:
#                    print 'Have a gup shup with friends, it will take lot of time to complete the process'
#                if i==1000:
#                    print 'Saif what are you doing? Are you done with dinner?'
#                    
#                if i==2000:
#                    print 'Saif yaar, I am getting Screwed up'
#                    
#                    
#                if i==3000:
#                     print 'Ab tou adat si hy'
#                if i==4000:
#                    print 'Getting sleepy now'
#                    
#                if i==5000:
#                    print 'jaan lo gae kia?'
#                    
#                if i==6000:
#                    print 'chalo tum tou so jaao yaar saif'
#                    
#                if i==9000:
#                    print 'Thanks God'
#                    
#                if i == 9220:





import cPickle
import gzip
import numpy as np
import theano
from theano import tensor as t
f=gzip.open('cultivated_adv_inpt.pkl.gz','rb')
test_data=cPickle.load(f)

#f=gzip.open('mnist.pkl.gz','rb')
#training_data, validation_data, test_data =cPickle.load(f)
#f.close()

#test_x,test_y=test_data
#a=test_x[8]
#a=a.reshape(28,28)
#import matplotlib.pyplot as plt
#a=plt.imshow(a,plt.cm.gray)

test_x,test_y=test_data



shared_x = theano.shared(
    np.asarray(test_x, dtype=theano.config.floatX), borrow=True)
shared_y = theano.shared(
            np.asarray(test_y, dtype=theano.config.floatX), borrow=True)


















