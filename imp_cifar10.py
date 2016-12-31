# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 13:20:36 2016

@author: engrs
"""

import pylab
import PIL
from PIL import Image
import theano
from theano import tensor as t
import numpy as np 
import cPickle
train_x=[]
train_y=[]
f=open('data_batch_1','rb')
dictt=cPickle.load(f)
for j in xrange(10000):
    a=dictt['data'][j]
    img=a.reshape(3,32,32).transpose(1,2,0)
    img =np.asarray(img, dtype='float64') / 256.
    b=dictt['labels'][j]
    train_x.append(img)
    train_y.append(b)





import convolution_layer_cifar
from convolution_layer_cifar import ConvPoolLayer
print 'wait'
training_data, test_data =convolution_layer_cifar.load_data_shared()
print 'wait_finished'
mini_batchsize=1


net=ConvPoolLayer(filter_shape=(2,3,5,5),
                           image_shape=(mini_batchsize,3,972,1296),
                            pool_size=(2,2))
                            
                            
                            
                            


img=train_x[0]
img = Image.open('adeel.bmp')

#img_=img.transpose(2,0,1).reshape(1,3,3840,5760)
img = np.asarray(img, dtype='float64') / 256.
pylab.subplot(1, 3, 1); pylab.axis('off'); pylab.imshow(img)
pylab.gray()
img_ = img.transpose(2, 0, 1).reshape(1, 3, 972, 1296)

x=t.tensor4('input')
net.set_inpt(x,1)
f=theano.function([x],net.output)
filtered_img=f(img_)
pylab.subplot(1, 3, 2); pylab.axis('off'); pylab.imshow(filtered_img[0, 0, :, :])
pylab.subplot(1, 3, 3); pylab.axis('off'); pylab.imshow(filtered_img[0, 1, :, :])
pylab.show()


            
inpt_y=theano.shared(
            np.asarray(train_y[0], dtype=theano.config.floatX), borrow=True)
inpt_y=t.cast(inpt_y, "int32")



training_x, training_y=training_data
a=net.set_inpt(x,1)
i=t.lscalar()

f=theano.function([i],[net.output,net.w,net.b],
                  givens={
                x:
                training_x[i*mini_batchsize: (i+1)*mini_batchsize]
            })
#
            
            
            
            
for j in xrange(50000):
    print j
    filtered_img,w,b=f(j)






#