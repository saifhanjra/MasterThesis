# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 17:30:05 2016

@author: saif
"""
import cPickle
#####################################################
input_file=open('params_cnn_gstrb_mb_10_tester.pkl','rb')
params=cPickle.load(input_file)
input_file.close()


# Theano and numpy
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
import matplotlib.pyplot as plt

# Activation functions

from theano.tensor.nnet import sigmoid
from theano.tensor.nnet import softmax



def ReLU(z): return T.maximum(0.0, z)





class Network():
    def __init__(self,layers,mini_batchsize):
        
        self.layers=layers##### number of layers
        self.mini_batchsize=mini_batchsize
        self.x=T.tensor4()  ##### defining the input
        self.y=T.ivector() #### actual output,from training_data, test_data, validation_data
        """ Next, i have to connect all defined layers so that i can propogate input and takes the ouput
        """
        
        init_layer=self.layers[0]### i consider from the list of layers, layers[0] as 1st layer
        init_layer.set_inpt(self.x,self.mini_batchsize) ### invoke the function  set_inpt from layers[0]
                                                        ### and provide the required parametrs.
        for j in xrange(1,len(self.layers)):        ###taking output from the previous layer and provide
                                                    ### and provide as an input to current layer
            layer, prev_layer = self.layers[j], self.layers[j-1]
            layer.set_inpt(prev_layer.output,self.mini_batchsize)
        self.output=self.layers[-1].output
        
    """ I have pretrained Network, with accuracy more than 99.09%. I need optimized cost function
        for the creation of adversial example. I calculate cost but only for one epoch using freezed 
        parameters. """




     
    def getting_cost(self,training_data,mini_batchsize):
        training_x, training_y=training_data

        
        
        self.cost_param = self.layers[-1].cost(self) ##### optimized cost
        i=T.lscalar()
        
        
        train_mb = theano.function(
            [i], self.cost_param,
            givens={
                self.x:
                training_x[i*self.mini_batchsize: (i+1)*self.mini_batchsize],
                self.y:
                training_y[i*self.mini_batchsize: (i+1)*self.mini_batchsize]
            })
            
        num_training_batches = size(training_data)/mini_batchsize
            
        for i in xrange(num_training_batches):
            self.cost_train=train_mb(i)
            
    """ Next step, i want to give input to my pretrained network and check weather it is rightly
     classified or not. if it is rightly classified then i need to make this input 
     ADVERSIAL so that network with very high performance can easily fooled"""
            
        
        
    def forwarding_testdata(self,test_data,mini_batchsize):
        
        
        
        test_x,test_y=test_data
        
        i=T.lscalar()
        
        
        test_network=theano.function(inputs=[i],outputs=self.layers[-1].accuracy(self.y),
                                     givens={
                                     self.x:
                                         test_x[i*mini_batchsize:(i+1)*mini_batchsize],
                                    self.y:
                                        test_y[i*mini_batchsize:(i+1)*mini_batchsize]})
                                        
        self.s=input('Please enter the Index of Test input for which you want to evaluate the Network = ')
        self.accurracy=test_network(self.s)

      

    """ Now, i have given my input to network , what i will do in the next step is to check
    weather this rightly classified or wrongly classified and if it is rightly classsified, 
    i followed the step to make it adversarial"""    
        
        
    def create_adversial_inpt(self,test_data,mini_batchsize,epislon):
        if self.accurracy==1: ###if given test input is correctly classified
            
            
            print 'Given  test input is correctly classified, you can make it adversarial'
            
            a=input('Press 1 if you want to make correctly classified input adversarial')
            if a==1:
            
                test_x,test_y = test_data ### breaking the test data into tuple 
        
        
                self.grad_x=T.grad(cost=self.cost_param, wrt=self.x) ### grad of cost function wrt self.x
            
                                                                #### which is input to network
                m=T.lscalar()
                f_grad_x=theano.function([m],outputs=self.grad_x,givens=
                                                            {self.x:
                                                                test_x[m*mini_batchsize:(m+1)*mini_batchsize],
                                                                self.y:
                                                                    test_y[m*mini_batchsize:(m+1)*mini_batchsize]
                                                                    })
                                                                    
                                                                    
                                                                    
                k=T.lscalar()
                prob_correctly_image=theano.function(inputs=[k],outputs=self.layers[-1].confidence(),
                                     givens={
                                     self.x:
                                         test_x[k*mini_batchsize:(k+1)*mini_batchsize]
                                         })
                                         
                self.prob_correctly_classified=prob_correctly_image(self.s)
            
                                                                    
                                                                    
                                                                    
                '''x_adv= x + epislon*signum*grad(cost,x)''' 
                
                #### Above stated formula for creating adversarial example is konow fast gradient 
                ### method. proposed in publication 'Explaining and Harnessing Adversarial Examples'                                                  
                                                                
                ### calculate gradient of cost function wrt given test input                                                 
                self.grad_inpt=f_grad_x(self.s)
                self.grad_norm_inp=np.linalg.norm(self.grad_inpt)
                
                

                
                                
                #### Now applying signum function .
                grad_inpt_sign=np.sign(self.grad_inpt)
               
                
                #### multipling it with  hyper paramteter epislon and offcourse choice of epislon offcourse
                ### is curicial
                grad_inpt_sign_epislon=epislon*grad_inpt_sign
                self.a_sa=grad_inpt_sign_epislon
                grad_inpt_sign_epislon=grad_inpt_sign_epislon.transpose(2,3,1,0).reshape(48,48,3)
                ''' Goodfellow Method of creating adversity'''
#                plt.suptitle('Goodfellow method of harnessing Advesarial inputs')
#                plt.subplot(132)
#                plt.title('Epsilon*(signum*(grad(cost,wrt=Test input)))')
#                plt.imshow(grad_inpt_sign_epislon)
                
                
                ## i have defined all the data set symbolically.Now,  i am intersted in the real values
                ## so i have to foolow the mechanism of theano for getting real values.
                
                j=T.lscalar()
            
                f_test_inpt=theano.function([j],outputs=test_x[j])
            
                test_inpt= f_test_inpt(self.s)
                                                ### is a list of real values of given test input
                                                ####now 
                                                ###epislon*signum*grad(cost,x)=grad_inpt_sign_epislon
                                                ####x= test_inpt

                '''x_adv= x + epislon*signum*grad(cost,x)'''
                
                ''' Sigum Method/Goodfellow et al.attack'''
#                self.adv_inpt=test_inpt + grad_inpt_sign_epislon
#                plt.subplot(131)
#                plt.suptitle('Goodfellow method of harnessing Advesarial inputs')
#                plt.title('Test input')
#                plt.imshow(test_inpt)
#                
#                plt.subplot(132)
#                plt.title('Epsilon*(signum*(grad(cost,wrt=Test input)))')
#                plt.imshow(grad_inpt_sign_epislon)                
#                plt.subplot(133)
#                plt.title('Adversarial Input')
#                plt.imshow(self.adv_inpt)
                
                
                ''' Tobias Method of creating Adversarity'''
                g=self.grad_inpt/self.grad_norm_inp
                g=g.transpose(2,3,1,0).reshape(48,48,3)
                self.adv_inpt=test_inpt+epislon*g
                plt.subplot(131)
                plt.suptitle('Tobias Method of harnessing Adversarial inputs')
                plt.title('Test input')
                plt.imshow(test_inpt)
                plt.subplot(132)
                plt.title('Epsilon*(grad(cost wrt=Test input)/norm(grad(cost wrt=Test input)))')
                plt.imshow(epislon*g)
                plt.subplot(133)
                plt.title('Adversarial input')
                plt.imshow(self.adv_inpt)

                
                
                
                
######################################################                
                
#                plt.suptitle('Adversarial image created using Fast gradient method.    Epislon=0.048')
#                plt.subplot(131)
#                plt.title('Given Test input')
#                plt.imshow(test_inpt.reshape(28,28), plt.cm.gray)
                
                
                
                
                
                
                
    '''I am done with implementation of proposed method for creating an adversarial example,
    named as Fast gradient method, lastly i want to check weather the created adversarial example is 
    really an adeversarial example with choosen hyper parameter epislon'''
            
            
    def check_adv_inpt(self,mini_batchsize):

        testImages, testLabels = readTrafficSigns_test('test_images')
    
        tes_data=(testImages, testLabels)
   
        test_x_x,test_y_y=tes_data        

        
        a=self.adv_inpt  ### created adv input
        test_x_x[self.s]=a

        self.my_adv_inpt=test_x_x[self.s]
        self.shared_x = theano.shared(
            np.asarray(test_x_x, dtype=theano.config.floatX), borrow=True)
            
        shared_y_y= theano.shared(
            np.asarray(test_y_y, dtype=theano.config.floatX), borrow=True)
        self.shared_y=T.cast(shared_y_y, "int32")
        
        
        i=T.lscalar()
        
        
        check_adv=theano.function(inputs=[i],outputs=self.layers[-1].accuracy(self.y),
                                     givens={
                                     self.x:
                                         self.shared_x[i*mini_batchsize:(i+1)*mini_batchsize],
                                    self.y:
                                        self.shared_y[i*mini_batchsize:(i+1)*mini_batchsize]})
                                        
                                        
        self.adv_check=check_adv(self.s)
        if self.adv_check==0:
            print 'Adversarial input has been created'
            
#            j=T.lscalar()
#            prob_wrongly_classified=theano.function(inputs=[j],outputs=self.layers[-1].confidence(),
#                                     givens={
#                                     self.x:
#                                         self.shared_x[j*mini_batchsize:(j+1)*mini_batchsize]
#                                         })
#                                        
#            self.prob_wrongly_classified=prob_wrongly_classified(self.s)
#            
#            
#            p=T.lscalar()
#            grad_adv=T.grad(self.cost_param,self.x)
#            grad_adv_inptt=theano.function(inputs=[p],outputs=grad_adv,
#                                          givens={
#                                     self.x:
#                                         self.shared_x[p*mini_batchsize:(p+1)*mini_batchsize],
#                                    self.y:
#                                        self.shared_y[p*mini_batchsize:(p+1)*mini_batchsize]})
#                                        
#            self.grad_adv_inpt=grad_adv_inptt(self.s)
#            self.grad_norm_adv_inpt=np.linalg.norm(self.grad_adv_inpt)
#            
#            
            
        
    
#            plt.subplot(133)
#            plt.title('Adversarial image')
#            plt.imshow(self.adv_inpt.reshape(28,28),plt.cm)













#### Define layer types

class ConvPoolLayer(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.
    """

    def __init__(self, filter_shape, image_shape, pool_size=(2, 2),
                 activation_fn=sigmoid,params=params):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.
        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.
        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.
        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.pool_size = pool_size
        self.activation_fn=activation_fn
        self.w=params[0]
        self.b=params[1]

        self.params = [self.w, self.b]

    def set_inpt(self, inpt, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = conv.conv2d(
            input=self.inpt, filters=self.w, filter_shape=self.filter_shape,
            image_shape=self.image_shape)
        pooled_out = downsample.max_pool_2d(
            input=conv_out, ds=self.pool_size, ignore_border=True)
        self.output = self.activation_fn(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            
class ConvPoolLayer1(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.
    """

    def __init__(self, filter_shape, image_shape, pool_size=(2, 2),
                 activation_fn=sigmoid,params=params):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.
        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.
        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.
        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.pool_size = pool_size
        self.activation_fn=activation_fn
        self.w=params[2]
        self.b=params[3]
       
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = conv.conv2d(
            input=self.inpt, filters=self.w, filter_shape=self.filter_shape,
            image_shape=self.image_shape)
        pooled_out = downsample.max_pool_2d(
            input=conv_out, ds=self.pool_size, ignore_border=True)
        self.output = self.activation_fn(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=sigmoid,params=params):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.w=params[4]
        self.b=params[5]
        
       
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))

class SoftmaxLayer(object):

    def __init__(self, n_in, n_out,params=params):
        self.n_in = n_in
        self.n_out = n_out
        self.w=params[6]
        self.b=params[7]
     
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax (T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        



    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))
        
    def cost(self, net):
        "Return the log-likelihood cost."
        return -T.mean(T.log(self.output)[T.arange(net.y.shape[0]), net.y])
        
    def confidence(self):
        
        output_conf = self.output

        
        
        return output_conf
        



### for importing data
import cPickle
import gzip


#### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]
#### Lthe MNIST data #### 
import csv
import sys
import matplotlib.pyplot as plt
from scipy.misc import imresize

# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 
def readTrafficSigns_training(rootpath_training='Images'):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    print rootpath_training
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    for c in range(0,43):
        
        print c
        prefix = "./" + rootpath_training + '/' + format(c, '05d') + '/' # subdirectory for class
        fullpath = prefix + 'GT-'+ format(c, '05d') + '.csv'        
        print (str(fullpath))
        gtFile = open(fullpath) # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        gtReader.next() # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
            labels.append(row[7]) # the 8th column is the label
        gtFile.close()
#    return images, labels
    
    

#### got trainig data in two list


    train_inpt=images


    train_x=[]
##### as the size of each impage is differet, worth to resize each image. here 48*48
    for i in xrange(39209):
        a=imresize(train_inpt[i],(48,48))
        img = np.asarray(a, dtype='float64') / 256.
        train_x.append(img)
    
    
    training_tuple_list=[]    
    for i in zip(train_x,labels):
        a=i
        training_tuple_list.append(a)
    
    
    
    np.random.shuffle(training_tuple_list)

### now again i want to break tuple in to two differetn lists

    training_x=[]
    training_y=[]

    for i in xrange(len(training_tuple_list)):
        b=training_tuple_list[i]
        c=b[0]
        d=b[1]
        training_x.append(c)
        training_y.append(d)
    
    
    
    return training_x, training_y


    
    
def readTrafficSigns_test(rootpath_test='test_images'):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    print rootpath_test
    images_test = [] # images
    labels_test = [] # corresponding labels
    # loop over all 42 classes
#    print c
    prefix = "./" + rootpath_test + '/' + format(0, '05d') + '/' # subdirectory for class
    fullpath = prefix + 'GT-'+ format(0, '05d') + '.csv'        
    print (str(fullpath))
    gtFile = open(fullpath) # annotations file
    gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
    gtReader.next() # skip header
#         loop over all images in current annotations file
    for row in gtReader:
        images_test.append(plt.imread(prefix + row[0])) # the 1th column is the filename
        labels_test.append(row[7]) # the 8th column is the label
    gtFile.close()
    
    test_inpt=images_test
    test_x=[]
    for i in xrange(len(test_inpt)):
        a=imresize(test_inpt[i],(48,48))
        img_test = np.asarray(a, dtype='float64') / 256.
        test_x.append(img_test)
        
    test_tuple_list=[]
    for i in zip(test_x,labels_test):
        a=i
        test_tuple_list.append(a)
        
    test_data_x=[]
    test_data_y=[]
    
    
    for i in xrange(len(test_tuple_list)):
        b=test_tuple_list[i]
        c=b[0]
        d=b[1]
        test_data_x.append(c)
        test_data_y.append(d)
        
        
    return test_data_x,test_data_y
    
    
    
    
trainImages,trainLabels = readTrafficSigns_training('Images')
    
testImages, testLabels = readTrafficSigns_test('test_images')
    
tes_data=(testImages, testLabels)
trai_data=(trainImages,trainLabels)
data=trai_data,tes_data
    
    
    
    
def load_data_shared(filename=data):
    training_data,test_data=filename
    
    
    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.
        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data), shared(test_data)]
    
    
