# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 13:27:36 2016

@author: engrs
"""

#### to unzip and unpickle the dataset
import gzip
import cPickle
#####################################################
""" For the creation of adversial example i will use pretrained Network i just need to import the 
parameters of trained network with exactly same architecture """


input_file=open('params_cnn_mb_1.pkl','rb')
params=cPickle.load(input_file)
input_file.close() ###### imported paramters
####################################################
### importing cost function
#input_file_cost=open('cost_conv_nn_mb_1.pkl','rb')
#cost=cPickle.load(input_file_cost)
#input_file_cost.close()
#######################################################


# Theano and numpy
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

######ploting library py lab and various function
import matplotlib.pyplot as plt

# Activation functions

from theano.tensor.nnet import sigmoid
from theano.tensor.nnet import softmax

""" Creation of Adversial Example  """


class Network():
    def __init__(self,layers,mini_batchsize):
        
        self.layers=layers##### number of layers
        self.mini_batchsize=mini_batchsize
        self.x=T.matrix()  ##### defining the input
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




     
    def getting_cost(self,training_data,test_data,mini_batchsize):
        training_x, training_y=training_data
        test_x, test_y=test_data
        
        
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
        
                                        
        self.index_correctly_classified_list=[]
        self.index_wrongly_classified_list=[]
                               
        for j in xrange (10000):
            f_test_network=test_network(j)
            if f_test_network==1:
                self.index_correctly_classified_list.append(j)
                
            if f_test_network==0:
                self.index_wrongly_classified_list.append(j)
            
                                        
#        self.s=input('Please enter the Index of Test input for which you want to evaluate the Network = ')
#        self.accurracy=test_network(self.s)

      

    """ Now, i have given my input to network , what i will do in the next step is to check
    weather this rightly classified or wrongly classified and if it is rightly classsified, 
    i followed the step to make it adversarial"""    
        
        
    def create_adversial_inpt(self,test_data,mini_batchsize,epislon):
        self.inpt_high_normalized_grad_all_list=[]
        self.inpt_high_normalized_grad_all_index=[]
        
        self.inpt_high_normalized_grad_hypothetic_list=[]
        self.inpt_high_normalized_grad_hypothetic_index=[]
        
        
        self.inpt_high_normalized_grad_adv_list=[]
        self.inpt_high_normalized_grad_adv_index=[]
        
        test_x,test_y=test_data ### breaking the test data into tuple
        self.grad_x=T.grad(cost=self.cost_param, wrt=self.x) ### grad of cost function wrt self.x ####symbollic gradient
        
        m=T.lscalar()
        f_grad_x=theano.function([m],outputs=self.grad_x,givens=                     ###function for calculating symbollic gradient
                                                            {self.x:
                                                                test_x[m*mini_batchsize:(m+1)*mini_batchsize],
                                                                self.y:
                                                                    test_y[m*mini_batchsize:(m+1)*mini_batchsize]
                                                                    })
        n=T.lscalar()
            
        self.f_test_inpt=theano.function([n],outputs=test_x[n])      ### getting input function
        self.adv_inpt_accum=[]
        self.epsilon_adv_inpts_list=[]
        self.norm_grad_adv_inpt_list=[]
        self.index_adv=[]##### index of adv inpts
        counter=0
        
        
        #for i in xrange(len(self.index_correctly_classified_list)):
        for i in xrange(1530):
            
            self.test_x_correctly_classified=self.index_correctly_classified_list[i]
            
            self.grad_correctly_classified=f_grad_x(self.test_x_correctly_classified)
            self.magnitude_grad_correctly_classified=np.linalg.norm(self.grad_correctly_classified)
            
            

            
            
            
            

            if self.magnitude_grad_correctly_classified>0.1:
                
                
                self.inpt_high_normalized_grad_all_list.append(self.magnitude_grad_correctly_classified)
                self.inpt_high_normalized_grad_all_index.append(self.test_x_correctly_classified)
                
                given_test_inpt=self.f_test_inpt(self.test_x_correctly_classified)

                self.hypothetic_adv_inpt=given_test_inpt+epislon*(self.grad_correctly_classified/self.magnitude_grad_correctly_classified)
                #self.hypothetic_adv_inpt=given_test_inpt+epislon*(np.sign(self.grad_correctly_classified))
                self.check_adversarity_hypothetic_inpt=self.check_adv_inpt(mini_batchsize)
                if self.check_adversarity_hypothetic_inpt==0:
                    counter=counter+1
                    print counter
                    print 'Correctly classified input is given for calculating Epislon and Normalized grad of cost wrt given input'
                    print 'Processing............'
                    
                    
                    a=self.harnesssing_adv_inpt()
                    self.adv_inpt_accum.append(a)
                    self.norm_grad_adv_inpt_list.append(self.magnitude_grad_correctly_classified)
                    print'The value of normalized Gradient is:',self.magnitude_grad_correctly_classified
                    
                    self.adv_inpt_assumed=given_test_inpt+epislon*(self.grad_correctly_classified/self.magnitude_grad_correctly_classified)
                    #self.adv_inpt_assumed=given_test_inpt+epislon*(np.sign(self.grad_correctly_classified))
                    self.pred_network_after=self.appending_adv_inpt_and_check_equality(mini_batchsize)                    
                    b=self.binary_search(epislon,mini_batchsize)
                    self.epsilon_adv_inpts_list.append(b)
                    print 'Value of new epislon is ',b,' for test input with index',self.test_x_correctly_classified
                    
                    self.inpt_high_normalized_grad_adv_list.append(self.magnitude_grad_correctly_classified)
                    self.inpt_high_normalized_grad_adv_index.append(self.test_x_correctly_classified)
                    self.index_adv.append(self.test_x_correctly_classified)
                    
                    
                    
                    
                if self.check_adversarity_hypothetic_inpt==1:
                    self.inpt_high_normalized_grad_hypothetic_list.append(self.magnitude_grad_correctly_classified)
                    self.inpt_high_normalized_grad_hypothetic_index.append(self.test_x_correctly_classified)
                    
                    
                    
                    

            
                    
                    
                    
                    
    def harnesssing_adv_inpt(self):
        

            
            import cPickle
            import gzip
            f=gzip.open('mnist.pkl.gz','rb')
            training_dataa, validation_dataa, test_dataa = cPickle.load(f)
        
            f.close()
            test_x,test_y=test_dataa
            
            
          
            
            test_x[self.test_x_correctly_classified]=self.hypothetic_adv_inpt
            adv_test_inpt=test_x[self.test_x_correctly_classified]
            adv_test_index=test_y[self.test_x_correctly_classified]
            test_data=(adv_test_inpt,adv_test_index)
            return test_data
            
            
            
            
            
            
            
            
    def binary_search(self,epislon,mini_batchsize):
        self.epislon_new=epislon
        while  self.pred_network_after==0:
                        
            self.epislon_new=0.5*self.epislon_new
            self.adv_inpt_assumed= self.f_test_inpt(self.test_x_correctly_classified)+self.epislon_new*(self.grad_correctly_classified/self.magnitude_grad_correctly_classified)
            #self.adv_inpt_assumed= self.f_test_inpt(self.test_x_correctly_classified)+self.epislon_new*(np.sign(self.grad_correctly_classified))
            self.pred_network_after=self.appending_adv_inpt_and_check_equality(mini_batchsize)
            print 'step 1: Value of new epislon is', self.epislon_new
        print 'step1 done'
            
            
            
            
        step_size=self.epislon_new
        while step_size>1/1000.0:
            epsilon_new_new=step_size+self.epislon_new
            self.adv_inpt_assumed= self.f_test_inpt(self.test_x_correctly_classified)+epsilon_new_new*(self.grad_correctly_classified/self.magnitude_grad_correctly_classified)
            self.pred_network_after=self.appending_adv_inpt_and_check_equality(mini_batchsize)
            if self.pred_network_after==1:
                self.epislon_new=epsilon_new_new+step_size
                
            step_size=0.5*step_size
            print 'New step Size is ',step_size
            
            
        return self.epislon_new
        
        
           
            
    

                
                
                
    def appending_adv_inpt_and_check_equality(self,mini_batchsize):
        
        import cPickle
        import gzip
        f=gzip.open('mnist.pkl.gz','rb')
        training_dataa, validation_dataa, test_dataa = cPickle.load(f)
        
        f.close()
        
        test_x, test_y= test_dataa
        
        a=self.adv_inpt_assumed  ### created adv input
        test_x[self.test_x_correctly_classified]=a
        shared_x = theano.shared(
        np.asarray(test_x, dtype=theano.config.floatX), borrow=True)
            
        shared_y_y= theano.shared(
        np.asarray(test_y, dtype=theano.config.floatX), borrow=True)
        shared_y=T.cast(shared_y_y, "int32")
        
        
        i=T.lscalar()
        
        
        check_adv=theano.function(inputs=[i],outputs=self.layers[-1].accuracy(self.y),
                                     givens={
                                     self.x:
                                         shared_x[i*mini_batchsize:(i+1)*mini_batchsize],
                                    self.y:
                                        shared_y[i*mini_batchsize:(i+1)*mini_batchsize]})
                                        
                                        
        adv_check=check_adv(self.test_x_correctly_classified)
        return adv_check            
    
    
    
    
    
    '''I am done with implementation of proposed method for creating an adversarial example,
    named as Fast gradient method, lastly i want to check weather the created adversarial example is 
    really an adeversarial example with choosen hyper parameter epislon'''
            
            
    def check_adv_inpt(self,mini_batchsize):
        
        import cPickle
        import gzip
        f=gzip.open('mnist.pkl.gz','rb')
        training_dataa, validation_dataa, test_dataa = cPickle.load(f)
        
        f.close()
        
        test_x, test_y= test_dataa
        
        a=self.hypothetic_adv_inpt  ### created adv input
        test_x[self.test_x_correctly_classified]=a
        self.shared_x = theano.shared(
            np.asarray(test_x, dtype=theano.config.floatX), borrow=True)
            
        shared_y_y= theano.shared(
            np.asarray(test_y, dtype=theano.config.floatX), borrow=True)
        self.shared_y=T.cast(shared_y_y, "int32")
        
        
        i=T.lscalar()
        
        
        check_adv=theano.function(inputs=[i],outputs=self.layers[-1].accuracy(self.y),
                                     givens={
                                     self.x:
                                         self.shared_x[i*mini_batchsize:(i+1)*mini_batchsize],
                                    self.y:
                                        self.shared_y[i*mini_batchsize:(i+1)*mini_batchsize]})
                                        
                                        
        self.adv_check=check_adv(self.test_x_correctly_classified)
        return self.adv_check

           
            
            
            
            
            
        
        
        
            
            
            
        
            
##############################################################################################
#################### Defining the different Layers############################################
##############################################################################################


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

    def set_inpt(self, inpt, mini_batchsize):
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

    def set_inpt(self, inpt, mini_batchsize):
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

    def set_inpt(self, inpt, mini_batchsize):
        self.inpt = inpt.reshape((mini_batchsize, self.n_in))
        self.output = self.activation_fn(T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.eq(y, self.y_out)

class SoftmaxLayer(object):

    def __init__(self, n_in, n_out,params=params):
        self.n_in = n_in
        self.n_out = n_out
        self.w=params[6]
        self.b=params[7]
     
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, mini_batchsize):
        self.inpt = inpt.reshape((mini_batchsize, self.n_in))
        self.output = softmax (T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        



    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        
        
        g=T.eq(y, self.y_out)
        return g
        
        
    def cost(self, net):
        "Return the log-likelihood cost."
        return -T.mean(T.log(self.output)[T.arange(net.y.shape[0]), net.y])
        
        
    def confidence(self):
        
        output_conf = self.output

        
        
        return output_conf

        
        








#### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]
    
    
    
    
    

#### Lthe MNIST data #### 
def load_data_shared(filename="mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.
        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data), shared(validation_data), shared(test_data)]
