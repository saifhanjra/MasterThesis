# -*- coding: utf-8 -*-
"""
Created on Mon May 16 13:53:38 2016

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
from pylab import imshow, show, cm

# Activation functions
import pylab

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
        
    """ I have pretrained Network, with accuracy more than 99%. I need optimized cost function
        for the creation of adversial example. I calculate cost but only for one epoch using freezed 
        parameters. """




     
    def getting_cost(self,training_data,test_data,mini_batchsize):
        training_x, training_y=training_data
        test_x, test_y=test_data
        
        
        self.cost_param = self.layers[-1].cost(self)
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
                                        
                                        
        self.index_wrongly_classified=[]
        self.index_correctly_classified=[]
       
                                        
                                        
        for z in xrange(10000):
            self.accuracy=test_network(z)
            
            if self.accuracy==0: #### wrongly classified x
                wrongly_classified=z
                self.index_wrongly_classified.append(wrongly_classified)
                
            if self.accuracy==1: ### correctly_classified x
                correctly_classified=z
                self.index_correctly_classified.append(correctly_classified)
                
                
                
                
    
        
                
                
                
                
                
                
                
                
                
                            
        
                
                                        
        
            
            

#        

    """ Now, i have given my input to network , what i will do in the next step is to check
    weather this rightly classified or wrongly classified and if it is rightly classsified, 
    i followed the step to make it adversarial"""    
        
        
    def create_adversial_inpt(self,test_data,mini_batchsize,epislon):

            
            test_x,test_y=test_data### braking the test data into tuple 
        
        
            self.grad_x=T.grad(cost=self.cost_param, wrt=self.x) ### grad of cost function wrt self.x
            
                                                                #### which is input to network
            m=T.lscalar()
            f_grad_x=theano.function([m],outputs=self.grad_x,givens=
                                                            {self.x:
                                                                test_x[m*mini_batchsize:(m+1)*mini_batchsize],
                                                                self.y:
                                                                    test_y[m*mini_batchsize:(m+1)*mini_batchsize]
                                                                    })
                                                                    
                                                                    
                                                                    
            j=T.lscalar()
            
            f_test_inpt=theano.function([j],outputs=test_x[j])
            
            prob_cc_inpt=T.lscalar()
            softmax_value=theano.function([prob_cc_inpt],outputs=self.layers[-1].confidence(),givens=
                                                            {self.x:
                                                                test_x[prob_cc_inpt*mini_batchsize:(prob_cc_inpt+1)*mini_batchsize]})
            self.adv_inpt_list=[]
            self.dangerous_inpt_list=[]
            self.dangerous_inpt_grad_list=[]
            self.adv_inpt_grad_list=[]
            self.dang_corr_clas_grad_list=[]
            self.index_adv=[]
            self.index_dang_corr_class=[]
            self.prob_corr_classified_list=[]
            self.prob_wrongly_classified_list=[]
            self.prob_correctly_classified_test_inpt_high_grad=[] #### check the prob for corr classfied test inputs
            self.predicted_class_correct=[]
            self.predicted_class_adversarial=[] ####
            self.norm_grad_inpt_adv_list=[]
            self.norm_grad_high_correctly_classified_list=[] ####
            
            
            
            
            
            for i in xrange(len(self.index_correctly_classified)):
                self.index_corr=self.index_correctly_classified[i]
                grad_correctly_classified_inpt=f_grad_x(self.index_corr)
                norm_grad_corr_classified_inpt=np.linalg.norm( grad_correctly_classified_inpt)
                if norm_grad_corr_classified_inpt>0.0001:
                    print 'Given input is a dangerous one, can be classified wrongly'
                    self.dangerous_inpt_grad_list.append(norm_grad_corr_classified_inpt)

                    
                    self.dangerous_inpt_list.append(self.index_corr)
                    given_test_inpt=f_test_inpt(self.index_corr)
                    self.adv_inpt_assumed=given_test_inpt+epislon*np.sign(grad_correctly_classified_inpt)
#                    self.adv_inpt_assumed=given_test_inpt+epislon*(grad_correctly_classified_inpt/ norm_grad_corr_classified_inpt)                    
                    
                    pred_network_after=self.appending_adv_inpt_and_check_equality(mini_batchsize)
                    if pred_network_after==0:
                        print 'Adversial input has been given to Network'
                        self.index_adv.append(self.index_corr)
                        
                        self.adv_inpt_grad_list.append(norm_grad_corr_classified_inpt)
                        self.adv_inpt_list.append(self.index_corr)
                        ### now check the confidence of cnn of prdecting the wrong descision
                        probability_wrongly_classified,pred_class_adversarial=self.check_confidence(mini_batchsize)
                        self.prob_wrongly_classified_list.append(probability_wrongly_classified)
                        self.predicted_class_adversarial.append(pred_class_adversarial)
                        
                        
                        ####prob of coreectly classified test inputs
                        prob_correctly_classified_test_inpt,pred_class_correct=softmax_value(self.index_corr)
                        self.prob_correctly_classified_test_inpt_high_grad.append(prob_correctly_classified_test_inpt)
                        self.predicted_class_correct.append(pred_class_correct)
                        ###### Noramlized gradient of adversarial inputs
                        self.norm_grad_inpt_adv_list.append(norm_grad_corr_classified_inpt)
                        
                        
                    if pred_network_after==1:
                        print 'Dangerous but Correctly classified input'
                        self.index_dang_corr_class.append(self.index_corr)
                        self.dang_corr_clas_grad_list.append(norm_grad_corr_classified_inpt)
                        probability_correctly_classified=self.check_confidence(1)
                        self.prob_corr_classified_list.append(probability_correctly_classified)
                        self.norm_grad_high_correctly_classified_list.append(norm_grad_corr_classified_inpt)
                        
                        
                        
                        
                        
                        
    def appending_adv_inpt_and_check_equality(self,mini_batchsize):
        
        import cPickle
        import gzip
        f=gzip.open('mnist.pkl.gz','rb')
        training_dataa, validation_dataa, test_dataa = cPickle.load(f)
        
        f.close()
        
        test_x, test_y= test_dataa
        
        a=self.adv_inpt_assumed  ### created adv input
        test_x[self.index_corr]=a
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
                                        
                                        
        self.adv_check=check_adv(self.index_corr)
        
        return self.adv_check
        
        
    
    
    
    
    
    
    def check_confidence(self,mini_batchsize):
        j=T.lscalar()
        confidence=theano.function(inputs=[j],outputs=self.layers[-1].confidence(),
                                     givens={
                                     self.x:
                                         self.shared_x[j*mini_batchsize:(j+1)*mini_batchsize]
                                         })
                                        
        conf_corr_wrong, pred_class_corr_wrong=confidence(self.index_corr)
        return (conf_corr_wrong, pred_class_corr_wrong)
        
        

        
        


                    


                

                
                
                
                

                        
                    
                
                
                            
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
        predicted_class= self.y_out
#        output_conf = self.output
        output_conf_max= T.max(self.output,axis=1)
        
        
        
        
        
        
        return (output_conf_max,predicted_class)






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