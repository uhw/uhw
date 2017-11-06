
# coding: utf-8

# In[134]:


import tensorflow as tf
from keras.utils import np_utils
import numpy as np
import datetime
from keras.models import Input, Model,Sequential
from keras.layers import Conv2D, MaxPool2D, Conv2DTranspose, UpSampling2D,LeakyReLU
from keras.layers import BatchNormalization, Dropout, Add
from keras.layers import Dense, Activation, Flatten, Reshape, LeakyReLU
from keras.initializers import RandomNormal
from keras import backend as K

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


from tensorflow.examples.tutorials.mnist import input_data


# In[127]:


class Subpixel(Conv2D):
    def __init__(self,
                 filters,
                 kernel_size,
                 r,
                 padding='same',
                 data_format=None,
                 strides=(1,1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Subpixel, self).__init__(
            filters=r*r*filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.r = r

    def _phase_shift(self, I):
        r = self.r
        bsize, a, b, c = I.get_shape().as_list()
        bsize = K.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
        X = K.reshape(I, [bsize, a, b, int(c/(r*r)),r, r]) # bsize, a, b, c/(r*r), r, r
        X = K.permute_dimensions(X, (0, 1, 2, 5, 4, 3))  # bsize, a, b, r, r, c/(r*r)
        #Keras backend does not support tf.split, so in future versions this could be nicer
        X = [X[:,i,:,:,:,:] for i in range(a)] # a, [bsize, b, r, r, c/(r*r)
        X = K.concatenate(X, 2)  # bsize, b, a*r, r, c/(r*r)
        X = [X[:,i,:,:,:] for i in range(b)] # b, [bsize, r, r, c/(r*r)
        X = K.concatenate(X, 2)  # bsize, a*r, b*r, c/(r*r)
        return X

    def call(self, inputs):
        return self._phase_shift(super(Subpixel, self).call(inputs))

    def compute_output_shape(self, input_shape):
        unshifted = super(Subpixel, self).compute_output_shape(input_shape)
        return (unshifted[0], int(self.r*unshifted[1]), int(self.r*unshifted[2]), int(unshifted[3]/(self.r*self.r)))

    def get_config(self):
        config = super(Conv2D, self).get_config()
        config.pop('rank')
        config.pop('dilation_rate')
        config['filters']/=int(self.r*self.r)
        config['r'] = self.r
        return config


# In[151]:


class AnimeGeneratorFactory():
    
    def build(self, input_shape):
        """
            Returns a generator Model described here: https://arxiv.org/pdf/1708.05509.pdf
            
            Args:
                input_same: A 3 length tuple describing (width, height, channel)
                
            Output:
                Keras Model
        """
        MOMENTUM = 0.9
        DIM = 16
        DEPTH = 64
        NUM_RESIDUAL = 16
        NUM_SUBPIXEL = 2
        FINAL_FILTERS = 3
        INITIAL_FILTERS = 64
        def residual_block(layer, filters, momentum):
            """
                Residual Block consisting of
                    Conv2D -> Batch Normalization -> relu -> Conv2D -> Batch Normalization -> Residual Addition
                
                Args:
                    layer:   Keras Layer
                    filters: output size as an integer
                    momentum: variable for batch normalization
                
                Returns:
                    Keras layer
            """
            shortcut = layer
            layer = Conv2DTranspose(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding="same")(layer)
            layer = BatchNormalization(momentum= momentum)(layer)
            layer = Activation('relu')(layer)
            layer = Conv2DTranspose(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding="same")(layer)
            layer = BatchNormalization(momentum= momentum)(layer)

            layer = Add()([layer, shortcut])
            return layer
        
        def residual_layer(layer, number, filters, momentum):
            """
                Facade for residual block.
                
                Creates Residual layer with specified number of residual blocks
                
                Args:
                    layer:   Keras layer
                    number:  number of residual blocks in layer
                    filters: output size as an integer
                    momentum: variable for batch normalization
                
                Returns:
                    Keras layer
            """
            for _ in range(number):
                layer = residual_block(layer, filters, momentum)
            return layer
        
        def subpixel_block(layer, filters, momentum):
            """
                sub-pixel block consisting of
                    Conv2D -> pixel shuffler x 2 -> Batch Normalization -> Relu
                    
                the code of subpixel layer is based on https://github.com/Tetrachrome/subpixel
                    
                Args:
                    layer:   Keras Layer
                    filters: output size as an integer
                    momentum: variable for batch normalization
                
                Returns:
                    Keras layer
            """  
            
            layer = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding="same")(layer)
            layer = Subpixel(filters, (3,3), 2)(layer)

            layer = BatchNormalization(momentum= momentum)(layer)
            layer = Activation('relu')(layer)
            return layer
        
        def subpixel_layer(layer, number, filters, momentum):
            """
                Facade for subpixel block.
                
                Creates subpixel layer with specified number of subpixel blocks
                
                Args:
                    layer:   Keras layer
                    number:  number of subpixel blocks in layer
                    filters: output size as an integer
                    momentum: variable for batch normalization
                
                Returns:
                    Keras layer
            """
            for _ in range(number):
                layer = subpixel_block(layer, filters, momentum)
            return layer
        inputs = Input(shape=input_shape)
        filters = INITIAL_FILTERS
        layer = Dense(DEPTH*DIM*DIM)(inputs)

        layer = BatchNormalization(momentum = MOMENTUM)(layer)
        layer = Activation('relu')(layer)
        layer = Reshape((DIM,DIM,DEPTH))(layer)
        old = layer
        print("starting layer built")
        # 16 residual layers
        layer = residual_layer(layer, NUM_RESIDUAL, filters, MOMENTUM)
    
        
        layer = BatchNormalization(momentum = MOMENTUM)(layer)
        layer = Activation('relu')(layer)
        layer = Add()([layer, old])
        
        print("residual layer built")

        filters *= 4
        # 3 sub-pixel layers
        layer = subpixel_layer(layer, NUM_SUBPIXEL, filters, MOMENTUM)

        print("sub-pixel layer built")
        
        layer = Conv2D(filters=FINAL_FILTERS, kernel_size=(9, 9), strides=(1, 1), padding="same")(layer)
        layer = Activation('tanh')(layer)
        
        print("final layer built")
        return Model(input = inputs, output = layer)
        


# In[157]:


generator = AnimeGeneratorFactory().build([128+34])


# In[153]:


print (generator.summary())


# In[158]:


zsamples = np.random.normal(size=[10 * 10, 162])
zsamples.shape
images = generator.predict(zsamples)

