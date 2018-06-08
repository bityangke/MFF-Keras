
# coding: utf-8
import pdb
import warnings
import sys
sys.path.append("../Keras-Group-Normalization")

import math
import numpy as np

from keras import backend as K
from keras import optimizers
from keras.applications import inception_v3, vgg19, resnet50
from keras.layers import Dense, Dropout, Activation, Average
from keras.layers import Concatenate, Conv2D, Input, Flatten
from keras.models import Sequential, Model
from keras.initializers import RandomNormal

from utils import *
from group_norm import GroupNormalization
# modules = {'VGG19': vgg19, 'ResNet50': resnet50, 'InceptionV3': inception_v3}
model_initializers = {
            'VGG19': vgg19.VGG19,
            'ResNet50': resnet50.ResNet50, 
            'InceptionV3': inception_v3.InceptionV3
        }

input_dims = {
        'VGG19': 224,
        'ResNet50': 224,
        'InceptionV3': 229
    }


class TSN():
    def __init__(self, num_class, num_segments, modality, architecture='resnet', new_length=None,
                 consensus_type='mlp', before_softmax=True, num_motion=3,
                 dropout=0.8, img_feature_dim=256, dataset='jester',
                 crop_num=1, partial_bn=True, print_spec=True, group_norm=True):
        
        self.modality = modality
        self.num_segments = num_segments
        self.num_motion = num_motion
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.dataset = dataset
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.img_feature_dim = img_feature_dim  # the dimension of the CNN feature to represent each frame
        self.num_class = num_class
        self._enable_pbn = partial_bn
        self.group_norm=group_norm

        self.image_dim = input_dims[architecture]

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        # Establish number of channels
        if new_length is None:
            if modality == "RGB":
                self.new_length = 1
                self.num_channels = 3
            elif modality == "Flow":
                self.new_length = self.num_motion
                self.num_channels = 2*self.new_length
            elif modality == "RGBFlow":
                self.new_length = self.num_motion
                self.num_channels = 2*self.new_length+3
        else:
            self.new_length = new_length
            
        
        # Set up base (vgg, inception, etc)
        self._prepare_base_model(architecture)
                
        # Convert from RGB to RGBFlow, RGBDiff or Flow
        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            raise ValueError("Flow only not implemented yet")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            raise ValueError("RGBDiff not implemented yet")
        elif self.modality == 'RGBFlow':
            self._construct_rgbflow_model()
            print("RGBFlow model ready.")
        else: 
            self._construct_rgb_model()
            print("RGB model ready.")
        
        self._prepare_tsn()
        self._prepare_siamese()
        self._prepare_consensus_layer()
        
        self.total_model = Model(inputs=self.input_layers, outputs=self.total_model)
        
        
        
    def _prepare_consensus_layer(self):
        if self.consensus_type.lower() == 'mlp':
            num_bottleneck = 512
            self.total_model = Activation('relu')(self.merged)
            self.total_model = Dense(num_bottleneck, activation='relu')(self.total_model)
            self.total_model = Dense(self.num_class, activation=None)(self.total_model)
            
        elif self.consensus_type == 'rnn':
            pass
        else:
            raise ValueError("Averaging not implemented yet")

    
    # Retrieves the base model from keras.appliactions
    def _prepare_base_model(self, architecture):
        #todo delete dis
        weights_input_shape = (self.image_dim, self.image_dim, 3)
        base_input_shape = (self.image_dim, self.image_dim, self.num_channels)
        
        model_initializer = model_initializers[architecture]

        if architecture == 'ResNet50' or architecture == 'VGG19':
            
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length
            
        elif architecture == 'InceptionV3':
            
            self.input_mean = [104, 117, 128]
            self.input_std = [1]
            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1+self.new_length)
                
        else:
            raise ValueError('Unknown architecture: {}'.format(architecture))

        self.weights_model = model_initializer(include_top=False, 
                    weights='imagenet', 
                    input_shape=weights_input_shape,
                    pooling=None, 
                    classes=self.num_class)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.base_model = model_initializer(include_top=False, 
                        input_shape=base_input_shape, 
                        weights=None,
                        pooling=None, 
                        classes=self.num_class)
        


        
        

    def _prepare_tsn(self):
        
        initializer = RandomNormal(mean=0.0, stddev=0.001, seed=None)
        
        self.base_model = Flatten()(self.base_model.output)
        self.base_model = Dropout(rate=self.dropout)(self.base_model)

        if self.consensus_type.lower() == 'mlp':
            # set the MFFs feature dimension
            self.base_model = Dense(self.img_feature_dim, kernel_initializer=initializer)(self.base_model)
        else:
            # the default consensus types in TSN
            self.base_model = Dense(self.num_class, kernel_initializer=initializer)(self.base_model)
        
        # Add softmax TODO change argument name
        if not self.before_softmax:
            self.base_model = Activation('softmax')(self.base_model)
            

    def _construct_rgbflow_model(self):        

        special_init_done = False
        for new_layer, weight_layer in zip(self.base_model.layers, self.weights_model.layers):
            # Find first conv layer
            if isinstance(new_layer, Conv2D) and not special_init_done:
                params = [x.copy() for x in weight_layer.get_weights()]
                weight = params[0]

                # RGB plus optical flow
                H, W, C, num_filters = weight.shape
                # Allocate new weights
                new_kernel_size = (H, W, self.num_channels, num_filters)
                new_kernels = np.zeros(new_kernel_size)

                # See Motion Fused Frames paper, section 3.3 for initialization technique
                new_kernels[:,:,:3,:] = weight
                new_kernels[:,:,3:,:] = np.tile(
                                            np.mean(weight, axis=2, keepdims=True), 
                                            (1, 1, 2*self.new_length,1))
                
                # Allocate new weights
                new_kernel_size = (H, W, self.num_channels, num_filters)
                new_kernels = np.zeros(new_kernel_size)
                params[0] = new_kernels

                new_layer.set_weights(params)
                special_init_done = True
            else:
                # copy over downstream layers
                new_layer.set_weights(weight_layer.get_weights())

        # save input layer for model conversion
        self.base_input_layer = self.base_model.input
        if self.group_norm:
            base_input_shape = (self.image_dim, self.image_dim, self.num_channels)
            gn_input = Input(shape=base_input_shape)
            gn_tensor = GroupNormalization(groups=self.num_channels)(gn_input)  
            self.base_input_layer = gn_input
            self.base_model = self.base_model(gn_tensor)
            self.base_model = Model(gn_input, self.base_model)

        # delete canibalized weights model
        del self.weights_model
       
    
    def _construct_rgb_model(self):
        self.base_model = self.weights_model
        self.base_input_layer = self.weights_model.input
   
    
    def _prepare_siamese(self):
       
        self.base_model = Model(self.base_input_layer, self.base_model, name="tsn")


        self.input_layers = [Input(shape=(self.image_dim, self.image_dim, self.num_channels), dtype='float32') for i in range(self.num_segments)]
        siamese_outputs = [self.base_model(input_) for input_ in self.input_layers]

        self.merged = Concatenate()(siamese_outputs)
    



    def get_augmentation(self):
        print("Augmentation not implemented yet")
        pass
