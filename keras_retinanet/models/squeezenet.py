"""
Author: Igor Muniz Soares
Date: 08/2018

"""


import keras
from keras.utils import get_file
from keras.models import Model
from keras.layers import Input, MaxPool2D,  Conv2D, Dropout, concatenate, Reshape, Lambda, AveragePooling2D
from keras import backend as K
from keras.initializers import TruncatedNormal
from keras.regularizers import l2
import numpy as np
import tensorflow as tf

from . import retinanet
from . import Backbone
from ..utils.image import preprocess_image


class SqueezeBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return squeeze_retinanet(*args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):
        """ Downloads ImageNet weights and returns path to weights file.
        Weights can be downloaded at https://github.com/fizyr/keras-models/releases .
        """
        if self.backbone == 'vgg16':
            resource = keras.applications.vgg16.WEIGHTS_PATH_NO_TOP
            checksum = '6d6bbae143d832006294945121d1f1fc'
        elif self.backbone == 'vgg19':
            resource = keras.applications.vgg19.WEIGHTS_PATH_NO_TOP
            checksum = '253f8cb515780f3b799900260a226db6'
        else:
            raise ValueError("Backbone '{}' not recognized.".format(self.backbone))

        return get_file(
            '{}_weights_tf_dim_ordering_tf_kernels_notop.h5'.format(self.backbone),
            resource,
            cache_subdir='models',
            file_hash=checksum
        )

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['squeezenet']

        if self.backbone not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(self.backbone, allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='caffe')


class SqueezeNet():
    # initialize model from config file
    def __init__(self, input=None, weight_decay=0.001):

        # hyperparameter config file
        if input == None:
            self.input = (None,None,3)
        else:
            self.input = input
        # create Keras model
        self.WEIGHT_DECAY = weight_decay
        self.model = self._create_model()

    # creates keras model
    def _create_model(self):
        
        input_layer = Input(shape=self.input,
                            name="input_1")

        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME", activation='relu',
                       use_bias=True, kernel_initializer=TruncatedNormal(stddev=0.001),
                       kernel_regularizer=l2(self.WEIGHT_DECAY))(input_layer)

        pool1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='SAME', name="pool1")(conv1)

        fire2 = self._fire_layer(name="fire2", input=pool1, s1x1=16, e1x1=64, e3x3=64)

        fire3 = self._fire_layer(
            'fire3', fire2, s1x1=16, e1x1=64, e3x3=64)
        pool3 = MaxPool2D(
            pool_size=(3, 3), strides=(2, 2), padding='SAME', name='pool3')(fire3)

        fire4 = self._fire_layer(
            'fire4', pool3, s1x1=32, e1x1=128, e3x3=128)
        fire5 = self._fire_layer(
            'fire5', fire4, s1x1=32, e1x1=128, e3x3=128)

        pool5 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='SAME', name="pool5")(fire5)

        fire6 = self._fire_layer(
            'fire6', pool5, s1x1=48, e1x1=192, e3x3=192)
        fire7 = self._fire_layer(
            'fire7', fire6, s1x1=48, e1x1=192, e3x3=192)
        fire8 = self._fire_layer(
            'fire8', fire7, s1x1=64, e1x1=256, e3x3=256)
        fire9 = self._fire_layer(
            'fire9', fire8, s1x1=64, e1x1=256, e3x3=256)

        pool9 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='SAME', name="pool9")(fire9)

        model = Model(inputs=input_layer, outputs=pool9)

        return model

    def _fire_layer(self, name, input, s1x1, e1x1, e3x3, stdd=0.01):
        """
        wrapper for fire layer constructions
        :param name: name for layer
        :param input: previous layer
        :param s1x1: number of filters for squeezing
        :param e1x1: number of filter for expand 1x1
        :param e3x3: number of filter for expand 3x3
        :param stdd: standard deviation used for intialization
        :return: a keras fire layer
        """

        sq1x1 = Conv2D(
            name=name + '/squeeze1x1', filters=s1x1, kernel_size=(1, 1), strides=(1, 1), use_bias=True,
            padding='SAME', kernel_initializer=TruncatedNormal(stddev=stdd), activation="relu",
            kernel_regularizer=l2(self.WEIGHT_DECAY))(input)

        ex1x1 = Conv2D(
            name=name + '/expand1x1', filters=e1x1, kernel_size=(1, 1), strides=(1, 1), use_bias=True,
            padding='SAME', kernel_initializer=TruncatedNormal(stddev=stdd), activation="relu",
            kernel_regularizer=l2(self.WEIGHT_DECAY))(sq1x1)

        ex3x3 = Conv2D(
            name=name + '/expand3x3', filters=e3x3, kernel_size=(3, 3), strides=(1, 1), use_bias=True,
            padding='SAME', kernel_initializer=TruncatedNormal(stddev=stdd), activation="relu",
            kernel_regularizer=l2(self.WEIGHT_DECAY))(sq1x1)

        return concatenate([ex1x1, ex3x3], axis=3)


def squeeze_retinanet(num_classes, backbone='squeezenet', inputs=None, modifier=None, **kwargs):
    """ Constructs a retinanet model using a squeeze backbone.

    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (one of ('vgg16', 'vgg19')).
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).

    Returns
        RetinaNet model with a VGG backbone.
    """
    # choose default input
    if inputs is None:
        inputs = (None, None, 3)

    # create the squeezenet backbone

    squeezenet = SqueezeNet(inputs).model
    #print(squeezenet.summary())

    if modifier:
        squeezenet = modifier(squeezenet)


    # create the full model
    layer_names = ["pool3", "pool5", "pool9"]
    layer_outputs = [squeezenet.get_layer(name).output for name in layer_names]

    return retinanet.retinanet(inputs=squeezenet.inputs, num_classes=num_classes, backbone_layers=layer_outputs, **kwargs)

