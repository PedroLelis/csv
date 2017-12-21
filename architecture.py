import tensorflow as tf
import pandas as pd
import numpy as np

import argparse
import sys

sys.path.append("../submodules/models/research/")
sys.path.append("../submodules/models/research/slim/")
sys.path.append("../submodules/models/research/slim/nets/")

from slim.nets import alexnet
from slim.nets import cifarnet
from slim.nets import inception
from slim.nets import lenet
from slim.nets import mobilenet_v1
from slim.nets import overfeat
from slim.nets import resnet_v1
from slim.nets import resnet_v2
from slim.nets import vgg
from slim.nets.nasnet import nasnet

from tensorflow.contrib import slim

networks_map = {
    'alexnet_v2': alexnet.alexnet_v2,
    'cifarnet': cifarnet.cifarnet,
    'overfeat': overfeat.overfeat,
    'vgg_a': vgg.vgg_a,
    'vgg_16': vgg.vgg_16,
    'vgg_19': vgg.vgg_19,
    'inception_v1': inception.inception_v1_base,
    'inception_v2': inception.inception_v2_base,
    'inception_v3': inception.inception_v3_base,
    'inception_v4': inception.inception_v4_base,
    'inception_resnet_v2': inception.inception_resnet_v2_base,
    'lenet': lenet.lenet,
    'resnet_v1_50': resnet_v1.resnet_v1_50,
    'resnet_v1_101': resnet_v1.resnet_v1_101,
    'resnet_v1_152': resnet_v1.resnet_v1_152,
    'resnet_v1_200': resnet_v1.resnet_v1_200,
    'resnet_v2_50': resnet_v2.resnet_v2_50,
    'resnet_v2_101': resnet_v2.resnet_v2_101,
    'resnet_v2_152': resnet_v2.resnet_v2_152,
    'resnet_v2_200': resnet_v2.resnet_v2_200,
    'mobilenet_v1': mobilenet_v1.mobilenet_v1,
    'mobilenet_v1_075': mobilenet_v1.mobilenet_v1_075,
    'mobilenet_v1_050': mobilenet_v1.mobilenet_v1_050,
    'mobilenet_v1_025': mobilenet_v1.mobilenet_v1_025,
    'nasnet_cifar': nasnet.build_nasnet_cifar,
    'nasnet_mobile': nasnet.build_nasnet_mobile,
    'nasnet_large': nasnet.build_nasnet_large
    }

arg_scopes_map = {
    'alexnet_v2': alexnet.alexnet_v2_arg_scope,
    'cifarnet': cifarnet.cifarnet_arg_scope,
    'overfeat': overfeat.overfeat_arg_scope,
    'vgg_a': vgg.vgg_arg_scope,
    'vgg_16': vgg.vgg_arg_scope,
    'vgg_19': vgg.vgg_arg_scope,
    'inception_v1': inception.inception_v3_arg_scope,
    'inception_v2': inception.inception_v3_arg_scope,
    'inception_v3': inception.inception_v3_arg_scope,
    'inception_v4': inception.inception_v4_arg_scope,
    'inception_resnet_v2': inception.inception_resnet_v2_arg_scope,
    'lenet': lenet.lenet_arg_scope,
    'resnet_v1_50': resnet_v1.resnet_arg_scope,
    'resnet_v1_101': resnet_v1.resnet_arg_scope,
    'resnet_v1_152': resnet_v1.resnet_arg_scope,
    'resnet_v1_200': resnet_v1.resnet_arg_scope,
    'resnet_v2_50': resnet_v2.resnet_arg_scope,
    'resnet_v2_101': resnet_v2.resnet_arg_scope,
    'resnet_v2_152': resnet_v2.resnet_arg_scope,
    'resnet_v2_200': resnet_v2.resnet_arg_scope,
    'mobilenet_v1': mobilenet_v1.mobilenet_v1_arg_scope,
    'mobilenet_v1_075': mobilenet_v1.mobilenet_v1_arg_scope,
    'mobilenet_v1_050': mobilenet_v1.mobilenet_v1_arg_scope,
    'mobilenet_v1_025': mobilenet_v1.mobilenet_v1_arg_scope,
    'nasnet_cifar': nasnet.nasnet_cifar_arg_scope,
    'nasnet_mobile': nasnet.nasnet_mobile_arg_scope,
    'nasnet_large': nasnet.nasnet_large_arg_scope
    }

scopes_name_map = {
    # 'alexnet_v2': ,
    # 'cifarnet': ,
    # 'overfeat': ,
    # 'vgg_a': ,
    # 'vgg_16': ,
    # 'vgg_19': ,
    'inception_v1': 'InceptionV1/',
    'inception_v2': 'InceptionV2/',
    'inception_v3': 'InceptionV3/',
    'inception_v4': 'InceptionV4/',
    # 'inception_resnet_v2': ,
    # 'lenet': ,
    # 'resnet_v1_50': ,
    # 'resnet_v1_101': ,
    # 'resnet_v1_152': ,
    # 'resnet_v1_200': ,
    # 'resnet_v2_50': ,
    # 'resnet_v2_101': ,
    # 'resnet_v2_152': ,
    # 'resnet_v2_200': ,
    # 'mobilenet_v1': ,
    # 'mobilenet_v1_075': ,
    # 'mobilenet_v1_050': ,
    # 'mobilenet_v1_025': ,
    # 'nasnet_cifar': ,
    # 'nasnet_mobile': ,
    # 'nasnet_large': ,
}


def cnn_architecture(inputs, num_labels, is_training, network_name):
    """
    Return network architecture

    Args:
        inputs: Tensor
            Input Tensor
        is_training: bool
            Whether the network will be used to train or not. Used for dropout operation

    Returns:
        Logits for each demographic network branch
    """
    # Load common model architeture - except Fully-Connected layers and logits
    with slim.arg_scope(arg_scopes_map[network_name]()):
        net, endpoints = networks_map[network_name](inputs)

    # Add Fully-Connected layers
    with tf.variable_scope('Fully-Connected_layers'):
        net = tf.layers.flatten(net)
        net = tf.layers.dense(
            net, num_labels, activation=None, name='FC1', trainable=True)

    return net
