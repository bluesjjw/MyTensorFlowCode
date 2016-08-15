# -*- coding: utf-8 -*-

""" AlexNet.
Applying 'Alexnet' to Oxford's 17 Category Flower Dataset classification task.
References:
    - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet
    Classification with Deep Convolutional Neural Networks. NIPS, 2012.
    - 17 Category Flower Dataset. Maria-Elena Nilsback and Andrew Zisserman.
Links:
    - [AlexNet Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
    - [Flower Dataset (17)](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/)
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

import tflearn.datasets.oxflower17 as oxflower17
X, Y = oxflower17.load_data(dirname='/home/jiawei/dataset/17flowers/', one_hot=True, resize_pics=(227, 227))

tflearn.config.init_graph (log_device=False, soft_placement=True)

# Building 'AlexNet'
network = input_data(shape=[None, 227, 227, 3])
network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 17, activation='softmax')
network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Job id
run_id = 'alexnet_oxflowers17'

tensorboard_verbose = 3

log_dir = 'tflearn_logs/'

# Device
gpu = '/gpu:0'
cpu = '/cpu:0'

# Checkpoint
check_path = 'model_alexnet'
max_checkpoints = 10
snapshot_step=200
is_snapshot_epoch=False

# Training parameters
n_epoch=100
valid_ratio=0.1
is_shuffle=True
is_show_metric=True
batch_size=256

# Training
with tf.device(gpu):
    # Force all Variables to reside on the CPU.
    # with tf.arg_ops([tflearn.variables.variable], device=cpu):
    model = tflearn.DNN(network, checkpoint_path=check_path, max_checkpoints=max_checkpoints, 
                        tensorboard_dir=log_dir, tensorboard_verbose=tensorboard_verbose)
    model.fit(X, Y, n_epoch=n_epoch, validation_set=valid_ratio, shuffle=is_shuffle,
              show_metric=is_show_metric, batch_size=batch_size, snapshot_step=snapshot_step,
              snapshot_epoch=is_snapshot_epoch, run_id=run_id)