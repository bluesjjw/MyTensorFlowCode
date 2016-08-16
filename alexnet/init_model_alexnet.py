from __future__ import division, print_function, absolute_import

import tensorflow as tf

import sys
sys.path.append("..")
from utils import Utils

model_path = 'init_mnist.ckpt'

conv1 = {
    'weights': tf.Variable(Utils.xavier_init(11, 11, 1,, n_hidden_1)),
    'encoder_h2': tf.Variable(Utils.xavier_init(n_hidden_1, n_hidden_2)),
    'decoder_h1': tf.Variable(Utils.xavier_init(n_hidden_2, n_hidden_1)),
    'decoder_h2': tf.Variable(Utils.xavier_init(n_hidden_1, n_input)),
}