# -*- coding: utf-8 -*-

""" Auto Encoder Example.
Using an auto encoder on MNIST handwritten digits.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Import sub-module
import sys
sys.path.append("..")
from utils import Utils

# Parameters
learning_rate = 0.01
momentum = 0.1
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10

# Device config
device = '/gpu:2'
log_device_placement = True

# Network Parameters
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 128 # 2nd layer num features
n_input = 784 # MNIST data input (img shape: 28*28)

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2

# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

with tf.device(device):
    # TF Graph input (only pictures)
    X = tf.placeholder("float", [None, n_input])

    weights = {
        'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
        'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
    }
    biases = {
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'decoder_b2': tf.Variable(tf.random_normal([n_input])),
    }

    # Construct model
    encoder_op = encoder(X)
    decoder_op = decoder(encoder_op)

    # Prediction
    y_pred = decoder_op
    # Targets (Labels) are the input data.
    y_true = X
    # Define loss (the squared error)
    cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))

with tf.device(device):
     # Optimizer algorithm
    optim_alg = tf.train.AdamOptimizer(learning_rate)

    # Save image
    save_image_name = 'mnist-adam.png'
    
    # Define optimizer, minimize the squared error
    optimizer = optim_alg.minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Model saver
saver = tf.train.Saver({"encoder_h1": weights['encoder_h1'], 
    "encoder_h2": weights['encoder_h2'],
    "decoder_h1": weights['decoder_h1'],
    "decoder_h2": weights['decoder_h2'],
    "encoder_b1": biases['encoder_b1'],
    "encoder_b2": biases['encoder_b2'],
    "decoder_b1": biases['decoder_b1'],
    "decoder_b2": biases['decoder_b2']
    })

# Launch the graph
with tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement)) as sess:
    sess.run(init)
    # Load initial weights and biases from local file
    load_path = saver.restore(sess, model_path)
    print "Model restore from file: %s" % load_path
    print (sess.run(weights['encoder_h1']))
    print (sess.run(biases['encoder_b1']))

    total_batch = int(mnist.train.num_examples/batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    # Applying encode and decode over test set
    encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[0][i].axis('off')
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
        a[1][i].axis('off')
    plt.savefig(save_image_name)
    #f.show()
    #plt.draw()
    #plt.waitforbuttonpress()
