import numpy as np

import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from Autoencoder import Autoencoder

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import datetime

mnist = input_data.read_data_sets('../mnist', one_hot = True)

def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_hidden = 256 # hidden layer num features

# tf Graph input
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
n_samples = int(mnist.train.num_examples)

# training hyper-parameter
training_epochs = 100
batch_size = 128
learning_rate = 0.001
momentum = 0.1
display_step = 1

# activation function and optimizer
transfer_function = tf.nn.softplus
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum = momentum)

# save image
save_image_name = 'mnist-momen.png'

# device config
device = '/gpu:2'
log_device_placement = True

with tf.device(device):
    autoencoder = Autoencoder(device = device, n_input = n_input, n_hidden = n_hidden, transfer_function = transfer_function, optimizer = optimizer)

    start_time = datetime.datetime.now()
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = get_random_block_from_data(X_train, batch_size)

            # Fit training using batch data
            cost = autoencoder.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch + 1), \
                "cost=", "{:.9f}".format(avg_cost)

    end_time = datetime.datetime.now()

    print "Total time of training: " + str(end_time - start_time)
    print "Total cost of test set: " + str(autoencoder.calc_total_cost(X_test))

examples_to_show = 10
# Applying encode and decode over test set
encode_decode = autoencoder.reconstruct(X_test[:examples_to_show])
# Compare original images with their reconstructions
f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(examples_to_show):
    ax1 = a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    #ax1.set_cmap('hot')
    a[0][i].axis('off')
    #ax1.axes.get_xaxis().set_visible(False)
    #ax1.axes.get_yaxis().set_visible(False)
    #ax1.axes.get_xaxis().set_ticks([])
    #ax1.axes.get_yaxis().set_ticks([])
    #ax1.axes.get_xaxis().set_ticklabels([])
    #ax1.axes.get_yaxis().set_ticklabels([])
    ax2 = a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    a[1][i].axis('off')

#f.show()
#plt.draw()
#plt.waitforbuttonpress()
plt.savefig(save_image_name)

#plt.figure(figsize=(10,2))
#for i in range(examples_to_show):
#    plt.subplot(2,10,i+1)
#    plt.imshow(np.reshape(mnist.test.images[i], (28, 28)))
#    plt.subplot(2,10,i+11)
#    plt.imshow(np.reshape(encode_decode[i], (28, 28)))