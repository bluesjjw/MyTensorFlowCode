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


import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tflearn

# Data loading and preprocessing
import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(dirname='/home/jiawei/dataset/mnist/', one_hot=True)

tflearn.config.init_graph (log_device=False, soft_placement=True)

# Training parameters
n_epoch=100
batch_size=256
optimizer_name = 'sgd'
if len(sys.argv) >= 2:
    optimizer_name = sys.argv[1]
learning_rate = 0.01
if len(sys.argv) >= 3:
    learning_rate = float(sys.argv[2])
loss_name = 'mean_square'

# Job id
run_id = 'autoencoder_mnist_' + optimizer_name + str(learning_rate)

# Tensorboard
tensorboard_mode = 0
log_dir = 'tflearn_logs/'

# Device
gpu = '/gpu:0'
cpu = '/cpu:0'

# Checkpoint
check_path = 'model_' + optimizer_name + str(learning_rate)
max_checkpoints = 10
snapshot_step=200
is_snapshot_epoch=False

# Building the encoder
encoder = tflearn.input_data(shape=[None, 784])
encoder = tflearn.fully_connected(encoder, 256)
encoder = tflearn.fully_connected(encoder, 128)

# Building the decoder
decoder = tflearn.fully_connected(encoder, 256)
decoder = tflearn.fully_connected(decoder, 784)

# Regression, with mean square error
net = tflearn.regression(decoder, optimizer=optimizer_name, learning_rate=learning_rate,
                         loss=loss_name, metric=None)

with tf.device(gpu):
	# Training the auto encoder
	model = tflearn.DNN(net, tensorboard_verbose=tensorboard_mode)
	model.fit(X, X, n_epoch=n_epoch, validation_set=(testX, testX),
		run_id=run_id, batch_size=batch_size)

# Encoding X[0] for test
print("\nTest encoding of X[0]:")
# New model, re-using the same session, for weights sharing
encoding_model = tflearn.DNN(encoder, session=model.session)
print(encoding_model.predict([X[0]]))

# Testing the image reconstruction on new data (test set)
print("\nVisualizing results after being encoded and decoded:")
testX = tflearn.data_utils.shuffle(testX)[0]
# Applying encode and decode over test set
encode_decode = model.predict(testX)
# Compare original images with their reconstructions
f, a = plt.subplots(2, 10, figsize=(10, 2))
examples_to_show = 10
for i in range(examples_to_show):
    a[0][i].imshow(np.reshape(testX[i], (28, 28)))
    a[0][i].axis('off')
    a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    a[1][i].axis('off')
save_image_name = 'image/tflearn-mnist-' + optimizer_name + str(learning_rate) + '.png'
plt.savefig(save_image_name)
#f.show()
#plt.draw()
#plt.waitforbuttonpress()