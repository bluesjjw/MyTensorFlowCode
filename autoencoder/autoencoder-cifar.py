from __future__ import division, absolute_import, print_function

import numpy as np
import tensorflow as tf

# Import sub-module
import sys
sys.path.append('..')
from datasets import cifar10

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import datetime

# Building the encoder
def encoder(x):
	with tf.device(gpu):
		# Encoder Hidden layer with sigmoid activation #1
		layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
								   biases['encoder_b1']))
		# Decoder Hidden layer with sigmoid activation #2
		layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
								   biases['encoder_b2']))
		return layer_2

# Building the decoder
def decoder(x):
	with tf.device(gpu):
		# Encoder Hidden layer with sigmoid activation #1
		layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
								   biases['decoder_b1']))
		# Decoder Hidden layer with sigmoid activation #2
		layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
								   biases['decoder_b2']))
		return layer_2

# initial model

# Train parameters
optimizer_name = 'sgd' # default optimization algorithm
learning_rate = 0.001
momentum = 0.9
training_epochs = 100
batch_size = 256
display_step = 1
examples_to_show = 10

# Device config
gpu = '/gpu:2'
cpu = '/cpu:0'
log_device_placement = True

# Network parameters
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 128 # 2nd layer num features
n_input = 1024 # cifear10 data input (img shape: 32*32, gray scale)

if __name__ == '__main__':
	print('====================================')
	print('Train autoencoder on cifar10 dataset')
	print('====================================')

	if len(sys.argv) >= 2:
		optimizer_name = sys.argv[1]
	print('Optimizer: ' + optimizer_name)
	if len(sys.argv) >= 3:
		learning_rate = float(sys.argv[2])
	print('Learning rate: ' + str(learning_rate))
	if len(sys.argv) >= 4:
		momentum = float(sys.argv[3])
	print('Momentum: ' + str(momentum))

	# load cifar10 dataset
	dataset_dir = '/home/jiawei/dataset/cifar-10/cifar-10-batches-py'
	train_set, test_set = cifar10.load_data(dataset_dir, False, True)
	X_train, Y_train = train_set
	X_test, Y_test = test_set
	print('Train set shape: ' + str(X_train.shape))
	print('Test set shape: ' + str(X_test.shape))

	with tf.device(gpu):
		# TF input placeholder
		X = tf.placeholder("float", [None, n_input])

		# Weights and biases
		#weights = {'encoder_h1': tf.Variable(tf.zeros([n_input, n_hidden_1], dtype=tf.float32)),
		#'encoder_h2': tf.Variable(tf.zeros([n_hidden_1, n_hidden_2], dtype=tf.float32)),
		#'decoder_h1': tf.Variable(tf.zeros([n_hidden_2, n_hidden_1], dtype=tf.float32)),
		#'decoder_h2': tf.Variable(tf.zeros([n_hidden_1, n_input], dtype=tf.float32)),
		#}

		#biases = {'encoder_b1': tf.Variable(tf.zeros([n_hidden_1], dtype=tf.float32)),
		#'encoder_b2': tf.Variable(tf.zeros([n_hidden_2], dtype=tf.float32)),
		#'decoder_b1': tf.Variable(tf.zeros([n_hidden_1], dtype=tf.float32)),
		#'decoder_b2': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
		#}

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

		# Optimizers
		optimizers = {
			'sgd': tf.train.GradientDescentOptimizer(learning_rate),
			'adadelta': tf.train.AdadeltaOptimizer(learning_rate),
			'adagrad': tf.train.AdagradOptimizer(learning_rate),
			'momentum': tf.train.MomentumOptimizer(learning_rate, momentum = momentum),
			'adam': tf.train.AdamOptimizer(learning_rate),
			'ftrl': tf.train.FtrlOptimizer(learning_rate),
			'rmsp': tf.train.RMSPropOptimizer(learning_rate, momentum = momentum),
		}
		
		# Define loss (the squared error)
		cost = tf.reduce_sum(tf.pow(tf.sub(y_true, y_pred), 2))
		
		# Optimizer algorithm
		optim_alg = optimizers[optimizer_name]
		# Save image
		save_image_name = 'image/cifar10-' + optimizer_name + '.png'
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

	with tf.device(gpu):  
		sess = tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement))
		sess.run(init)

		# Load initial weights and biases from local file
		load_path = saver.restore(sess, model_path)
		print("Model restore from file: %s" % model_path)

		start_time = datetime.datetime.now()
	
		n_samples = X_train.shape[0]
		total_batch = int(n_samples/batch_size)
		# Training cycle
		for epoch in range(training_epochs):
			avg_cost = 0.
			# Loop over all batches
			for i in range(total_batch):
				batch_xs = X_train[i * batch_size : i * batch_size + batch_size]
				# Run optimization op (backprop) and cost op (to get loss value)
				_, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
				# Compute average loss
				avg_cost += c / n_samples * batch_size
			# Display logs per epoch step
			if epoch % display_step == 0:
				print("Epoch:", '%04d' % (epoch+1),
					  "cost=", "{:.9f}".format(avg_cost))
		end_time = datetime.datetime.now()
		print("Optimization Finished!")
		print("Total time of training: %s" % str(end_time - start_time))
		print("Total cost of test set: %f" % sess.run(cost, feed_dict={X: X_test}))

		# Applying encode and decode over test set
		encode_decode = sess.run(y_pred, feed_dict={X: X_test[:examples_to_show]})
		# Compare original images with their reconstructions
		f, a = plt.subplots(2, 10, figsize=(10, 2))
		for i in range(examples_to_show):
			a[0][i].imshow(np.reshape(X_test[i], (32, 32)))
			a[0][i].axis('off')
			a[1][i].imshow(np.reshape(encode_decode[i], (32, 32)))
			a[1][i].axis('off')

		plt.savefig(save_image_name)