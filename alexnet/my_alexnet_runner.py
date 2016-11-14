from __future__ import division, print_function, absolute_import

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tflearn
from tflearn.layers.core import input_data
import tflearn.datasets.oxflower17 as oxflower17

import datetime

from .my_alexnet import MyAlexNet
# Import sub-module
import sys
sys.path.append('..')
from network import Network
from datasets import cifar10

# Input data
n_input = 28
n_class = 10

# Initial model
model_path = "init_alexnet.ckpt"

# Train parameters
optimizer_name = 'sgd' # default optimization algorithm
learning_rate = 0.001
momentum = 0.9
training_epochs = 100
batch_size = 256
display_step = 1

# Device config
gpu = '/gpu:0'
cpu = '/cpu:0'
log_device_placement = True
soft_placement = True


if __name__ == '__main__':
	print('====================================')
	print('Train AlexNet on mnist dataset')
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

	# Load ox flower dataset
	# dataset_dir = '/home/jiawei/dataset/17flowers/'
	# X, Y = oxflower17.load_data(dirname='/home/jiawei/dataset/17flowers/', one_hot=True, resize_pics=(227, 227))
	# Load mnist dataset
	dataset_dir = '/home/jiawei/dataset/mnist'
	mnist = input_data.read_data_sets(dataset_dir, one_hot=True)
	# Load cifar10 dataset
	# dataset_dir = '/home/jiawei/dataset/cifar-10/cifar-10-batches-py'
	# train_set, test_set = cifar10.load_data(dataset_dir, False, False, True)

	images = tf.placeholder(tf.float32, [batch_size, n_input, n_input, 1])
	labels = tf.placeholder(tf.float32, [batch_size, n_class])
	net = MyAlexNet({'data': images})

	pred = net.layers['fc8']
	# pred = tf.nn.softmax(fc8)

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, labels), 0)
	
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

	# Optimizer algorithm
	optim_alg = optimizers[optimizer_name]
	optimizer = optim_alg.minimize(loss)

	# Evaluate model
	correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(labels,1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	# Initializing the variables
	init = tf.initialize_all_variables()

	with tf.device(gpu):
		sess = tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement,
			allow_soft_placement=soft_placement))

	    sess.run(init)
	    net.load(model_path, sess)

	    start_time = datetime.datetime.now()
    
    	n_samples = int(mnist.train.num_examples)
    	total_batch = int(n_samples / batch_size)
    	# Training cycle
	    for epoch in range(training_epochs):
	        avg_cost = 0.
	        # Loop over all batches
	        for i in range(total_batch):
	            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
	            # Run optimization op (backprop) and cost op (to get loss value)
	            loss, acc, _ = sess.run([loss, accuracy, optimizer], 
	            	feed_dict={images: batch_xs, labels: batch_ys})
	            # Compute average loss
	            avg_cost += loss / n_samples * batch_size
	        # Display logs per epoch step
	        if epoch % display_step == 0:
	            print("Epoch:" + "{:4d}".format(epoch+1) + \
	                  "cost=" + "{:.9f}".format(avg_cost))

	    end_time = datetime.datetime.now()
	    print("Optimization Finished!")
	    print("Total time of training: %s" % str(end_time - start_time))

	    # Calculate loss and accuracy for mnist test images
	    print("Test set------cost: %f, accuracy: %f" % sess.run([loss, accuracy], 
	    	feed_dict={images: mnist.test.images, labels: mnist.test.labels}))

