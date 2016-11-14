from __future__ import division, print_function, absolute_import

import tensorflow as tf

import sys
sys.path.append("..")
from utils import Utils

model_path = 'init_imagenet.ckpt'

# Input 
# 224 * 224 * 224

conv1 = {
    'weights': tf.Variable(tf.random_normal([11, 11, 3, 96], 
    	mean=0.0, stddev=0.01, dtype=tf.float32)),
    'biases': tf.Variable(tf.zeros([96], dtype=tf.float32))
}

conv2 = {
	'weights': tf.Variable(tf.random_normal([5, 5, 48, 256], 
    	mean=0.0, stddev=0.01, dtype=tf.float32)),
    'biases': tf.Variable(tf.ones([256], dtype=tf.float32))
}

conv3 = {
	'weights': tf.Variable(tf.random_normal([3, 3, 256, 384], 
    	mean=0.0, stddev=0.01, dtype=tf.float32)),
    'biases': tf.Variable(tf.zeros([384], dtype=tf.float32))
}

conv4 = {
	'weights': tf.Variable(tf.random_normal([3, 3, 384, 384], 
    	mean=0.0, stddev=0.01, dtype=tf.float32)),
    'biases': tf.Variable(tf.ones([384], dtype=tf.float32))
}

conv5 = {
	'weights': tf.Variable(tf.random_normal([3, 3, 384, 256], 
    	mean=0.0, stddev=0.01, dtype=tf.float32)),
    'biases': tf.Variable(tf.ones([256], dtype=tf.float32))
}

fc6 = {
	'weights': tf.Variable(tf.random_normal([13*13*256, 4096], 
    	mean=0.0, stddev=0.01, dtype=tf.float32)),
    'biases': tf.Variable(tf.zeros([4096], dtype=tf.float32))
}

fc7 = {
	'weights': tf.Variable(tf.random_normal([4096, 4096], 
    	mean=0.0, stddev=0.01, dtype=tf.float32)),
    'biases': tf.Variable(tf.zeros([4096], dtype=tf.float32))
}

fc8 = {
	'weights': tf.Variable(tf.random_normal([4096, 1024], 
    	mean=0.0, stddev=0.01, dtype=tf.float32)),
    'biases': tf.Variable(tf.zeros([1024], dtype=tf.float32))
}

init = tf.initialize_all_variables()
saver = tf.train.Saver({
	"conv1": conv1,
	"conv2": conv2,
	"conv3": conv3,
	"conv4": conv4,
	"conv5": conv5,
	"fc6": fc6,
	"fc7": fc7,
	"fc8": fc8
	})

with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess, model_path)
    print "Model saved in file: %s" % save_path
    load_path = saver.restore(sess, model_path)
    print "Model restore from file: %s" % model_path
    print (sess.run(fc8['weights']))
    print (sess.run(fc8['biases']))
