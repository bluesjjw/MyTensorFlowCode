# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../mnist", one_hot=True)

import tensorflow as tf
import sys
sys.path.append("..")
from utils import Utils

# Network Parameters
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 128 # 2nd layer num features
n_input = 784 # MNIST data input (img shape: 28*28)

model_path = "init_mlp.ckpt"

# tf Graph input


# Ctrate model


weights = {
    'encoder_h1': tf.Variable(Utils.xavier_init(n_input, n_hidden_1)),
    'encoder_h2': tf.Variable(Utils.xavier_init(n_hidden_1, n_hidden_2)),
    'decoder_h1': tf.Variable(Utils.xavier_init(n_hidden_2, n_hidden_1)),
    'decoder_h2': tf.Variable(Utils.xavier_init(n_hidden_1, n_input)),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}

# Construct model

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Define loss and optimizer


init = tf.initialize_all_variables()
saver = tf.train.Saver({"encoder_h1": weights['encoder_h1'], 
	"encoder_h2": weights['encoder_h2'],
	"decoder_h1": weights['decoder_h1'],
	"decoder_h2": weights['decoder_h2'],
	"encoder_b1": biases['encoder_b1'],
	"encoder_b2": biases['encoder_b2'],
	"decoder_b1": biases['decoder_b1'],
	"decoder_b2": biases['decoder_b2']
	})

with tf.Session() as sess:
	sess.run(init)
	save_path = saver.save(sess, model_path)
	print "Model saved in file: %s" % save_path
