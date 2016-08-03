# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../../../../dataset/", one_hot=True)

import tensorflow as tf

# Network parameters
n_hidden1 = 256
n_hidden2 = 256
n_input = 784
n_classes = 10

model_path = "../data/init_mlp.ckpt"

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Ctrate model
def multilayer_perceptron(x, weight, biases):
        # Hidden layer with Relu activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with Relu activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_2, weight['wout']) + biases['bout']
        return out_layer

# Layer weights and biases
weights = {
	'h1': tf.Variable(tf.random_normal([n_input, n_hidden1])),
	'h2': tf.Variable(tf.random_normal([n_hidden1, n_hidden2])),
	'wout': tf.Variable(tf.random_normal([n_hidden2, n_classes]))
}
biases = {
	'b1': tf.Variable(tf.random_normal([n_hidden1])),
	'b2': tf.Variable(tf.random_normal([n_hidden2])),
	'bout': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.initialize_all_variables()
saver = tf.train.Saver({"h1": weights['h1'], 
	"h2": weights['h2'],
	"wout": weights['wout'],
	"b1": biases['b1'],
	"b2": biases['b2'],
	"bout": biases['bout'] 
	})

with tf.Session() as sess:
	sess.run(init)
	# Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print "Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
	save_path = saver.save(sess, model_path)
	print "Model saved in file: %s" % save_path
