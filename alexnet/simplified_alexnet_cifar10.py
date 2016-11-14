import datetime
import tensorflow as tf

#from tensorflow.models.image.cifar10 import cifar10

import sys
sys.path.append('..')
from datasets import cifar10

# Conv2D wrapper, with bias and relu activation
def conv2d(name, l_input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'),b), name=name)

# MaxPool2D wrapper
def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

# normalization
def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

# define the whole network
def alex_net(_X, _weights, _biases, _dropout):

    # Reshape input picture
    _X = tf.reshape(_X, shape=[-1, 32, 32, 3])

    # Convolution Layer
    conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])
    # Max Pooling (down-sampling)
    pool1 = max_pool('pool1', conv1, k=2)
    # normalization
    norm1 = norm('norm1', pool1, lsize=4)
    # Dropout
    norm1 = tf.nn.dropout(norm1, _dropout)

    conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
    pool2 = max_pool('pool2', conv2, k=2)
    norm2 = norm('norm2', pool2, lsize=4)
    norm2 = tf.nn.dropout(norm2, _dropout)

    conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
    pool3 = max_pool('pool3', conv3, k=2)
    norm3 = norm('norm3', pool3, lsize=4)
    norm3 = tf.nn.dropout(norm3, _dropout)

    # Fully connected layer
    # First reshape conv2 output to fit fully connected layer input
    dense1 = tf.reshape(norm3, [-1, _weights['wd1'].get_shape().as_list()[0]]) 
    dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1') 
    # Another fully connected layer
    dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation

    # Output, class prediction
    out = tf.matmul(dense2, _weights['out']) + _biases['out']
    return out

def grad_to_file(grads, w_grad_file, b_grad_file):
    w_grad = grads[0]
    b_grad = grads[1]
    print w_grad.shape
    print b_grad.shape
    w_grad_f = open(w_grad_file, "a")
    b_grad_f = open(b_grad_file, "a")
    w_grad = w_grad.flatten()
    b_grad = b_grad.flatten()
    for i in range(len(w_grad)):
        w_grad_f.write(str(w_grad[i]) + "\n")
    for i in range(len(b_grad)):
        b_grad_f.write(str(b_grad[i]) + "\n")
    w_grad_f.close()
    b_grad_f.close()

def grad_to_file(grads, grad_file):
    print "Gradient of shape %s to file %s" % (str(grads.shape), grad_file)
    #print grads
    f = open(grad_file, "a")
    grads = grads.flatten()
    for i in range(len(grads)):
        f.write(str(grads[i]) + "\n")
    f.close()

# input data
# load cifar10 dataset
dataset_dir = '/Users/jiangjiawei/Downloads/cifar-10-batches-py'
train_set, test_set = cifar10.load_data(dataset_dir, True, False, True)
X_train, Y_train = train_set
print "Train instances shape: " + str(X_train.shape)
print "Train labels shape: " + str(Y_train.shape)
X_test, Y_test = test_set
print "Test instances shape: " + str(X_train.shape)
print "Test labels shape: " + str(Y_train.shape)

# training hyper-parameter
learning_rate = 0.005
training_epochs = 200
batch_size = 256
display_step = 1

# Network Parameters
n_input = 3072
n_classes = 10
dropout = 0.5

# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 3, 64])),
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    'wd1': tf.Variable(tf.random_normal([4*4*256, 1024])),
    'wd2': tf.Variable(tf.random_normal([1024, 1024])),
    'out': tf.Variable(tf.random_normal([1024, 10]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([128])),
    'bc3': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'bd2': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
# dropout (keep probability)
keep_prob = tf.placeholder(tf.float32)    

# Reshape input picture
X = tf.reshape(x, shape=[-1, 32, 32, 3])

# Convolution Layer
conv1 = conv2d('conv1', X, weights['wc1'], biases['bc1'])
# Max Pooling (down-sampling)
pool1 = max_pool('pool1', conv1, k=2)
# normalization
norm1 = norm('norm1', pool1, lsize=4)
# Dropout
norm1 = tf.nn.dropout(norm1, dropout)

conv2 = conv2d('conv2', norm1, weights['wc2'], biases['bc2'])
pool2 = max_pool('pool2', conv2, k=2)
norm2 = norm('norm2', pool2, lsize=4)
norm2 = tf.nn.dropout(norm2, dropout)

conv3 = conv2d('conv3', norm2, weights['wc3'], biases['bc3'])
pool3 = max_pool('pool3', conv3, k=2)
norm3 = norm('norm3', pool3, lsize=4)
norm3 = tf.nn.dropout(norm3, dropout)

# Fully connected layer
# First reshape conv2 output to fit fully connected layer input
dense1 = tf.reshape(norm3, [-1, weights['wd1'].get_shape().as_list()[0]]) 
dense1 = tf.nn.relu(tf.matmul(dense1, weights['wd1']) + biases['bd1'], name='fc1') 
# Another fully connected layer
dense2 = tf.nn.relu(tf.matmul(dense1, weights['wd2']) + biases['bd2'], name='fc2') # Relu activation

# Output, class prediction
pred = tf.matmul(dense2, weights['out']) + biases['out']

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer_alg = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer = optimizer_alg.minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# number of samples to compute gradients
num_sample_grad = batch_size
# gradient of conv layers
conv1_w_grad = tf.mul(tf.gradients(conv1, [weights['wc1']]), 1.0/num_sample_grad)
conv1_b_grad = tf.mul(tf.gradients(conv1, [biases['bc1']]), 1.0/num_sample_grad)
conv2_w_grad = tf.mul(tf.gradients(conv2, [weights['wc2']]), 1.0/num_sample_grad)
conv2_b_grad = tf.mul(tf.gradients(conv2, [biases['bc2']]), 1.0/num_sample_grad)
conv3_w_grad = tf.mul(tf.gradients(conv3, [weights['wc3']]), 1.0/num_sample_grad)
conv3_b_grad = tf.mul(tf.gradients(conv3, [biases['bc3']]), 1.0/num_sample_grad)
dense1_w_grad = tf.mul(tf.gradients(dense1, [weights['wd1']]), 1.0/num_sample_grad)
dense1_b_grad = tf.mul(tf.gradients(dense1, [biases['bd1']]), 1.0/num_sample_grad)
dense2_w_grad = tf.mul(tf.gradients(dense2, [weights['wd2']]), 1.0/num_sample_grad)
dense2_b_grad = tf.mul(tf.gradients(dense2, [biases['bd2']]), 1.0/num_sample_grad)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    n_samples = X_train.shape[0]
    total_batch = int(n_samples / batch_size)

    # Training cycle
    for epoch in range(training_epochs):
        start_time = datetime.datetime.now()
        avg_cost = 0.
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = X_train[i * batch_size : i * batch_size + batch_size]
            batch_ys = Y_train[i * batch_size : i * batch_size + batch_size]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
            # Compute average loss
            avg_cost += c / n_samples * batch_size
        end_time = datetime.datetime.now()
        print "Epoch " + str(epoch) + ", time= " + str(end_time - start_time)
        # Display logs per epoch step
        if epoch % display_step == 0:
            # Calculate batch loss and accuracy
            #acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            #loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            print "Epoch " + str(epoch) + ", Loss= " + "{:.6f}".format(avg_cost)
            print "Test Accuracy:", sess.run(accuracy, feed_dict={x: X_test, y: Y_test, keep_prob: 1.})
        if epoch % 20 == 0:
            # the output of 1st conv layer
            conv1_w_grad_result = sess.run(conv1_w_grad, feed_dict={x: X_test[:num_sample_grad], y: Y_test[:num_sample_grad], keep_prob: 1.})
            conv1_b_grad_result = sess.run(conv1_b_grad, feed_dict={x: X_test[:num_sample_grad], y: Y_test[:num_sample_grad], keep_prob: 1.})
            grad_to_file(conv1_w_grad_result, "grad_file_cezhang/conv1_w_grad_epoch_" + str(epoch) + ".txt")
            grad_to_file(conv1_b_grad_result, "grad_file_cezhang/conv1_b_grad.txt")

            # the output of 2nd conv layer
            conv2_w_grad_result = sess.run(conv2_w_grad, feed_dict={x: X_test[:num_sample_grad], y: Y_test[:num_sample_grad], keep_prob: 1.})
            conv2_b_grad_result = sess.run(conv2_b_grad, feed_dict={x: X_test[:num_sample_grad], y: Y_test[:num_sample_grad], keep_prob: 1.})
            grad_to_file(conv2_w_grad_result, "grad_file_cezhang/conv2_w_grad_epoch_" + str(epoch) + ".txt")
            grad_to_file(conv2_b_grad_result, "grad_file_cezhang/conv2_b_grad_epoch_" + str(epoch) + ".txt")

            # the output of 3rd conv layer
            conv3_w_grad_result = sess.run(conv3_w_grad, feed_dict={x: X_test[:num_sample_grad], y: Y_test[:num_sample_grad], keep_prob: 1.})
            conv3_b_grad_result = sess.run(conv3_b_grad, feed_dict={x: X_test[:num_sample_grad], y: Y_test[:num_sample_grad], keep_prob: 1.})
            grad_to_file(conv3_w_grad_result, "grad_file_cezhang/conv3_w_grad_epoch_" + str(epoch) + ".txt")
            grad_to_file(conv3_b_grad_result, "grad_file_cezhang/conv3_b_grad_epoch_" + str(epoch) + ".txt")
            
            # the output of 1st dense layer
            dense1_w_grad_result = sess.run(dense1_w_grad, feed_dict={x: X_test[:num_sample_grad], y: Y_test[:num_sample_grad], keep_prob: 1.})
            dense1_b_grad_result = sess.run(dense1_b_grad, feed_dict={x: X_test[:num_sample_grad], y: Y_test[:num_sample_grad], keep_prob: 1.})
            grad_to_file(dense1_w_grad_result, "grad_file_cezhang/dense1_w_grad_epoch_" + str(epoch) + ".txt")
            grad_to_file(dense1_b_grad_result, "grad_file_cezhang/dense1_b_grad_epoch_" + str(epoch) + ".txt")

            # the output of 1st dense layer
            dense2_w_grad_result = sess.run(dense2_w_grad, feed_dict={x: X_test[:num_sample_grad], y: Y_test[:num_sample_grad], keep_prob: 1.})
            dense2_b_grad_result = sess.run(dense2_b_grad, feed_dict={x: X_test[:num_sample_grad], y: Y_test[:num_sample_grad], keep_prob: 1.})
            grad_to_file(dense2_w_grad_result, "grad_file_cezhang/dense2_w_grad_epoch_" + str(epoch) + ".txt")
            grad_to_file(dense2_b_grad_result, "grad_file_cezhang/dense2_b_grad_epoch_" + str(epoch) + ".txt")
     
    print "Optimization Finished!"

    # Calculate accuracy for 256 mnist test images
    # print "Final test Accuracy:", sess.run(accuracy, feed_dict={x: X_test, y: Y_test, keep_prob: 1.})

    # the output of 1st conv layer
    conv1_w_grad_result = sess.run(conv1_w_grad, feed_dict={x: X_test[:num_sample_grad], y: Y_test[:num_sample_grad], keep_prob: 1.})
    conv1_b_grad_result = sess.run(conv1_b_grad, feed_dict={x: X_test[:num_sample_grad], y: Y_test[:num_sample_grad], keep_prob: 1.})
    grad_to_file(conv1_w_grad_result, "grad_file_cezhang/conv1_w_grad.txt")
    grad_to_file(conv1_b_grad_result, "grad_file_cezhang/conv1_b_grad.txt")

    # the output of 2nd conv layer
    conv2_w_grad_result = sess.run(conv2_w_grad, feed_dict={x: X_test[:num_sample_grad], y: Y_test[:num_sample_grad], keep_prob: 1.})
    conv2_b_grad_result = sess.run(conv2_b_grad, feed_dict={x: X_test[:num_sample_grad], y: Y_test[:num_sample_grad], keep_prob: 1.})
    grad_to_file(conv2_w_grad_result, "grad_file_cezhang/conv2_w_grad.txt")
    grad_to_file(conv2_b_grad_result, "grad_file_cezhang/conv2_b_grad.txt")

    # the output of 3rd conv layer
    conv3_w_grad_result = sess.run(conv3_w_grad, feed_dict={x: X_test[:num_sample_grad], y: Y_test[:num_sample_grad], keep_prob: 1.})
    conv3_b_grad_result = sess.run(conv3_b_grad, feed_dict={x: X_test[:num_sample_grad], y: Y_test[:num_sample_grad], keep_prob: 1.})
    grad_to_file(conv3_w_grad_result, "grad_file_cezhang/conv3_w_grad.txt")
    grad_to_file(conv3_b_grad_result, "grad_file_cezhang/conv3_b_grad.txt")