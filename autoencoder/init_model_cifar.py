import tensorflow as tf
import sys
sys.path.append("..")
from utils import Utils

# Network Parameters
n_hidden_1 = 1024 # 1st layer num features
n_hidden_2 = 512 # 2nd layer num features
n_input = 3072 # cifar10 data input (img shape: 32*32)

model_path = "init_cifar.ckpt"

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
    load_path = saver.restore(sess, model_path)
    print "Model restore from file: %s" % load_path
    print (sess.run(weights['encoder_h1']))
    print (sess.run(biases['encoder_b1']))
