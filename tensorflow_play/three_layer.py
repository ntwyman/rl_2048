""" Modification of mnist_softmax to add extra layer of 15
neurons per http://neuralnetworksanddeeplearning.com/chap1.html
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

pixels = tf.placeholder(tf.float32, [None, 784])
input_weights = weight_variable([784, 15])
input_biases = bias_variable([15])

layer_two = tf.sigmoid(tf.matmul(pixels, input_weights) + input_biases)

layer_weights = weight_variable([15, 10])
layer_biases = bias_variable([10])

model = tf.nn.softmax(tf.matmul(layer_two, layer_weights) + layer_biases)
actual = tf.placeholder(tf.float32, [None, 10])


# The raw formulation of cross-entropy,
#
#
# cross_entropy = tf.reduce_mean(
#       -tf.reduce_sum(actual * tf.log(model), reduction_indices=[1]))
#
# can be numerically unstable.
#
# So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
# outputs of 'model', and then average across the batch

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=actual, logits=model))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(actual, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()
for _ in range(100):
    print("Running 1000 batches")
    for _ in range(1000):
        batch_pixels, batch_actuals = mnist.train.next_batch(100)
        sess.run(train_step,
                 feed_dict={pixels: batch_pixels, actual: batch_actuals})
    print(sess.run(accuracy,
                   feed_dict={pixels: mnist.test.images,
                              actual: mnist.test.labels}))
