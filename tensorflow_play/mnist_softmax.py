""" Code from the Tensorflow MNIST for beginners example.

See https://www.tensorflow.org/get_started/mnist/beginners
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

pixels = tf.placeholder(tf.float32, [None, 784])
weights = tf.Variable(tf.zeros([784, 10]))
biases = tf.Variable(tf.zeros([10]))
model = tf.nn.softmax(tf.matmul(pixels, weights) + biases)
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

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()
for _ in range(10000):
    batch_pixels, batch_actuals = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={pixels: batch_pixels, actual: batch_actuals})

correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(actual, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy,
               feed_dict={pixels: mnist.test.images, actual: mnist.test.labels}))
