""" Second example from tensorflow site.

Deep MNIST for experts. """

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()
in_data = tf.placeholder(tf.float32, shape=[None, 784])
expected = tf.placeholder(tf.float32, shape=[None, 10])


in_image = tf.reshape(in_data, [-1, 28, 28, 1])  # shape incoming data as 28 * 28 images with 1 channel.

# First convolution and pooling layer
weight_conv1 = weight_variable([5, 5, 1, 32])
bias_conv1 = bias_variable([32])
convol_layer1 = tf.nn.relu(conv2d(in_image, weight_conv1) + bias_conv1)  # [28, 28, 32]
pool_layer1 = max_pool_2x2(convol_layer1)  # [14, 14, 32]

# second convolution & pooling layer
weight_conv2 = weight_variable([5, 5, 32, 64])
bias_conv2 = bias_variable([64])
convol_layer2 = tf.nn.relu(conv2d(pool_layer1, weight_conv2) + bias_conv2) # [14, 14, 64]
pool_layer2 = max_pool_2x2(convol_layer2) # [7, 7, 64]

# Densely connected layer of 1024
full_weight = weight_variable([7 * 7 * 64, 1024])
full_bias = bias_variable([1024])
pool2_flat = tf.reshape(pool_layer2, [-1, 7*7*64])
full_layer = tf.nn.relu(tf.matmul(pool2_flat, full_weight) + full_bias)

# Dropout to prevent over-fitting
keep_prob = tf.placeholder(tf.float32)
full_layer_drop = tf.nn.dropout(full_layer, keep_prob)


# Finally a softmax output layer
out_weight = weight_variable([1024, 10])
out_bias = bias_variable([10])
output = tf.matmul(full_layer_drop, out_weight) + out_bias


cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=expected, logits=output))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(expected,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={in_data: batch[0], expected: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={in_data: batch[0], expected: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={in_data: mnist.test.images, expected: mnist.test.labels, keep_prob: 1.0}))
