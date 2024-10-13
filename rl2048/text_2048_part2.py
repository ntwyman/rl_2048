#
# Attempt at learning based on part_2 from rlwtf
# Doesn't seem to learn.
# Thoughts:
#     a) Loss function isn't any good
#     b) Simple policy network really isn't enough
#        for this. Check part 3 DQN.

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
from gym.envs.registration import register
# import matplotlib.pyplot as plt

register(id='Text2048-v0',
         entry_point='env:Text2048')

register(id='Graphics2048-v0',
         entry_point='env:Graphics2048')


env = gym.make('Text2048-v0')
gamma = 0.99


def discount_rewards(r):
    """ Take 1d array of float rewards and return discounted rewards """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class agent():
    def __init__(self, lr, s_size, a_size, h_size):
        # These lines establish the feed-forward part of the network.
        # The agent takes a state and produces an action
        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in,
                                      h_size,
                                      biases_initializer=None,
                                      activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden,
                                           a_size,
                                           biases_initializer=None,
                                           activation_fn=tf.nn.softmax)
        self.chosen_action = tf.argmax(self.output, 1)

        # The next six lines establish the training procedure.
        # We feed the reward and chosen action into the networks and compute
        # the loss, and use it to update the network
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
        self.indexes = (tf.range(0, tf.shape(self.output)[0]) *
                        tf.shape(self.output)[1] + self.action_holder)
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]),
                                             self.indexes)
        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) *
                                    self.reward_holder)
        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32,
                                         name=str(idx) + '_holder')
            self.gradient_holders.append(placeholder)
        self.gradients = tf.gradients(self.loss, tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(
            zip(self.gradient_holders, tvars))


tf.reset_default_graph()
myAgent = agent(lr=1e-2, s_size=16, a_size=4, h_size=32)
total_episodes = 5000
max_ep = 10000
update_frequency = 5
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    i = 0
    total_rewards = []
    total_length = []

    gradBuffer = sess.run(tf.trainable_variables())
    for idx, grad in enumerate(gradBuffer):
        gradBuffer[idx] = grad * 0

    while True:
        s = env.reset().flatten()
        running_reward = 0
        ep_history = []
        nudge = False
        for j in range(max_ep):
            # Get feed forward action wieghts
            if nudge:
                a = env.action_space.sample()
            else:
                a_dist = sess.run(myAgent.output, feed_dict={myAgent.state_in: [s]})
                a = np.random.choice(a_dist[0], p=a_dist[0])
                a = np.argmax(a_dist == a)

            s1, r, d, _ = env.step(a)
            s1 = s1.flatten()
            ep_history.append([s, a, r, s1])
            if np.array_equal(s, s1):
                nudge = True
            else:
                s = s1
                nudge = False
            running_reward += r
            if d is True:
                # Update the network
                ep_history = np.array(ep_history)
                ep_history[:, 2] = discount_rewards(ep_history[:, 2])
                feed_dict = {myAgent.reward_holder: ep_history[:, 2],
                             myAgent.action_holder: ep_history[:, 1],
                             myAgent.state_in: np.vstack(ep_history[:, 0])}
                grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
                for idx, grad in enumerate(grads):
                    gradBuffer[idx] += grad

                if (i > 0) and (i % update_frequency == 0):
                    feed_dict = dict(zip(myAgent.gradient_holders, gradBuffer))
                    _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
                    for idx, grad in enumerate(gradBuffer):
                        gradBuffer[idx] = grad * 0

                total_rewards.append(running_reward)
                total_length.append(j)
                break
        if i % 100 == 0:
            print(np.mean(total_rewards[-100:]))
        i += 1
