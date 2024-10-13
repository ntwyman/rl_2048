import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

tf.reset_default_graph()

# Set up feed forward version of Q-table
inputs1 = tf.placeholder(shape=[1,16], dtype = tf.float32)
W = tf.Variable(tf.random_uniform([16,4], 0, 0.01))
Qout = tf.matmul(inputs1, W)
predict = tf.argmax(Qout, 1)


# We get the loss by taking sum of squares difference between
# the target and the prediction Q-values.
nextQ = tf.placeholder(shape=[1, 4], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
update_model = trainer.minimize(loss)

# Set learning parameters
y = 0.99
e = 0.1
num_episodes = 2000
jList = []
rList = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        while steps < 99:
            steps += 1
            new_input = np.identity(16)[state:state+1]
            # print(new_input)
            action, allQ = sess.run([predict, Qout],
                                    feed_dict={
                                        inputs1: new_input
                                    })
            # print(action)
            if np.random.rand(1) < e:
                action[0] = env.action_space.sample()
            new_state, reward, done, _ = env.step(action[0])
            # print("s - {}, r - {}, d - {}".format(new_state, reward, done))
            # Get the Q' values by running the new state file through our model
            Q1 = sess.run(Qout, feed_dict=
                          {inputs1: np.identity(16)[new_state: new_state+1]})

            # Now work out max Q1
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0, action[0]] = reward + y * maxQ1

            #Train the network using the target and predicted Q values
            _, W1 = sess.run([update_model, W],
                             feed_dict={inputs1:np.identity(16)[state:state+1],
                                        nextQ:targetQ})
            total_reward += reward
            state = new_state
            if done:
                # reduce the randomness as we train the model
                e = 1./((i/50)+10)
                break
        print("Episode {}, steps {}, reward {}".format(i, steps, total_reward))
        jList.append(steps)
        rList.append(total_reward)
    print("Percent of succesful episodes: {}%" .format(sum(rList)/num_episodes))
