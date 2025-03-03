import tensorflow as tf
import numpy as np

bandits = [0.2, 0, -0.2, -5]

num_bandits = len(bandits)

def pullBandit (bandit):
    result = np.random.randn(1)
    if result > bandit:
        return 1
    return -1

tf.reset_default_graph()

# These two lines established the feed-forward part of the network.
# This does the actual choosing.
weights = tf.Variable(tf.ones([num_bandits]))
chosen_action = tf.argmax(weights,0)

# The next six lines establish the training procedure.
# We feed the reward and chosen action into the network
# to compute the loss, and use it to update the network.
reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
action_holder = tf.placeholder(shape=[1],dtype=tf.int32)
responsible_weight = tf.slice(weights,action_holder,[1])
loss = -(tf.log(responsible_weight)*reward_holder)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
update = optimizer.minimize(loss)


total_episodes = 1000 #Set total number of episodes to train agent on.
total_reward = np.zeros(num_bandits) #Set scoreboard for bandits to 0.
e = 0.1 #Set the chance of taking a random action.

init = tf.initialize_all_variables()

# Launch the tensorflow graph
with tf.Session() as sess:
    sess.run(init)
    i = 0
    while i < total_episodes:
        #Choose either a random action or one from our network.
        if np.random.rand(1) < e:
            action = np.random.randint(num_bandits)
        else:
            action = sess.run(chosen_action)
        reward = pullBandit(bandits[action]) #Get our reward from picking one of the bandits.

        #Update the network.
        _,resp,ww = sess.run([update,responsible_weight,weights],
                             feed_dict={reward_holder:[reward],
                                        action_holder:[action]})

        #Update our running tally of scores.
        total_reward[action] += reward
        if i % 50 == 0:
            w = sess.run(weights)
            print("Running reward for the {} bandits: {} W - {}".format(num_bandits,
                                                                        np.rint(100 * total_reward/i),
                                                                        w))
        i+=1
print("The agent thinks bandit {} is the most promising....".format(
    np.argmax(ww)+1))
if np.argmax(ww) == np.argmax(-np.array(bandits)):
      print("...and it was right!")
else:
      print("...and it was wrong!")
3
