import gym
import logging
import numpy as np
import os
import random
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
from gym.envs.registration import register

ENV_ACTION_COUNT = 4

register(id='Text2048-v0',
         entry_point='env:Text2048')


class Qnetwork():
    def __init__(self, h_size):
        # The network receives a frame from the game, flattened into an array.
        # It then re-sizes it and processes it through four convolution layers.
        self.scalarInput =  tf.placeholder(shape=[None, 16], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput,shape=[-1, 4, 4])
        self.conv1 = slim.conv2d(inputs=self.imageIn,
                                 num_outputs=64,
                                 kernel_size=2,
                                 stride=1,
                                 padding='SAME',
                                 biases_initializer=None)
        self.conv2 = slim.conv2d(inputs=self.conv1,
                                 num_outputs=64,
                                 kernel_size=3,
                                 stride=1,
                                 padding='VALID',
                                 biases_initializer=None)
        self.conv3 = slim.conv2d(inputs=self.conv2,
                                 num_outputs=h_size,
                                 kernel_size=2,
                                 stride=1,
                                 padding='VALID',
                                 biases_initializer=None)

        # We take the output from the final convolution layer and split it
        # into separate advantage and value streams.
        self.streamAC, self.streamVC = tf.split(self.conv3, 2, 2)
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([h_size//2, ENV_ACTION_COUNT]))
        self.VW = tf.Variable(xavier_init([h_size//2,1]))
        self.Advantage = tf.matmul(self.streamA,self.AW)
        self.Value = tf.matmul(self.streamV,self.VW)

        # Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage,
                                             tf.reduce_mean(self.Advantage,
                                                            axis=1,
                                                            keep_dims=True))
        self.predict = tf.argmax(self.Qout,1)

        # Below we obtain the loss by taking the sum of squares difference
        # between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,
                                         ENV_ACTION_COUNT,
                                         dtype=tf.float32)
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot),
                               axis=1)
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)


class experience_buffer():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self,size):
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,5])


def processState(states):
    return np.reshape(states, [16])


def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign(
            (var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)


if __name__ == '__main__':
    batch_size = 32 #How many experiences to use for each training step.
    update_freq = 4 #How often to perform a training step.
    y = .99 #Discount factor on the target Q-values
    startE = 1 #Starting chance of random action
    endE = 0.1 #Final chance of random action
    annealing_steps = 100000. #How many steps of training to reduce startE to endE.
    num_episodes = 10000 #How many episodes of game environment to train network with.
    pre_train_steps = 10000 #How many steps of random actions before training begins.
    max_epLength = 10000 #The max allowed length of our episode.
    load_model = False #Whether to load a saved model.
    path = "./dqn" #The path to save our model to.
    h_size = 512 #The size of the final convolution layer before splitting it into Advantage and Value streams.
    tau = 0.001 #Rate to update target network toward primary network

    gym.undo_logger_setup()
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # You can set the level to logging.DEBUG or logging.WARN if you
    # want to change the amount of output.
    logger.setLevel(logging.INFO)

    env = gym.make('Text2048-v0')

    tf.reset_default_graph()
    mainQN = Qnetwork(h_size)
    targetQN = Qnetwork(h_size)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    trainables = tf.trainable_variables()

    targetOps = updateTargetGraph(trainables, tau)

    myBuffer = experience_buffer()

    # Set the rate of random action decrease.
    e = startE
    stepDrop = (startE - endE)/annealing_steps

    # create lists to contain total rewards and steps per episode
    jList = []
    rList = []
    total_steps = 0

    # Make a path for our model to be saved in.
    if not os.path.exists(path):
        os.makedirs(path)

    with tf.Session() as sess:
        sess.run(init)
        if load_model:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(path)
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(num_episodes):
            episodeBuffer = experience_buffer()
            #Reset environment and get first new observation
            s = env.reset()
            s = processState(s)
            done = False
            rAll = 0
            j = 0
            #The Q-Network
            while not done:
                #If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
                j+=1
                #Choose an action by greedily (with e chance of random action) from the Q-network
                if np.random.rand(1) < e or total_steps < pre_train_steps:
                    a = np.random.randint(0,4)
                else:
                    a = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:[s]})[0]

                s1, r, done, _ = env.step(a)
                s1 = processState(s1)
                total_steps += 1
                # Save the experience to our episode buffer.
                episodeBuffer.add(np.reshape(np.array([s,a,r,s1,done]),[1,5]))

                if total_steps > pre_train_steps:
                    # if e > endE:
                    #    e -= stepDrop

                    if total_steps % (update_freq) == 0:
                        trainBatch = myBuffer.sample(batch_size) #Get a random batch of experiences.
                        #Below we perform the Double-DQN update to the target Q-values
                        Q1 = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,3])})
                        Q2 = sess.run(targetQN.Qout,feed_dict={targetQN.scalarInput:np.vstack(trainBatch[:,3])})
                        end_multiplier = -(trainBatch[:,4] - 1)
                        doubleQ = Q2[range(batch_size),Q1]
                        targetQ = trainBatch[:,2] + (y*doubleQ * end_multiplier)
                        #Update the network with our target values.
                        _ = sess.run(mainQN.updateModel, \
                                     feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0]),mainQN.targetQ:targetQ, mainQN.actions:trainBatch[:,1]})

                        updateTarget(targetOps,sess) #Update the target network toward the primary network.

                rAll += r
                s = s1

            myBuffer.add(episodeBuffer.buffer)
            jList.append(j)
            rList.append(rAll)
            #Periodically save the model.
            if i % 1000 == 0:
                saver.save(sess,path+'/model-'+str(i)+'.ckpt')
                print("Saved Model")
            if len(rList) % 10 == 0:
                print(len(rList), np.mean(jList[-10:]), np.mean(rList[-10:]), e)
        saver.save(sess,path+'/model-'+str(i)+'.ckpt')
    print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")
