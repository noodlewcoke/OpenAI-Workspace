import tensorflow as tf
import numpy as np
import random


# REPLAY BUFFER -> (s, a, s', r, d)
def discount(replay_buffer, gamma):
    disrb = replay_buffer[:]
    for i,rb in enumerate(disrb):
        for j,drb in enumerate(disrb[(i):]):
            rb[3] += drb[3]*gamma**(j) if not j==0 else 0
            if drb[4]:
                break
    return disrb

def v0_reward(replay_buffer):
    disrb = replay_buffer[:]
    for i,rb in enumerate(disrb):
        r0 = rb[3]
        for j,drb in enumerate(disrb[(i):]):
            rb[3] += drb[3] if not j==0 else 0
            drb[3] += r0 if j==1 else 0
            if drb[4]:
                break
    return disrb

def scramble(orig):
    scrambled = orig[:]
    random.shuffle(scrambled)
    return scrambled

class continuous_spg():
    def __init__(self, initus, exitus):
        self.initus = initus
        self.exitus = exitus

        self.net_in = tf.placeholder(tf.float32, [None, initus], name="Network_Input") #network input
        self.lr = tf.placeholder(tf.float32) #learning rate
        self.disreward = tf.placeholder(tf.float32) #discounted reward
        self.sampled_action = tf.placeholder(tf.float32, [exitus], name="Sampled_Action")

        weight_init = tf.contrib.layers.xavier_initializer()

        fc_1 = tf.contrib.layers.fully_connected(inputs=self.net_in, num_outputs=50, activation_fn=tf.nn.relu, weights_initializer=weight_init)
        fc_2 = tf.contrib.layers.fully_connected(inputs=fc_1, num_outputs=30, activation_fn=tf.nn.relu, weights_initializer=weight_init)
        self.exit = tf.contrib.layers.fully_connected(inputs=fc_2, num_outputs=2*self.exitus, activation_fn=tf.nn.tanh, weights_initializer=weight_init)

        self.mvn = tf.contrib.distributions.MultivariateNormalDiag(self.exit[-1, :self.exitus],tf.abs(self.exit[-1,self.exitus:]))

        self.action = self.mvn.sample()   #for run function

        self.loss = -tf.reduce_mean(self.disreward*tf.log(self.mvn.prob(self.sampled_action)+1e-06))#for train function
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def run(self, session, state):
            return session.run(self.action, feed_dict={self.net_in: state})

    def train(self, session, replay_buffer, batch_size, learning_rate):
        rpb = discount(replay_buffer, 0.99)
        drpb = scramble(rpb)
        for rb in drpb[:batch_size]:
            session.run([self.loss, self.opt], feed_dict={self.net_in: rb[0], self.sampled_action:rb[1], self.disreward:rb[3], self.lr:learning_rate})
