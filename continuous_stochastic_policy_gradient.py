import tensorflow as tf
import numpy as np
import random

def make_np_array(orig):
    npar = np.zeros((len(orig),len(orig[0][0])))
    for i,obj in enumerate(orig):
        npar[i] = np.array(obj, dtype=np.float32)
    return npar

class continuous_spg():
    def __init__(self, initus, exitus):
        self.initus = initus
        self.exitus = exitus

        self.net_in = tf.placeholder(tf.float32, [None, initus], name="Network_Input") #network input
        self.lr = tf.placeholder(tf.float32) #learning rate
        self.disreward = tf.placeholder(tf.float32) #discounted reward
        self.sampled_action = tf.placeholder(tf.float32, [None, exitus], name="Sampled_Action")

        weight_init = tf.contrib.layers.xavier_initializer()

        fc_1 = tf.contrib.layers.fully_connected(inputs=self.net_in, num_outputs=64, activation_fn=tf.nn.relu, weights_initializer=weight_init)
        fc_2 = tf.contrib.layers.fully_connected(inputs=fc_1, num_outputs=32, activation_fn=tf.nn.relu, weights_initializer=weight_init)
        self.mu_exit = tf.contrib.layers.fully_connected(inputs=fc_2, num_outputs=self.exitus, activation_fn=tf.nn.tanh, weights_initializer=weight_init)
        self.sigma_exit = tf.contrib.layers.fully_connected(inputs=fc_2, num_outputs=self.exitus, activation_fn=tf.nn.tanh, weights_initializer=weight_init)
        self.mvn = tf.contrib.distributions.MultivariateNormalDiag(self.mu_exit,tf.abs(self.sigma_exit))

        self.action = self.mvn.sample()   #for run function

        self.loss = -self.disreward*tf.log(self.mvn.prob(self.sampled_action)+1e-06)#for train function
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def run(self, session, state):
        return session.run(self.action, feed_dict={self.net_in: state})

    def train(self, session, replay_buffer, batch_size, learning_rate):
        drpb = np.array(replay_buffer[-batch_size:])
        session.run([self.loss, self.opt], feed_dict={self.net_in: make_np_array(drpb[:,0]), self.sampled_action:make_np_array(drpb[:,1]), self.disreward:drpb[:,3], self.lr:learning_rate})
