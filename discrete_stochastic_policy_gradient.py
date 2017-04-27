import tensorflow as tf
import numpy as np
import random

def make_np_array(orig):
    npar = np.zeros((len(orig),len(orig[0][0])))
    for i,obj in enumerate(orig):
        npar[i] = np.array(obj, dtype=np.float32)
    return npar

def index_mat(indices, batch_size, exitus):
    dummy = np.zeros((batch_size,exitus), dtype=np.int32)
    for i in range(batch_size):
        dummy[i] = np.array([i, indices[i]])
    return dummy

class discrete_spg():
    def __init__(self, initus, exitus):
        self.initus = initus
        self.exitus = exitus

        self.net_in = tf.placeholder(tf.float32, [None, initus], name="Network_Input") #network input
        self.lr = tf.placeholder(tf.float32) #learning rate
        self.disreward = tf.placeholder(tf.float32, [None]) #discounted reward
        self.sampled_action = tf.placeholder(tf.int32, [None,2*exitus])
        self.sampled_action_ce = tf.placeholder(tf.int32, [None])

        weight_init = tf.contrib.layers.xavier_initializer()

        fc_1 = tf.contrib.layers.fully_connected(inputs=self.net_in, num_outputs=32, activation_fn=tf.nn.relu, weights_initializer=weight_init)
        fc_2 = tf.contrib.layers.fully_connected(inputs=fc_1, num_outputs=32, activation_fn=tf.nn.relu, weights_initializer=weight_init)

        #"manual" implementation
        self.exit = tf.contrib.layers.fully_connected(inputs=fc_2, num_outputs=2*self.exitus, activation_fn=tf.nn.softmax, weights_initializer=weight_init)
        self.loss  = -self.disreward*tf.log(tf.gather_nd(self.exit, self.sampled_action))

        # #softmax cross-entropy method
        # self.exit = tf.contrib.layers.fully_connected(inputs=fc_2, num_outputs=2*self.exitus, activation_fn=None, weights_initializer=weight_init)
        # self.loss = -self.disreward*tf.log(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.exit, labels=self.sampled_action_ce))


        self.action = tf.argmax(self.exit, axis=1)
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def run(self, session, state):
        return session.run([self.exit,self.action], feed_dict={self.net_in:state})

    def train(self, session, replay_buffer, batch_size, learning_rate):
        drpb = np.array(replay_buffer[-(batch_size):])

        #"manual" implementation
        sam_act = index_mat(drpb[:,1], batch_size, 2*self.exitus)
        session.run([self.loss,self.opt], feed_dict={self.net_in: make_np_array(drpb[:,0]), self.sampled_action:sam_act, self.disreward:drpb[:,3], self.lr:learning_rate})[0]

        # #softmax cross-entropy method
        # session.run([self.loss,self.opt], feed_dict={self.net_in: make_np_array(drpb[:,0]), self.sampled_action_ce:drpb[:,1], self.disreward:drpb[:,3], self.lr:learning_rate})[0]
