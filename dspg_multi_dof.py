import tensorflow as tf
import numpy as np


class discrete_spg():
    def __init__(self, initus, exitus):
        self.initus = initus
        self.exitus = exitus

        self.net_in = tf.placeholder(tf.float32, [None, initus], name="Network_Input") #network input
        self.lr = tf.placeholder(tf.float32) #learning rate
        self.disreward = tf.placeholder(tf.float32) #discounted reward

        weight_init = tf.contrib.layers.xavier_initializer()

        fc_1 = tf.contrib.layers.fully_connected(inputs=self.net_in, num_outputs=50, activation_fn=tf.nn.relu, weights_initializer=weight_init)
        fc_2 = tf.contrib.layers.fully_connected(inputs=fc_1, num_outputs=30, activation_fn=tf.nn.relu, weights_initializer=weight_init)
        self.exit0 = tf.contrib.layers.fully_connected(inputs=fc_2, num_outputs=2*self.exitus, activation_fn=lambda i:i, weights_initializer=weight_init)

        self.exit = tf.softmax(tf.reshape(self.exit0,(-1,self.exitus,2)), dim=2)
        #self.actions = tf.layers.max_pooling1d(self.exit, 2, 2, padding='same', name="Actions_selected")
        self.actions = tf.argmax(self.exit, axis=1)

        self.loss = tf.reduce_mean(self.exit)

        def run(self, session, state):
            return session.run(self.actions, feed_dict={self.net_in:state})

        def train(self, session, replay_buffer, learning_rate):
            pass
