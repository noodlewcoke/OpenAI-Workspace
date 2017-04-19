import tensorflow as tf
import numpy as np
import random

def scramble(orig):
    scrambled = orig[:]
    random.shuffle(scrambled)
    return scrambled

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

def make_np_array(orig):
    npar = np.zeros((len(orig),4))
    for i,obj in enumerate(orig):
        npar[i] = np.array(obj, dtype=np.float32)
    return npar


class discrete_spg():
    def __init__(self, initus, exitus):
        self.initus = initus
        self.exitus = exitus

        self.net_in = tf.placeholder(tf.float32, [None, initus], name="Network_Input") #network input
        self.lr = tf.placeholder(tf.float32) #learning rate
        self.disreward = tf.placeholder(tf.float32, [None]) #discounted reward
        self.sampled_action = tf.placeholder(tf.int32, [None])

        weight_init = tf.contrib.layers.xavier_initializer()

        fc_1 = tf.contrib.layers.fully_connected(inputs=self.net_in, num_outputs=50, activation_fn=tf.nn.relu, weights_initializer=weight_init)
        fc_2 = tf.contrib.layers.fully_connected(inputs=fc_1, num_outputs=30, activation_fn=tf.nn.relu, weights_initializer=weight_init)
        self.exit = tf.contrib.layers.fully_connected(inputs=fc_2, num_outputs=2*self.exitus, activation_fn=None, weights_initializer=weight_init)

        self.action = tf.argmax(self.exit, axis=1)

        self.loss = -self.disreward*tf.log(self.exit[-1,self.sampled_action[-1]]+1e-06)
        #self.loss = self.disreward*tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.exit, labels=self.sampled_action)
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def run(self, session, state):
        return session.run([self.exit,self.action], feed_dict={self.net_in:state})

    def train(self, session, replay_buffer, batch_size, learning_rate):
        rpb = discount(replay_buffer, 0.99)
        #rpb = v0_reward(replay_buffer)
        result=[]
        drpb = np.array(scramble(rpb)[:batch_size])
        #for rb in drpb[:batch_size]:
        #session.run([self.loss,self.opt], feed_dict={self.net_in: rb[0], self.sampled_action:rb[1], self.disreward:rb[3], self.lr:learning_rate})[0]
        session.run([self.loss,self.opt], feed_dict={self.net_in: make_np_array(drpb[:,0]), self.sampled_action:drpb[:,1], self.disreward:drpb[:,3], self.lr:learning_rate})[0]
        #print min(result)
