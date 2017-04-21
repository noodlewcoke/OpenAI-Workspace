import gym
import tensorflow as tf
import numpy as np
import continuous_stochastic_policy_gradient as cspg
import discrete_stochastic_policy_gradient as dspg
import random


total_timesteps = 400
episode_number = 10000
max_timestep = 300
batch_size = 1000

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

def discrete_spg_cartpole():
    env = gym.make('CartPole-v0')

    sess = tf.InteractiveSession()
    spg = dspg.discrete_spg(4, 1)
    sess.run(tf.global_variables_initializer())
    replay_buffer = []
    mini_buffer = []
    trained = 0
    for episode in xrange(episode_number):
        observation = env.reset()
        R = 0
        flag = 0
        for t in xrange(total_timesteps):
            #env.render()
            if len(replay_buffer)>=batch_size:
                main_buffer = scramble(replay_buffer)
                spg.train(sess, main_buffer, batch_size, 0.0001)
                trained += 1
                #print trained
            prob, action = spg.run(sess, np.expand_dims(observation, 0))
            old_obs = observation
            #print prob
            observation, reward, done, info = env.step(np.squeeze(action))
            R += reward
            flag +=1
            mini_buffer.append([np.expand_dims(old_obs, 0), np.squeeze(action), np.expand_dims(observation, 0), reward, done])

            if flag==max_timestep:# or t==total_timesteps-1:
                done = 1
            if done:
                replay_buffer += discount(mini_buffer, 0.99)
                mini_buffer = []
                print episode, R, len(replay_buffer)
                break
        #spg.train(sess, main_buffer, 0.001)
    sess.close()


def continuous_spg_bipedalwalker():
    env = gym.make('BipedalWalker-v2')

    sess = tf.InteractiveSession()
    spg = cspg.continuous_spg(24, 4)
    sess.run(tf.global_variables_initializer())
    replay_buffer = []
    for episode in xrange(episode_number):
        observation = env.reset()
        R = 0
        flag = 0
        for t in xrange(total_timesteps):
            #env.render()
            if len(replay_buffer)>=batch_size:
                spg.train(sess, replay_buffer, batch_size, 0.01)
            action = spg.run(sess, np.expand_dims(observation, 0))
            old_obs = observation
            observation, reward, done, info = env.step(np.squeeze(action))
            R += reward
            flag += 1
            replay_buffer.append([np.expand_dims(old_obs, 0), np.squeeze(action), np.expand_dims(observation, 0), reward, done])
            if flag==max_timestep:# or t==total_timesteps-1:
                done=1
            if done:
                print episode, R
                break

        #spg.train(sess, main_buffer, 0.01)
    sess.close()


if __name__ == "__main__":
    #continuous_spg_bipedalwalker()
    discrete_spg_cartpole()
