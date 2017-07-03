import better_exceptions
import os
import tensorflow as tf
import gym
from gym import wrappers

from async_agent import *

GAME = 'Breakout-v4'
LOG_DIR = './log-a3c-%s'%GAME
MODEL_NAME = 'model.ckpt-240000' #'last.ckpt' #'model.ckpt-20000'

sample_env = wrappers.Monitor(gym.make(GAME), os.path.join('/tmp/',LOG_DIR),force=True)
nA = sample_env.action_space.n

ac= ActorCritic(nA)
agent= ACAgent(sample_env,
               ac,
               unroll_step=999999)

init_op = tf.global_variables_initializer()
saver = tf.train.Saver(var_list=ac.train_vars,max_to_keep = 3)

sess = tf.Session()
sess.graph.finalize()

sess.run(init_op)
agent.ac.initialize(sess)
saver.restore(sess,os.path.join(LOG_DIR,MODEL_NAME))

print agent.test_run(render=False)
sess.close()
