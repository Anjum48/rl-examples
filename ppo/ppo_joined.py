"""
A simple version of Proximal Policy Optimization (PPO) using single thread.

Based on:
1. Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]
2. Proximal Policy Optimization Algorithms (OpenAI): [https://arxiv.org/abs/1707.06347]
3. Generalized Advantage Estimation [https://arxiv.org/abs/1506.02438]

Thanks to OpenAI and morvanzhou for their examples
"""

import tensorflow as tf
import numpy as np
import gym
import os
import scipy.signal
from gym import wrappers
from datetime import datetime
from time import time
from Forex.fxsettings import OUTPUT_RESULTS_DIR

# ENVIRONMENT = 'Pendulum-v0'
# ENVIRONMENT = 'MountainCarContinuous-v0'
# ENVIRONMENT = 'LunarLanderContinuous-v2'
# ENVIRONMENT = 'BipedalWalker-v2'
# ENVIRONMENT = 'BipedalWalkerHardcore-v2'
ENVIRONMENT = 'CarRacing-v0'

EP_MAX = 10000
GAMMA = 0.99
LAMBDA = 0.95
ENTROPY_BETA = 0.0
LR = 0.0001
BATCH = 8192
MINIBATCH = 64
EPOCHS = 10
EPSILON = 0.2
VF_COEFF = 1.0
L2_REG = 0.001

TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
SUMMARY_DIR = os.path.join(OUTPUT_RESULTS_DIR, "PPO_Join", ENVIRONMENT, TIMESTAMP)


class PPO(object):
    def __init__(self, environment):
        self.s_dim, self.a_dim = environment.observation_space.shape, environment.action_space.shape[0]
        self.a_bound = environment.action_space.high
        self.cnn = len(self.s_dim) == 3

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        config = tf.ConfigProto(log_device_placement=False, device_count={'GPU': 1})  # 1 for GPU, 0 for CPU
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
        self.sess = tf.Session(config=config)
        self.state = tf.placeholder(tf.float32, [None] + list(self.s_dim), 'state')
        self.actions = tf.placeholder(tf.float32, [None, self.a_dim], 'action')
        self.advantage = tf.placeholder(tf.float32, [None, 1], 'advantage')
        self.rewards = tf.placeholder(tf.float32, [None, 1], 'discounted_r')

        self.dataset = tf.data.Dataset.from_tensor_slices({"state": self.state, "actions": self.actions,
                                                           "rewards": self.rewards, "advantage": self.advantage})
        self.dataset = self.dataset.shuffle(buffer_size=10000)
        self.dataset = self.dataset.batch(MINIBATCH)
        self.dataset = self.dataset.cache()
        self.dataset = self.dataset.repeat(EPOCHS)
        self.iterator = self.dataset.make_initializable_iterator()
        batch = self.iterator.get_next()

        pi_old, vf_old, old_params = self._build_net(batch["state"], 'oldnet')
        pi, self.v, params = self._build_net(batch["state"], 'net')
        pi_eval, self.vf_eval, _ = self._build_net(self.state, 'net', reuse=True)

        self.sample_op = tf.squeeze(pi_eval.sample(1), axis=0, name="sample_action")
        self.eval_action = pi_eval.loc
        self.global_step = tf.train.get_or_create_global_step()

        with tf.variable_scope('loss'):
            with tf.variable_scope('policy'):
                # Use floor functions for the probabilities to prevent NaNs when prob = 0
                ratio = tf.maximum(pi.prob(batch["actions"]), 1e-6) / tf.maximum(pi_old.prob(batch["actions"]), 1e-6)
                ratio = tf.clip_by_value(ratio, 0, 10)
                surr1 = batch["advantage"] * ratio
                surr2 = batch["advantage"] * tf.clip_by_value(ratio, 1 - EPSILON, 1 + EPSILON)
                loss_pi = -tf.reduce_mean(tf.minimum(surr1, surr2))
                tf.summary.scalar("loss", loss_pi)

            with tf.variable_scope('value_function'):
                clipped_value_estimate = vf_old + tf.clip_by_value(self.v - vf_old, -EPSILON, EPSILON)
                loss_vf1 = tf.squared_difference(clipped_value_estimate, batch["rewards"])
                loss_vf2 = tf.squared_difference(self.v, batch["rewards"])
                loss_vf = tf.reduce_mean(tf.maximum(loss_vf1, loss_vf2)) * 0.5
                tf.summary.scalar("loss", loss_vf)

            with tf.variable_scope('entropy_bonus'):
                entropy = pi.entropy()
                pol_entpen = -ENTROPY_BETA * tf.reduce_mean(entropy)

            loss = loss_pi + loss_vf * VF_COEFF + pol_entpen
            tf.summary.scalar("total", loss)

        with tf.variable_scope('train'):
            opt = tf.train.AdamOptimizer(LR)
            self.train_op = opt.minimize(loss, global_step=self.global_step, var_list=params)

            # grads, vs = zip(*opt.compute_gradients(loss, var_list=params))
            # grads, _ = tf.clip_by_global_norm(grads, 10.0)
            # self.train_op = opt.apply_gradients(zip(grads, vs), global_step=self.global_step)

        with tf.variable_scope('update_old'):
            self.update_old_op = [oldp.assign(p) for p, oldp in zip(params, old_params)]

        self.writer = tf.summary.FileWriter(SUMMARY_DIR, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        tf.summary.scalar("Value", tf.reduce_mean(self.v))
        tf.summary.scalar("Sigma", tf.reduce_mean(pi.scale))
        self.summarise = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))

    def update(self, s, a, r, adv):
        start, e_time = time(), []

        self.sess.run([self.update_old_op, self.iterator.initializer],
                      feed_dict={self.state: s, self.actions: a, self.rewards: r, self.advantage: adv})

        while True:
            try:
                e_start = time()
                summary, step, _ = self.sess.run([self.summarise, self.global_step, self.train_op])
                e_time.append(time() - e_start)
            except tf.errors.OutOfRangeError:
                break
        print("Trained in %.3fs. Average %.3fs/batch. Global step %i" % (time() - start, np.mean(e_time), step))
        return summary

    def _build_net(self, state_in, name, reuse=False):
        w_reg = tf.contrib.layers.l2_regularizer(L2_REG)

        with tf.variable_scope(name, reuse=reuse):
            if self.cnn:
                scaled = tf.cast(state_in, tf.float32) / 255.
                conv1 = tf.layers.conv2d(inputs=scaled, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu)
                conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)
                conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)
                state_in = tf.layers.flatten(conv3)

            l1 = tf.layers.dense(state_in, 400, tf.nn.relu, kernel_regularizer=w_reg, name="pi_l1")
            l2 = tf.layers.dense(l1, 400, tf.nn.relu, kernel_regularizer=w_reg, name="pi_l2")
            mu = tf.layers.dense(l2, self.a_dim, tf.nn.tanh, kernel_regularizer=w_reg, name="pi_mu")
            log_sigma = tf.get_variable(name="pi_sigma", shape=self.a_dim, initializer=tf.zeros_initializer())
            norm_dist = tf.distributions.Normal(loc=mu * self.a_bound, scale=tf.maximum(tf.exp(log_sigma), 0.1))
            vf = tf.layers.dense(l2, 1, kernel_regularizer=w_reg, name="vf_output")
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, vf, params

    def eval_state(self, state, stochastic=True):
        if stochastic:
            action, value = self.sess.run([self.sample_op, self.vf_eval], {self.state: state[np.newaxis, :]})
        else:
            action, value = self.sess.run([self.eval_action, self.vf_eval], {self.state: state[np.newaxis, :]})
        return action[0], value[0]


def add_histogram(writer, tag, values, step, bins=1000):
    """
    Logs the histogram of a list/vector of values.
    From: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
    """

    # Create histogram using numpy
    counts, bin_edges = np.histogram(values, bins=bins)

    # Fill fields of histogram proto
    hist = tf.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values ** 2))

    # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
    # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
    # Therefore we drop the start of the first bin
    bin_edges = bin_edges[1:]

    # Add bin edges and counts
    for edge in bin_edges:
        hist.bucket_limit.append(edge)
    for c in counts:
        hist.bucket.append(c)

    # Create and write Summary
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
    writer.add_summary(summary, step)


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


env = gym.make(ENVIRONMENT)
env = wrappers.Monitor(env, os.path.join(SUMMARY_DIR, ENVIRONMENT), video_callable=None)
ppo = PPO(env)

t = 0
buffer_s, buffer_a, buffer_r, buffer_v, terminals = [], [], [], [], []

for episode in range(EP_MAX + 1):
    s = env.reset()
    ep_r, ep_t, terminal = 0, 0, False
    ep_a = []

    while True:
        a, v = ppo.eval_state(s)

        if terminal:
            print('Episode: %i' % episode, "| Reward: %.2f" % ep_r, '| Steps: %i' % ep_t)

            # Update PPO
            if t >= BATCH:
                print("Training using %i steps..." % t)
                v = [v[0] * (1 - terminal)]  # v = 0 if terminal, otherwise use the predicted v
                rewards = np.array(buffer_r)
                vpred = np.array(buffer_v + v)
                terminals_array = np.array(terminals + [0])

                # Generalized Advantage Estimation - https://arxiv.org/abs/1506.02438
                delta = rewards + GAMMA * vpred[1:] * (1 - terminals_array[1:]) - vpred[:-1]
                advantage = discount(delta, GAMMA * LAMBDA)
                td_lambda_returns = advantage + np.array(buffer_v)
                advantage = (advantage - advantage.mean()) / np.maximum(advantage.std(), 1e-6)

                bs, ba, br, badv = np.reshape(buffer_s, (t,) + ppo.s_dim), np.vstack(buffer_a), \
                                   np.vstack(td_lambda_returns), np.vstack(advantage)

                graph_summary = ppo.update(bs, ba, br, badv)
                buffer_s, buffer_a, buffer_r, buffer_v, terminals = [], [], [], [], []
                t = 0

            # End of episode summary
            worker_summary = tf.Summary()
            worker_summary.value.add(tag="Reward", simple_value=ep_r)

            # Create Action histograms for each dimension
            actions = np.array(ep_a)
            for a in range(ppo.a_dim):
                add_histogram(ppo.writer, "Action/Dim" + str(a), actions[:, a], episode)

            try:
                ppo.writer.add_summary(graph_summary, episode)
            except NameError:
                pass
            ppo.writer.add_summary(worker_summary, episode)
            ppo.writer.flush()
            break

        buffer_s.append(s)
        buffer_a.append(a)
        buffer_v.append(v[0])
        terminals.append(terminal)
        ep_a.append(a)

        s, r, terminal, _ = env.step(np.clip(a, -ppo.a_bound, ppo.a_bound))
        buffer_r.append(r)

        ep_r += r
        ep_t += 1
        t += 1

env.close()

env = gym.make(ENVIRONMENT)
while True:
    s = env.reset()
    ep_r, ep_t = 0, 0
    while True:
        env.render()
        a, v = ppo.eval_state(s, stochastic=False)
        s, r, terminal, _ = env.step(np.clip(a, -ppo.a_bound, ppo.a_bound))
        ep_r += r
        ep_t += 1
        if terminal:
            print("Reward: %.2f" % ep_r, '| Steps: %i' % ep_t)
            break
