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
OUTPUT_RESULTS_DIR = "./"

ENVIRONMENT = 'Pendulum-v0'
# ENVIRONMENT = 'MountainCarContinuous-v0'
# ENVIRONMENT = 'LunarLanderContinuous-v2'
# ENVIRONMENT = 'BipedalWalker-v2'
# ENVIRONMENT = 'BipedalWalkerHardcore-v2'

EP_MAX = 10000
GAMMA = 0.99
LAMBDA = 0.95
ENTROPY_BETA = 0.0
LR = 0.0001
BATCH = 4096
MINI_BATCH = 128
EPOCHS = 10
EPSILON = 0.2
L2_REG = 0.01

TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
SUMMARY_DIR = os.path.join(OUTPUT_RESULTS_DIR, "PPO", ENVIRONMENT, TIMESTAMP)


class PPO(object):
    def __init__(self, environment):
        self.s_dim, self.a_dim = environment.observation_space.shape[0], environment.action_space.shape[0]
        self.a_bound = environment.action_space.high

        assert BATCH % MINI_BATCH == 0

        config = tf.ConfigProto(device_count={'GPU': 0})
        self.sess = tf.Session(config=config)
        self.state = tf.placeholder(tf.float32, [None, self.s_dim], 'state')  # TODO Normalise the state
        self.actions = tf.placeholder(tf.float32, [None, self.a_dim], 'action')
        self.advantage = tf.placeholder(tf.float32, [None, 1], 'advantage')
        self.rewards = tf.placeholder(tf.float32, [None, 1], 'discounted_r')

        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)

        self.v, vf_params = self._build_cnet("vf", trainable=True)
        oldvf, oldvf_params = self._build_cnet('oldvf', trainable=False)

        self.sample_op = tf.squeeze(pi.sample(1), axis=0, name="sample_action")

        with tf.variable_scope('update_old_functions'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]
            self.update_oldvf_op = [oldp.assign(p) for p, oldp in zip(vf_params, oldvf_params)]

        with tf.variable_scope('loss'):
            with tf.variable_scope('actor_loss'):
                # self.ratio = pi.prob(self.actions) / tf.maximum(oldpi.prob(self.actions), 1e-8)
                self.ratio = tf.maximum(pi.prob(self.actions), 1e-9) / tf.maximum(oldpi.prob(self.actions), 1e-9)
                self.ratio = tf.clip_by_value(self.ratio, 0, 2)
                surr1 = self.ratio * self.advantage
                surr2 = tf.clip_by_value(self.ratio, 1 - EPSILON, 1 + EPSILON) * self.advantage
                self.aloss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            with tf.variable_scope('critic_loss'):
                self.closs = tf.reduce_mean(tf.square(self.v - self.rewards))

            with tf.variable_scope('entropy_bonus'):
                entropy = pi.entropy()
                self.pol_entpen = -ENTROPY_BETA * tf.reduce_mean(entropy)

            self.loss = self.aloss + self.closs + self.pol_entpen

        with tf.variable_scope('train'):
            self.grads = tf.gradients(self.loss, pi_params + vf_params)
            self.grads, _ = tf.clip_by_global_norm(self.grads, 5.0)
            self.train_op = tf.train.AdamOptimizer(LR).apply_gradients(zip(self.grads, pi_params + vf_params))

            # Separate actor/critic optimisers
            a_opt = tf.train.AdamOptimizer(LR * 0.5)
            c_opt = tf.train.AdamOptimizer(LR)
            a_grads, a_vs = zip(*a_opt.compute_gradients(self.aloss, var_list=pi_params))
            c_grads, c_vs = zip(*c_opt.compute_gradients(self.closs, var_list=vf_params))
            a_grads, _ = tf.clip_by_global_norm(a_grads, 5.0)
            c_grads, _ = tf.clip_by_global_norm(c_grads, 5.0)
            self.atrain_op = a_opt.apply_gradients(zip(a_grads, a_vs))
            self.ctrain_op = c_opt.apply_gradients(zip(c_grads, c_vs))

        self.writer = tf.summary.FileWriter(SUMMARY_DIR, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        tf.summary.scalar("Loss/Actor", self.aloss)
        tf.summary.scalar("Loss/Critic", self.closs)
        tf.summary.scalar("Value", tf.reduce_mean(self.v))
        tf.summary.scalar("Ratio", tf.reduce_mean(self.ratio))
        tf.summary.scalar("LogStd", tf.reduce_mean(pi.scale))
        self.summarise = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))

    def update(self, s, a, r, adv, copy_func=True):
        if copy_func:
            self.sess.run([self.update_oldpi_op, self.update_oldvf_op])

        s_split = np.split(s, BATCH/MINI_BATCH)
        a_split = np.split(a, BATCH/MINI_BATCH)
        r_split = np.split(r, BATCH/MINI_BATCH)
        adv_split = np.split(adv, BATCH/MINI_BATCH)
        indexes = np.arange(BATCH/MINI_BATCH, dtype=int)

        for _ in range(EPOCHS):
            # np.random.shuffle(indexes)  # uncomment this to shuffle the batches
            for i in indexes:
                self.sess.run(self.train_op, {self.state: s_split[i], self.actions: a_split[i],
                                              self.advantage: adv_split[i], self.rewards: r_split[i]})

        return self.sess.run(self.summarise, {self.state: s, self.actions: a, self.advantage: adv, self.rewards: r})

    def _build_anet(self, name, trainable):
        w_reg = tf.contrib.layers.l2_regularizer(L2_REG)

        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.state, 400, tf.nn.relu, trainable=trainable,
                                 kernel_regularizer=w_reg, name="pi_l1")
            l2 = tf.layers.dense(l1, 400, tf.nn.relu, trainable=trainable, kernel_regularizer=w_reg, name="pi_l2")
            mu = tf.layers.dense(l2, self.a_dim, tf.nn.tanh, trainable=trainable,
                                 kernel_regularizer=w_reg, name="pi_mu_out")
            log_sigma = tf.get_variable(name="pi_log_sigma_out", shape=self.a_dim, trainable=trainable,
                                        initializer=tf.zeros_initializer(), regularizer=w_reg)
            norm_dist = tf.distributions.Normal(loc=mu * self.a_bound, scale=tf.exp(log_sigma))
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def _build_cnet(self, name, trainable):
        w_reg = tf.contrib.layers.l2_regularizer(L2_REG)

        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.state, 400, tf.nn.relu, trainable=trainable,
                                 kernel_regularizer=w_reg, name="vf_l1")
            l2 = tf.layers.dense(l1, 400, tf.nn.relu, trainable=trainable, kernel_regularizer=w_reg, name="vf_l2")
            vf = tf.layers.dense(l2, 1, trainable=trainable, kernel_regularizer=w_reg, name="vf_out")
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return vf, params

    def eval_state(self, state):
        action, value = self.sess.run([self.sample_op, self.v], {self.state: state[np.newaxis, :]})
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

for episode in range(EP_MAX):
    s = env.reset()
    ep_r, ep_t, terminal = 0, 0, True
    ep_a = []

    while True:
        a, v = ppo.eval_state(s)

        if ep_t > 0 and (t % BATCH == 0 or terminal):
            v = [v[0] * (1 - terminal)]  # v = 0 if terminal, otherwise use the predicted v
            rewards = np.array(buffer_r)
            vpred = np.array(buffer_v + v)
            terminals_array = np.array(terminals + [0])

            # Generalized Advantage Estimation - https://arxiv.org/abs/1506.02438
            advantage = discount(rewards + GAMMA * vpred[1:] * (1 - terminals_array[1:]) - vpred[:-1], GAMMA * LAMBDA)
            tdlamret = advantage + np.array(buffer_v)
            advantage = (advantage - advantage.mean()) / np.maximum(advantage.std(), 1e-6)

            bs, ba, br, badv = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(tdlamret), np.vstack(advantage)

            # Update PPO
            if t % BATCH == 0:
                print("Training...")
                graph_summary = ppo.update(bs, ba, br, badv)
                buffer_s, buffer_a, buffer_r, buffer_v, terminals = [], [], [], [], []

            # End of episode summary
            if terminal and ep_t > 0:
                worker_summary = tf.Summary()
                worker_summary.value.add(tag="Reward", simple_value=ep_r)

                # Create Action histograms for each dimension
                actions = np.array(ep_a)
                for a in range(ppo.a_dim):
                    add_histogram(ppo.writer, "Action/Dim" + str(a), actions[:, a], episode)

                graph_summary = ppo.sess.run(ppo.summarise, feed_dict={ppo.state: bs, ppo.actions: ba,
                                                                       ppo.advantage: badv, ppo.rewards: br})
                ppo.writer.add_summary(graph_summary, episode)
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

    print('Episode: %i' % episode, "| Reward: %.2f" % ep_r, '| Steps: %i' % ep_t)
env.close()

env = gym.make(ENVIRONMENT)
while True:
    s = env.reset()
    while True:
        env.render()
        a, v = ppo.eval_state(s)
        s, r, terminal, _ = env.step(np.clip(a, -ppo.a_bound, ppo.a_bound))
        if terminal:
            break
