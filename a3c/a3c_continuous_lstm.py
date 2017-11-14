"""
Asynchronous Advantage Actor Critic, A3C + RNN in continuous action space (https://arxiv.org/abs/1602.01783)
with Generalized Advantage Estimation, GAE (https://arxiv.org/abs/1506.02438)
Actor and Critic share similarities with the DDPG architecture (https://arxiv.org/abs/1509.02971)

Special thanks to the following GitHub users for their blogs & examples on A3C:
Morvan Zhou (morvanzhou), Arthur Juliani (awjuliani), Andrew Liou (andrewliao11), Jaromír (jaara),
Denny Britz (dennybritz), Corey Lynch (coreylynch), NVlabs, OpenAI
"""

import multiprocessing
import threading
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
# ENVIRONMENT = 'BipedalWalker-v2'
# ENVIRONMENT = 'BipedalWalkerHardcore-v2'
# ENVIRONMENT = 'LunarLanderContinuous-v2'
RENDER = False
RESTORE_DATE = None

if RESTORE_DATE is not None:
    SUMMARY_DIR = os.path.join(OUTPUT_RESULTS_DIR, 'A3C_LSTM', "gym", ENVIRONMENT, RESTORE_DATE)
else:
    TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
    SUMMARY_DIR = os.path.join(OUTPUT_RESULTS_DIR, "A3C_LSTM", "gym", ENVIRONMENT, TIMESTAMP)

N_WORKERS = multiprocessing.cpu_count() * 2
# N_WORKERS = 1
MAX_GLOBAL_EP = 10000 * N_WORKERS
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 5  # 5 for Pendulum
GAMMA = 0.99  # Paper uses 0.99
LAMBDA = 1.00  # 1.00 for Pendulum
ENTROPY_BETA = 0.0001  # Paper uses 0.0001
LR_A = 0.0001  # learning rate for actor - 0.000001 for Pendulum
LR_C = 0.0001  # learning rate for critic - 0.0001 for Pendulum
CELL_SIZE = 128
GLOBAL_EP = 0
RANDOM_SEED = 12345

env = gym.make(ENVIRONMENT)
N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]
with tf.name_scope('env_bounds'):
    S_UPPER = tf.Variable(env.observation_space.high, dtype=tf.float32, name="state_upper")
    S_LOWER = tf.Variable(env.observation_space.low, dtype=tf.float32, name="state_lower")
    A_UPPER = tf.Variable(env.action_space.high, dtype=tf.float32, name="action_upper")
    A_LOWER = tf.Variable(env.action_space.low, dtype=tf.float32, name="action_lower")


class ACNet(object):
    def __init__(self, scope, global_net=None):

        if scope == GLOBAL_NET_SCOPE:  # Create global network. This isn't used for prediction, only parameter updates
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S_global')
                self.is_training = False
                self._build_net()
                self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        else:   # Create local net and loss ops
            with tf.variable_scope(scope):
                with tf.name_scope('input_norm'):
                    self.s = 2 * (tf.placeholder(tf.float32, [None, N_S], 'S') - S_LOWER) / (S_UPPER - S_LOWER) - 1
                self.a_history = tf.placeholder(tf.float32, [None, N_A], 'A_history')
                self.advantage = tf.placeholder(tf.float32, [None, 1], 'Advantage')
                self.R_discounted = tf.placeholder(tf.float32, [None, 1], 'R_discounted')
                self.is_training = tf.placeholder(bool)

                mu, sigma, self.v = self._build_net()

                with tf.name_scope('action_prep'):
                    with tf.name_scope('wrap_a_out'):
                        mu, sigma = mu * A_UPPER, sigma + 1e-4

                    normal_dist = tf.contrib.distributions.Normal(mu, sigma)

                    with tf.name_scope('choose_a'):  # use local params to choose action
                        self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1, seed=RANDOM_SEED), axis=0),
                                                  A_LOWER, A_UPPER, name="a_chosen")

                with tf.name_scope('loss'):
                    with tf.name_scope('a_loss'):
                        self.log_prob = normal_dist.log_prob(self.a_history)
                        self.entropy = tf.reduce_sum(normal_dist.entropy())
                        self.a_loss = -tf.reduce_sum(self.log_prob * self.advantage) - self.entropy * ENTROPY_BETA
                    with tf.name_scope('c_loss'):
                        self.c_loss = 0.5 * tf.reduce_sum(tf.square(self.R_discounted - self.v))

                with tf.name_scope('local_grad'):
                    self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                    self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
                    a_grads = tf.gradients(self.a_loss, self.a_params)
                    c_grads = tf.gradients(self.c_loss, self.c_params)
                    # Gradient clipping
                    self.a_grads, _ = tf.clip_by_global_norm(a_grads, 10.0)
                    self.c_grads, _ = tf.clip_by_global_norm(c_grads, 10.0)

                    for grad, var in list(zip(self.a_grads, self.a_params)):
                        tf.summary.histogram(var.name + '/gradient', grad)
                    for grad, var in list(zip(self.c_grads, self.c_params)):
                        tf.summary.histogram(var.name + '/gradient', grad)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, global_net.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, global_net.c_params)]
                with tf.name_scope('push'):
                    # Use a globally shared pair of Adam optimisers
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, global_net.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, global_net.c_params))

                    # Each worker has an independent set of Adam optimiser parameters
                    # opt_a = tf.train.AdamOptimizer(LR_A, name='AdamA')
                    # opt_b = tf.train.AdamOptimizer(LR_C, name='AdamC')
                    # self.update_a_op = opt_a.apply_gradients(zip(self.a_grads, global_net.a_params))
                    # self.update_c_op = opt_b.apply_gradients(zip(self.c_grads, global_net.c_params))

            tf.summary.scalar("Actor/Loss/" + scope, self.a_loss)
            tf.summary.scalar("Actor/Advantage/" + scope, tf.reduce_sum(self.advantage))
            tf.summary.scalar("Actor/Entropy/" + scope, tf.reduce_sum(self.entropy * ENTROPY_BETA))
            tf.summary.scalar("Critic/Loss/" + scope, self.c_loss)
            tf.summary.scalar("Critic/Value/" + scope, tf.reduce_sum(self.v))
            tf.summary.scalar("Critic/Discounted_Reward/" + scope, tf.reduce_sum(self.R_discounted))
            summary_list = [s for s in tf.get_collection(tf.GraphKeys.SUMMARIES) if scope in s.name]
            self.summarise = tf.summary.merge(summary_list)

    def _build_net(self):
        w_init = tf.contrib.layers.variance_scaling_initializer(seed=RANDOM_SEED)
        w_init_final = tf.random_uniform_initializer(minval=-0.003, maxval=0.003, seed=RANDOM_SEED)
        w_reg = tf.contrib.layers.l2_regularizer(0.01)

        # Both actor and critic are separate networks with 2 dense layers as per DDPG. LSTMs added on final layer
        with tf.variable_scope('actor'):
            a_hidden1 = tf.layers.dense(self.s, 400, tf.nn.relu6, kernel_initializer=w_init,
                                        kernel_regularizer=w_reg, name='hidden_1')
            # a_hidden1 = tf.layers.batch_normalization(a_hidden1, training=self.is_training)
            # a_hidden1 = tf.layers.dropout(a_hidden1, training=self.is_training)
            a_hidden2 = tf.layers.dense(a_hidden1, 300, tf.nn.relu6, kernel_initializer=w_init,
                                        kernel_regularizer=w_reg, name='hidden_2')
            # a_hidden2 = tf.layers.batch_normalization(a_hidden2, training=self.is_training)
            # a_hidden2 = tf.layers.dropout(a_hidden2, training=self.is_training)
            # [time_step, feature] => [time_step, batch, feature]
            a_cell_in = tf.expand_dims(a_hidden2, axis=1, name='timely_input')
            a_rnn_cell = tf.contrib.rnn.BasicRNNCell(CELL_SIZE)
            self.a_init_state = a_rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
            a_outputs, self.a_final_state = tf.nn.dynamic_rnn(
                cell=a_rnn_cell, inputs=a_cell_in, initial_state=self.a_init_state, time_major=True)
            a_cell_out = tf.reshape(a_outputs, [-1, CELL_SIZE], name='flatten_rnn_outputs')
            # mu = tf.layers.dense(a_cell_out, N_A, tf.nn.tanh, kernel_initializer=w_init_final,
            #                      kernel_regularizer=w_reg, name='mu')
            mu = tf.layers.dense(a_cell_out, N_A, kernel_initializer=w_init_final, kernel_regularizer=w_reg,
                                 use_bias=False, name='mu')
            sigma = tf.layers.dense(a_cell_out, N_A, tf.nn.softplus, kernel_initializer=w_init_final,
                                    kernel_regularizer=w_reg, name='sigma')

        with tf.variable_scope('critic'):
            c_hidden1 = tf.layers.dense(self.s, 400, tf.nn.relu6, kernel_initializer=w_init,
                                        kernel_regularizer=w_reg, name='hidden_1')
            # c_hidden1 = tf.layers.batch_normalization(c_hidden1, training=self.is_training)
            # c_hidden1 = tf.layers.dropout(c_hidden1, training=self.is_training)
            c_hidden2 = tf.layers.dense(c_hidden1, 300, tf.nn.relu6, kernel_initializer=w_init,
                                        kernel_regularizer=w_reg, name='hidden_2')
            # c_hidden2 = tf.layers.batch_normalization(c_hidden2, training=self.is_training)
            # c_hidden2 = tf.layers.dropout(c_hidden2, training=self.is_training)
            # [time_step, feature] => [time_step, batch, feature]
            c_cell_in = tf.expand_dims(c_hidden2, axis=1, name='timely_input')
            c_rnn_cell = tf.contrib.rnn.BasicRNNCell(CELL_SIZE)
            self.c_init_state = c_rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
            c_outputs, self.c_final_state = tf.nn.dynamic_rnn(
                cell=c_rnn_cell, inputs=c_cell_in, initial_state=self.c_init_state, time_major=True)
            c_cell_out = tf.reshape(c_outputs, [-1, CELL_SIZE], name='flatten_rnn_outputs')
            v = tf.layers.dense(c_cell_out, 1, kernel_initializer=w_init, name='v')  # state value
        return mu, sigma, v

    def update_global(self, feed_dict):
        return SESS.run([self.update_a_op, self.update_c_op, self.summarise], feed_dict)[2]

    def pull_global(self):
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def eval_state(self, s, a_state, c_state):
        a, v, a_cell_state, c_cell_state = SESS.run([self.A, self.v, self.a_final_state, self.c_final_state],
                                                    {self.s: s[np.newaxis, :], self.a_init_state: a_state,
                                                     self.c_init_state: c_state, self.is_training: False})
        return a[0], v[0], a_cell_state, c_cell_state

    def get_value(self, s, cell_state):
        return SESS.run(self.v, {self.s: s[np.newaxis, :], self.c_init_state: cell_state, self.is_training: False})[0]


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
    # Thus, we drop the start of the first bin
    bin_edges = bin_edges[1:]

    # Add bin edges and counts
    for edge in bin_edges:
        hist.bucket_limit.append(edge)
    for c in counts:
        hist.bucket.append(c)

    # Create and write Summary
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
    writer.add_summary(summary, step)


class Worker(object):
    def __init__(self, name, global_net):
        self.env = gym.make(ENVIRONMENT)
        self.name = name
        self.ep_count = 0
        self.AC = ACNet(name, global_net)

        # if self.name == 'Worker_0':
        #     self.env = wrappers.Monitor(self.env, os.path.join(SUMMARY_DIR, ENVIRONMENT+'_'+self.name))

    @staticmethod
    def discount(x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    def work(self):
        global GLOBAL_EP
        buffer_s, buffer_a, buffer_r, buffer_v = [], [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            self.env.seed(RANDOM_SEED + GLOBAL_EP)
            s = self.env.reset()
            ep_r, ep_t = 0, 0
            ep_a, ep_v = [], []
            a_rnn_state = SESS.run(self.AC.a_init_state)  # Zero RNN state at beginning
            c_rnn_state = SESS.run(self.AC.c_init_state)
            a_keep_state = a_rnn_state.copy()
            c_keep_state = c_rnn_state.copy()

            while True:
                if (self.name == 'Worker_1' or N_WORKERS == 1) and RENDER:
                    self.env.render()

                a, v, a_rnn_state_, c_rnn_state_ = self.AC.eval_state(s, a_rnn_state, c_rnn_state)
                s2, r, terminal, info = self.env.step(a)

                ep_r += r
                ep_t += 1
                # r = np.clip(r, -1, 1)  # clip reward
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)
                buffer_v.append(v)
                ep_a.append(a)
                ep_v.append(v)

                if ep_t % UPDATE_GLOBAL_ITER == 0 or terminal:
                    if terminal:
                        r_next = 0  # Next reward is zero if terminal
                    else:
                        r_next = self.AC.get_value(s2, c_rnn_state_)[0]

                    # Generalized Advantage Estimation - https://arxiv.org/abs/1506.02438
                    rewards = np.array(buffer_r)
                    discounted_rewards = self.discount(np.append(rewards, r_next), GAMMA)[:-1]
                    values = np.array(buffer_v + [r_next])
                    advantages = self.discount(rewards + GAMMA * values[1:] - values[:-1], GAMMA * LAMBDA)

                    feed_dict = {
                        self.AC.s: np.asarray(buffer_s),
                        self.AC.a_history: np.asarray(buffer_a),
                        self.AC.advantage: np.vstack(advantages),
                        self.AC.R_discounted: np.vstack(discounted_rewards),
                        self.AC.a_init_state: a_keep_state,
                        self.AC.c_init_state: c_keep_state,
                        self.AC.is_training: False  # For small windows I doubt BN will work. Dropout may though
                    }

                    graph_summary = self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r, buffer_v = [], [], [], []
                    self.AC.pull_global()
                    # Replace the keep_state as the new initial rnn state_
                    a_keep_state = a_rnn_state_.copy()
                    c_keep_state = c_rnn_state_.copy()

                s = s2
                # Renew RNN states
                a_rnn_state = a_rnn_state_
                c_rnn_state = c_rnn_state_

                if terminal:
                    print(self.name, "| Local Ep:", self.ep_count,
                          "| Global Ep:", GLOBAL_EP,
                          "| Reward: %.2f" % ep_r,
                          "| Reward/step: %.3f" % (ep_r / ep_t),
                          "| Steps:", ep_t)

                    # Add summaries for TensorBoard
                    if self.name == 'Worker_0' and self.ep_count % 5 == 0:
                        worker_summary = tf.Summary()
                        worker_summary.value.add(tag="Reward/" + self.name, simple_value=ep_r)
                        # worker_summary.value.add(tag="Steps/" + self.name, simple_value=ep_t)
                        add_histogram(WRITER, "Critic/Value/" + self.name, np.ravel(ep_v), self.ep_count)

                        # Create Action histograms for each dimension
                        actions = np.array(ep_a)
                        for a in range(N_A):
                            add_histogram(WRITER, "Action/Dim"+str(a)+"/" + self.name, actions[:, a], self.ep_count)

                        WRITER.add_summary(worker_summary, self.ep_count)
                        WRITER.add_summary(graph_summary, self.ep_count)
                        WRITER.flush()

                    GLOBAL_EP += 1
                    self.ep_count += 1
                    break

if __name__ == "__main__":
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.01
    SESS = tf.Session(config=config)

    with tf.device("/cpu:0"):
        # OPT_A = tf.train.RMSPropOptimizer(LR_A, decay=0.99, name='RMSPropA')
        # OPT_C = tf.train.RMSPropOptimizer(LR_C, decay=0.99, name='RMSPropC')
        OPT_A = tf.train.AdamOptimizer(LR_A, name='AdamA')
        OPT_C = tf.train.AdamOptimizer(LR_C, name='AdamC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
        workers = []
        # Create workers
        for i in range(N_WORKERS):
            i_name = 'Worker_%i' % i   # worker name
            workers.append(Worker(i_name, GLOBAL_AC))

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())
    WRITER = tf.summary.FileWriter(SUMMARY_DIR, SESS.graph)

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)
