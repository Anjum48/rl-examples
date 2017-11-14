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


# ENVIRONMENT = 'Pendulum-v0'  # Make sure to divide reward by 10
# ENVIRONMENT = 'MountainCarContinuous-v0'
ENVIRONMENT = 'LunarLanderContinuous-v2'
RENDER = False
RESTORE_DATE = None

if RESTORE_DATE is not None:
    SUMMARY_DIR = os.path.join(OUTPUT_RESULTS_DIR, 'A3C', "gym", ENVIRONMENT, RESTORE_DATE)
else:
    TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
    SUMMARY_DIR = os.path.join(OUTPUT_RESULTS_DIR, "A3C", "gym", ENVIRONMENT, TIMESTAMP)

N_WORKERS = multiprocessing.cpu_count()
MAX_GLOBAL_EP = 10000 * N_WORKERS
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 5
GAMMA = 0.99
LAMBDA = 1.0
ENTROPY_BETA = 0.0001
LR_A = 0.0001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0

env = gym.make(ENVIRONMENT)
N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]
with tf.name_scope('env_bounds'):
    S_UPPER = tf.Variable(env.observation_space.high, dtype=tf.float32, name="state_upper")
    S_LOWER = tf.Variable(env.observation_space.low, dtype=tf.float32, name="state_lower")
    A_UPPER = tf.Variable(env.action_space.high, dtype=tf.float32, name="action_upper")
    A_LOWER = tf.Variable(env.action_space.low, dtype=tf.float32, name="action_lower")


class ACNet(object):
    def __init__(self, scope, globalAC=None):

        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self._build_net()
                self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                with tf.name_scope('input_norm'):
                    self.s = 2 * (tf.placeholder(tf.float32, [None, N_S], 'S') - S_LOWER) / (S_UPPER - S_LOWER) - 1
                self.a_history = tf.placeholder(tf.float32, [None, N_A], 'A')
                self.advantage = tf.placeholder(tf.float32, [None, 1], 'Advantage')
                self.R_discounted = tf.placeholder(tf.float32, [None, 1], 'R_discounted')

                mu, sigma, self.v = self._build_net()

                with tf.name_scope('action_prep'):
                    with tf.name_scope('wrap_a_out'):
                        mu, sigma = mu * A_UPPER, sigma + 1e-4

                    normal_dist = tf.contrib.distributions.Normal(mu, sigma)

                    with tf.name_scope('choose_a'):  # use local params to choose action
                        self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1, seed=1234), axis=0),
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
                    self.a_grads, _ = tf.clip_by_global_norm(a_grads, 40.0)
                    self.c_grads, _ = tf.clip_by_global_norm(c_grads, 40.0)

                    for grad, var in list(zip(self.a_grads, self.a_params)):
                        tf.summary.histogram(var.name + '/gradient', grad)
                    for grad, var in list(zip(self.c_grads, self.c_params)):
                        tf.summary.histogram(var.name + '/gradient', grad)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

            tf.summary.scalar("Actor/Loss/" + scope, self.a_loss)
            tf.summary.scalar("Actor/Advantage/" + scope, tf.reduce_sum(self.advantage))
            tf.summary.scalar("Actor/Entropy/" + scope, tf.reduce_sum(self.entropy * ENTROPY_BETA))
            tf.summary.scalar("Critic/Loss/" + scope, self.c_loss)
            tf.summary.scalar("Critic/Value/" + scope, tf.reduce_sum(self.v))
            tf.summary.scalar("Critic/Discounted_Reward/" + scope, tf.reduce_sum(self.R_discounted))
            summary_list = [s for s in tf.get_collection(tf.GraphKeys.SUMMARIES) if scope in s.name]
            self.summarise = tf.summary.merge(summary_list)

    def _build_net(self ):
        # w_init = tf.random_normal_initializer(0., .1)
        w_init = tf.contrib.layers.variance_scaling_initializer()
        w_init_final = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        w_reg = tf.contrib.layers.l2_regularizer(0.01)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, kernel_regularizer=w_reg, name='la')
            mu = tf.layers.dense(l_a, N_A, tf.nn.tanh, kernel_initializer=w_init_final, kernel_regularizer=w_reg, name='mu')
            sigma = tf.layers.dense(l_a, N_A, tf.nn.softplus, kernel_initializer=w_init, kernel_regularizer=w_reg, name='sigma')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, kernel_regularizer=w_reg, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, kernel_regularizer=w_reg, name='v')  # state value
        return mu, sigma, v

    def update_global(self, feed_dict):
        return SESS.run([self.update_a_op, self.update_c_op, self.summarise], feed_dict)[2]

    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def eval_state(self, s):
        a, v, = SESS.run([self.A, self.v], {self.s: s[np.newaxis, :]})
        return a[0], v[0][0]

    def get_value(self, s):
        return SESS.run(self.v, {self.s: s[np.newaxis, :]})[0]


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
    def __init__(self, name, globalAC):
        self.env = gym.make(ENVIRONMENT)
        self.name = name
        self.AC = ACNet(name, globalAC)
        self.ep_count = 0

        if self.name == 'Worker_0':
            self.env = wrappers.Monitor(self.env, os.path.join(SUMMARY_DIR, ENVIRONMENT+'_'+self.name))

    @staticmethod
    def discount(x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        buffer_s, buffer_a, buffer_r, buffer_v = [], [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r, ep_t = 0, 0
            ep_a, ep_v = [], []
            while True:
                if (self.name == 'Worker_1' or N_WORKERS == 1) and RENDER:
                    self.env.render()
                a, v = self.AC.eval_state(s)
                s2, r, terminal, info = self.env.step(a)

                ep_r += r
                ep_t += 1
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r/10)
                buffer_v.append(v)
                ep_a.append(a)
                ep_v.append(v)

                if ep_t % UPDATE_GLOBAL_ITER == 0 or terminal:   # update global and assign to local net
                    if terminal:
                        v_next = 0   # terminal
                    else:
                        v_next = self.AC.get_value(s2)[0]

                    # Generalized Advantage Estimation - https://arxiv.org/abs/1506.02438
                    rewards = np.array(buffer_r)
                    discounted_rewards = self.discount(np.append(rewards, v_next), GAMMA)[:-1]
                    values = np.array(buffer_v + [v_next])
                    advantages = self.discount(rewards + GAMMA * values[1:] - values[:-1], GAMMA * LAMBDA)

                    feed_dict = {
                        self.AC.s: np.asarray(buffer_s),
                        self.AC.a_history: np.asarray(buffer_a),
                        self.AC.advantage: np.vstack(advantages),
                        self.AC.R_discounted: np.vstack(discounted_rewards)
                    }

                    graph_summary = self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r, buffer_v = [], [], [], []
                    self.AC.pull_global()

                s = s2
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

                        add_histogram(WRITER, "Critic/Value/" + self.name, np.ravel(ep_v), self.ep_count)

                        # Create Action histograms for each dimension
                        actions = np.array(ep_a)
                        for a in range(N_A):
                            add_histogram(WRITER, "Action/Dim" + str(a) + "/" + self.name, actions[:, a], self.ep_count)

                        WRITER.add_summary(worker_summary, self.ep_count)
                        WRITER.add_summary(graph_summary, self.ep_count)
                        WRITER.flush()

                    GLOBAL_EP += 1
                    self.ep_count += 1
                    break

if __name__ == "__main__":
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        # OPT_A = tf.train.RMSPropOptimizer(LR_A, decay=0.99, name='RMSPropA')
        # OPT_C = tf.train.RMSPropOptimizer(LR_C, decay=0.99, name='RMSPropC')
        OPT_A = tf.train.AdamOptimizer(LR_A, name='AdamA')
        OPT_C = tf.train.AdamOptimizer(LR_C, name='AdamC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
        workers = []
        # Create workers
        for i in range(N_WORKERS):
            i_name = 'Worker_%i' % i  # worker name
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
