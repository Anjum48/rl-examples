"""
A simple version of Proximal Policy Optimization (PPO) using single thread and an LSTM layer.

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
OUTPUT_RESULTS_DIR = "./"

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
LR = 0.00005
BATCH = 8192
MINIBATCH = 64
EPOCHS = 10
EPSILON = 0.1
VF_COEFF = 1.0
L2_REG = 0.001
LSTM_UNITS = 128
LSTM_LAYERS = 1
KEEP_PROB = 0.8

TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
SUMMARY_DIR = os.path.join(OUTPUT_RESULTS_DIR, "PPO_LSTM", ENVIRONMENT, TIMESTAMP)


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
        self.keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # Use the TensorFlow Dataset API
        self.dataset = tf.data.Dataset.from_tensor_slices({"state": self.state, "actions": self.actions,
                                                           "rewards": self.rewards, "advantage": self.advantage})
        self.dataset = self.dataset.batch(MINIBATCH)
        self.iterator = self.dataset.make_initializable_iterator()
        batch = self.iterator.get_next()
        self.global_step = tf.train.get_or_create_global_step()

        # Create an old & new policy function but also
        # make separate value & policy functions for evaluation & training (with shared variables)
        pi_old, pi_old_params, _, _ = self._build_anet(batch["state"], 'oldpi')
        pi, pi_params, self.pi_i_state, self.pi_f_state = self._build_anet(batch["state"], 'pi')
        pi_eval, _, self.pi_eval_i_state, self.evalpi_f_state = self._build_anet(self.state, 'pi', reuse=True, batch_size=1)

        vf_old, vf_old_params, _, _ = self._build_cnet(batch["state"], "oldvf")
        self.v, vf_params, self.vf_i_state, self.vf_f_state = self._build_cnet(batch["state"], "vf")
        self.vf_eval, _, self.vf_eval_i_state, self.vf_eval_f_state = self._build_cnet(self.state, 'vf', reuse=True, batch_size=1)

        self.sample_op = tf.squeeze(pi_eval.sample(1), axis=0, name="sample_action")
        self.eval_action = pi_eval.loc

        with tf.variable_scope('loss'):
            epsilon_decay = tf.train.polynomial_decay(EPSILON, self.global_step, 1e5, 0.01, power=0.0)

            with tf.variable_scope('policy'):
                # Use floor functions for the probabilities to prevent NaNs when prob = 0
                ratio = tf.maximum(pi.prob(batch["actions"]), 1e-6) / tf.maximum(pi_old.prob(batch["actions"]), 1e-6)
                ratio = tf.clip_by_value(ratio, 0, 10)
                surr1 = batch["advantage"] * ratio
                surr2 = batch["advantage"] * tf.clip_by_value(ratio, 1 - epsilon_decay, 1 + epsilon_decay)
                loss_pi = -tf.reduce_mean(tf.minimum(surr1, surr2))
                tf.summary.scalar("loss", loss_pi)

            with tf.variable_scope('value_function'):
                clipped_value_estimate = vf_old + tf.clip_by_value(self.v - vf_old, -epsilon_decay, epsilon_decay)
                loss_vf1 = tf.squared_difference(clipped_value_estimate, batch["rewards"])
                loss_vf2 = tf.squared_difference(self.v, batch["rewards"])
                loss_vf = tf.reduce_mean(tf.maximum(loss_vf1, loss_vf2)) * 0.5
                # loss_vf = tf.reduce_mean(tf.square(self.v - batch["rewards"])) * 0.5
                tf.summary.scalar("loss", loss_vf)

            with tf.variable_scope('entropy_bonus'):
                entropy = pi.entropy()
                pol_entpen = -ENTROPY_BETA * tf.reduce_mean(entropy)

            loss = loss_pi + loss_vf * VF_COEFF + pol_entpen
            tf.summary.scalar("total", loss)
            tf.summary.scalar("epsilon", epsilon_decay)

        with tf.variable_scope('train'):
            opt = tf.train.AdamOptimizer(LR)
            self.train_op = opt.minimize(loss, global_step=self.global_step, var_list=pi_params + vf_params)

            # grads, vs = zip(*opt.compute_gradients(loss, var_list=pi_params + vf_params))
            # Need to split the two networks so that clip_by_global_norm works properly
            # pi_grads, pi_vs = grads[:len(pi_params)], vs[:len(pi_params)]
            # vf_grads, vf_vs = grads[len(pi_params):], vs[len(pi_params):]
            # pi_grads, _ = tf.clip_by_global_norm(pi_grads, 10)
            # vf_grads, _ = tf.clip_by_global_norm(vf_grads, 10)
            # self.train_op = opt.apply_gradients(zip(pi_grads + vf_grads, pi_vs + vf_vs), global_step=self.global_step)

            # for grad, var in zip(pi_grads + vf_grads, pi_vs + vf_vs):
            #     tf.summary.histogram(var.name, grad)

        with tf.variable_scope('update_old'):
            self.update_pi_old_op = [oldp.assign(p) for p, oldp in zip(pi_params, pi_old_params)]
            self.update_vf_old_op = [oldp.assign(p) for p, oldp in zip(vf_params, vf_old_params)]

        self.writer = tf.summary.FileWriter(SUMMARY_DIR, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        tf.summary.scalar("Value", tf.reduce_mean(self.v))
        tf.summary.scalar("Sigma", tf.reduce_mean(pi.scale))
        self.summarise = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))

    def _build_anet(self, state_in, name, reuse=False, batch_size=MINIBATCH):
        w_reg = tf.contrib.layers.l2_regularizer(L2_REG)

        with tf.variable_scope(name, reuse=reuse):
            if self.cnn:
                scaled = tf.cast(state_in, tf.float32) / 255.
                conv1 = tf.layers.conv2d(inputs=scaled, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu)
                conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)
                conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)
                state_in = tf.layers.flatten(conv3)

            l1 = tf.layers.dense(state_in, 400, tf.nn.relu, kernel_regularizer=w_reg, name="pi_l1")
            l2 = tf.layers.dense(l1, LSTM_UNITS, tf.nn.relu, kernel_regularizer=w_reg, name="pi_l2")

            # LSTM layer
            a_lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units=LSTM_UNITS)
            a_lstm = tf.nn.rnn_cell.DropoutWrapper(a_lstm, output_keep_prob=self.keep_prob)
            a_lstm = tf.nn.rnn_cell.MultiRNNCell(cells=[a_lstm] * LSTM_LAYERS)

            a_init_state = a_lstm.zero_state(batch_size=batch_size, dtype=tf.float32)
            lstm_in = tf.expand_dims(l2, axis=1)

            a_outputs, a_final_state = tf.nn.dynamic_rnn(cell=a_lstm, inputs=lstm_in, initial_state=a_init_state)
            a_cell_out = tf.reshape(a_outputs, [-1, LSTM_UNITS], name='flatten_lstm_outputs')

            mu = tf.layers.dense(a_cell_out, self.a_dim, tf.nn.tanh, kernel_regularizer=w_reg,
                                 name="pi_mu", use_bias=False)
            log_sigma = tf.get_variable(name="pi_sigma", shape=self.a_dim, initializer=tf.zeros_initializer())
            norm_dist = tf.distributions.Normal(loc=mu * self.a_bound, scale=tf.maximum(tf.exp(log_sigma), 0.1))
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params, a_init_state, a_final_state

    def _build_cnet(self, state_in, name, reuse=False, batch_size=MINIBATCH):
        w_reg = tf.contrib.layers.l2_regularizer(L2_REG)

        with tf.variable_scope(name, reuse=reuse):
            if self.cnn:
                scaled = tf.cast(state_in, tf.float32) / 255.
                conv1 = tf.layers.conv2d(inputs=scaled, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu)
                conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)
                conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)
                state_in = tf.layers.flatten(conv3)

            l1 = tf.layers.dense(state_in, 400, tf.nn.relu, kernel_regularizer=w_reg, name="vf_l1")
            l2 = tf.layers.dense(l1, LSTM_UNITS, tf.nn.relu, kernel_regularizer=w_reg, name="vf_l2")

            # LSTM layer
            c_lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units=LSTM_UNITS)
            c_lstm = tf.nn.rnn_cell.DropoutWrapper(c_lstm, output_keep_prob=self.keep_prob)
            c_lstm = tf.nn.rnn_cell.MultiRNNCell([c_lstm] * LSTM_LAYERS)

            c_init_state = c_lstm.zero_state(batch_size=batch_size, dtype=tf.float32)
            lstm_in = tf.expand_dims(l2, axis=1)

            c_outputs, c_final_state = tf.nn.dynamic_rnn(cell=c_lstm, inputs=lstm_in, initial_state=c_init_state)
            c_cell_out = tf.reshape(c_outputs, [-1, LSTM_UNITS], name='flatten_lstm_outputs')

            vf = tf.layers.dense(c_cell_out, 1, kernel_regularizer=w_reg, name="vf_out")
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return vf, params, c_init_state, c_final_state

    def update(self, exp):
        start, e_time = time(), []
        self.sess.run([self.update_pi_old_op, self.update_vf_old_op])

        for _ in range(EPOCHS):
            np.random.shuffle(exp)
            for ep_s, ep_a, ep_r, ep_adv in exp:

                # Trim down to round minibatches
                trim_index = (ep_s.shape[0] // MINIBATCH) * MINIBATCH
                ep_s = ep_s[:trim_index]
                ep_a = ep_a[:trim_index]
                ep_r = ep_r[:trim_index]
                ep_adv = ep_adv[:trim_index]

                self.sess.run(self.iterator.initializer, feed_dict={self.state: ep_s, self.actions: ep_a,
                                                                    self.rewards: ep_r, self.advantage: ep_adv})

                a_state, c_state = self.sess.run([self.pi_i_state, self.vf_i_state])
                train_ops = [self.summarise, self.global_step, self.pi_f_state, self.vf_f_state, self.train_op]

                while True:
                    try:
                        e_start = time()
                        feed_dict = {self.pi_i_state: a_state, self.vf_i_state: c_state, self.keep_prob: KEEP_PROB}
                        summary, step, a_state, c_state, _ = self.sess.run(train_ops, feed_dict=feed_dict)
                        e_time.append(time() - e_start)
                    except tf.errors.OutOfRangeError:
                        break

        print("Trained in %.3fs. Average %.3fs/batch. Global step %i" % (time() - start, np.mean(e_time), step))
        return summary

    def eval_state(self, state, a_state, c_state, stochastic=True):
        if stochastic:
            eval_ops = [self.sample_op, self.vf_eval, self.evalpi_f_state, self.vf_eval_f_state]
        else:
            eval_ops = [self.eval_action, self.vf_eval, self.evalpi_f_state, self.vf_eval_f_state]

        a, v, a_state, c_state = self.sess.run(eval_ops,
                                               {self.state: state[np.newaxis, :],
                                                self.pi_eval_i_state: a_state, self.vf_eval_i_state: c_state,
                                                self.keep_prob: 1.0})
        return a[0], v[0], a_state, c_state


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


class RunningStats(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    # https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / (self.count + batch_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        self.mean = new_mean
        self.var = new_var
        self.count = batch_count + self.count


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


env = gym.make(ENVIRONMENT)
env = wrappers.Monitor(env, os.path.join(SUMMARY_DIR, ENVIRONMENT), video_callable=None)
ppo = PPO(env)

t = 0
buffer_s, buffer_a, buffer_r, buffer_v, terminals = [], [], [], [], []
experience = []
stats = RunningStats()

for episode in range(EP_MAX + 1):

    a_lstm_state, c_lstm_state = ppo.sess.run([ppo.pi_eval_i_state, ppo.vf_eval_i_state])  # Zero LSTM state

    s = env.reset()
    ep_r, ep_t, terminal = 0, 0, False
    ep_a = []

    while True:
        a, v, a_lstm_state, c_lstm_state = ppo.eval_state(s, a_lstm_state, c_lstm_state)

        if terminal:
            print('Episode: %i' % episode, "| Reward: %.2f" % ep_r, '| Steps: %i' % ep_t)

            v = [v[0] * (1 - terminal)]  # v = 0 if terminal, otherwise use the predicted v
            rewards = np.array(buffer_r)
            vpred = np.array(buffer_v + v)
            terminals_array = np.array(terminals + [0])

            # Generalized Advantage Estimation - https://arxiv.org/abs/1506.02438
            delta = rewards + GAMMA * vpred[1:] * (1 - terminals_array[1:]) - vpred[:-1]
            advantage = discount(delta, GAMMA * LAMBDA)
            td_lambda_returns = advantage + np.array(buffer_v)
            advantage = (advantage - np.mean(advantage)) / np.maximum(np.std(advantage), 1e-6)

            if len(rewards) >= MINIBATCH:  # Ignore episodes shorter than a minibatch
                bs, ba, br, badv = np.reshape(buffer_s, (ep_t,) + ppo.s_dim), np.vstack(buffer_a), \
                                   np.vstack(td_lambda_returns), np.vstack(advantage)
                experience.append([bs, ba, br, badv])
            else:
                t -= len(rewards)

            buffer_s, buffer_a, buffer_r, buffer_v, terminals = [], [], [], [], []

            # Update PPO
            if t >= BATCH:
                # Per batch normalisation of advantages
                # advs = np.concatenate(list(zip(*experience))[3])
                # for x in experience:
                #     x[3] = (x[3] - np.mean(advs)) / np.maximum(np.std(advs), 1e-6)

                print("Training using %i episodes and %i steps..." % (len(experience), t))
                graph_summary = ppo.update(experience)
                t, experience = 0, []

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
    a_lstm_state, c_lstm_state = ppo.sess.run([ppo.pi_eval_i_state, ppo.vf_eval_i_state])
    while True:
        env.render()
        a, v, a_lstm_state, c_lstm_state = ppo.eval_state(s, a_lstm_state, c_lstm_state, stochastic=False)
        s, r, terminal, _ = env.step(np.clip(a, -ppo.a_bound, ppo.a_bound))
        ep_r += r
        ep_t += 1
        if terminal:
            print("Reward: %.2f" % ep_r, '| Steps: %i' % ep_t)
            break
