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
ENVIRONMENT = 'BipedalWalkerHardcore-v2'

EP_MAX = 10000
GAMMA = 0.99
LAMBDA = 0.95
ENTROPY_BETA = 0.0
LR = 0.0001
BATCH = 8192
MINIBATCH = 64
EPOCHS = 10
EPSILON = 0.1
VF_COEFF = 0.5
L2_REG = 0.001
LSTM_UNITS = 128
LSTM_LAYERS = 1
KEEP_PROB = 0.8

TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
SUMMARY_DIR = os.path.join(OUTPUT_RESULTS_DIR, "PPO_LSTM", ENVIRONMENT, TIMESTAMP)


class PPO(object):
    def __init__(self, environment):
        self.s_dim, self.a_dim = environment.observation_space.shape[0], environment.action_space.shape[0]
        self.a_bound = environment.action_space.high

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        config = tf.ConfigProto(log_device_placement=False, device_count={'GPU': 0})  # 1 for GPU, 0 for CPU
        self.sess = tf.Session(config=config)
        self.state = tf.placeholder(tf.float32, [None, self.s_dim], 'state')
        self.actions = tf.placeholder(tf.float32, [None, self.a_dim], 'action')
        self.advantage = tf.placeholder(tf.float32, [None, 1], 'advantage')
        self.rewards = tf.placeholder(tf.float32, [None, 1], 'discounted_r')

        self.is_training = tf.placeholder(tf.bool, name='training_flag')
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
        oldpi, oldpi_params, _, _ = self._build_anet(batch["state"], 'oldpi')
        pi, pi_params, self.pi_i_state, self.pi_f_state = self._build_anet(batch["state"], 'pi')
        evalpi, _, self.evalpi_i_state, self.evalpi_f_state = self._build_anet(self.state, 'pi', reuse=True, batch_size=1)

        self.v, vf_params, self.vf_i_state, self.vf_f_state = self._build_cnet(batch["state"], "vf")
        self.evalvf, _, self.evalvf_i_state, self.evalvf_f_state = self._build_cnet(self.state, 'vf', reuse=True, batch_size=1)

        self.sample_op = tf.squeeze(evalpi.sample(1), axis=0, name="sample_action")
        self.eval_action = evalpi.loc

        with tf.variable_scope('loss'):
            with tf.variable_scope('actor_loss'):
                # Use floor functions for the probabilities to prevent NaNs when prob = 0
                ratio = tf.maximum(pi.prob(batch["actions"]), 1e-9) / tf.maximum(oldpi.prob(batch["actions"]), 1e-9)
                ratio = tf.clip_by_value(ratio, 0, 10)  # If the ratio denominator is tiny, the ratio can explode
                surr1 = ratio * batch["advantage"]
                surr2 = tf.clip_by_value(ratio, 1 - EPSILON, 1 + EPSILON) * batch["advantage"]
                a_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            with tf.variable_scope('critic_loss'):
                c_loss = tf.reduce_mean(tf.square(self.v - batch["rewards"])) * 0.5

            with tf.variable_scope('entropy_bonus'):
                entropy = pi.entropy()
                pol_entpen = -ENTROPY_BETA * tf.reduce_mean(entropy)

            self.loss = a_loss + c_loss * VF_COEFF + pol_entpen

        with tf.variable_scope('train'):
            opt = tf.train.AdamOptimizer(LR)
            grads, vs = zip(*opt.compute_gradients(self.loss, var_list=pi_params + vf_params))
            # grads, _ = tf.clip_by_global_norm(grads, 50.0)
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(self.update_ops):
                self.train_op = opt.apply_gradients(zip(grads, vs), global_step=self.global_step)

        with tf.variable_scope('update_old_pi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.writer = tf.summary.FileWriter(SUMMARY_DIR, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        tf.summary.scalar("Loss/Actor", a_loss)
        tf.summary.scalar("Loss/Critic", c_loss)
        tf.summary.scalar("Value", tf.reduce_mean(self.v))
        tf.summary.scalar("Ratio", tf.reduce_mean(ratio))
        tf.summary.scalar("Sigma", tf.reduce_mean(pi.scale))
        self.summarise = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))

    def _build_anet(self, state_in, name, reuse=False, batch_size=MINIBATCH):
        w_reg = tf.contrib.layers.l2_regularizer(L2_REG)

        with tf.variable_scope(name, reuse=reuse):
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

            mu = tf.layers.dense(a_cell_out, self.a_dim, tf.nn.tanh, kernel_regularizer=w_reg, name="pi_mu")
            sigma = tf.get_variable(name="pi_sigma", shape=self.a_dim, initializer=tf.ones_initializer())
            norm_dist = tf.distributions.Normal(loc=mu * self.a_bound, scale=tf.maximum(sigma, 0.1))
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params, a_init_state, a_final_state

    def _build_cnet(self, state_in, name, reuse=False, batch_size=MINIBATCH):
        w_reg = tf.contrib.layers.l2_regularizer(L2_REG)

        with tf.variable_scope(name, reuse=reuse):
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
        self.sess.run([self.update_oldpi_op])

        for _ in range(EPOCHS):
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
            eval_ops = [self.sample_op, self.evalvf, self.evalpi_f_state, self.evalvf_f_state]
        else:
            eval_ops = [self.eval_action, self.evalvf, self.evalpi_f_state, self.evalvf_f_state]

        a, v, a_state, c_state = self.sess.run(eval_ops,
                                               {self.state: state[np.newaxis, :],
                                                self.evalpi_i_state: a_state, self.evalvf_i_state: c_state,
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


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


env = gym.make(ENVIRONMENT)
env = wrappers.Monitor(env, os.path.join(SUMMARY_DIR, ENVIRONMENT), video_callable=None)
ppo = PPO(env)

t = 0
buffer_s, buffer_a, buffer_r, buffer_v, terminals = [], [], [], [], []
experience = []

for episode in range(EP_MAX + 1):

    a_lstm_state, c_lstm_state = ppo.sess.run([ppo.evalpi_i_state, ppo.evalvf_i_state])  # Zero LSTM state at beginning

    s = env.reset()
    ep_r, ep_t, terminal = 0, 0, True
    ep_a = []

    while True:
        a, v, a_lstm_state, c_lstm_state = ppo.eval_state(s, a_lstm_state, c_lstm_state)

        if ep_t > 0 and terminal:
            print('Episode: %i' % episode, "| Reward: %.2f" % ep_r, '| Steps: %i' % ep_t)

            v = [v[0] * (1 - terminal)]  # v = 0 if terminal, otherwise use the predicted v
            rewards = np.array(buffer_r)
            vpred = np.array(buffer_v + v)
            terminals_array = np.array(terminals + [0])

            # Generalized Advantage Estimation - https://arxiv.org/abs/1506.02438
            advantage = discount(rewards + GAMMA * vpred[1:] * (1 - terminals_array[1:]) - vpred[:-1], GAMMA * LAMBDA)
            tdlamret = advantage + np.array(buffer_v)
            advantage = (advantage - advantage.mean()) / np.maximum(advantage.std(), 1e-6)

            if len(rewards) >= MINIBATCH:  # Ignore episodes shorter than a minibatch
                bs, ba, br, badv = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(tdlamret), np.vstack(advantage)
                experience.append((bs, ba, br, badv))
            else:
                t -= len(rewards)

            buffer_s, buffer_a, buffer_r, buffer_v, terminals = [], [], [], [], []

            # Update PPO
            if t >= BATCH:
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
    a_lstm_state, c_lstm_state = ppo.sess.run([ppo.evalpi_i_state, ppo.evalvf_i_state])
    while True:
        env.render()
        a, v, a_lstm_state, c_lstm_state = ppo.eval_state(s, a_lstm_state, c_lstm_state, stochastic=False)
        s, r, terminal, _ = env.step(np.clip(a, -ppo.a_bound, ppo.a_bound))
        ep_r += r
        ep_t += 1
        if terminal:
            print("Reward: %.2f" % ep_r, '| Steps: %i' % ep_t)
            break
