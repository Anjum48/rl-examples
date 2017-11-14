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


ENVIRONMENT = 'CartPole-v0'
# ENVIRONMENT = 'LunarLander-v2'
RENDER = False
RESTORE_DATE = None

if RESTORE_DATE is not None:
    SUMMARY_DIR = os.path.join(OUTPUT_RESULTS_DIR, 'A3C_LSTM', "gym", ENVIRONMENT, RESTORE_DATE)
else:
    TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
    SUMMARY_DIR = os.path.join(OUTPUT_RESULTS_DIR, "A3C_LSTM", "gym", ENVIRONMENT, TIMESTAMP)

# N_WORKERS = multiprocessing.cpu_count()
N_WORKERS = 1
MAX_GLOBAL_EP = 10000 * N_WORKERS
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 5
GAMMA = 0.99
LAMBDA = 1.00
ENTROPY_BETA = 0.01
LR = 0.001    # learning rate for actor
GLOBAL_EP = 0
CELL_SIZE = 256

env = gym.make(ENVIRONMENT)
N_S = env.observation_space.shape
N_A = env.action_space.n


class ACNet(object):
    def __init__(self, scope, globalAC=None):
        if scope == GLOBAL_NET_SCOPE:  # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None] + list(N_S), 'S')
                self._build_net()
                self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/network')
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None] + list(N_S), 'S')
                self.a_history = tf.placeholder(tf.int32, [None], 'A_history')
                self.advantage = tf.placeholder(tf.float32, [None], 'Advantage')
                self.R_discounted = tf.placeholder(tf.float32, [None], 'R_discounted')

                self.a_prob, self.v = self._build_net()

                with tf.name_scope('loss'):
                    with tf.name_scope('c_loss'):
                        self.c_loss = 0.5 * tf.reduce_sum(tf.square(self.R_discounted - self.v))

                    with tf.name_scope('a_loss'):
                        ac_one_hot = tf.one_hot(self.a_history, N_A, dtype=tf.float32)
                        self.a_loss = -tf.reduce_sum(tf.reduce_sum(tf.log(self.a_prob) * ac_one_hot, axis=1) * self.advantage)

                    self.entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob))
                    self.loss = self.a_loss + 0.5 * self.c_loss - ENTROPY_BETA * self.entropy

                with tf.name_scope('local_grad'):
                    self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/network')
                    self.grads = tf.gradients(self.loss, self.params)
                    self.grads, _ = tf.clip_by_global_norm(self.grads, 40.0)

                    for grad, var in list(zip(self.grads, self.params)):
                        tf.summary.histogram(var.name + '/gradient', grad)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.params, globalAC.params)]
                with tf.name_scope('push'):
                    self.update_op = OPT.apply_gradients(zip(self.grads, globalAC.params))

            tf.summary.scalar("Loss/" + scope, self.loss)
            tf.summary.scalar("Actor/Loss/" + scope, self.a_loss)
            tf.summary.scalar("Actor/Entropy/" + scope, tf.reduce_sum(self.entropy * ENTROPY_BETA))
            tf.summary.scalar("Critic/Loss/" + scope, self.c_loss)
            tf.summary.scalar("Critic/Value/" + scope, tf.reduce_sum(self.v))
            tf.summary.scalar("Critic/Discounted_Reward/" + scope, tf.reduce_sum(self.R_discounted))
            summary_list = [s for s in tf.get_collection(tf.GraphKeys.SUMMARIES) if scope in s.name]
            self.summarise = tf.summary.merge(summary_list)

    def _build_net(self):
        with tf.variable_scope('network'):
            w_init = tf.contrib.layers.variance_scaling_initializer()
            w_reg = tf.contrib.layers.l2_regularizer(0.01)
            hidden = tf.layers.dense(self.s, 200, tf.nn.relu, kernel_initializer=w_init,
                                     kernel_regularizer=w_reg, name='hidden')

            # Introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
            cell_in = tf.expand_dims(hidden, axis=0)
            lstm = tf.nn.rnn_cell.LSTMCell(CELL_SIZE)

            # LSTM initialisation
            self.state_init = lstm.zero_state(batch_size=1, dtype=tf.float32)
            step = tf.shape(self.s)[:1]

            # Calculate outputs
            lstm_outputs, self.state_out = tf.nn.dynamic_rnn(lstm, cell_in, initial_state=self.state_init)
            cell_out = tf.reshape(lstm_outputs, [-1, CELL_SIZE])

            a_prob = tf.layers.dense(cell_out, N_A, tf.nn.softmax, kernel_initializer=w_init, kernel_regularizer=w_reg,
                                     name='ap')
            v = tf.layers.dense(cell_out, 1, kernel_initializer=w_init, kernel_regularizer=w_reg, name='v')
        return a_prob, v

    def update_global(self, feed_dict):  # run by a local
        return SESS.run([self.update_op, self.summarise], feed_dict)[1]

    def pull_global(self):  # run by a local
        SESS.run([self.pull_params_op])

    def eval_state(self, s, lstm_state):
        prob_weights, v, new_lstm_state = SESS.run([self.a_prob, self.v, self.state_out],
                                                   {self.s: [s], self.state_init: lstm_state})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action, v[0][0], new_lstm_state

    def get_value(self, s, lstm_state):
        return SESS.run(self.v, {self.s: [s], self.state_init: lstm_state})[0]


class Worker(object):
    def __init__(self, name, globalAC):
        self.env = gym.make(ENVIRONMENT)
        self.name = name
        self.ep_count = 0
        self.AC = ACNet(name, globalAC)

        if self.name == 'Worker_0':
            self.env = wrappers.Monitor(self.env, os.path.join(SUMMARY_DIR, ENVIRONMENT+'_'+self.name))

    @staticmethod
    def discount(x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    def work(self):
        global GLOBAL_EP
        buffer_s, buffer_a, buffer_r, buffer_v = [], [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r, ep_t = 0, 0
            ep_a, ep_v = [], []
            # last_lstm_state = self.AC.state_init  # LSTM state at beginning
            last_lstm_state = SESS.run(self.AC.state_init)  # LSTM state at beginning

            while True:
                if (self.name == 'Worker_1' or N_WORKERS == 1) and RENDER:
                    self.env.render()

                a, v, lstm_state = self.AC.eval_state(s, last_lstm_state)
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

                if ep_t % UPDATE_GLOBAL_ITER == 0 or terminal:  # update global and assign to local net
                    if terminal:
                        r_next = 0  # Next reward is zero if terminal
                    else:
                        r_next = self.AC.get_value(s2, lstm_state)[0]

                    # Generalized Advantage Estimation - https://arxiv.org/abs/1506.02438
                    rewards = np.array(buffer_r)
                    discounted_rewards = self.discount(np.append(rewards, r_next), GAMMA)[:-1]
                    values = np.array(buffer_v + [r_next])
                    advantages = self.discount(rewards + GAMMA * values[1:] - values[:-1], GAMMA * LAMBDA)

                    feed_dict = {
                        self.AC.s: np.asarray(buffer_s),
                        self.AC.a_history: np.asarray(buffer_a),
                        self.AC.advantage: advantages,
                        self.AC.R_discounted: discounted_rewards,
                        # self.AC.state_in[0]: last_lstm_state[0],
                        # self.AC.state_in[1]: last_lstm_state[1],
                        self.AC.state_init: last_lstm_state,
                    }
                    graph_summary = self.AC.update_global(feed_dict)

                    buffer_s, buffer_a, buffer_r, buffer_v = [], [], [], []
                    self.AC.pull_global()

                s = s2
                # Renew RNN states
                last_lstm_state = lstm_state

                if terminal:
                    print(self.name, "| Local Ep:", self.ep_count,
                          "| Global Ep:", GLOBAL_EP,
                          "| Reward: %.2f" % ep_r,
                          "| Reward/step: %.3f" % (ep_r / ep_t),
                          "| Steps:", ep_t)

                    # Add summaries for TensorBoard
                    if self.name == 'Worker_0' and self.ep_count % 5 == 0:
                        worker_summary = tf.Summary()
                        worker_summary.value.add(tag="Reward/"+self.name, simple_value=ep_r)
                        WRITER.add_summary(worker_summary, self.ep_count)
                        WRITER.add_summary(graph_summary, self.ep_count)
                        WRITER.flush()

                    GLOBAL_EP += 1
                    self.ep_count += 1
                    break


if __name__ == "__main__":
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        # OPT = tf.train.RMSPropOptimizer(LR, name='RMSProp')
        OPT = tf.train.AdamOptimizer(LR, name='Adam')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
        workers = []
        # Create worker
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
