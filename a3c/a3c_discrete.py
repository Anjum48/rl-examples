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
from datetime import datetime
OUTPUT_RESULTS_DIR = "./"


ENVIRONMENT = 'CartPole-v0'
RENDER = False
RESTORE_DATE = None

if RESTORE_DATE is not None:
    SUMMARY_DIR = os.path.join(OUTPUT_RESULTS_DIR, 'A3C', "gym", ENVIRONMENT, RESTORE_DATE)
else:
    TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
    SUMMARY_DIR = os.path.join(OUTPUT_RESULTS_DIR, "A3C", "gym", ENVIRONMENT, TIMESTAMP)

# N_WORKERS = multiprocessing.cpu_count()
N_WORKERS = 1
MAX_GLOBAL_EP = 10000 * N_WORKERS
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 5
GAMMA = 0.99
ENTROPY_BETA = 0.01
LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GLOBAL_EP = 0

env = gym.make(ENVIRONMENT)

N_S = env.observation_space.shape[0]
N_A = env.action_space.n


class ACNet(object):
    def __init__(self, scope, globalAC=None):
        if scope == GLOBAL_NET_SCOPE:  # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self._build_net()
                self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                self.a_prob, self.v = self._build_net()

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(tf.log(self.a_prob) * tf.one_hot(self.a_his, N_A, dtype=tf.float32),
                                             axis=1, keep_dims=True)
                    exp_v = log_prob * td
                    self.entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob), axis=1,  keep_dims=True)  # encourage exploration
                    self.exp_v = ENTROPY_BETA * self.entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('local_grad'):
                    self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                    self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

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
            tf.summary.scalar("Actor/Entropy/" + scope, tf.reduce_sum(self.entropy * ENTROPY_BETA))
            tf.summary.scalar("Critic/Loss/" + scope, self.c_loss)
            tf.summary.scalar("Critic/Value/" + scope, tf.reduce_sum(self.v))
            summary_list = [s for s in tf.get_collection(tf.GraphKeys.SUMMARIES) if scope in s.name]
            self.summarise = tf.summary.merge(summary_list)

    def _build_net(self):
        # TODO: Add LSTM
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            a_prob = tf.layers.dense(l_a, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        return a_prob, v

    def update_global(self, feed_dict):  # run by a local
        return SESS.run([self.update_a_op, self.update_c_op, self.summarise], feed_dict)[2]  # local grads applies to global net

    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def eval_state(self, s):  # run by a local
        prob_weights, v = SESS.run([self.a_prob, self.v], feed_dict={self.s: s[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action, v[0]

    def get_value(self, s):
        return SESS.run(self.v, {self.s: s[np.newaxis, :]})[0]


class Worker(object):
    def __init__(self, name, globalAC):
        self.env = gym.make(ENVIRONMENT)
        self.name = name
        self.ep_count = 0
        self.AC = ACNet(name, globalAC)

        # if self.name == 'Worker_0':
        #     self.env = wrappers.Monitor(self.env, os.path.join(SUMMARY_DIR, ENVIRONMENT+'_'+self.name))

    def work(self):
        global GLOBAL_EP
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r, ep_t = 0, 0
            while True:
                if (self.name == 'Worker_1' or N_WORKERS == 1) and RENDER:
                    self.env.render()

                a, v = self.AC.eval_state(s)
                s2, r, terminal, info = self.env.step(a)

                ep_r += r
                ep_t += 1
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if ep_t % UPDATE_GLOBAL_ITER == 0 or terminal:  # update global and assign to local net
                    if terminal:
                        v_s_ = 0  # terminal
                    else:
                        v_s_ = self.AC.get_value(s2)[0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:  # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)

                    feed_dict = {
                        self.AC.s: np.asarray(buffer_s),
                        self.AC.a_his: np.asarray(buffer_a),
                        self.AC.v_target: np.vstack(buffer_v_target[::-1]),
                    }
                    graph_summary = self.AC.update_global(feed_dict)

                    buffer_s, buffer_a, buffer_r = [], [], []
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
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
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
