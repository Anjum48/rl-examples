"""
A simple version of Proximal Policy Optimization (PPO) using single thread.

Based on:
1. Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]
2. Proximal Policy Optimization Algorithms (OpenAI): [https://arxiv.org/abs/1707.06347]
3. Generalized Advantage Estimation [https://arxiv.org/abs/1506.02438]

Thanks to OpenAI and morvanzhou for their examples
"""
import sys
sys.path.append("/home/anjum/PycharmProjects/rl-examples/")
import argparse
import tensorflow as tf
import numpy as np
import gym
import os
import scipy.signal
from gym import wrappers
OUTPUT_RESULTS_DIR = "./"

# ENVIRONMENT = 'Pendulum-v0'
# ENVIRONMENT = 'MountainCarContinuous-v0'
# ENVIRONMENT = 'LunarLanderContinuous-v2'
ENVIRONMENT = 'BipedalWalker-v2'
# ENVIRONMENT = 'BipedalWalkerHardcore-v2'

EP_MAX = 10000
GAMMA = 0.99
LAMBDA = 0.95
ENTROPY_BETA = 0.0
LR = 0.0001
BATCH = 8192
MINIBATCH = 64
EPOCHS = 10
EPSILON = 0.2
VF_COEFF = 0.5
L2_REG = 0.001
LSTM_UNITS = 128
LSTM_LAYERS = 1
KEEP_PROB = 0.8
PORT_OFFSET = 0  # number to offset the port from 2222. Useful for multiple runs


class PPO(object):
    def __init__(self, environment, wid):
        self.s_dim, self.a_dim = environment.observation_space.shape[0], environment.action_space.shape[0]
        self.a_bound = environment.action_space.high
        self.wid = wid
        is_chief = wid == 0

        self.state = tf.placeholder(tf.float32, [None, self.s_dim], 'state')
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
        oldpi, oldpi_params, _, _ = self._build_anet(batch["state"], 'oldpi')
        pi, pi_params, self.pi_i_state, self.pi_f_state = self._build_anet(batch["state"], 'pi')
        evalpi, _, self.evalpi_i_state, self.evalpi_f_state = self._build_anet(self.state, 'pi', reuse=True, batch_size=1)

        self.v, vf_params, self.vf_i_state, self.vf_f_state = self._build_cnet(batch["state"], "vf")
        self.evalvf, _, self.evalvf_i_state, self.evalvf_f_state = self._build_cnet(self.state, 'vf', reuse=True, batch_size=1)

        self.sample_op = tf.squeeze(evalpi.sample(1), axis=0, name="sample_action")
        self.eval_action = evalpi.loc

        with tf.variable_scope('loss'):
            with tf.variable_scope('policy'):
                # Use floor functions for the probabilities to prevent NaNs when prob = 0
                ratio = pi.prob(batch["actions"]) / tf.maximum(oldpi.prob(batch["actions"]), 1e-6)
                surr1 = batch["advantage"] * ratio
                surr2 = batch["advantage"] * tf.clip_by_value(ratio, 1 - EPSILON, 1 + EPSILON)
                loss_pi = -tf.reduce_mean(tf.minimum(surr1, surr2))
                tf.summary.scalar("loss", loss_pi)
                # tf.summary.histogram("Ratio", ratio)

            with tf.variable_scope('value_function'):
                loss_vf = tf.reduce_mean(tf.square(self.v - batch["rewards"])) * 0.5
                tf.summary.scalar("loss", loss_vf)

            with tf.variable_scope('entropy_bonus'):
                entropy = pi.entropy()
                pol_entpen = -ENTROPY_BETA * tf.reduce_mean(entropy)

            self.loss = loss_pi + loss_vf * VF_COEFF + pol_entpen
            tf.summary.scalar("total", self.loss)

        with tf.variable_scope('train'):
            self.global_step = tf.train.get_or_create_global_step()
            opt = tf.train.AdamOptimizer(LR)
            opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=N_AGG, total_num_replicas=N_WORKER)

            grads, vs = zip(*opt.compute_gradients(self.loss, var_list=pi_params + vf_params))
            # Need to split the two networks so that clip_by_global_norm works properly
            pi_grads, pi_vs = grads[:len(pi_params)], vs[:len(pi_params)]
            vf_grads, vf_vs = grads[len(pi_params):], vs[len(pi_params):]
            pi_grads, _ = tf.clip_by_global_norm(pi_grads, 10.0)
            vf_grads, _ = tf.clip_by_global_norm(vf_grads, 10.0)
            self.train_op = opt.apply_gradients(zip(pi_grads + vf_grads, pi_vs + vf_vs), global_step=self.global_step)

            self.sync_replicas_hook = opt.make_session_run_hook(is_chief)

        with tf.variable_scope('update_old_pi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        tf.summary.scalar("Value", tf.reduce_mean(self.v))
        tf.summary.scalar("Sigma", tf.reduce_mean(pi.scale))
        self.summarise = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))

    def _build_anet(self, state_in, name, reuse=False, batch_size=MINIBATCH):
        w_reg = tf.contrib.layers.l2_regularizer(L2_REG)

        with tf.variable_scope(name, reuse=reuse):
            l1 = tf.layers.dense(state_in, 400, tf.nn.relu, kernel_regularizer=w_reg, name="pi_l1")
            l2 = tf.layers.dense(l1, LSTM_UNITS, tf.nn.relu, kernel_regularizer=w_reg, name="pi_l2")

            # LSTM layer
            a_lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units=LSTM_UNITS)
            a_lstm = tf.nn.rnn_cell.DropoutWrapper(a_lstm, output_keep_prob=self.keep_prob, seed=42)
            a_lstm = tf.nn.rnn_cell.MultiRNNCell(cells=[a_lstm] * LSTM_LAYERS)

            a_init_state = a_lstm.zero_state(batch_size=batch_size, dtype=tf.float32)
            lstm_in = tf.expand_dims(l2, axis=1)

            a_outputs, a_final_state = tf.nn.dynamic_rnn(cell=a_lstm, inputs=lstm_in, initial_state=a_init_state)
            a_cell_out = tf.reshape(a_outputs, [-1, LSTM_UNITS], name='flatten_lstm_outputs')

            mu = tf.layers.dense(a_cell_out, self.a_dim, tf.nn.tanh, kernel_regularizer=w_reg,
                                 name="pi_mu", use_bias=False)
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
            c_lstm = tf.nn.rnn_cell.DropoutWrapper(c_lstm, output_keep_prob=self.keep_prob, seed=42)
            c_lstm = tf.nn.rnn_cell.MultiRNNCell([c_lstm] * LSTM_LAYERS)

            c_init_state = c_lstm.zero_state(batch_size=batch_size, dtype=tf.float32)
            lstm_in = tf.expand_dims(l2, axis=1)

            c_outputs, c_final_state = tf.nn.dynamic_rnn(cell=c_lstm, inputs=lstm_in, initial_state=c_init_state)
            c_cell_out = tf.reshape(c_outputs, [-1, LSTM_UNITS], name='flatten_lstm_outputs')

            vf = tf.layers.dense(c_cell_out, 1, kernel_regularizer=w_reg, name="vf_out")
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return vf, params, c_init_state, c_final_state

    def update(self, exp, sess):
        start, e_time = time(), []
        sess.run([self.update_oldpi_op])

        for k in range(EPOCHS):
            for ep_s, ep_a, ep_r, ep_adv in exp:

                # Trim down to round minibatches
                trim_index = (ep_s.shape[0] // MINIBATCH) * MINIBATCH
                ep_s = ep_s[:trim_index]
                ep_a = ep_a[:trim_index]
                ep_r = ep_r[:trim_index]
                ep_adv = ep_adv[:trim_index]

                sess.run(self.iterator.initializer, feed_dict={self.state: ep_s, self.actions: ep_a,
                                                               self.rewards: ep_r, self.advantage: ep_adv})

                a_state, c_state = sess.run([self.pi_i_state, self.vf_i_state])
                train_ops = [self.summarise, self.global_step, self.pi_f_state, self.vf_f_state, self.train_op]

                while True:
                    try:
                        e_start = time()
                        feed_dict = {self.pi_i_state: a_state, self.vf_i_state: c_state, self.keep_prob: KEEP_PROB}
                        summary, step, a_state, c_state, _ = sess.run(train_ops, feed_dict=feed_dict)
                        e_time.append(time() - e_start)
                    except tf.errors.OutOfRangeError:
                        break
        print("Worker_%i Trained in %.3fs at %.3fs/batch. Global step %i" % (self.wid, time() - start, np.mean(e_time), step))
        return summary

    def eval_state(self, state, a_state, c_state, sess, stochastic=True):
        if stochastic:
            eval_ops = [self.sample_op, self.evalvf, self.evalpi_f_state, self.evalvf_f_state]
        else:
            eval_ops = [self.eval_action, self.evalvf, self.evalpi_f_state, self.evalvf_f_state]

        a, v, a_state, c_state = sess.run(eval_ops,
                                          {self.state: state[np.newaxis, :], self.evalpi_i_state: a_state,
                                           self.evalvf_i_state: c_state, self.keep_prob: 1.0})
        return a[0], v[0], a_state, c_state


def start_parameter_server(pid, spec):
    cluster = tf.train.ClusterSpec(spec)
    server = tf.train.Server(cluster, job_name="ps", task_index=pid)
    print("Starting PS #{}".format(pid))
    server.join()


class Worker(object):
    def __init__(self, wid, spec):
        self.wid = wid
        self.env = gym.make(ENVIRONMENT)

        print("Starting Worker #{}".format(wid))
        cluster = tf.train.ClusterSpec(spec)
        self.server = tf.train.Server(cluster, job_name="worker", task_index=wid)

        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % wid, cluster=cluster)):
            if self.wid == 0:
                self.env = wrappers.Monitor(self.env, os.path.join(SUMMARY_DIR, ENVIRONMENT), video_callable=None)
            self.ppo = PPO(self.env, self.wid)

    @staticmethod
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

    @staticmethod
    def discount(x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    def work(self):
        hooks = [self.ppo.sync_replicas_hook]
        sess = tf.train.MonitoredTrainingSession(master=self.server.target, is_chief=(self.wid == 0),
                                                 checkpoint_dir=SUMMARY_DIR,
                                                 save_summaries_steps=None, save_summaries_secs=None, hooks=hooks)
        if self.wid == 0:
            writer = SummaryWriterCache.get(SUMMARY_DIR)

        t, episode = 0, 0
        buffer_s, buffer_a, buffer_r, buffer_v, terminals = [], [], [], [], []
        experience = []

        while not sess.should_stop() and not (episode > EP_MAX and self.wid == 0):
            a_lstm_state, c_lstm_state = sess.run([self.ppo.evalpi_i_state, self.ppo.evalvf_i_state])
            s = self.env.reset()
            ep_r, ep_t, terminal = 0, 0, True
            ep_a = []

            while True:
                a, v, a_lstm_state, c_lstm_state = self.ppo.eval_state(s, a_lstm_state, c_lstm_state, sess)

                if ep_t > 0 and terminal:
                    # End of episode summary
                    print('{0:.1f}%'.format(episode / EP_MAX * 100), '| Worker_%i' % self.wid,
                          '| Episode: %i' % episode, "| Reward: %.2f" % ep_r, '| Steps: %i' % ep_t)

                    v = [v[0] * (1 - terminal)]  # v = 0 if terminal, otherwise use the predicted v
                    rewards = np.array(buffer_r)
                    vpred = np.array(buffer_v + v)
                    terminals_array = np.array(terminals + [0])

                    # Generalized Advantage Estimation - https://arxiv.org/abs/1506.02438
                    advantage = self.discount(rewards + GAMMA * vpred[1:] * (1 - terminals_array[1:]) - vpred[:-1], GAMMA * LAMBDA)
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
                        print("Worker_%i Training using %i episodes and %i steps..." % (self.wid, len(experience), t))

                        graph_summary = self.ppo.update(experience, sess)
                        t, experience = 0, []

                    if self.wid == 0:
                        worker_summary = tf.Summary()
                        worker_summary.value.add(tag="Reward", simple_value=ep_r)

                        # Create Action histograms for each dimension
                        actions = np.array(ep_a)
                        for a in range(self.ppo.a_dim):
                            self.add_histogram(writer, "Action/Dim" + str(a), actions[:, a], episode)

                        try:
                            writer.add_summary(graph_summary, episode)
                        except NameError:
                            pass
                        writer.add_summary(worker_summary, episode)
                        writer.flush()

                    episode += 1
                    break

                buffer_s.append(s)
                buffer_a.append(a)
                buffer_v.append(v[0])
                terminals.append(terminal)
                ep_a.append(a)

                s, r, terminal, _ = self.env.step(np.clip(a, -self.ppo.a_bound, self.ppo.a_bound))
                buffer_r.append(r)

                ep_r += r
                ep_t += 1
                t += 1

        self.env.close()
        print("Worker_%i finished" % self.wid)


def main(_):
    pass


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_name', action='store', dest='job_name', help='Either "ps" or "worker"')
    parser.add_argument('--task_index', action='store', dest='task_index', help='ID number of the job')
    parser.add_argument('--timestamp', action='store', dest='timestamp', help='Timestamp for output directory')
    parser.add_argument('--workers', action='store', dest='n_workers', help='Number of workers')
    parser.add_argument('--agg', action='store', dest='n_agg', help='Number of gradients to aggregate')
    parser.add_argument('--ps', action='store', dest='ps', help='Number of parameter servers')
    args = parser.parse_args()

    N_WORKER = int(args.n_workers)
    N_AGG = int(args.n_agg)
    PS = int(args.ps)
    TIMESTAMP = args.timestamp
    SUMMARY_DIR = os.path.join(OUTPUT_RESULTS_DIR, "DPPO_LSTM", ENVIRONMENT, TIMESTAMP)

    if PS == 0:
        spec = {"worker": ["localhost:" + str(2222 + PS + i + PORT_OFFSET) for i in range(N_WORKER)]}
    else:
        spec = {"ps": ["localhost:" + str(2222 + i + PORT_OFFSET) for i in range(PS)],
                "worker": ["localhost:" + str(2222 + PS + i + PORT_OFFSET) for i in range(N_WORKER)]}

    if args.job_name == "ps":
        tf.app.run(start_parameter_server(int(args.task_index), spec))
    elif args.job_name == "worker":
        w = Worker(int(args.task_index), spec)
        tf.app.run(w.work())
