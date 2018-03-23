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
VF_COEFF = 1.0
L2_REG = 0.001
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

        self.dataset = tf.data.Dataset.from_tensor_slices({"state": self.state, "actions": self.actions,
                                                           "rewards": self.rewards, "advantage": self.advantage})
        self.dataset = self.dataset.shuffle(buffer_size=10000)
        self.dataset = self.dataset.batch(MINIBATCH)
        self.dataset = self.dataset.cache()
        self.dataset = self.dataset.repeat(EPOCHS)
        self.iterator = self.dataset.make_initializable_iterator()
        batch = self.iterator.get_next()

        # Create an old & new policy function but also
        # make separate value & policy functions for evaluation & training (with shared variables)
        oldpi, oldpi_params = self._build_anet(batch["state"], 'oldpi')
        pi, pi_params = self._build_anet(batch["state"], 'pi')
        evalpi, _ = self._build_anet(self.state, 'pi', reuse=True)

        self.v, vf_params = self._build_cnet(batch["state"], "vf")
        self.evalvf, _ = self._build_cnet(self.state, 'vf', reuse=True)

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
            # pi_grads, _ = tf.clip_by_global_norm(pi_grads, 10.0)
            # vf_grads, _ = tf.clip_by_global_norm(vf_grads, 10.0)
            self.train_op = opt.apply_gradients(zip(pi_grads + vf_grads, pi_vs + vf_vs), global_step=self.global_step)

            self.sync_replicas_hook = opt.make_session_run_hook(is_chief)

        with tf.variable_scope('update_old_pi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        tf.summary.scalar("Value", tf.reduce_mean(self.v))
        tf.summary.scalar("Sigma", tf.reduce_mean(pi.scale))
        self.summarise = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))

    def _build_anet(self, state_in, name, reuse=False):
        w_reg = tf.contrib.layers.l2_regularizer(L2_REG)

        with tf.variable_scope(name, reuse=reuse):
            l1 = tf.layers.dense(state_in, 400, tf.nn.relu, kernel_regularizer=w_reg, name="pi_l1")
            l2 = tf.layers.dense(l1, 400, tf.nn.relu, kernel_regularizer=w_reg, name="pi_l2")
            mu = tf.layers.dense(l2, self.a_dim, tf.nn.tanh, kernel_regularizer=w_reg, name="pi_mu")
            sigma = tf.get_variable(name="pi_sigma", shape=self.a_dim, initializer=tf.ones_initializer())
            norm_dist = tf.distributions.Normal(loc=mu * self.a_bound, scale=tf.maximum(sigma, 0.1))
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def _build_cnet(self, state_in, name, reuse=False):
        w_reg = tf.contrib.layers.l2_regularizer(L2_REG)

        with tf.variable_scope(name, reuse=reuse):
            l1 = tf.layers.dense(state_in, 400, tf.nn.relu, kernel_regularizer=w_reg, name="vf_l1")
            l2 = tf.layers.dense(l1, 400, tf.nn.relu, kernel_regularizer=w_reg, name="vf_l2")
            vf = tf.layers.dense(l2, 1, kernel_regularizer=w_reg, name="vf_output")

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return vf, params

    def update(self, s, a, r, adv, sess):
        start = time()
        e_time = []

        sess.run([self.update_oldpi_op, self.iterator.initializer],
                 feed_dict={self.state: s, self.actions: a, self.rewards: r, self.advantage: adv})

        while True:
            try:
                e_start = time()
                summary, step, _ = sess.run([self.summarise, self.global_step, self.train_op])
                e_time.append(time() - e_start)
            except tf.errors.OutOfRangeError:
                break
        print("Worker_%i Trained in %.3fs at %.3fs/batch. Global step %i" % (self.wid, time() - start, np.mean(e_time), step))
        return summary

    def eval_state(self, state, sess, stochastic=True):
        if stochastic:
            action, value = sess.run([self.sample_op, self.evalvf], {self.state: state[np.newaxis, :]})
        else:
            action, value = sess.run([self.eval_action, self.evalvf], {self.state: state[np.newaxis, :]})
        return action[0], value[0]


def start_parameter_server(pid, spec):
    cluster = tf.train.ClusterSpec(spec)
    server = tf.train.Server(cluster, job_name="ps", task_index=pid)
    print("Starting PS #{}".format(pid))
    server.join()


# class StopAtEpisodeHook(tf.train.SessionRunHook):
#     def __init__(self, max_episodes, worker):
#         self.ep_max = max_episodes
#         self.wid = worker
#
#     def after_run(self, run_context, run_values):
#         global episode
#         if episode >= self.ep_max and self.wid == 0:
#             print("Reached %i episodes" % episode)
#             run_context.request_stop()


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

        while not sess.should_stop() and not (episode > EP_MAX and self.wid == 0):
            s = self.env.reset()
            ep_r, ep_t, terminal = 0, 0, True
            ep_a = []

            while True:
                a, v = self.ppo.eval_state(s, sess)

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

                    bs, ba, br, badv = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(tdlamret), np.vstack(advantage)

                    # Update PPO
                    if t >= BATCH:
                        print("Worker_%i Training using %i steps..." % (self.wid, t))
                        graph_summary = self.ppo.update(bs, ba, br, badv, sess)
                        buffer_s, buffer_a, buffer_r, buffer_v, terminals = [], [], [], [], []
                        t = 0

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
    SUMMARY_DIR = os.path.join(OUTPUT_RESULTS_DIR, "DPPO", ENVIRONMENT, TIMESTAMP)

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
