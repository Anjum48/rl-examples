"""
Implementation of DDPG - Deep Deterministic Policy Gradient
Algorithm and hyperparameter details can be found here: http://arxiv.org/pdf/1509.02971v2.pdf
Variance scaling paper: https://arxiv.org/pdf/1502.01852v1.pdf
Thanks to GitHub users yanpanlau, pemami4911, songrotek and JunhongXu for their DDPG examples

Batch normalisation on the actor accelerates learning but has poor long term stability. Applying to the critic breaks
it, particularly on the state branch. Not sure why but I think this issue might be specific to Pendulum-v0
"""
import numpy as np
import tensorflow as tf
import random
import os
import gym
from gym import wrappers
from time import time
from datetime import datetime
from collections import deque
OUTPUT_RESULTS_DIR = "./"

# ==========================
#   Training Parameters
# ==========================
MAX_EPISODES = 1000  # Max training steps
LR_A = 0.0001  # Actor learning rate - Paper uses 0.0001
LR_C = 0.001  # Critic learning rate - Paper uses 0.001
L2_DECAY = 0.01  # L2 weight decay for Q - Paper uses 0.01
GAMMA = 0.99  # Discount factor - Paper uses 0.99
TAU = 0.001  # Soft target update param - Paper uses 0.001

# Exploration noise parameters
OU_MU = 0.0
OU_THETA = 0.15  # Paper uses 0.15
OU_SIGMA = 0.20  # Paper uses 0.20
TAU2 = 25

BUFFER_SIZE = 1000000  # Size of replay buffer
MINIBATCH_SIZE = 64

ENVIRONMENT = 'Pendulum-v0'
# ENVIRONMENT = 'MountainCarContinuous-v0'
# ENVIRONMENT = 'SemisuperPendulumNoise-v0'
# ENVIRONMENT = 'SemisuperPendulumRandom-v0'
# ENVIRONMENT = 'SemisuperPendulumDecay-v0'
# ENVIRONMENT = 'BipedalWalker-v2'
# ENVIRONMENT = 'BipedalWalkerHardcore-v2'
# ENVIRONMENT = 'LunarLanderContinuous-v2'

TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
SUMMARY_DIR = os.path.join(OUTPUT_RESULTS_DIR, "DDPG", "gym", ENVIRONMENT, TIMESTAMP)


class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=None):
        """
        The right side of the deque contains the most recent experiences
        The buffer stores a number of past experiences to stochastically sample from
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque(maxlen=self.buffer_size)
        self.seed = random_seed
        if self.seed is not None:
            random.seed(self.seed)

    def add(self, state, action, reward, t, s2):
        experience = (state, action, reward, t, s2)
        self.buffer.append(experience)
        self.count += 1

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch]).reshape(batch_size, -1)
        t_batch = np.array([_[3] for _ in batch]).reshape(batch_size, -1)
        s2_batch = np.array([_[4] for _ in batch])
        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0


class OUNoise:
    """Ornstein-Uhlenbeck noise"""
    def __init__(self, action_dimension, mu=0.0, theta=0.15, sigma=0.3, seed=123):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()
        np.random.seed(seed)

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class DDPG:
    def __init__(self, environment):
        self.s_dim, self.a_dim = environment.observation_space.shape[0], environment.action_space.shape[0]
        self.a_bound = environment.action_space.high

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        config = tf.ConfigProto(log_device_placement=False, device_count={'GPU': 0})  # 1 for GPU, 0 for CPU
        self.sess = tf.Session(config=config)
        self.state = tf.placeholder(tf.float32, [None, self.s_dim], 'state')
        self.action = tf.placeholder(tf.float32, [None, self.a_dim], 'action')
        self.y_t = tf.placeholder(tf.float32, [None, self.a_dim], name="y_t")  # Result from equation 5 in paper

        # Create base and target networks
        self.mu, mu_params = self.create_actor_network("mu")
        self.mu_target, mu_target_params = self.create_actor_network("mu_target")
        self.q, q_params = self.create_critic_network("q")
        self.q_target, q_target_params = self.create_critic_network("q_target")

        # Ops for updating the target networks
        with tf.variable_scope('update_target'):
            self.update_mu = [pt.assign(p * TAU + pt * (1-TAU)) for p, pt in zip(mu_params, mu_target_params)]
            self.update_q = [pt.assign(p * TAU + pt * (1-TAU)) for p, pt in zip(q_params, q_target_params)]

        with tf.variable_scope('train'):
            loss_q = tf.losses.mean_squared_error(self.y_t, self.q)
            tf.summary.scalar("loss_critic", loss_q)
            # TODO: Add gradient clipping
            policy_gradient = tf.gradients(self.q, self.action)[0]
            actor_gradient = tf.gradients(self.mu, mu_params, -policy_gradient)

            self.train_critic = tf.train.AdamOptimizer(LR_C).minimize(loss_q)
            self.train_actor = tf.train.AdamOptimizer(LR_A).apply_gradients(zip(actor_gradient, mu_params))

        self.writer = tf.summary.FileWriter(SUMMARY_DIR, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.summarise = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))

    def create_actor_network(self, name):
        with tf.variable_scope(name):
            # TODO: Add L2 regularization
            l1 = tf.layers.dense(self.state, 400, tf.nn.relu, name='actor_L1')
            l2 = tf.layers.dense(l1, 400, tf.nn.relu, name='actor_L2')

            w_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
            action = tf.layers.dense(l2, self.a_dim, tf.nn.tanh, kernel_initializer=w_init, name='actor_output')
            scaled_action = tf.multiply(action, self.a_bound, name='actor_output_scaled')

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return scaled_action, params

    def create_critic_network(self, name):
        with tf.variable_scope(name):
            # TODO: Add L2 regularization
            net = tf.layers.dense(self.state, 400, tf.nn.relu, name='critic_L1')
            initializer = tf.variance_scaling_initializer()
            s_union_weights = tf.Variable(initializer.__call__([400, 300]), name='critic_L2_Ws')
            a_union_weights = tf.Variable(initializer.__call__([self.a_dim, 300]), name='critic_L2_Wa')
            union_biases = tf.Variable(tf.zeros([300]), name='critic_L2_b')

            net = tf.nn.relu(tf.matmul(net, s_union_weights) +
                             tf.matmul(self.action, a_union_weights) + union_biases,
                             name='critic_L2')

            w_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
            value = tf.layers.dense(net, self.a_dim, kernel_initializer=w_init, name='critic_output')
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return value, params

    def update_target(self):
        self.sess.run([self.update_mu, self.update_q])

    def train(self, state, action, y):
        return self.sess.run([self.train_critic, self.train_actor, self.summarise],
                             {self.state: state, self.action: action, self.y_t: y})[2]

    def predict_action(self, state, target=False):
        if target:
            return self.sess.run(self.mu_target, feed_dict={self.state: state})
        else:
            return self.sess.run(self.mu, feed_dict={self.state: state})

    def predict_value(self, state, action, target=False):
        if target:
            return self.sess.run(self.q_target, feed_dict={self.state: state, self.action: action})
        else:
            return self.sess.run(self.q, feed_dict={self.state: state, self.action: action})


# ===========================
#   TensorFlow Summary Ops
# ===========================
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


def main(_):
    env = gym.make(ENVIRONMENT)
    env = wrappers.Monitor(env, os.path.join(SUMMARY_DIR, ENVIRONMENT), video_callable=None)
    ddpg = DDPG(env)

    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    for i in range(MAX_EPISODES):
        start = time()
        ep_rewards = []
        ep_action_dist = []
        s = env.reset()

        exploration_noise = OUNoise(ddpg.a_dim, OU_MU, OU_THETA, OU_SIGMA)

        while True:
            if i > 100:
                env.render()

            a = ddpg.predict_action(s[np.newaxis, :])

            # Add exploration noise and decay exponentially using epsilon
            epsilon = np.exp(-i/TAU2)
            a += epsilon * exploration_noise.noise() / env.action_space.high

            # Step forward in the environment
            a = np.clip(a, env.action_space.low, env.action_space.high)
            s2, r, terminal, info = env.step(a[0])
            ep_action_dist.append(a[0])

            # Add to the replay buffer: previous state, action, reward, terminal state (bool), new state
            replay_buffer.add(np.hstack(s), np.hstack(a), r, terminal, np.hstack(s2))

            if replay_buffer.size() > MINIBATCH_SIZE:
                s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(MINIBATCH_SIZE)

                # Calculate target q
                target_q = ddpg.predict_value(s2_batch, ddpg.predict_action(s2_batch, target=True), target=True)
                y = r_batch + GAMMA * target_q * ~t_batch

                # Update the networks
                graph_summary = ddpg.train(s_batch, a_batch, y)
                ddpg.update_target()

            s = s2  # new state is the output from this step
            ep_rewards.append(r)

            if terminal:
                # Add results to summaries
                episode_summary = tf.Summary()
                episode_summary.value.add(tag="Reward", simple_value=np.sum(ep_rewards))
                episode_summary.value.add(tag="Epsilon", simple_value=epsilon)
                add_histogram(ddpg.writer, "Actions", np.ravel(ep_action_dist), i)
                add_histogram(ddpg.writer, "Rewards", np.array(ep_rewards), i)

                ddpg.writer.add_summary(episode_summary, i)
                ddpg.writer.add_summary(graph_summary, i)
                ddpg.writer.flush()

                print('Episode', i, '\tReward: %.2f' % np.sum(ep_rewards), '\tTime: %.1fs' % (time() - start),
                      '\tEpsilon: %.3f' % epsilon)
                exploration_noise.reset()
                break
    env.close()


if __name__ == '__main__':
    tf.app.run()
