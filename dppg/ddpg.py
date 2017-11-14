"""
Implementation of DDPG - Deep Deterministic Policy Gradient
Algorithm and hyperparameter details can be found here: http://arxiv.org/pdf/1509.02971v2.pdf
Variance scaling paper: https://arxiv.org/pdf/1502.01852v1.pdf
Thanks to GitHub users yanpanlau, pemami4911, songrotek and JunhongXu for their DDPG examples

Batch normalisation on the actor accelerates learning but has poor long term stability. Applying to the critic breaks
it, particularly on the state branch. Not sure why but I think this issue is specific to this environment 
"""
import numpy as np
import tensorflow as tf
import tflearn
import random
import os
import pickle
import gym
from gym import wrappers
from time import time
from datetime import datetime
from collections import deque
OUTPUT_RESULTS_DIR = "./"
# OUTPUT_RESULTS_DIR = os.pardir

# ==========================
#   Training Parameters
# ==========================
# Max training steps
MAX_EPISODES = 1000
# Max episode length
MAX_EP_STEPS = 2000
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001  # Paper uses 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001  # Paper uses 0.001
# L2 weight decay for Q
L2_DECAY = 0.01  # Paper uses 0.01
# Discount factor
GAMMA = 0.99  # Paper uses 0.99
# Soft target update param
TAU = 0.001  # Paper uses 0.001
# Exploration parameters
OU_MU = 0.0
OU_THETA = 0.15  # Paper uses 0.15
OU_SIGMA = 0.20  # Paper uses 0.20
TAU2 = 25
RESTORE_DATE = None
# RESTORE_DATE = "20170415-213317"

# ===========================
#   Utility Parameters
# ===========================
# Directory for storing TensorBoard summary results
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 1000000
MINIBATCH_SIZE = 64
ENVIRONMENT = 'Pendulum-v0'
# ENVIRONMENT = 'MountainCarContinuous-v0'
# ENVIRONMENT = 'SemisuperPendulumNoise-v0'
# ENVIRONMENT = 'SemisuperPendulumRandom-v0'
# ENVIRONMENT = 'SemisuperPendulumDecay-v0'
# ENVIRONMENT = 'BipedalWalker-v2'
# ENVIRONMENT = 'BipedalWalkerHardcore-v2'
# ENVIRONMENT = 'LunarLanderContinuous-v2'
if RESTORE_DATE is not None:
    SUMMARY_DIR = os.path.join(OUTPUT_RESULTS_DIR, 'DDPG', "gym", ENVIRONMENT, RESTORE_DATE)
else:
    TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
    SUMMARY_DIR = os.path.join(OUTPUT_RESULTS_DIR, "DDPG", "gym", ENVIRONMENT, TIMESTAMP)


# ===========================
#   Replay Buffer
# ===========================
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


# ===========================
#   Ornstein-Uhlenbeck noise
# ===========================
class OUNoise:
    """docstring for OUNoise"""
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


# ===========================
#   Actor and Critic DNNs
# ===========================
# TODO: Remove the tflearn dependency and replace with pure TensorFlow
class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, restore=False):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau

        self.is_training = tf.placeholder(tf.bool, name='Actor_is_training')

        if not restore:
            # Actor network
            self.inputs, self.outputs, self.scaled_outputs = self.create_actor_network()
            self.net_params = tf.trainable_variables()  # Returns a list of Variables where trainable=True

            # Target network
            self.target_inputs, self.target_outputs, self.target_scaled_outputs = self.create_actor_network("_target")
            self.target_net_params = tf.trainable_variables()[len(self.net_params):]

            # Temporary placeholder action gradient - this gradient will be provided by the critic network
            self.action_gradients = tf.placeholder(tf.float32, [None, self.action_dim], name="actor_action_gradient")

            # Combine dnetScaledOut/dnetParams with criticToActionGradient to get actorGradient
            self.actor_gradients = tf.gradients(self.scaled_outputs, self.net_params, -self.action_gradients,
                                                name="actor_gradient")

            self.optimize = tf.train.AdamOptimizer(self.learning_rate, name='Adam_Actor'). \
                apply_gradients(zip(self.actor_gradients, self.net_params))

            tf.add_to_collection('Actor_action_gradients', self.action_gradients)
            tf.add_to_collection('Actor_optimize', self.optimize)
        else:
            # Load Actor network
            self.inputs, self.out, self.scaled_outputs = self.load_actor_network()
            # Filter the loaded trainable variables for those belonging only to the actor network
            self.net_params = [v for v in tf.trainable_variables() if "actor" in v.name and "target" not in v.name]

            # Load Target network
            self.target_inputs, self.target_outputs, self.target_scaled_outputs = self.load_actor_network(True)
            # Filter the loaded trainable variables for those belonging only to the target actor network
            self.target_net_params = [v for v in tf.trainable_variables() if "actor" in v.name and "target" in v.name]

            self.action_gradients = tf.get_collection('Actor_action_gradients')[0]
            self.optimize = tf.get_collection('Actor_optimize')[0]

        # Op for periodically updating target network with online network weights
        self.update_target_net_params = \
            [self.target_net_params[i].assign(tf.multiply(self.net_params[i], self.tau) +
                                              tf.multiply(self.target_net_params[i], 1. - self.tau))
             for i in range(len(self.target_net_params))]

        self.num_trainable_vars = len(self.net_params) + len(self.target_net_params)

    def create_actor_network(self, suffix=""):
        state = tflearn.input_data(shape=[None, self.state_dim], name='actor_input'+suffix)
        # state_bn = tf.layers.batch_normalization(state, training=self.is_training, scale=False,
        #                                          name='actor_BN_input'+suffix)
        net = tflearn.fully_connected(state, 400, activation='relu', name='actor_L1'+suffix,
                                      weights_init=tflearn.initializations.variance_scaling(seed=RANDOM_SEED))
        if suffix == "":
            tf.summary.histogram("Actor/Layer1", net.W)
        # net = tf.layers.batch_normalization(net, training=self.is_training, scale=False,
        #                                     name='actor_BN1'+suffix)
        net = tflearn.fully_connected(net, 300, activation='relu', name='actor_L2'+suffix,
                                      weights_init=tflearn.initializations.variance_scaling(seed=RANDOM_SEED))
        if suffix == "":
            tf.summary.histogram("Actor/Layer2", net.W)
        # net = tf.layers.batch_normalization(net, training=self.is_training, scale=True,
        #                                     name='actor_BN2'+suffix)
        # Final layer weights are initialized to Uniform[-3e-3, 3e-3]
        weight_init_final = tflearn.initializations.uniform(minval=-0.003, maxval=0.003, seed=RANDOM_SEED)
        action = tflearn.fully_connected(net, self.action_dim, activation='tanh', weights_init=weight_init_final,
                                         name='actor_output'+suffix)
        # Scale output to [-action_bound, action_bound]
        scaled_action = tf.multiply(action, self.action_bound, name='actor_output_scaled'+suffix)
        return state, action, scaled_action

    @staticmethod
    def load_actor_network(target=False):
        suffix = "_target" if target else ""
        inputs = tf.get_default_graph().get_tensor_by_name("actor_input"+suffix+"/X:0")
        out = tf.get_default_graph().get_tensor_by_name("actor_output"+suffix+"/Tanh:0")
        scaled_out = tf.get_default_graph().get_tensor_by_name("actor_output_scaled"+suffix+":0")
        return inputs, out, scaled_out

    def train(self, inputs, action_gradients):
        # Extra ops for BN. Parameters associated with the target network are ignored
        extra_update_ops = [v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if
                            "actor" in v.name and "target" not in v.name]
        return self.sess.run([self.optimize, extra_update_ops],
                             feed_dict={self.inputs: inputs, self.action_gradients: action_gradients,
                                        self.is_training: True})

    def predict(self, inputs):
        return self.sess.run(self.scaled_outputs, feed_dict={self.inputs: inputs,
                                                             self.is_training: False})

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_outputs, feed_dict={self.target_inputs: inputs,
                                                                    self.is_training: False})

    def update_target_network(self):
        self.sess.run(self.update_target_net_params, feed_dict={self.is_training: False})

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, restore=False):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau

        self.is_training = tf.placeholder(tf.bool, name='Critic_is_training')

        if not restore:
            # Create the Critic network
            self.inputs, self.action, self.outputs = self.create_critic_network()
            self.network_params = [v for v in tf.trainable_variables() if "critic" in v.name]

            # Create the Target Network
            self.target_inputs, self.target_action, self.target_outputs = self.create_critic_network("_target")
            self.target_network_params = [v for v in tf.trainable_variables() if
                                          "critic" in v.name and "target" in v.name]

            # Network target (y_i) - Obtained from the target networks
            self.q_value = tf.placeholder(tf.float32, [None, self.action_dim], name="critic_q_value")
            self.L2 = tf.add_n([L2_DECAY * tf.nn.l2_loss(v) for v in self.network_params if "/W" in v.name])
            self.loss = tf.losses.mean_squared_error(self.q_value, self.outputs) + self.L2
            self.optimize = tf.train.AdamOptimizer(self.learning_rate, name='Adam_Critic').minimize(self.loss)

            tf.add_to_collection('Critic_q_value', self.q_value)
            tf.add_to_collection('Critic_loss', self.loss)
            tf.add_to_collection('Critic_optimize', self.optimize)

        else:
            # Load the Critic network
            self.inputs, self.action, self.outputs = self.load_critic_network()
            # Filter the loaded trainable variables for those belonging only to the critic network
            self.network_params = [v for v in tf.trainable_variables() if "critic" in v.name and "target" not in v.name]

            # Load the Target Network
            self.target_inputs, self.target_action, self.target_outputs = self.load_critic_network(True)
            # Filter the loaded trainable variables for those belonging only to the target critic network
            self.target_network_params = [v for v in tf.trainable_variables() if
                                          "critic" in v.name and "target" in v.name]

            self.predicted_q_value = tf.get_collection('Critic_predicted_q_value')[0]
            self.L2 = tf.add_n([L2_DECAY * tf.nn.l2_loss(v) for v in self.network_params if "/W" in v.name])
            self.loss = tf.get_collection('Critic_loss')[0] + self.L2
            self.optimize = tf.get_collection('Critic_optimize')[0]

        # Op for periodically updating target network with online network weights
        self.update_target_net_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # Get the gradient of the critic w.r.t. the action
        self.action_grads = tf.gradients(self.outputs, self.action, name="critic_action_gradient")
        tf.summary.scalar("L2", self.L2)

    def create_critic_network(self, suffix=""):
        # Critic breaks when BN is added to the state in Pendulum-v0. Not sure why :(
        state = tflearn.input_data(shape=[None, self.state_dim], name="critic_input_state"+suffix)
        # state_bn = tf.layers.batch_normalization(state, training=self.is_training, scale=False,
        #                                          name='critic_BN_input'+suffix)
        action = tflearn.input_data(shape=[None, self.action_dim], name="critic_input_action"+suffix)
        # action_bn = tf.layers.batch_normalization(action, training=self.is_training, scale=False,
        #                                           name='critic_BN_action'+suffix)
        net = tflearn.fully_connected(state, 400, activation='relu', name='critic_L1'+suffix,
                                      weights_init=tflearn.initializations.variance_scaling(seed=RANDOM_SEED))
        if suffix == "":
            tf.summary.histogram("Critic/Layer1", net.W)
        # net = tf.layers.batch_normalization(net, training=self.is_training, scale=False,
        #                                     name='critic_BN1'+suffix)
        # Add the action tensor in the 2nd hidden layer and create variables for W's and b
        s_union = tflearn.fully_connected(net, 300, name="critic_L2_state" + suffix,
                                          weights_init=tflearn.initializations.variance_scaling(seed=RANDOM_SEED))
        a_union = tflearn.fully_connected(action, 300, name="critic_L2_action" + suffix,
                                          weights_init=tflearn.initializations.variance_scaling(seed=RANDOM_SEED))
        net = tf.nn.relu(tf.matmul(net, s_union.W) + tf.matmul(action, a_union.W) + s_union.b,
                         name='critic_L2' + suffix)
        if suffix == "":
            tf.summary.histogram("Critic/Layer2/state", s_union.W)
            tf.summary.histogram("Critic/Layer2/action", a_union.W)

        # Linear layer connected to action_dim outputs representing Q(s,a). Weights are init to Uniform[-3e-3, 3e-3]
        weight_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003, seed=RANDOM_SEED)
        q_value = tflearn.fully_connected(net, self.action_dim, activation="linear",
                                          weights_init=weight_init, name='critic_output'+suffix)
        return state, action, q_value

    @staticmethod
    def load_critic_network(target=False):
        suffix = "_target" if target else ""
        inputs = tf.get_default_graph().get_tensor_by_name("critic_input_state"+suffix+"/X:0")
        action = tf.get_default_graph().get_tensor_by_name("critic_input_action"+suffix+"/X:0")
        out = tf.get_default_graph().get_tensor_by_name("critic_output"+suffix+"/BiasAdd:0")
        return inputs, action, out

    def train(self, inputs, action, target_q_value):
        # Extra ops for BN. Parameters associated with the target network are ignored
        extra_update_ops = [v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if
                            "critic" in v.name and "target" not in v.name]
        return self.sess.run([self.optimize, self.loss, extra_update_ops], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.q_value: target_q_value,
            self.is_training: True
        })[:2]

    def predict(self, inputs, action):
        return self.sess.run(self.outputs, feed_dict={self.inputs: inputs, self.action: action,
                                                      self.is_training: False})

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_outputs, feed_dict={self.target_inputs: inputs, self.target_action: action,
                                                             self.is_training: False})

    def action_gradients(self, inputs, action):
        return self.sess.run(self.action_grads, feed_dict={self.inputs: inputs, self.action: action,
                                                           self.is_training: False})

    def update_target_network(self):
        self.sess.run(self.update_target_net_params, feed_dict={self.is_training: False})


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


# ===========================
#   Agent Training
# ===========================
def train(sess, env, actor, critic, saver, replay_buffer):
    """
    Train Actor and Critic networks and save checkpoints
    :param sess: TensorFlow session
    :param env: environment to be used for training
    :param actor: Actor network
    :param critic: Critic network
    :param saver: TensorFlow saver object
    :param replay_buffer: Replay buffer to store experience
    :return:
    """
    # Initialise variables
    if RESTORE_DATE is None:
        sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

    # Initialize target network weights & noise function
    actor.update_target_network()
    critic.update_target_network()

    for i in range(MAX_EPISODES):
        start = time()
        ep_rewards = []
        ep_q_rmse = []
        ep_action_dist = []
        ep_loss = []
        env.seed(RANDOM_SEED + i)
        s = env.reset()

        exploration_noise = OUNoise(actor.action_dim, OU_MU, OU_THETA, OU_SIGMA, RANDOM_SEED + i)

        while True:
            # if i > 100:
            # env.render()

            a = actor.predict(np.reshape(s, (1, actor.state_dim)))  # Reshape state into a column

            # Add exploration noise
            if RESTORE_DATE is None:
                epsilon = np.exp(-i/TAU2)
                a += epsilon * exploration_noise.noise() / env.action_space.high
            else:
                epsilon = 0

            # Step forward in the environment
            a = np.clip(a, env.action_space.low, env.action_space.high)
            s2, r, terminal, info = env.step(a[0])
            ep_action_dist.append(a[0])

            replay_buffer.add(np.reshape(s, (actor.state_dim,)),  # Previous state
                              np.reshape(a, (actor.action_dim,)),  # Action
                              r,  # Reward
                              terminal,  # Terminal state (bool)
                              np.reshape(s2, (actor.state_dim,)))  # New state

            # Keep adding experience to the memory until there are at least 50 episodes of samples before training
            if replay_buffer.size() > MINIBATCH_SIZE:
                s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(MINIBATCH_SIZE)

                # Calculate target q
                target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))
                ep_q_rmse.append(np.sqrt(np.mean((target_q - r_batch) ** 2, axis=0)))
                y = r_batch + GAMMA * target_q * ~t_batch

                # Update the critic given the targets
                _, loss = critic.train(s_batch, a_batch, y)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                # grads = np.clip(grads, -1, 1)  # Gradient clipping to prevent exploding gradients
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()
            else:
                loss = 0

            s = s2  # new state is the output from this step
            ep_rewards.append(r)
            ep_loss.append(loss)

            if terminal:
                # Add results to summaries
                episode_summary = tf.Summary()
                episode_summary.value.add(tag="Reward", simple_value=np.sum(ep_rewards))
                episode_summary.value.add(tag="Q_RMSE", simple_value=np.mean(ep_q_rmse))
                episode_summary.value.add(tag="Epsilon", simple_value=epsilon)
                episode_summary.value.add(tag="Loss", simple_value=loss)

                # Hack to add histograms
                add_histogram(writer, "Actions", np.ravel(ep_action_dist), i)
                add_histogram(writer, "Rewards", np.array(ep_rewards), i)

                summary_str = sess.run(tf.summary.merge_all())
                writer.add_summary(episode_summary, i)
                writer.add_summary(summary_str, i)
                writer.flush()
                print('Reward: %.2f' % np.sum(ep_rewards), '\t Episode', i,
                      '\tQ RMSE: %.2f' % np.mean(ep_q_rmse),
                      '\tTime: %.1f' % (time() - start),
                      '\tEpsilon: %.3f' % epsilon,
                      '\tLoss: %.3f' % np.mean(ep_loss)),
                exploration_noise.reset()
                break

        # Save model every 50 steps
        if i % 50 == 0 and i != 0:
            save_start = time()
            save_path = saver.save(sess, os.path.join(SUMMARY_DIR, "ddpg_model"))
            pickle.dump(replay_buffer, open(os.path.join(SUMMARY_DIR, "replay_buffer.pkl"), "wb"))
            print("Model saved in %.1f" % (time() - save_start), "seconds. Path: %s" % save_path)


def main(_):
    # Need to split actor & critic into different graphs/sessions to prevent serialisation errors
    # See https://github.com/tflearn/tflearn/issues/381
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.05
    with tf.Session(config=config) as sess:
        env = gym.make(ENVIRONMENT)
        # env = wrappers.Monitor(env, os.path.join(SUMMARY_DIR, ENVIRONMENT+'-experiment'), force=True)
        tf.set_random_seed(RANDOM_SEED)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high
        # Ensure action bound is symmetric
        # assert (env.action_space.high == -env.action_space.low)

        # Restore networks and replay buffer from disk otherwise create new ones
        if RESTORE_DATE is not None:
            saver = tf.train.import_meta_graph(os.path.join(SUMMARY_DIR, "ddpg_model.meta"))
            saver.restore(sess, os.path.join(SUMMARY_DIR, "ddpg_model"))
            actor = ActorNetwork(sess, state_dim, action_dim, action_bound, ACTOR_LEARNING_RATE, TAU, restore=True)
            critic = CriticNetwork(sess, state_dim, action_dim, CRITIC_LEARNING_RATE, TAU, restore=True)

            # Initialise the uninitialised variables
            uninitialized_vars = []
            for var in tf.global_variables():
                try:
                    sess.run(var)
                except tf.errors.FailedPreconditionError:
                    uninitialized_vars.append(var)
            sess.run(tf.variables_initializer(uninitialized_vars))

            replay_buffer = pickle.load(open(os.path.join(SUMMARY_DIR, "replay_buffer.pkl"), "rb"))

            print("Model restored from %s" % os.path.join(SUMMARY_DIR, "ddpg_model"))
        else:
            actor = ActorNetwork(sess, state_dim, action_dim, action_bound, ACTOR_LEARNING_RATE, TAU)
            critic = CriticNetwork(sess, state_dim, action_dim, CRITIC_LEARNING_RATE, TAU)
            saver = tf.train.Saver()
            replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

        # Start training given a session, environment, actor & critic
        train(sess, env, actor, critic, saver, replay_buffer)
        env.close()
        # gym.upload(os.path.join(SUMMARY_DIR, ENVIRONMENT+'-experiment'), api_key='XXXXXXXXXXXXXXXXXXXXXX')


if __name__ == '__main__':
    # Quick wrapper that handles flag parsing and then dispatches to your own main
    # http://stackoverflow.com/questions/33703624/how-does-tf-app-run-work
    tf.app.run()
