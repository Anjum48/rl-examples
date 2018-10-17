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
from utils import RunningStats, discount, add_histogram
OUTPUT_RESULTS_DIR = "./"


EP_MAX = 10000
GAMMA = 0.99
LAMBDA = 0.95
ENTROPY_BETA = 0.01  # 0.01 for discrete, 0.0 for continuous
LR = 0.00005
BATCH = 8192  # 128 for discrete, 8192 for continuous
MINIBATCH = 32
EPOCHS = 10
EPSILON = 0.2
VF_COEFF = 1.0
L2_REG = 0.001
LSTM_UNITS = 256
LSTM_LAYERS = 1
KEEP_PROB = 1.0
SIGMA_FLOOR = 0.1  # Useful to set this to a non-zero number since the LSTM can make sigma drop too quickly

# MODEL_RESTORE_PATH = "/path/to/saved/model"
MODEL_RESTORE_PATH = None


class PPO(object):
    def __init__(self, environment, summary_dir="./", gpu=False, greyscale=True):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        config = tf.ConfigProto(log_device_placement=False, device_count={'GPU': gpu})
        config.gpu_options.per_process_gpu_memory_fraction = 0.1

        if len(environment.action_space.shape) > 0:
            self.discrete = False
            self.s_dim, self.a_dim = environment.observation_space.shape, environment.action_space.shape[0]
            self.a_bound = (environment.action_space.high - environment.action_space.low) / 2
            self.actions = tf.placeholder(tf.float32, [None, self.a_dim], 'action')
        else:
            self.discrete = True
            self.s_dim, self.a_dim = environment.observation_space.shape, environment.action_space.n
            self.actions = tf.placeholder(tf.int32, [None, 1], 'action')
        self.cnn = len(self.s_dim) == 3
        self.greyscale = greyscale  # If not greyscale and using RGB, make sure to divide the images by 255

        self.sess = tf.Session(config=config)
        self.state = tf.placeholder(tf.float32, [None] + list(self.s_dim), 'state')
        self.advantage = tf.placeholder(tf.float32, [None, 1], 'advantage')
        self.rewards = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # Use the TensorFlow Dataset API
        self.dataset = tf.data.Dataset.from_tensor_slices({"state": self.state, "actions": self.actions,
                                                           "rewards": self.rewards, "advantage": self.advantage})
        self.dataset = self.dataset.batch(MINIBATCH, drop_remainder=True)
        self.iterator = self.dataset.make_initializable_iterator()
        batch = self.iterator.get_next()
        self.global_step = tf.train.get_or_create_global_step()

        # Create an old & new policy function but also
        # make separate value & policy functions for evaluation & training (with shared variables)
        pi_old, vf_old, old_params, _, _ = self._build_net(batch["state"], 'oldnet')
        pi, self.v, params, self.i_state, self.f_state = self._build_net(batch["state"], 'net')
        pi_eval, self.vf_eval, _, self.eval_i_state, self.eval_f_state = self._build_net(self.state, 'net', reuse=True, batch_size=1)

        self.sample_op = tf.squeeze(pi_eval.sample(1), axis=0, name="sample_action")
        self.eval_action = pi_eval.mode()  # Used mode for discrete case. Mode should equal mean in continuous
        self.saver = tf.train.Saver()

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

            with tf.variable_scope('entropy'):
                entropy = pi.entropy()
                pol_entpen = -ENTROPY_BETA * tf.reduce_mean(entropy)

            loss = loss_pi + loss_vf * VF_COEFF + pol_entpen
            tf.summary.scalar("total", loss)
            # tf.summary.scalar("epsilon", epsilon_decay)

        with tf.variable_scope('train'):
            opt = tf.train.AdamOptimizer(LR)
            self.train_op = opt.minimize(loss, global_step=self.global_step, var_list=params)

            # grads, vs = zip(*opt.compute_gradients(loss, var_list=params))
            # grads, _ = tf.clip_by_global_norm(grads, 1.0)
            # self.train_op = opt.apply_gradients(zip(grads, vs), global_step=self.global_step)

        with tf.variable_scope('update_old'):
            self.update_old_op = [oldp.assign(p) for p, oldp in zip(params, old_params)]

        self.writer = tf.summary.FileWriter(summary_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        tf.summary.scalar("value", tf.reduce_mean(self.v))
        tf.summary.scalar("policy_entropy", tf.reduce_mean(entropy))
        if not self.discrete:
            tf.summary.scalar("sigma", tf.reduce_mean(pi.scale))
        self.summarise = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))

    def _build_net(self, state_in, name, reuse=False, batch_size=MINIBATCH):
        w_reg = tf.contrib.layers.l2_regularizer(L2_REG)

        with tf.variable_scope(name, reuse=reuse):
            if self.cnn:
                if self.greyscale:
                    state_in = tf.image.rgb_to_grayscale(state_in)
                conv1 = tf.layers.conv2d(inputs=state_in, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu)
                conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)
                conv3 = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)
                state_in = tf.layers.flatten(conv3)

            layer1 = tf.layers.dense(state_in, 400, tf.nn.relu, kernel_regularizer=w_reg, name="l1")
            layer2 = tf.layers.dense(layer1, LSTM_UNITS, tf.nn.relu, kernel_regularizer=w_reg, name="l2")

            # LSTM layer
            lstm = tf.nn.rnn_cell.LSTMCell(num_units=LSTM_UNITS, name='basic_lstm_cell')
            lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=self.keep_prob)
            lstm = tf.nn.rnn_cell.MultiRNNCell(cells=[lstm] * LSTM_LAYERS)

            init_state = lstm.zero_state(batch_size=batch_size, dtype=tf.float32)
            lstm_in = tf.expand_dims(layer2, axis=1)

            outputs, final_state = tf.nn.dynamic_rnn(cell=lstm, inputs=lstm_in, initial_state=init_state)
            cell_out = tf.reshape(outputs, [-1, LSTM_UNITS], name='flatten_lstm_outputs')

            vf = tf.layers.dense(cell_out, 1, kernel_regularizer=w_reg, name="vf_out")

            if self.discrete:
                a_logits = tf.layers.dense(cell_out, self.a_dim, kernel_regularizer=w_reg, name="pi_logits")
                dist = tf.distributions.Categorical(logits=a_logits)
            else:
                mu = tf.layers.dense(cell_out, self.a_dim, tf.nn.tanh, kernel_regularizer=w_reg, name="pi_mu")
                log_sigma = tf.get_variable(name="pi_sigma", shape=self.a_dim, initializer=tf.zeros_initializer())
                dist = tf.distributions.Normal(loc=mu * self.a_bound, scale=tf.maximum(tf.exp(log_sigma), SIGMA_FLOOR))

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return dist, vf, params, init_state, final_state

    def save_model(self, model_path, step=None):
        save_path = self.saver.save(self.sess, os.path.join(model_path, "model.ckpt"), global_step=step)
        return save_path

    def restore_model(self, model_path):
        self.saver.restore(self.sess, os.path.join(model_path, "model.ckpt"))
        print("Model restored from", model_path)

    def update(self, exp):
        start, e_time = time(), []
        e_time, summaries = [], []
        self.sess.run([self.update_old_op])

        for epoch in range(EPOCHS):
            np.random.shuffle(exp)  # Shuffle the episodes
            for ep_s, ep_a, ep_r, ep_adv in exp:

                self.sess.run(self.iterator.initializer, feed_dict={self.state: ep_s, self.actions: ep_a,
                                                                    self.rewards: ep_r, self.advantage: ep_adv})

                state = self.sess.run([self.i_state])
                train_ops = [self.summarise, self.global_step, self.f_state, self.train_op]

                while True:
                    try:
                        e_start = time()
                        summary, step, state, _ = self.sess.run(train_ops,
                                                                feed_dict={self.i_state: state, self.keep_prob: KEEP_PROB})
                        e_time.append(time() - e_start)
                    except tf.errors.OutOfRangeError:
                        break

        print("Trained in %.3fs. Average %.3fs/batch. Global step %i" % (time() - start, np.mean(e_time), step))
        return summary

    def eval_state(self, state, lstm_state_in, stochastic=True):
        if stochastic:
            eval_ops = [self.sample_op, self.vf_eval, self.eval_f_state]
        else:
            eval_ops = [self.eval_action, self.vf_eval, self.eval_f_state]

        a, v, lstm_state_new = self.sess.run(eval_ops,
                                             {self.state: state[np.newaxis, :], self.eval_i_state: lstm_state_in, self.keep_prob: 1.0})
        return a[0], np.squeeze(v), lstm_state_new


if __name__ == '__main__':
    # Discrete environments
    # ENVIRONMENT = 'CartPole-v1'  # Switch off reward scaling for this
    # ENVIRONMENT = 'MountainCar-v0'
    # ENVIRONMENT = 'LunarLander-v2'
    # ENVIRONMENT = 'Pong-v0'

    # Continuous environments
    # ENVIRONMENT = 'Pendulum-v0'
    # ENVIRONMENT = 'MountainCarContinuous-v0'
    # ENVIRONMENT = 'LunarLanderContinuous-v2'
    # ENVIRONMENT = 'BipedalWalker-v2'
    # ENVIRONMENT = 'BipedalWalkerHardcore-v2'
    ENVIRONMENT = 'CarRacing-v0'

    TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
    SUMMARY_DIR = os.path.join(OUTPUT_RESULTS_DIR, "PPO_LSTM", ENVIRONMENT, TIMESTAMP)

    env = gym.make(ENVIRONMENT)
    env = wrappers.Monitor(env, os.path.join(SUMMARY_DIR, ENVIRONMENT), video_callable=None)
    ppo = PPO(env, SUMMARY_DIR, gpu=False)

    if MODEL_RESTORE_PATH is not None:
        ppo.restore_model(MODEL_RESTORE_PATH)

    t, terminal = 0, False
    buffer_s, buffer_a, buffer_r, buffer_v, buffer_terminal = [], [], [], [], []
    rolling_r = RunningStats()
    experience, batch_rewards = [], []

    for episode in range(EP_MAX):

        lstm_state = ppo.sess.run(ppo.eval_i_state)  # Zero LSTM state at beginning

        s = env.reset()
        ep_r, ep_t, ep_a = 0, 0, []

        while True:
            a, v, lstm_state = ppo.eval_state(s, lstm_state)

            if terminal:
                # Normalise rewards
                rewards = np.array(buffer_r)
                rewards = np.clip(rewards / rolling_r.std, -10, 10)
                batch_rewards = batch_rewards + buffer_r

                v_final = [v * (1 - terminal)]  # v = 0 if terminal, otherwise use the predicted v
                values = np.array(buffer_v + v_final)
                terminals = np.array(buffer_terminal + [terminal])

                # Generalized Advantage Estimation - https://arxiv.org/abs/1506.02438
                delta = rewards + GAMMA * values[1:] * (1 - terminals[1:]) - values[:-1]
                advantage = discount(delta, GAMMA * LAMBDA, terminals)
                returns = advantage + np.array(buffer_v)
                # Per episode normalisation of advantages
                # advantage = (advantage - advantage.mean()) / np.maximum(advantage.std(), 1e-6)

                bs, ba, br, badv = np.reshape(buffer_s, (len(buffer_s),) + ppo.s_dim), np.vstack(buffer_a), \
                                   np.vstack(returns), np.vstack(advantage)
                experience.append([bs, ba, br, badv])

                buffer_s, buffer_a, buffer_r, buffer_v, buffer_terminal = [], [], [], [], []

                # Update ppo
                if t >= BATCH:
                    # Per batch normalisation of advantages
                    advs = np.concatenate(list(zip(*experience))[3])
                    for x in experience:
                        x[3] = (x[3] - np.mean(advs)) / np.maximum(np.std(advs), 1e-6)

                    # Update rolling reward stats
                    rolling_r.update(np.array(batch_rewards))

                    print("Training using %i episodes and %i steps..." % (len(experience), t))
                    graph_summary = ppo.update(experience)
                    t, experience, batch_rewards = 0, [], []

            buffer_s.append(s)
            buffer_a.append(a)
            buffer_v.append(v)
            buffer_terminal.append(terminal)
            ep_a.append(a)

            if not ppo.discrete:
                a = np.clip(a, env.action_space.low, env.action_space.high)
            s, r, terminal, _ = env.step(a)
            buffer_r.append(r)

            ep_r += r
            ep_t += 1
            t += 1

            if terminal:
                # End of episode summary
                print('Episode: %i' % episode, "| Reward: %.2f" % ep_r, '| Steps: %i' % ep_t)

                worker_summary = tf.Summary()
                worker_summary.value.add(tag="Reward", simple_value=ep_r)
                worker_summary.value.add(tag="Reward/mean", simple_value=rolling_r.mean)
                worker_summary.value.add(tag="Reward/std", simple_value=rolling_r.std)

                # Create Action histograms for each dimension
                actions = np.array(ep_a)
                if ppo.discrete:
                    add_histogram(ppo.writer, "Action", actions, episode, bins=ppo.a_dim)
                else:
                    for a in range(ppo.a_dim):
                        add_histogram(ppo.writer, "Action/Dim" + str(a), actions[:, a], episode)

                try:
                    ppo.writer.add_summary(graph_summary, episode)
                except NameError:
                    pass
                ppo.writer.add_summary(worker_summary, episode)
                ppo.writer.flush()

                # Save the model
                if episode % 100 == 0 and episode > 0:
                    path = ppo.save_model(SUMMARY_DIR, episode)
                    print('Saved model at episode', episode, 'in', path)

                break

    env.close()

    env = gym.make(ENVIRONMENT)
    env = wrappers.Monitor(env, os.path.join(SUMMARY_DIR, ENVIRONMENT + "_trained"), video_callable=None)

    while True:
        s = env.reset()
        ep_r, ep_t = 0, 0
        lstm_state = ppo.sess.run(ppo.eval_i_state)
        while True:
            env.render()
            a, v, lstm_state = ppo.eval_state(s, lstm_state, stochastic=False)
            if not ppo.discrete:
                a = np.clip(a, env.action_space.low, env.action_space.high)
            s, r, terminal, _ = env.step(a)
            ep_r += r
            ep_t += 1
            if terminal:
                print("Reward: %.2f" % ep_r, '| Steps: %i' % ep_t)
                break
