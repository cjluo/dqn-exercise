import os
os.environ["KERAS_BACKEND"] = "tensorflow"

from environment import Environment
from model import QNetwork
import tensorflow as tf
import numpy as np
from keras import backend as K
import random
import time

flags = tf.app.flags

flags.DEFINE_string('experiment', 'dqn_breakout',
                    'Name of the current experiment')
flags.DEFINE_string('game', 'Breakout-v0',
                    'Name of the atari game to play.'
                    'Full list here: https://gym.openai.com/envs#atari')

flags.DEFINE_integer('resized_width', 84, 'Scale screen to this width.')
flags.DEFINE_integer('resized_height', 84, 'Scale screen to this height.')
flags.DEFINE_integer('agent_history_length', 4,
                     'Use this number of recent screens as the environment'
                     'state.')
flags.DEFINE_boolean('display', False,
                     'Whether to do display the game screen or not')

flags.DEFINE_integer('tmax', int(50 * 1e4), 'Number of training timesteps.')
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_float('gamma', 0.99, 'Reward discount rate.')

flags.DEFINE_float('start_ep', 0.5, 'Start ep for exploring')
flags.DEFINE_float('end_ep', 0.1, 'End ep for exploring')
flags.DEFINE_float('end_ep_t', 1 * 1e4, 'Steps to decay ep')

flags.DEFINE_integer('replay_size', 1 * 1e4, 'Size of replay memory')
flags.DEFINE_integer('batch_size', 32, 'Size of mini batch')

flags.DEFINE_string('summary_dir', './summaries',
                    'Directory for storing tensorboard summaries')
flags.DEFINE_string('checkpoint_dir', './checkpoints',
                    'Directory for storing model checkpoints')
flags.DEFINE_integer('summary_interval', 5,
                     'Save training summary to file every n seconds (rounded '
                     'up to statistics interval.')
flags.DEFINE_integer('checkpoint_interval', 600,
                     'Checkpoint the model (i.e. save the parameters) every n '
                     'seconds (rounded up to statistics interval.')

FLAGS = flags.FLAGS


class DQN(object):
    def __init__(self):
        self._env = Environment(FLAGS.game, FLAGS.resized_width,
                                FLAGS.resized_height,
                                FLAGS.agent_history_length, FLAGS.display,
                                FLAGS.replay_size)
        self._q_network = QNetwork(
            self._env.action_size, FLAGS.agent_history_length,
            FLAGS.resized_width, FLAGS.resized_height, FLAGS.learning_rate)

        self._setup_summary()

        if not os.path.exists(FLAGS.checkpoint_dir):
            os.makedirs(FLAGS.checkpoint_dir)
        if not os.path.exists(FLAGS.summary_dir):
            os.makedirs(FLAGS.summary_dir)

    def _train_q_network(self, session):
        prev_state_batch, action_batch, reward_batch, current_state_batch,\
            terminal_batch = self._env.sample(FLAGS.batch_size)
        if len(prev_state_batch) == 0:
            return

        y_batch = []
        q_batch = self._q_network.eval(session, current_state_batch)

        for i in xrange(FLAGS.batch_size):
            if terminal_batch[i]:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(
                    reward_batch[i] + FLAGS.gamma * np.max(q_batch[i]))

        return self._q_network.train(
            session, prev_state_batch, action_batch, y_batch)

    def _e_greedy(self, session, state, t):
        ep = np.interp(t, [0, FLAGS.end_ep_t], [FLAGS.start_ep, FLAGS.end_ep])

        if random.random() < ep:
            action = random.randrange(self._env.action_size)
            q_max = -1
        else:
            q = self._q_network.eval(session, [state])[0]
            action = np.argmax(q)
            q_max = q[action]

        return action, q_max

    def _setup_summary(self):
        reward = tf.Variable(0.)
        tf.scalar_summary("Episode Reward", reward)
        max_q_avg = tf.Variable(0.)
        tf.scalar_summary("Max Q Value", max_q_avg)
        loss = tf.Variable(0.)
        tf.scalar_summary("Loss", loss)
        summary_vars = [reward, max_q_avg, loss]
        self._summary_placeholders = [
            tf.placeholder("float")] * len(summary_vars)
        self._update_ops = [summary_vars[i].assign(
            self._summary_placeholders[i]) for i in range(len(summary_vars))]
        self._summary_op = tf.merge_all_summaries()

    def _update_summary(self, session, reward, max_q_avg, loss):
        summary_values = [reward, max_q_avg, loss]
        for i in range(len(summary_values)):
            session.run(
                self._update_ops[i],
                feed_dict={
                    self._summary_placeholders[i]: float(summary_values[i])})

    def train(self, session, saver, writer):
        session.run(tf.initialize_all_variables())

        total_reward = 0
        step_count = 0
        q_max_sum = 0
        episode = 0
        self._env.new_game()
        record_time = time.time()
        for t in xrange(FLAGS.tmax):
            action, q_max = self._e_greedy(
                session, self._env.get_frame_history(), t)
            _, reward, terminal, _ = self._env.step(action)

            total_reward += reward
            q_max_sum += q_max
            step_count += 1

            loss = self._train_q_network(session)
            if t % FLAGS.checkpoint_interval == 0:
                saver.save(
                    session,
                    FLAGS.checkpoint_dir + "/" + FLAGS.experiment + ".ckpt",
                    global_step=t)

            if terminal:
                print(
                    "Episode %d: time=%d total_reward=%d q_max_avg=%f loss=%f"
                    % (episode, t, total_reward, q_max_sum / step_count,
                       loss))
                self._update_summary(
                    session, total_reward, q_max_sum / step_count, loss)
                episode += 1
                total_reward = 0
                step_count = 0
                q_max_sum = 0
                self._env.new_game()

            now_time = time.time()
            if now_time - record_time > FLAGS.summary_interval:
                summary = session.run(self._summary_op)
                writer.add_summary(summary)
                record_time = now_time


def main(_):
    dqn = DQN()
    with tf.Session() as session:
        K.set_session(session)
        saver = tf.train.Saver()
        writer = tf.train.SummaryWriter(FLAGS.summary_dir, session.graph)
        dqn.train(session, saver, writer)


if __name__ == "__main__":
    tf.app.run()
