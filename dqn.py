from environment import Environment
from model import QNetwork
import tensorflow as tf
import numpy as np
import random
import time
import sys
import os

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

flags.DEFINE_integer('tmax', int(300 * 1e4), 'Number of training timesteps.')
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_float('gamma', 0.99, 'Reward discount rate.')
flags.DEFINE_integer('train_frequency', 4,
                     'Train mini-batch after # of steps.')
flags.DEFINE_integer('update_frequency', 5e3,
                     'Steps between target q network parameter updates')

flags.DEFINE_float('start_ep', 1, 'Start ep for exploring')
flags.DEFINE_float('end_ep', 0.1, 'End ep for exploring')
flags.DEFINE_float('end_ep_t', 6 * 1e4, 'Steps to decay ep')

flags.DEFINE_integer('replay_size', int(6 * 1e4), 'Size of replay memory')
flags.DEFINE_integer('batch_size', 32, 'Size of mini batch')
flags.DEFINE_float('alpha', 0, 'Alpha factor in prioritized sampling, '
                   '0 means equal priority')

flags.DEFINE_boolean('double', True, 'Double DQN training')
flags.DEFINE_boolean('dueling', True, 'Dueling DQN training')

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

flags.DEFINE_boolean('play', False, 'If true, run gym evaluation')
flags.DEFINE_integer('play_round', 10, 'Rounds to play in evaluation')


FLAGS = flags.FLAGS


def log_in_line(str):
    sys.stdout.write(str)
    sys.stdout.write("\r")
    sys.stdout.flush()


class DQN(object):
    def __init__(self):
        self._env = Environment(FLAGS.game, FLAGS.resized_width,
                                FLAGS.resized_height,
                                FLAGS.agent_history_length, FLAGS.display,
                                FLAGS.replay_size, FLAGS.alpha)
        # Training Q network:
        # 1) generate action
        # 2) value updated in training
        self._q_network = QNetwork(
            'online', self._env.action_size, FLAGS.agent_history_length,
            FLAGS.resized_width, FLAGS.resized_height, FLAGS.learning_rate,
            FLAGS.dueling)
        # Target Q network:
        # 1) estimate y value
        # 2) value updated from training Q network
        self._target_q_network = QNetwork(
            'target', self._env.action_size, FLAGS.agent_history_length,
            FLAGS.resized_width, FLAGS.resized_height, FLAGS.learning_rate,
            FLAGS.dueling)

        self._setup_summary()
        self._setup_global_step()

        if not os.path.exists(FLAGS.checkpoint_dir):
            os.makedirs(FLAGS.checkpoint_dir)
        if not os.path.exists(FLAGS.summary_dir):
            os.makedirs(FLAGS.summary_dir)

    def _train_q_network(self, session):
        prev_state_batch, action_batch, reward_batch, current_state_batch,\
            terminal_batch, sample_batch = self._env.sample(FLAGS.batch_size)
        if len(prev_state_batch) == 0:
            return

        y_batch = []
        q_batch = self._target_q_network.eval(session, current_state_batch)
        terminal_batch = np.array(terminal_batch, dtype=int)

        if FLAGS.double:
            action_next_batch = np.argmax(
                self._q_network.eval(session, current_state_batch), axis=1)
            q_action_batch = q_batch[range(FLAGS.batch_size),
                                     action_next_batch]
            y_batch = reward_batch + FLAGS.gamma * np.multiply(
                1 - terminal_batch, q_action_batch)
        else:
            y_batch = reward_batch + FLAGS.gamma * np.multiply(
                1 - terminal_batch, np.max(q_batch, axis=1))

        if FLAGS.alpha is not 0:
            # Updates priority
            priority = np.absolute(np.max(q_batch, axis=1) - y_batch)
            i = 0
            for sample in sample_batch:
                sample['priority'] = priority[i]
                i += 1

        return self._q_network.train(
            session, prev_state_batch, action_batch, y_batch)

    def _e_greedy(self, session, state, t):
        ep = np.interp(t, [0, FLAGS.end_ep_t], [FLAGS.start_ep, FLAGS.end_ep])

        if random.random() < ep:
            action = random.randrange(self._env.action_size)
            q_max = 0
        else:
            q = self._q_network.eval(session, [state])[0]
            action = np.argmax(q)
            q_max = q[action]

        return action, q_max, ep

    def _setup_global_step(self):
        self._global_step = tf.Variable(0, name='global_step', trainable=False)

    def _setup_summary(self):
        reward = tf.Variable(0., trainable=False)
        tf.scalar_summary("Episode Reward", reward)
        max_q_avg = tf.Variable(0., trainable=False)
        tf.scalar_summary("Max Q Value", max_q_avg)
        loss = tf.Variable(0., trainable=False)
        tf.scalar_summary("Loss", loss)
        ep = tf.Variable(0., trainable=False)
        tf.scalar_summary("Epsilon", ep)
        summary_vars = [reward, max_q_avg, loss, ep]
        self._summary_placeholders = [
            tf.placeholder("float")] * len(summary_vars)
        self._update_ops = [summary_vars[i].assign(
            self._summary_placeholders[i]) for i in range(len(summary_vars))]
        self._summary_op = tf.merge_all_summaries()

    def _update_summary(self, session, reward, max_q_avg, loss, ep):
        summary_values = [reward, max_q_avg, loss, ep]
        for i in range(len(summary_values)):
            session.run(
                self._update_ops[i],
                feed_dict={
                    self._summary_placeholders[i]: float(summary_values[i])})

    def load_model(self, session):
        print("Loading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            fname = os.path.join(FLAGS.checkpoint_dir, ckpt_name)
            # Only restore saved variables
            reader = tf.train.NewCheckpointReader(fname)
            save_var_map = reader.get_variable_to_shape_map()
            all_var_map = tf.all_variables()
            restore_var_list = [variable for variable in all_var_map
                                if variable.name.split(':')[0] in save_var_map]
            saver = tf.train.Saver(restore_var_list)
            saver.restore(session, fname)
            print("Load SUCCESS: %s" % fname)
        else:
            print("Load FAILED: %s" % FLAGS.checkpoint_dir)

    def train(self, session, saver, writer):
        total_reward = 0
        q_max_list = []
        episode = 0
        loss = -1
        self._env.new_game()
        episode_time = time.time()
        record_time = time.time()
        t_terminal = 0

        t_start = self._global_step.eval(session=session)
        print("Training started with global step %d" % t_start)

        # Fill in the initial replay buffer
        for t in xrange(FLAGS.replay_size):
            action, _, _ = self._e_greedy(session, self._env.get_frames(), t)
            _, _, terminal, _ = self._env.step(action)
            if terminal:
                self._env.new_game()
                log_in_line("Fill in replay buffer %d percent..."
                            % (t * 100 / FLAGS.replay_size))

        for t in xrange(t_start, FLAGS.tmax):
            action, q_max, ep = self._e_greedy(
                session, self._env.get_frames(), t)
            _, reward, terminal, _ = self._env.step(action)

            total_reward += reward
            if q_max != 0:
                q_max_list.append(q_max)

            if t % FLAGS.train_frequency == 0:
                loss = self._train_q_network(session)

            if t % FLAGS.update_frequency == 0:
                self._target_q_network.update_network_params(
                    session, self._q_network)

            if t % FLAGS.checkpoint_interval == 0:
                self._global_step.assign(t).eval(session=session)
                saver.save(
                    session,
                    FLAGS.checkpoint_dir + "/" + FLAGS.experiment + ".ckpt",
                    global_step=self._global_step)

            if terminal:
                new_episode_time = time.time()
                steps_per_sec = (
                    t - t_terminal) / (new_episode_time - episode_time)

                q_max = np.mean(q_max_list)

                log_in_line(
                    "Episode %d (%f steps/sec): step=%d total_reward=%d "
                    "q_max_avg=%f loss=%f ep=%f"
                    % (episode, steps_per_sec, t, total_reward,
                       q_max, loss, ep))
                self._update_summary(
                    session, total_reward, q_max, loss, ep)

                episode_time = new_episode_time
                t_terminal = t

                episode += 1
                total_reward = 0
                q_max_list = []
                loss = -1
                self._env.new_game()

            now_time = time.time()
            if now_time - record_time > FLAGS.summary_interval:
                summary = session.run(self._summary_op)
                writer.add_summary(summary, float(t))
                record_time = now_time

    def play(self, session):
        total_reward_list = []
        for episode in xrange(FLAGS.play_round):
            self._env.new_game()
            total_reward = 0

            while True:
                q = self._q_network.eval(
                    session, [self._env.get_frames()])[0]
                action = np.argmax(q)
                _, reward, terminal, _ = self._env.step(action)

                total_reward += reward
                if terminal:
                    break

            print("Episode %d: total_reward=%d" % (episode, total_reward))
            total_reward_list.append(total_reward)

        print("Total round %d: mean_reward=%f, max_reward=%d" % (
            FLAGS.play_round, np.mean(total_reward_list),
            np.max(total_reward_list)))


def main(_):
    dqn = DQN()
    with tf.Session() as session:
        saver = tf.train.Saver()
        writer = tf.train.SummaryWriter(FLAGS.summary_dir, session.graph)
        session.run(tf.initialize_all_variables())
        dqn.load_model(session)
        if FLAGS.play:
            dqn.play(session)
        else:
            dqn.train(session, saver, writer)


if __name__ == "__main__":
    tf.app.run()
