from environment import Environment
from model import QNetwork
import tensorflow as tf
import numpy as np
import random
import time
import sys
import os
from threading import Thread, Lock
from collections import deque


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

flags.DEFINE_integer('tmax', int(1e7), 'Number of training timesteps.')
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_float('gamma', 0.99, 'Reward discount rate.')
flags.DEFINE_integer('update_frequency', 1e4,
                     'Steps between target q network parameter updates')

flags.DEFINE_float('start_ep', 1, 'Start ep for exploring')
flags.DEFINE_float('end_ep_t', 8e5, 'Steps to decay ep')

flags.DEFINE_integer('async_update', 32, 'Frequency of async update')

flags.DEFINE_boolean('double', True, 'Double DQN training')
flags.DEFINE_boolean('dueling', True, 'Dueling DQN training')

flags.DEFINE_string('summary_dir', './summaries',
                    'Directory for storing tensorboard summaries')
flags.DEFINE_string('checkpoint_dir', './checkpoints',
                    'Directory for storing model checkpoints')
flags.DEFINE_integer('checkpoint_interval', 5e3,
                     'Checkpoint the model (i.e. save the parameters) every n '
                     'steps.')
flags.DEFINE_integer('print_interval', 10,
                     'Print the training result every n episodes.')

flags.DEFINE_boolean('play', False, 'If true, run gym evaluation')
flags.DEFINE_integer('play_round', 10, 'Rounds to play in evaluation')

flags.DEFINE_integer('thread', 8, 'Training thread count')


FLAGS = flags.FLAGS


def log_in_line(str):
    sys.stdout.write(str)
    sys.stdout.write("\r")
    sys.stdout.flush()


def sample_final_epsilon():
    """
    Sample a final epsilon value to anneal towards from a distribution.
    These values are specified in section 5.1 of
    http://arxiv.org/pdf/1602.01783v1.pdf
    """
    final_epsilons = np.array([.1, .01, .5])
    probabilities = np.array([0.4, 0.3, 0.3])
    return np.random.choice(final_epsilons, 1, p=list(probabilities))[0]


class AtomicInt(object):
    def __init__(self, t):
        self._lock = Lock()
        self._t = t
        self._time = time.time()
        self._speed = 0

    def get(self):
        return self._t

    def increment(self):
        self._lock.acquire()
        self._t += 1
        self._lock.release()
        if self._t % FLAGS.checkpoint_interval == 0:
            time_now = time.time()
            self._speed = FLAGS.checkpoint_interval / (time_now - self._time)
            self._time = time_now

    def get_speed(self):
        return self._speed


class DQN(object):
    def __init__(self):
        self._envs = [Environment(
            FLAGS.game, FLAGS.resized_width, FLAGS.resized_height,
            FLAGS.agent_history_length, 0, 0)
            for i in range(FLAGS.thread)]
        self._action_size = self._envs[0].action_size

        # Training Q network:
        # 1) generate action
        # 2) value updated in training
        self._q_network = QNetwork(
            'online', self._action_size, FLAGS.agent_history_length,
            FLAGS.resized_width, FLAGS.resized_height, FLAGS.learning_rate,
            FLAGS.dueling)
        # Target Q network:
        # 1) estimate y value
        # 2) value updated from training Q network
        self._target_q_network = QNetwork(
            'target', self._action_size, FLAGS.agent_history_length,
            FLAGS.resized_width, FLAGS.resized_height, FLAGS.learning_rate,
            FLAGS.dueling)

        self._update_network_params_op =\
            self._target_q_network.get_update_network_params_op(
                self._q_network)

        self._setup_summary()
        self._setup_global_step()

        if not os.path.exists(FLAGS.checkpoint_dir):
            os.makedirs(FLAGS.checkpoint_dir)
        if not os.path.exists(FLAGS.summary_dir):
            os.makedirs(FLAGS.summary_dir)

    def _train_q_network(self, session, prev_state_batch, action_batch,
                         y_batch):
        return self._q_network.train(
            session, prev_state_batch, action_batch, y_batch)

    def _e_greedy(self, session, state, t, end_ep):
        ep = np.interp(t, [0, FLAGS.end_ep_t], [FLAGS.start_ep, end_ep])

        if random.random() < ep:
            action = random.randrange(self._action_size)
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

    def _train_thread(self, thread_id, atomic_t, session, saver,
                      writer):
        total_reward = 0
        q_max_list = []
        episode = 0
        loss = -1
        thread_t = 0

        env = self._envs[thread_id]
        env.new_game()

        prev_state_batch = deque()
        action_batch = deque()
        y_batch = deque()

        end_ep = sample_final_epsilon()

        print("Thread %d starting... final ep %f" % (thread_id, end_ep))
        time.sleep(0)

        while True:
            t = atomic_t.get()
            if t > FLAGS.tmax:
                break

            prev_state = env.get_frames()
            action, q_max, ep = self._e_greedy(session, prev_state, t,
                                               end_ep)
            current_state, reward, terminal, _ = env.step(action)

            total_reward += reward
            if q_max != 0:
                q_max_list.append(q_max)

            q = self._target_q_network.eval(session, [current_state])[0]

            if FLAGS.double:
                action_next = np.argmax(
                    self._q_network.eval(session, [current_state])[0])
                q_action = q[action_next]
                y = reward + FLAGS.gamma * (1 - terminal) * q_action
            else:
                y = reward + FLAGS.gamma * (1 - terminal) * np.max(q)

            prev_state_batch.append(prev_state)
            action_batch.append(action)
            y_batch.append(y)

            thread_t += 1
            atomic_t.increment()

            if thread_t == FLAGS.async_update or terminal:
                thread_t = 0
                loss = self._train_q_network(
                    session, prev_state_batch, action_batch, y_batch)
                prev_state_batch.clear()
                action_batch.clear()
                y_batch.clear()

            if t % FLAGS.update_frequency == 0:
                session.run(self._update_network_params_op)

            if t % FLAGS.checkpoint_interval == 0:
                try:
                    self._global_step.assign(t).eval(session=session)
                except:
                    print "Count not assign global step!"

                saver.save(
                    session,
                    FLAGS.checkpoint_dir + "/" + FLAGS.experiment + ".ckpt",
                    global_step=self._global_step)
                summary = session.run(self._summary_op)
                writer.add_summary(summary, float(t))

            if terminal:
                q_max = np.mean(q_max_list) if q_max_list else 0

                if episode % FLAGS.print_interval == 0:
                    log_in_line(
                        "Thread %d Episode %d (%f steps/sec): step=%d "
                        "total_reward=%d q_max_avg=%f loss=%f ep=%f"
                        % (thread_id, episode, atomic_t.get_speed(), t,
                           total_reward, q_max, loss, ep))

                self._update_summary(
                    session, total_reward, q_max, loss, ep)

                episode += 1
                total_reward = 0
                q_max_list = []
                loss = -1
                env.new_game()

    def train(self, session, saver, writer):
        t_start = self._global_step.eval(session=session)
        atomic_t = AtomicInt(t_start)

        training_threads = [Thread(
            target=self._train_thread,
            args=(thread_id, atomic_t, session, saver, writer))
            for thread_id in range(FLAGS.thread)]

        for thread in training_threads:
            thread.start()

        while atomic_t.get() < FLAGS.tmax:
            if FLAGS.display:
                for env in self._envs:
                    env.render()
            time.sleep(1)

        for thread in training_threads:
            thread.join()

    def play(self, session):
        total_reward_list = []
        for episode in xrange(FLAGS.play_round):
            env = self._envs[0]
            env.new_game()
            total_reward = 0

            while True:
                q = self._q_network.eval(
                    session, [env.get_frames()])[0]
                action = np.argmax(q)
                _, reward, terminal, _ = env.step(action)

                if FLAGS.display:
                    env.render()

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
