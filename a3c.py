from environment import Environment
from model_a3c import A3CNetwork
import tensorflow as tf
import numpy as np
import time
import sys
import os
from threading import Thread, Lock
from collections import deque


flags = tf.app.flags

flags.DEFINE_string('experiment', 'breakout',
                    'Name of the current experiment')
flags.DEFINE_string('game', 'Breakout-v0',
                    'Name of the atari game to play.'
                    'Full list here: https://gym.openai.com/envs#atari')

flags.DEFINE_integer('resized_width', 84, 'Scale screen to this width.')
flags.DEFINE_integer('resized_height', 84, 'Scale screen to this height.')
flags.DEFINE_integer('agent_history_length', 4,
                     'Use this number of recent screens as the environment'
                     'state.')
flags.DEFINE_integer('action_repeat', 4, 'Repeat same action for number of'
                     'frames.')
flags.DEFINE_boolean('display', False,
                     'Whether to do display the game screen or not')

flags.DEFINE_integer('tmax', int(1e7), 'Number of training timesteps.')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
flags.DEFINE_float('gamma', 0.99, 'Reward discount rate.')

flags.DEFINE_integer('async_update', 32, 'Frequency of async update')
flags.DEFINE_float('beta', 0.01, 'Beta factor in policy entropy weight')
flags.DEFINE_float('grad_clip', 100, 'Maximum gradient in norm.')

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


class A3C(object):
    def __init__(self):
        self._envs = [Environment(
            FLAGS.game, FLAGS.resized_width, FLAGS.resized_height,
            FLAGS.agent_history_length, 0, 0, FLAGS.action_repeat)
            for i in range(FLAGS.thread)]
        self._action_size = self._envs[0].action_size

        self._global_a3c_network = A3CNetwork(
            'global', self._action_size, FLAGS.agent_history_length,
            FLAGS.resized_width, FLAGS.resized_height, FLAGS.learning_rate,
            FLAGS.beta, FLAGS.grad_clip)

        self._local_a3c_networks = [A3CNetwork(
            'thread' + str(i), self._action_size, FLAGS.agent_history_length,
            FLAGS.resized_width, FLAGS.resized_height, FLAGS.learning_rate,
            FLAGS.beta, FLAGS.grad_clip) for i in range(FLAGS.thread)]

        self._sync_network_params_ops =\
            [local_network.get_update_network_params_op(
                self._global_a3c_network)
                for local_network in self._local_a3c_networks]

        self._train_global_network_ops =\
            [self._global_a3c_network.apply_grads(local_network.get_grads())
             for local_network in self._local_a3c_networks]

        self._setup_summary()
        self._setup_global_step()

        if not os.path.exists(FLAGS.checkpoint_dir):
            os.makedirs(FLAGS.checkpoint_dir)
        if not os.path.exists(FLAGS.summary_dir):
            os.makedirs(FLAGS.summary_dir)

    def _sample_action(self, session, network, state):
        p = network.eval_p(session, [state])[0]
        action = np.random.choice(self._action_size, p=p)
        p_action = p[action]
        return action, p_action

    def _setup_global_step(self):
        self._global_step = tf.Variable(0, name='global_step', trainable=False)

    def _setup_summary(self):
        reward = tf.Variable(0., trainable=False)
        tf.scalar_summary("Episode Reward", reward)
        action_p_avg = tf.Variable(0., trainable=False)
        tf.scalar_summary("Action P Value", action_p_avg)
        v_avg = tf.Variable(0., trainable=False)
        tf.scalar_summary("V Value", v_avg)
        summary_vars = [reward, action_p_avg, v_avg]
        self._summary_placeholders = [
            tf.placeholder("float")] * len(summary_vars)
        self._update_ops = [summary_vars[i].assign(
            self._summary_placeholders[i]) for i in range(len(summary_vars))]
        self._summary_op = tf.merge_all_summaries()

    def _update_summary(self, session, reward, action_p_avg, v_avg):
        summary_values = [reward, action_p_avg, v_avg]
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

    def _train_thread(self, thread_id, atomic_t, session, local_network,
                      sync_op, train_op, saver, writer):
        total_reward = 0
        p_action_list = []
        v_list = []
        episode = 0
        thread_t = 0

        env = self._envs[thread_id]
        env.new_game()

        prev_state_batch = deque()
        action_batch = deque()
        reward_t_batch = deque()

        print("Thread %d starting..." % thread_id)
        time.sleep(0)

        session.run(sync_op)

        while True:
            t = atomic_t.get()
            if t > FLAGS.tmax:
                break

            prev_state = env.get_frames()
            action, p_action = self._sample_action(
                session, local_network, prev_state)
            current_state, reward, terminal, _ = env.step(action)

            total_reward += reward
            p_action_list.append(p_action)

            prev_state_batch.append(prev_state)
            action_batch.append(action)
            reward_t_batch.append(reward)

            thread_t += 1
            atomic_t.increment()

            if thread_t == FLAGS.async_update or terminal:
                batch_size = thread_t
                thread_t = 0
                if batch_size > 0:
                    if terminal:
                        reward_final = 0
                    else:
                        reward_final = local_network.eval_v(
                            session, [current_state])[0]
                        v_list.append(reward_final)

                    reward_batch = np.zeros(batch_size)
                    for i in reversed(range(batch_size)):
                        reward_batch[i] = FLAGS.gamma * reward_final\
                            + reward_t_batch.pop()
                        reward_final = reward_batch[i]

                    session.run(
                        train_op,
                        feed_dict=local_network.get_feed_dict(
                            prev_state_batch, action_batch, reward_batch))

                    prev_state_batch.clear()
                    action_batch.clear()
                    reward_t_batch.clear()

                    session.run(sync_op)

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
                p_action = np.mean(p_action_list) if p_action_list else 0
                v = np.mean(v_list) if v_list else 0

                if episode % FLAGS.print_interval == 0:
                    log_in_line(
                        "Thread %d Episode %d (%f steps/sec): step=%d "
                        "total_reward=%d p_max_avg=%f v=%f"
                        % (thread_id, episode, atomic_t.get_speed(), t,
                           total_reward, p_action, v))

                self._update_summary(session, total_reward, p_action, v)

                episode += 1
                total_reward = 0
                p_action_list = []
                v_list = []
                env.new_game()

    def train(self, session, saver, writer):
        t_start = self._global_step.eval(session=session)
        atomic_t = AtomicInt(t_start)

        training_threads = [Thread(
            target=self._train_thread,
            args=(thread_id, atomic_t, session,
                  self._local_a3c_networks[thread_id],
                  self._sync_network_params_ops[thread_id],
                  self._train_global_network_ops[thread_id],
                  saver, writer))
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
                action, _ = self._sample_action(
                    session, self._global_a3c_network, env.get_frames())
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
    network = A3C()
    with tf.Session() as session:
        saver = tf.train.Saver()
        writer = tf.train.SummaryWriter(FLAGS.summary_dir, session.graph)
        session.run(tf.initialize_all_variables())
        network.load_model(session)
        if FLAGS.play:
            network.play(session)
        else:
            network.train(session, saver, writer)


if __name__ == "__main__":
    tf.app.run()
