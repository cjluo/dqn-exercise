import tensorflow as tf
import tensorflow.contrib.layers as layers
import threading


class QNetwork(object):
    def __init__(self, namespace, num_actions, agent_history_length,
                 resized_width, resized_height, learning_rate, dueling=False):
        assert isinstance(threading.current_thread(), threading._MainThread)

        # Inputs in NHWC format
        self._state_input = tf.placeholder(
            "float",
            [None, agent_history_length, resized_width, resized_height])
        self._action_input = tf.placeholder("int32", [None])
        self._y_input = tf.placeholder("float", [None])

        nhwc_input = tf.transpose(self._state_input, [0, 2, 3, 1])

        with tf.variable_scope(namespace):
            with tf.variable_scope('conv_layer1'):
                h_conv1 = layers.conv2d(nhwc_input, 32, [8, 8], [4, 4],
                                        padding='VALID')
            with tf.variable_scope('conv_layer2'):
                h_conv2 = layers.conv2d(h_conv1, 64, [4, 4], [2, 2],
                                        padding='VALID')
            with tf.variable_scope('conv_layer3'):
                h_conv3 = layers.conv2d(h_conv2, 64, [3, 3], [1, 1],
                                        padding='VALID')
            with tf.variable_scope('flatten_layer3'):
                h_flatten3 = layers.flatten(h_conv3)

            if dueling:
                h_state_fc4 = layers.fully_connected(
                    h_flatten3, num_outputs=512, activation_fn=tf.nn.relu)
                h_action_fc4 = layers.fully_connected(
                    h_flatten3, num_outputs=512, activation_fn=tf.nn.relu)
                h_state_fc5 = layers.fully_connected(
                    h_state_fc4, num_outputs=1, activation_fn=None)
                h_action_fc5 = layers.fully_connected(
                    h_action_fc4, num_outputs=num_actions, activation_fn=None)
                self._q_output = h_state_fc5 + h_action_fc5 - tf.reduce_mean(
                    h_action_fc5, 1, keep_dims=True)
            else:
                h_fc4 = layers.fully_connected(
                    h_flatten3, num_outputs=512, activation_fn=tf.nn.relu)
                self._q_output = layers.fully_connected(
                    h_fc4, num_outputs=num_actions, activation_fn=None)

        # Loss
        action_onehot = tf.one_hot(self._action_input, num_actions)
        q_action = tf.reduce_sum(tf.mul(self._q_output, action_onehot), 1)
        self._loss = tf.nn.l2_loss(q_action - self._y_input)
        self._optimizer = tf.train.AdamOptimizer(
            learning_rate).minimize(self._loss)

        self._trainable_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, namespace)

    def eval(self, session, state_batch):
        return self._q_output.eval(
            session=session,
            feed_dict={self._state_input: state_batch})

    def train(self, session, state_batch, action_batch, y_batch):
        _, loss_value = session.run([self._optimizer, self._loss], feed_dict={
            self._state_input: state_batch,
            self._action_input: action_batch,
            self._y_input: y_batch})
        return loss_value

    @property
    def network_params(self):
        return self._trainable_variables

    def get_update_network_params_op(self, source_dqn):
        target_network_params = self.network_params
        source_network_params = source_dqn.network_params
        assert len(target_network_params) > 0
        assert len(source_network_params) > 0
        return [target_network_params[i].assign(source_network_params[i])
                for i in range(len(target_network_params))]
