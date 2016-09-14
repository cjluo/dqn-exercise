import tensorflow as tf
import tensorflow.contrib.layers as layers
import threading


class A3CNetwork(object):
    def __init__(self, namespace, num_actions, agent_history_length,
                 resized_width, resized_height, learning_rate, beta,
                 grad_clip):
        assert isinstance(threading.current_thread(), threading._MainThread)

        self._namespace = namespace
        # Inputs in NHWC format
        self._state_input = tf.placeholder(
            "float",
            [None, agent_history_length, resized_width, resized_height])
        self._action_input = tf.placeholder("int32", [None])
        self._r_input = tf.placeholder("float", [None])

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

            with tf.variable_scope('value'):
                h_state_fc4 = layers.fully_connected(
                    h_flatten3, num_outputs=512, activation_fn=tf.nn.relu)
                h_state_fc5 = layers.fully_connected(
                    h_state_fc4, num_outputs=1, activation_fn=None)
                self._v_output = tf.reshape(h_state_fc5, [-1])

            with tf.variable_scope('policy'):
                h_action_fc4 = layers.fully_connected(
                    h_flatten3, num_outputs=512, activation_fn=tf.nn.relu)
                h_action_fc5 = layers.fully_connected(
                    h_action_fc4, num_outputs=num_actions, activation_fn=None)
                self._p_output = tf.nn.softmax(h_action_fc5)

        # Loss
        action_onehot = tf.one_hot(self._action_input, num_actions)

        log_p = tf.log(tf.clip_by_value(self._p_output, 1e-15, 1.0))

        entropy = -tf.reduce_sum(self._p_output * log_p, reduction_indices=1)

        log_p_action = tf.reduce_sum(
            tf.mul(log_p, action_onehot), reduction_indices=1)

        td = self._r_input - self._v_output

        p_loss = -tf.reduce_sum(log_p_action * td + entropy * beta)

        v_loss = 0.5 * tf.nn.l2_loss(td)

        loss = p_loss + v_loss
        self._optimizer = tf.train.RMSPropOptimizer(learning_rate)

        self._trainable_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, namespace)

        grads_and_vars = self._optimizer.compute_gradients(
            loss, self._trainable_variables)

        self._grads = [tf.clip_by_norm(gv[0], grad_clip)
                       for gv in grads_and_vars]

    def eval_p(self, session, state_batch):
        return self._p_output.eval(
            session=session,
            feed_dict={self._state_input: state_batch})

    def eval_v(self, session, state_batch):
        return self._v_output.eval(
            session=session,
            feed_dict={self._state_input: state_batch})

    def get_grads(self):
        return self._grads

    def apply_grads(self, grads):
        return self._optimizer.apply_gradients(zip(grads, self.network_params))

    def get_feed_dict(self, state_batch, action_batch, r_batch):
        return {self._state_input: state_batch,
                self._action_input: action_batch,
                self._r_input: r_batch}

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
