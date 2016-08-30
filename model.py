import tensorflow as tf
import keras.backend as K
from keras.layers import Convolution2D, Flatten, Dense, Input, Lambda, merge
from keras.models import Model
from keras.utils.visualize_util import plot

class QNetwork(object):
    def __init__(self, num_actions, agent_history_length, resized_width,
                 resized_height, learning_rate, dueling=False):
        # Inputs
        self._state_input = tf.placeholder(
            "float",
            [None, agent_history_length, resized_width, resized_height])
        self._action_input = tf.placeholder("int32", [None])
        self._y_input = tf.placeholder("float", [None])

        # Forward pass
        inputs = Input(shape=(agent_history_length,
                              resized_width, resized_height,))
        model = Convolution2D(nb_filter=16, nb_row=8, nb_col=8, subsample=(
            4, 4), activation='relu', border_mode='same')(inputs)
        model = Convolution2D(nb_filter=32, nb_row=4, nb_col=4, subsample=(
            2, 2), activation='relu', border_mode='same')(model)
        model = Flatten()(model)

        if dueling:
            state_model = Dense(output_dim=256, activation='relu')(model)
            action_model = Dense(output_dim=256, activation='relu')(model)
            state_value = Dense(output_dim=1, activation='linear')(state_model)
            action_value = Dense(
                output_dim=num_actions, activation='linear')(action_model)
            action_value_normalize = Lambda(
                lambda x: x - K.mean(x, axis=1, keepdims=True))(action_value)
            q_values = merge(
                [state_value, action_value_normalize],
                mode=lambda x: x[1] + x[0], output_shape=lambda x: x[1])
        else:
            model = Dense(output_dim=256, activation='relu')(model)
            q_values = Dense(
                output_dim=num_actions, activation='linear')(model)
        self._q_model = Model(
            input=inputs, output=q_values)
        plot(self._q_model, to_file='model.png')
        self._q_output = self._q_model(self._state_input)

        # Loss
        action_onehot = tf.one_hot(self._action_input, num_actions)
        q_action = tf.reduce_sum(tf.mul(self._q_output, action_onehot), 1)
        self._loss = tf.nn.l2_loss(q_action - self._y_input)
        self._optimizer = tf.train.AdamOptimizer(
            learning_rate).minimize(self._loss)

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
        return self._q_model.trainable_weights

    def update_network_params(self, session, source_dqn):
        target_network_params = self.network_params
        source_network_params = source_dqn.network_params
        update_target_network_params = [
            target_network_params[i].assign(source_network_params[i])
            for i in range(len(target_network_params))]
        session.run(update_target_network_params)
