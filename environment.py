import gym
import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque


class Environment(object):
    def __init__(self, env_name, resized_width, resized_height,
                 agent_history_length, display, replay_size):
        self._env = gym.make(env_name)
        self._width = resized_width
        self._height = resized_height
        self._history_length = agent_history_length
        self._display = display
        self._state_buffer = deque(maxlen=replay_size)

    @property
    def action_size(self):
        return self._env.action_space.n

    def new_game(self):
        frame = self._get_frame(self._env.reset())

        state = {'frame': frame,
                 'action': 0,
                 'reward': 0,
                 'terminal': False,
                 'info': None}
        state['prev_state'] = state
        self._state = state

    def step(self, action):
        frame, reward, terminal, info = self._env.step(action)
        frame = self._get_frame(frame)

        new_state = {'frame': frame,
                     'reward': np.clip(reward, -1, 1),
                     'action': action,
                     'terminal': terminal,
                     'info': info,
                     'prev_state': self._state}

        self._state_buffer.append(new_state)
        self._state = new_state

        if self._display:
            self._env.render()
        return frame, reward, terminal, info

    def _get_frame(self, frame):
        return resize(rgb2gray(frame), (self._width, self._height))

    def sample(self, batch_size):
        buffer_size = len(self._state_buffer)
        if buffer_size < batch_size:
            return [], [], [], [], []
        else:
            batch = np.random.choice(self._state_buffer, batch_size)

            prev_state_batch = []
            action_batch = []
            reward_batch = []
            current_state_batch = []
            terminal_batch = []

            for state in batch:
                history = []
                current_state = state
                for _ in xrange(self._history_length + 1):
                    history.append(current_state['frame'])
                    current_state = current_state['prev_state']

                prev_state_batch.append(history[1:])
                current_state_batch.append(history[:-1])
                action_batch.append(state['action'])
                reward_batch.append(state['reward'])
                terminal_batch.append(state['terminal'])

            return prev_state_batch, action_batch, reward_batch,\
                current_state_batch, terminal_batch

    def get_frame_history(self):
        history = []
        current_state = self._state
        for _ in xrange(self._history_length):
            history.append(current_state['frame'])
            current_state = current_state['prev_state']
        return history
