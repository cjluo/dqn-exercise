import gym
import numpy as np
import cv2
from collections import deque


class Environment(object):
    def __init__(self, env_name, resized_width, resized_height,
                 agent_history_length, replay_size, alpha, reward_clip=1):
        self._env = gym.make(env_name)
        self._width = resized_width
        self._height = resized_height
        self._history_length = agent_history_length
        self._replay_size = replay_size
        self._state_buffer = deque(maxlen=replay_size)
        self._default_priority = 0
        self._alpha = alpha
        self._reward_clip = reward_clip

    @property
    def action_size(self):
        return self._env.action_space.n

    def new_game(self):
        frame = self._process_frame(self._env.reset())
        self._frames = [frame] * self._history_length

    def step(self, action):
        frame, reward, terminal, info = self._env.step(action)
        frame = self._process_frame(frame)
        if self._reward_clip > 0:
            reward = np.clip(reward, -1 * self._reward_clip, self._reward_clip)

        prev_frames = self._frames
        frames = prev_frames[1:] + [frame]
        self._frames = frames

        if self._replay_size > 0:
            self._state_buffer.append({
                'frames': frames,
                'prev_frames': prev_frames,
                'action': action,
                'reward': reward,
                'terminal': terminal,
                'priority': self._default_priority})

        return list(frames), reward, terminal, info

    def render(self):
        self._env.render()

    def _process_frame(self, frame):
        return cv2.resize(cv2.cvtColor(
            frame, cv2.COLOR_RGB2GRAY) / 255., (self._width, self._height))

    def _get_sample_probability(self):
        priority = np.zeros(len(self._state_buffer))
        i = 0
        for state in self._state_buffer:
            priority[i] = state['priority']
            if self._default_priority < priority[i]:
                self._default_priority = priority[i]
            i += 1

        probability = np.power(priority + 1e-7, self._alpha)
        return probability / np.sum(probability)

    def sample(self, batch_size):
        if self._replay_size < 0:
            raise Exception('replay_size = 0!')

        buffer_size = len(self._state_buffer)
        if buffer_size < batch_size:
            return [], [], [], [], [], []
        else:
            prev_frames_batch = []
            current_frames_batch = []
            action_batch = []
            reward_batch = []
            terminal_batch = []

            if self._alpha == 0:
                state_batch = np.random.choice(
                    self._state_buffer, batch_size)
            else:
                state_batch = np.random.choice(
                    self._state_buffer, batch_size,
                    p=self._get_sample_probability())

            for state in state_batch:
                prev_frames_batch.append(state['prev_frames'])
                current_frames_batch.append(state['frames'])
                action_batch.append(state['action'])
                reward_batch.append(state['reward'])
                terminal_batch.append(state['terminal'])

            return prev_frames_batch, action_batch, reward_batch,\
                current_frames_batch, terminal_batch, state_batch

    def get_frames(self):
        return list(self._frames)
