import random


class ReplayMemory():
    ''' Replay memory D in article. '''

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        # self.buffer = [[row['state'], row['action'], row['reward'], row['n_state']] for _, row in data.iterrows()][-self.buffer_size:] TODO: empty or not?
        self.buffer = []

    def add(self, state, action, reward, n_state):
        self.buffer.append([state, action, reward, n_state])
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def size(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)